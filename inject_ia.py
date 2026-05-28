#!/usr/bin/env python
"""
inject_ia.py
============
Inject intrinsic alignments (IA) into a diffsky lightcone catalog and save
the IA-injected WL source sample to disk.

Running modes
-------------
MPI mode (default)
    mpirun -n N python inject_ia.py --data_dir /path/to/data \\
        --output_dir /path/to/output --z_min 0.5 --z_max 1.5 \\
        --central_alignment 1.0 --satellite_alignment 1.0

    Each MPI rank processes one z-slice; [z_min, z_max] is divided into N
    equal slices (one per rank).  High CPU and memory, short wall time.

Serial mode
    python inject_ia.py --serial --n_chunks N --data_dir /path/to/data \\
        --output_dir /path/to/output --z_min 0.5 --z_max 1.5 \\
        --central_alignment 1.0 --satellite_alignment 1.0

    A single process loops through N z-slices one at a time, loading and
    processing each chunk sequentially.  Low CPU, moderate memory, long
    wall time.

Output per chunk
----------------
  <output_dir>/catalog_z{z_lo:.3f}_{z_hi:.3f}_chunk{chunk:04d}.hdf5
  <output_dir>/figures/sanity_chunk{chunk:04d}_z{z_lo:.3f}_{z_hi:.3f}_*.png

Catalog columns saved
---------------------
  All loaded diffsky columns (unit-stripped), plus:
    synthetic            : bool — gal_id absent from synth_cores=False load
    valid_host_halo_shape: bool — host halo eigenvectors are usable
    galaxy_axisA/B/C_x/y/z : 3D orientation axes from IA injection
    disk_2d_*  / bulge_2d_* : projected 2D ellipse parameters
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
from jax import random as jran

try:
    from mpi4py import MPI
    _comm = MPI.COMM_WORLD
    _rank = _comm.Get_rank()
    _size = _comm.Get_size()
except ImportError:
    _comm = None
    _rank = 0
    _size = 1

import opencosmo as oc

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# modular_alignment lives in modular_alignments/; ellipse_proj_kernels_v2 at the top level
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "modular_alignments"))
from modular_alignment import align_to_halo, align_radially
from ellipse_proj_kernels_v2 import Ellipse2DParams, compute_ellipse2d, mc_mu_phi_gamma

# ─── Column selection ─────────────────────────────────────────────────────────

LOAD_COLUMNS = (
    "gal_id", "redshift", "redshift_true",
    "*bulge*", "*disk*",
    "central", "logm*",
    "roman_F129", "roman_F158",
    "top_host_idx", "top_host_infall_fof_halo_eigS*",
    "x", "y", "z", "ra*", "dec*",
    "r50_*", "*_host",
)

# ─── Utilities ────────────────────────────────────────────────────────────────

def _val(col):
    """Strip Astropy unit wrapper; return plain numpy array."""
    return col.value if hasattr(col, "value") else np.asarray(col)


def _log(rank, msg):
    print(f"[rank {rank:04d}] {msg}", flush=True)


# ─── Catalog helper functions ─────────────────────────────────────────────────

def mask_bad_halocat(
    data,
    fill_value=None,
    halo_axis_keys=(
        "top_host_infall_fof_halo_eigS3X",
        "top_host_infall_fof_halo_eigS3Y",
        "top_host_infall_fof_halo_eigS3Z",
    ),
    atol=1e-8,
):
    """Return True for galaxies with missing or fill-value host halo major axis."""
    bad = (
        np.isclose(_val(data[halo_axis_keys[0]]), 0, atol=atol)
        | np.isclose(_val(data[halo_axis_keys[1]]), 0, atol=atol)
        | np.isclose(_val(data[halo_axis_keys[2]]), 0, atol=atol)
        | (np.abs(_val(data[halo_axis_keys[0]])) > 100.0)
        | (np.abs(_val(data[halo_axis_keys[1]])) > 100.0)
        | (np.abs(_val(data[halo_axis_keys[2]])) > 100.0)
    )
    if fill_value is not None:
        bad |= (
            np.isclose(_val(data[halo_axis_keys[0]]), fill_value, atol=atol)
            | np.isclose(_val(data[halo_axis_keys[1]]), fill_value, atol=atol)
            | np.isclose(_val(data[halo_axis_keys[2]]), fill_value, atol=atol)
        )
    return bad


def select_wl_sample(dataset, J_depth=26.5, H_depth=26.5, snr_min=18.0, R_min=0.4):
    """
    Select weak-lensing source sample based on combined J+H SNR and spatial
    resolution factor R (see notebook for full derivation).

    Roman WFI PSF EE50 radii: F129=0.089", F158=0.103" (SCA01, field center).
    """
    EE50PSF_J = 0.089
    EE50PSF_H = 0.103

    data = dataset.data
    snr_J = 5.0 * 10.0 ** ((J_depth - _val(data["roman_F129"])) / 2.5)
    snr_H = 5.0 * 10.0 ** ((H_depth - _val(data["roman_F158"])) / 2.5)
    snr_JH = np.sqrt(snr_J**2 + snr_H**2)

    snr_J2, snr_H2 = snr_J**2, snr_H**2
    EE50PSF_eff2 = (snr_J2 * EE50PSF_J**2 + snr_H2 * EE50PSF_H**2) / (snr_J2 + snr_H2)

    # r50_disk is in comoving kpc (after .with_units("comoving"))
    r_eff = data["r50_disk"]
    comoving_dist = dataset.cosmology.comoving_distance(_val(data["redshift"]))
    # kpc / (Mpc * 1e3 kpc/Mpc) = dimensionless (radians) → arcsec
    r_eff_arcsec = _val((r_eff / comoving_dist).to('1')) * 206265.0

    R = 1.0 / (1.0 + EE50PSF_eff2 / r_eff_arcsec**2)
    return (snr_JH > snr_min) & (R > R_min)


# ─── ZXZ Euler angles ─────────────────────────────────────────────────────────

def euler_angles_ZXZ(A, B, C):
    """Recover ZXZ intrinsic Euler angles from orthonormal axis triad (A, B, C)."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    cos_beta = np.clip(C[..., 2], -1.0, 1.0)
    beta = np.arccos(cos_beta)
    sin_beta = np.sin(beta)
    no_lock = np.abs(sin_beta) > 1e-10
    alpha = np.where(no_lock,
                     np.arctan2(C[..., 0], -C[..., 1]),
                     np.arctan2(A[..., 1],  A[..., 0]))
    gamma = np.where(no_lock,
                     np.arctan2(A[..., 2], B[..., 2]),
                     0.0)
    return alpha, beta, gamma


def _ZXZ_matrix(alpha, beta, gamma):
    """Build ZXZ rotation matrix; inputs may be scalar or array (→ shape (3,3,N))."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta),  np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    return np.array([
        [ca * cg - sa * cb * sg,  -ca * sg - sa * cb * cg,  sa * sb],
        [sa * cg + ca * cb * sg,  -sa * sg + ca * cb * cg, -ca * sb],
        [sb * sg,                   sb * cg,                  cb   ],
    ])


# ─── IA injection ─────────────────────────────────────────────────────────────

def run_ia_injection(
    data,
    central_mask,
    satellite_mask,
    central_alignment,
    satellite_alignment,
    Lbox,
    envelope=True,
    random_alignment=False,
    rng_key=None,
):
    """
    Inject IA into a batch of galaxies.

    Parameters
    ----------
    data : Astropy-table-like
        Galaxy columns for the batch.
    central_mask, satellite_mask : bool arrays matching len(data)
    central_alignment, satellite_alignment : float
        DW alignment strength in [-1, 1].
    Lbox : float  Box size (same units as x/y/z).
    envelope : bool  True → Schur-complement 2D projection; False → z=0 slice.
    random_alignment : bool  Ignore alignment strengths; use random orientations.
    rng_key : JAX PRNGKey or None

    Returns
    -------
    galaxy_axes : dict[str, np.ndarray]  Keys galaxy_axisA/B/C_x/y/z.
    E2D_disk, E2D_bulge : Ellipse2DParams namedtuples.
    """
    n = len(_val(data["x"]))
    if rng_key is None:
        rng_key = jran.key(514)

    if random_alignment:
        mu_ran, phi_ran, gamma_ran = mc_mu_phi_gamma(n, rng_key)
        # _ZXZ_matrix broadcasts: result shape (3, 3, N)
        R = _ZXZ_matrix(
            np.array(phi_ran) + np.pi / 2.0,
            np.arccos(np.clip(np.array(mu_ran), -1.0, 1.0)),
            np.array(gamma_ran),
        )
        galaxy_axes = {
            f"galaxy_axis{ax}_{dim}": R[j, i]
            for i, ax in enumerate("ABC")
            for j, dim in enumerate("xyz")
        }
        mu_j, phi_j, gamma_j = mu_ran, phi_ran, gamma_ran
    else:
        # Central: align to host halo major axis
        central_axes = align_to_halo(
            _val(data["top_host_infall_fof_halo_eigS3X"][central_mask]),
            _val(data["top_host_infall_fof_halo_eigS3Y"][central_mask]),
            _val(data["top_host_infall_fof_halo_eigS3Z"][central_mask]),
            central_alignment,
            prim_gal_axis="A",
        )
        # Satellite: align radially toward host center
        satellite_axes = align_radially(
            _val(data["x_host"][satellite_mask]),
            _val(data["y_host"][satellite_mask]),
            _val(data["z_host"][satellite_mask]),
            _val(data["x"][satellite_mask]),
            _val(data["y"][satellite_mask]),
            _val(data["z"][satellite_mask]),
            [Lbox] * 3,
            satellite_alignment,
            prim_gal_axis="A",
        )
        galaxy_axes = {}
        for i, ax in enumerate("ABC"):
            for j, dim in enumerate("xyz"):
                col = f"galaxy_axis{ax}_{dim}"
                arr = np.zeros(n)
                arr[central_mask] = central_axes[i][:, j]
                arr[satellite_mask] = satellite_axes[i][:, j]
                galaxy_axes[col] = arr

        gal_A = np.stack([galaxy_axes["galaxy_axisA_x"],
                          galaxy_axes["galaxy_axisA_y"],
                          galaxy_axes["galaxy_axisA_z"]], axis=-1)
        gal_B = np.stack([galaxy_axes["galaxy_axisB_x"],
                          galaxy_axes["galaxy_axisB_y"],
                          galaxy_axes["galaxy_axisB_z"]], axis=-1)
        gal_C = np.stack([galaxy_axes["galaxy_axisC_x"],
                          galaxy_axes["galaxy_axisC_y"],
                          galaxy_axes["galaxy_axisC_z"]], axis=-1)
        alpha, beta, gamma_arr = euler_angles_ZXZ(gal_A, gal_B, gal_C)
        mu_j = np.cos(beta)
        phi_j = alpha - np.pi / 2.0
        gamma_j = gamma_arr

    a_disk = np.array(_val(data["r50_disk"]), dtype=float)
    b_disk = a_disk * np.array(_val(data["b_over_a_disk"]), dtype=float)
    c_disk = a_disk * np.array(_val(data["c_over_a_disk"]), dtype=float)

    a_bulge = np.array(_val(data["r50_bulge"]), dtype=float)
    b_bulge = a_bulge * np.array(_val(data["b_over_a_bulge"]), dtype=float)
    c_bulge = a_bulge * np.array(_val(data["c_over_a_bulge"]), dtype=float)

    E2D_disk  = compute_ellipse2d(a_disk,  b_disk,  c_disk,  mu_j, phi_j, gamma_j, envelop=envelope)
    E2D_bulge = compute_ellipse2d(a_bulge, b_bulge, c_bulge, mu_j, phi_j, gamma_j, envelop=envelope)

    return galaxy_axes, E2D_disk, E2D_bulge


def _combine_field(val_good, val_bad, good_mask, bad_mask, n_total):
    """Scatter val_good and val_bad into a full array of length n_total."""
    shape = (n_total,) + np.asarray(val_good).shape[1:]
    arr = np.zeros(shape, dtype=np.asarray(val_good).dtype)
    arr[good_mask] = np.asarray(val_good)
    if val_bad is not None:
        arr[bad_mask] = np.asarray(val_bad)
    return arr


def combine_ia_results(result_good, result_bad, good_mask, bad_mask, n_total):
    """
    Merge IA results from the good-halo and bad-halo subsets into full
    arrays of length n_total.
    """
    axes_good, E2D_disk_good, E2D_bulge_good = result_good
    axes_bad  = result_bad[0] if result_bad is not None else None
    e_disk_bad  = result_bad[1] if result_bad is not None else None
    e_bulge_bad = result_bad[2] if result_bad is not None else None

    combined_axes = {
        col: _combine_field(
            axes_good[col],
            axes_bad[col] if axes_bad else None,
            good_mask, bad_mask, n_total,
        )
        for col in axes_good
    }

    def _merge_ellipse(e_good, e_bad):
        return Ellipse2DParams(**{
            field: _combine_field(
                getattr(e_good, field),
                getattr(e_bad, field) if e_bad is not None else None,
                good_mask, bad_mask, n_total,
            )
            for field in Ellipse2DParams._fields
        })

    return combined_axes, _merge_ellipse(E2D_disk_good, e_disk_bad), _merge_ellipse(E2D_bulge_good, e_bulge_bad)


# ─── Output ───────────────────────────────────────────────────────────────────

def save_catalog(data, synthetic_flag, valid_halo_flag, galaxy_axes, E2D_disk, E2D_bulge, output_path):
    """Write the IA-injected WL catalog to HDF5."""
    try:
        colnames = data.colnames
    except AttributeError:
        colnames = list(data.keys())

    with h5py.File(output_path, "w") as f:
        grp = f.create_group("galaxies")

        for col in colnames:
            try:
                grp.create_dataset(col, data=_val(data[col]))
            except Exception as e:
                warnings.warn(f"Could not save column '{col}': {e}")

        grp.create_dataset("synthetic",             data=synthetic_flag.astype(bool))
        grp.create_dataset("valid_host_halo_shape", data=valid_halo_flag.astype(bool))

        for col, arr in galaxy_axes.items():
            grp.create_dataset(col, data=arr)

        for field in Ellipse2DParams._fields:
            grp.create_dataset(f"disk_2d_{field}",  data=np.asarray(getattr(E2D_disk,  field)))
            grp.create_dataset(f"bulge_2d_{field}", data=np.asarray(getattr(E2D_bulge, field)))


# ─── Sanity figures ───────────────────────────────────────────────────────────

def make_sanity_plots(
    data,
    good_mask,
    central_mask,
    satellite_mask,
    E2D_disk_noIA_slice,
    E2D_disk_noIA_env,
    E2D_disk_IA_env,
    E2D_bulge_noIA_slice,
    E2D_bulge_noIA_env,
    E2D_bulge_IA_env,
    gal_axes_IA_env,
    E2D_halo,
    fig_prefix,
    central_alignment,
    satellite_alignment,
    z_lo,
    z_hi,
):
    """
    Produce four sanity-check figures for the current redshift chunk.
    All E2D_* objects and gal_axes_IA_env cover only the good_mask subset.
    """
    central_mask_good = central_mask[good_mask]
    satellite_mask_good = satellite_mask[good_mask]
    rng = np.random.default_rng(42)

    # ── Fig 1: Host halo eigenvalue sizes ─────────────────────────────────────
    def _eig_size(key):
        return np.sqrt(
            _val(data[f"{key}X"])**2 +
            _val(data[f"{key}Y"])**2 +
            _val(data[f"{key}Z"])**2
        )
    base = "top_host_infall_fof_halo_eigS"
    S1 = _eig_size(f"{base}1")
    S2 = _eig_size(f"{base}2")
    S3 = _eig_size(f"{base}3")

    n_plot = min(50_000, len(S1))
    idx = rng.choice(len(S1), n_plot, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(S1[idx], S2[idx], s=1, alpha=0.3)
    axes[0].set(xlabel="S1 (Mpc)", ylabel="S2 (Mpc)", title="S1 vs S2")
    axes[0].plot([0, S1[idx].max()], [0, S1[idx].max()], "k--", lw=0.8)
    axes[1].scatter(S2[idx], S3[idx], s=1, alpha=0.3)
    axes[1].set(xlabel="S2 (Mpc)", ylabel="S3 (Mpc)", title="S2 vs S3")
    axes[1].plot([0, S2[idx].max()], [0, S2[idx].max()], "k--", lw=0.8)
    fig.suptitle(f"Host halo eigenvalue sizes  z=[{z_lo:.3f}, {z_hi:.3f})")
    fig.savefig(f"{fig_prefix}_halo_eigenvalues.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 2: Ellipticity distributions ──────────────────────────────────────
    ellip_bins = np.linspace(0, 1, 30)
    bc = 0.5 * (ellip_bins[:-1] + ellip_bins[1:])

    label_IA = f"IA cen={central_alignment:.1f} sat={satellite_alignment:.1f}"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, comp, e_noIA_s, e_noIA_e, e_IA_e in [
        (axes[0], "disk",  E2D_disk_noIA_slice,  E2D_disk_noIA_env,  E2D_disk_IA_env),
        (axes[1], "bulge", E2D_bulge_noIA_slice, E2D_bulge_noIA_env, E2D_bulge_IA_env),
    ]:
        ax.hist(_val(data[f"ellipticity_{comp}"])[good_mask],
                bins=ellip_bins, label="Diffsky 3D", color="gray", alpha=0.6)
        ax.plot(bc, np.histogram(np.asarray(e_noIA_s.ellipticity), bins=ellip_bins)[0],
                drawstyle="steps-mid", lw=2, label="No IA (slice)")
        ax.plot(bc, np.histogram(np.asarray(e_noIA_e.ellipticity), bins=ellip_bins)[0],
                drawstyle="steps-mid", lw=2, label="No IA (envelope)")
        ax.plot(bc, np.histogram(np.asarray(e_IA_e.ellipticity), bins=ellip_bins)[0],
                drawstyle="steps-mid", lw=2, label=label_IA)
        ax.legend(fontsize=8)
        ax.set(xlabel="2D Ellipticity", ylabel="Count",
               title=f"{comp.capitalize()} — z=[{z_lo:.3f},{z_hi:.3f})", xlim=(0, 1))
    fig.savefig(f"{fig_prefix}_ellipticity.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 3: Misalignment angle distribution ─────────────────────────────────
    psi_disk = np.asarray(E2D_disk_IA_env.psi)
    psi_halo = np.asarray(E2D_halo.psi)

    # 3D misalignment: central vs halo major axis
    cen_A = np.stack([gal_axes_IA_env[f"galaxy_axisA_{d}"][central_mask_good]
                      for d in "xyz"], axis=-1)
    halo_A_cen = np.stack([
        _val(data[f"{base}3X"][good_mask])[central_mask_good],
        _val(data[f"{base}3Y"][good_mask])[central_mask_good],
        _val(data[f"{base}3Z"][good_mask])[central_mask_good],
    ], axis=-1)
    cos_3D_cen = np.abs(
        np.einsum("ij,ij->i", cen_A, halo_A_cen)
        / (np.linalg.norm(cen_A, axis=1) * np.linalg.norm(halo_A_cen, axis=1) + 1e-30)
    )
    cos_2D_cen = np.abs(np.cos(psi_disk[central_mask_good] - psi_halo[central_mask_good]))

    # 3D misalignment: satellite vs radial vector to host
    sat_A = np.stack([gal_axes_IA_env[f"galaxy_axisA_{d}"][satellite_mask_good]
                      for d in "xyz"], axis=-1)
    dx = _val(data["x_host"][good_mask])[satellite_mask_good] - _val(data["x"][good_mask])[satellite_mask_good]
    dy = _val(data["y_host"][good_mask])[satellite_mask_good] - _val(data["y"][good_mask])[satellite_mask_good]
    dz = _val(data["z_host"][good_mask])[satellite_mask_good] - _val(data["z"][good_mask])[satellite_mask_good]
    rad_vec = np.stack([dx, dy, dz], axis=-1)
    cos_3D_sat = np.abs(
        np.einsum("ij,ij->i", sat_A, rad_vec)
        / (np.linalg.norm(sat_A, axis=1) * np.linalg.norm(rad_vec, axis=1) + 1e-30)
    )
    cos_2D_sat = np.abs(np.cos(psi_disk[satellite_mask_good] - psi_halo[satellite_mask_good]))

    mu_bins = np.linspace(0, 1, 100)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cos3d, cos2d, title in [
        (axes[0], cos_3D_cen, cos_2D_cen, f"Central (p={central_alignment:.1f})"),
        (axes[1], cos_3D_sat, cos_2D_sat, f"Satellite (p={satellite_alignment:.1f})"),
    ]:
        ax.hist(cos3d, bins=mu_bins, label="3D MA", alpha=0.7)
        ax.hist(cos2d, bins=mu_bins, label="2D MA", alpha=0.5)
        ax.set(xlabel=r"$|\cos\theta_\mathrm{MA}|$", title=title, xlim=(-0.05, 1.05))
        ax.legend()
    fig.suptitle(f"Misalignment angle  z=[{z_lo:.3f}, {z_hi:.3f})")
    fig.savefig(f"{fig_prefix}_misalignment.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 4: log(Mhost) vs redshift ─────────────────────────────────────────
    # Try common column name variants
    logm_col = None
    for candidate in ("logmp_obs_host", "logm_host", "logmvir_host"):
        try:
            _ = data[candidate]
            logm_col = candidate
            break
        except (KeyError, Exception):
            pass

    if logm_col is not None:
        n_plot = min(50_000, np.sum(good_mask))
        idx = rng.choice(np.sum(good_mask), n_plot, replace=False)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(
            _val(data["redshift"])[good_mask][idx],
            _val(data[logm_col])[good_mask][idx],
            s=1, alpha=0.3,
        )
        ax.set(xlabel="Redshift", ylabel=f"log(Mhost) [{logm_col}]",
               title=f"Host halo mass  z=[{z_lo:.3f},{z_hi:.3f})")
        fig.savefig(f"{fig_prefix}_logm_redshift.png", dpi=100, bbox_inches="tight")
        plt.close(fig)


# ─── Argument parser ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir",   required=True,
                   help="Directory containing lc_cores*.hdf5 files")
    p.add_argument("--output_dir", required=True,
                   help="Directory for output catalogs and figures")
    p.add_argument("--z_min",  type=float, default=0.0,
                   help="Minimum redshift of the range to process (default 0.0)")
    p.add_argument("--z_max",  type=float, default=3.0,
                   help="Maximum redshift of the range to process (default 3.0)")
    p.add_argument("--central_alignment",   type=float, default=1.0,
                   help="DW alignment strength for central galaxies in [-1,1] (default 1.0)")
    p.add_argument("--satellite_alignment", type=float, default=1.0,
                   help="DW alignment strength for satellite galaxies in [-1,1] (default 1.0)")
    p.add_argument("--J_depth",  type=float, default=27.4,
                   help="Roman F129 5σ AB depth (default 27.4)")
    p.add_argument("--H_depth",  type=float, default=27.4,
                   help="Roman F158 5σ AB depth (default 27.4)")
    p.add_argument("--snr_min",  type=float, default=18.0,
                   help="Min combined J+H SNR for WL selection (default 18.0)")
    p.add_argument("--R_min",    type=float, default=0.4,
                   help="Min resolution factor R for WL selection (default 0.4)")
    p.add_argument("--seed",     type=int,   default=42,
                   help="Base random seed for JAX operations (default 42)")
    p.add_argument("--no_envelope", action="store_true",
                   help="Use z=0 cross-section projection instead of envelope (default: envelope)")
    p.add_argument("--serial", action="store_true",
                   help="Run in serial mode: one process loops through chunks sequentially "
                        "(no MPI required)")
    p.add_argument("--n_chunks", type=int, default=1,
                   help="Number of redshift chunks to use in serial mode (default 1)")
    return p.parse_args()


# ─── Per-chunk processing ─────────────────────────────────────────────────────

def process_chunk(z_lo, z_hi, chunk_id, total_chunks, args, files, envelope):
    """
    Load, select, and inject IA for a single redshift slice [z_lo, z_hi).

    Parameters
    ----------
    z_lo, z_hi : float  Redshift bounds for this chunk.
    chunk_id   : int    Index of this chunk (used for file/figure naming and logging).
    total_chunks : int  Total number of chunks (for log messages only).
    args       : argparse.Namespace  Parsed CLI arguments.
    files      : list[Path]  Sorted list of input HDF5 files.
    envelope   : bool   True → Schur-complement 2D projection; False → z=0 slice.
    """
    _log(chunk_id, f"Processing z=[{z_lo:.4f}, {z_hi:.4f})  "
                   f"(chunk {chunk_id}/{total_chunks})")

    # ── Load synth_cores=False to build the set of real gal_ids ───────────
    _log(chunk_id, "Loading synth_cores=False for gal_id...")
    t0 = time.time()
    ds_real = oc.open(files, synth_cores=False)
    ds_real = ds_real.select("gal_id")
    ds_real = ds_real.with_redshift_range(z_lo, z_hi)
    ds_real = ds_real.with_units("comoving")
    _data_real = ds_real.data
    gal_id_col = _data_real["gal_id"] if hasattr(_data_real, "colnames") else _data_real
    real_gal_ids = set(_val(gal_id_col).tolist())
    del ds_real
    _log(chunk_id, f"  {len(real_gal_ids):,} real gal_ids  ({time.time()-t0:.1f}s)")

    # ── Load full catalog (synth_cores=True) ──────────────────────────────
    _log(chunk_id, "Loading full catalog (synth_cores=True)...")
    t0 = time.time()
    dataset = oc.open(files, synth_cores=True)
    dataset = dataset.select(*LOAD_COLUMNS)
    dataset = dataset.with_redshift_range(z_lo, z_hi)
    dataset = dataset.with_units("comoving")
    n_before_wl = len(dataset)
    _log(chunk_id, f"  {n_before_wl:,} galaxies before WL cut  ({time.time()-t0:.1f}s)")

    # ── WL selection ──────────────────────────────────────────────────────
    _log(chunk_id, "Applying WL selection...")
    wl_mask = select_wl_sample(
        dataset,
        J_depth=args.J_depth, H_depth=args.H_depth,
        snr_min=args.snr_min, R_min=args.R_min,
    )
    dataset = dataset.with_new_columns(wl_sample=wl_mask)
    dataset = dataset.filter(oc.col("wl_sample") == True)
    n = len(dataset)
    _log(chunk_id, f"  {n:,} WL-selected ({100*n/n_before_wl:.1f}%)")

    data = dataset.data

    # ── Compute flags ─────────────────────────────────────────────────────
    _log(chunk_id, "Computing synthetic and valid_host_halo_shape flags...")
    gal_ids = _val(data["gal_id"])
    synthetic_flag  = np.array([gid not in real_gal_ids for gid in gal_ids])
    bad_halo_flag   = mask_bad_halocat(data, fill_value=-1.4779781)
    valid_halo_flag = ~bad_halo_flag

    central_mask   = (_val(data["central"]).astype(int) == 1)
    satellite_mask = ~central_mask

    n_synth = np.sum(synthetic_flag)
    n_bad   = np.sum(bad_halo_flag)
    _log(chunk_id, f"  synthetic={n_synth:,} ({100*n_synth/n:.1f}%)  "
                   f"bad_halo={n_bad:,} ({100*n_bad/n:.1f}%)")
    _log(chunk_id, f"  central={np.sum(central_mask):,}  "
                   f"satellite={np.sum(satellite_mask):,}")

    Lbox = np.inf  # lightcone → no periodic wrapping needed for radial vectors

    good_mask = valid_halo_flag
    bad_mask  = bad_halo_flag

    # ── IA injection ──────────────────────────────────────────────────────
    _log(chunk_id, f"Running IA injection (envelope={envelope})...")
    t0 = time.time()

    data_good = data[good_mask]
    cen_good  = central_mask[good_mask]
    sat_good  = satellite_mask[good_mask]

    axes_noIA_slice, E2D_disk_noIA_slice, E2D_bulge_noIA_slice = run_ia_injection(
        data_good, cen_good, sat_good,
        0.0, 0.0, Lbox,
        envelope=False, rng_key=jran.key(args.seed),
    )
    axes_noIA_env, E2D_disk_noIA_env, E2D_bulge_noIA_env = run_ia_injection(
        data_good, cen_good, sat_good,
        0.0, 0.0, Lbox,
        envelope=True, rng_key=jran.key(args.seed + 10),
    )
    axes_IA_env, E2D_disk_IA_env, E2D_bulge_IA_env = run_ia_injection(
        data_good, cen_good, sat_good,
        args.central_alignment, args.satellite_alignment, Lbox,
        envelope=envelope, rng_key=jran.key(args.seed + 20),
    )

    result_bad = None
    if np.sum(bad_mask) > 0:
        data_bad = data[bad_mask]
        result_bad = run_ia_injection(
            data_bad, central_mask[bad_mask], satellite_mask[bad_mask],
            0.0, 0.0, Lbox,
            envelope=envelope, random_alignment=True,
            rng_key=jran.key(args.seed + 30),
        )

    _log(chunk_id, f"  IA done in {time.time()-t0:.1f}s")

    # ── Halo 2D shape (for misalignment figure) ───────────────────────────
    base = "top_host_infall_fof_halo_eigS"
    hA = np.stack([_val(data[f"{base}3X"][good_mask]),
                   _val(data[f"{base}3Y"][good_mask]),
                   _val(data[f"{base}3Z"][good_mask])], axis=-1)
    hB = np.stack([_val(data[f"{base}2X"][good_mask]),
                   _val(data[f"{base}2Y"][good_mask]),
                   _val(data[f"{base}2Z"][good_mask])], axis=-1)
    hC = np.stack([_val(data[f"{base}1X"][good_mask]),
                   _val(data[f"{base}1Y"][good_mask]),
                   _val(data[f"{base}1Z"][good_mask])], axis=-1)
    alpha_h, beta_h, gamma_h = euler_angles_ZXZ(hA, hB, hC)
    E2D_halo = compute_ellipse2d(
        np.sqrt(np.sum(hA**2, axis=-1)),
        np.sqrt(np.sum(hB**2, axis=-1)),
        np.sqrt(np.sum(hC**2, axis=-1)),
        np.cos(beta_h), alpha_h - np.pi / 2.0, gamma_h,
        envelop=True,
    )

    # ── Combine good + bad for the saved catalog ───────────────────────────
    result_good = (axes_IA_env, E2D_disk_IA_env, E2D_bulge_IA_env)
    galaxy_axes, E2D_disk_out, E2D_bulge_out = combine_ia_results(
        result_good, result_bad, good_mask, bad_mask, n,
    )

    # ── Create output directories ─────────────────────────────────────────
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Sanity figures ────────────────────────────────────────────────────
    _log(chunk_id, "Generating sanity figures...")
    fig_prefix = str(fig_dir / f"sanity_chunk{chunk_id:04d}_z{z_lo:.3f}_{z_hi:.3f}")
    make_sanity_plots(
        data,
        good_mask,
        central_mask,
        satellite_mask,
        E2D_disk_noIA_slice, E2D_disk_noIA_env, E2D_disk_IA_env,
        E2D_bulge_noIA_slice, E2D_bulge_noIA_env, E2D_bulge_IA_env,
        axes_IA_env,
        E2D_halo,
        fig_prefix,
        args.central_alignment,
        args.satellite_alignment,
        z_lo,
        z_hi,
    )

    # ── Save catalog ──────────────────────────────────────────────────────
    catalog_path = out_dir / f"catalog_z{z_lo:.3f}_{z_hi:.3f}_chunk{chunk_id:04d}.hdf5"
    _log(chunk_id, f"Saving catalog → {catalog_path}")
    save_catalog(data, synthetic_flag, valid_halo_flag, galaxy_axes,
                 E2D_disk_out, E2D_bulge_out, catalog_path)
    _log(chunk_id, f"Done. {n:,} galaxies written.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    envelope = not args.no_envelope

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("lc_cores*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No lc_cores*.hdf5 found in {data_dir}")

    if args.serial:
        # ── Serial mode: one process, N chunks processed sequentially ─────
        n_chunks = args.n_chunks
        print(f"[serial] Serial mode: {n_chunks} chunk(s) over "
              f"z=[{args.z_min}, {args.z_max}]", flush=True)
        print(f"[serial] Found {len(files)} input file(s)", flush=True)
        dz = (args.z_max - args.z_min) / n_chunks
        for chunk_id in range(n_chunks):
            z_lo = args.z_min + chunk_id * dz
            z_hi = args.z_min + (chunk_id + 1) * dz
            t_chunk = time.time()
            process_chunk(z_lo, z_hi, chunk_id, n_chunks, args, files, envelope)
            print(f"[serial] Chunk {chunk_id}/{n_chunks} finished in "
                  f"{time.time()-t_chunk:.1f}s", flush=True)
    else:
        # ── MPI mode: each rank handles one z-slice in parallel ───────────
        rank = _rank
        size = _size
        if rank == 0:
            print(f"[mpi] MPI mode: {size} rank(s) over "
                  f"z=[{args.z_min}, {args.z_max}]", flush=True)
            print(f"[mpi] Found {len(files)} input file(s)", flush=True)
        dz = (args.z_max - args.z_min) / size
        z_lo = args.z_min + rank * dz
        z_hi = args.z_min + (rank + 1) * dz
        process_chunk(z_lo, z_hi, rank, size, args, files, envelope)


if __name__ == "__main__":
    main()
