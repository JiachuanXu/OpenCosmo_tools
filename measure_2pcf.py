#!/usr/bin/env python
"""
measure_2pcf.py
===============
Measure projected two-point correlation functions (w_gg, w_g+, w_++) from
IA-injected galaxy catalogs produced by inject_ia.py.

Usage
-----
    python measure_2pcf.py --input_dir /path/to/catalogs \\
        --data_dir /path/to/lc_cores_files \\
        --output_dir /path/to/output \\
        --z_min 0.5 --z_max 1.0 \\
        [--num_threads 8] [--patch_threshold 500000] [--n_patches 50]

Input catalog naming convention
--------------------------------
    catalog_z{z_lo:.3f}_{z_hi:.3f}_chunk{chunk:04d}.hdf5

Chunks whose z-range overlaps [z_min, z_max) are loaded and concatenated.

Galaxy selection
----------------
* Satellites  : all (radial alignment is always valid)
* Centrals    : only those with ``valid_host_halo_shape == True``

Measurements
------------
w_gg  : cen-cen, cen-sat, sat-sat
w_g+  : cen-cen, sat-sat, cen-sat, sat-cen, halo-halo
w_++  : cen-cen, sat-sat, cen-sat, halo-halo

Output files
------------
  <output_dir>/wgg_{pair}.txt
  <output_dir>/wgp_{pair}.txt
  <output_dir>/wpp_{pair}.txt
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import h5py
import treecorr
import opencosmo as oc

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "modular_alignments"))
from ellipse_proj_kernels_v2 import compute_ellipse2d

from random_catalog import make_random_catalog


# ── cosmology ─────────────────────────────────────────────────────────────────

def load_cosmology(data_dir):
    """
    Read the fiducial cosmology from the original diffsky lc_cores*.hdf5 files.

    The cosmology is stored as metadata in the opencosmo dataset and is
    retrieved via ``dataset.cosmology``, which returns an astropy cosmology
    object.  This mirrors how inject_ia.py accesses it via
    ``dataset.cosmology.comoving_distance(...)``.
    """
    files = sorted(Path(data_dir).glob("lc_cores*.hdf5"))
    if not files:
        raise FileNotFoundError(
            f"No lc_cores*.hdf5 files found in {data_dir}. "
            "Pass the directory of the original diffsky mock via --data_dir."
        )
    cosmo = oc.open(files[0]).cosmology
    print(f"Cosmology loaded from {files[0].name}: {cosmo}")
    return cosmo


# ── filename parsing ──────────────────────────────────────────────────────────

_CHUNK_RE = re.compile(r"catalog_z(\d+\.\d+)_(\d+\.\d+)_chunk(\d+)\.hdf5$")


def _parse_chunk_zrange(path):
    """Return (z_lo, z_hi) parsed from the catalog filename, or None."""
    m = _CHUNK_RE.search(Path(path).name)
    if m is None:
        return None
    return float(m.group(1)), float(m.group(2))


def _find_overlapping_chunks(input_dir, z_min, z_max):
    """Return sorted paths of chunks whose z-range overlaps [z_min, z_max)."""
    files = sorted(Path(input_dir).glob("catalog_z*.hdf5"))
    out = []
    for f in files:
        zr = _parse_chunk_zrange(f)
        if zr is None:
            continue
        z_lo, z_hi = zr
        if z_lo < z_max and z_hi > z_min:
            out.append(f)
    return out


# ── HDF5 loading ──────────────────────────────────────────────────────────────

_NEEDED_COLS = [
    "redshift",
    "ra_obs", "dec_obs",
    "central",
    "valid_host_halo_shape",
    "disk_2d_ellipticity",  "disk_2d_psi",
    "bulge_2d_ellipticity", "bulge_2d_psi",
    "top_host_infall_fof_halo_eigS1X", "top_host_infall_fof_halo_eigS1Y", "top_host_infall_fof_halo_eigS1Z",
    "top_host_infall_fof_halo_eigS2X", "top_host_infall_fof_halo_eigS2Y", "top_host_infall_fof_halo_eigS2Z",
    "top_host_infall_fof_halo_eigS3X", "top_host_infall_fof_halo_eigS3Y", "top_host_infall_fof_halo_eigS3Z",
]


def _load_chunk(path, z_min, z_max):
    """Load required columns from one HDF5 chunk, filtered to [z_min, z_max)."""
    with h5py.File(path, "r") as f:
        grp = f["galaxies"]
        z = grp["redshift"][:]
        mask = (z >= z_min) & (z < z_max)
        if not np.any(mask):
            return None
        out = {}
        for col in _NEEDED_COLS:
            if col in grp:
                out[col] = grp[col][:][mask]
            else:
                print(f"  [warn] column '{col}' missing in {Path(path).name}")
    return out


def load_catalog(input_dir, z_min, z_max):
    """Load and concatenate all chunks that overlap [z_min, z_max)."""
    files = _find_overlapping_chunks(input_dir, z_min, z_max)
    if not files:
        raise FileNotFoundError(
            f"No catalog chunks found in {input_dir} overlapping z=[{z_min:.3f}, {z_max:.3f})"
        )
    print(f"Found {len(files)} overlapping chunk(s) for z=[{z_min:.3f}, {z_max:.3f})")
    parts = []
    for f in files:
        t0 = time.time()
        chunk = _load_chunk(f, z_min, z_max)
        if chunk is not None:
            parts.append(chunk)
            print(f"  {f.name}: {len(chunk['redshift']):,} galaxies  ({time.time()-t0:.1f}s)")
        else:
            print(f"  {f.name}: no galaxies in target range, skipping")
    if not parts:
        raise ValueError(f"No galaxies found in z=[{z_min:.3f}, {z_max:.3f})")
    return {col: np.concatenate([p[col] for p in parts]) for col in parts[0]}


# ── galaxy selection ──────────────────────────────────────────────────────────

def select_galaxies(cat):
    """
    Return boolean masks for aligned centrals and satellites.

    Satellites: all — radial alignment is always valid.
    Centrals:   only those with valid_host_halo_shape == True.
    """
    is_central = cat["central"].astype(bool)
    valid_halo = cat["valid_host_halo_shape"].astype(bool)
    central_mask = is_central & valid_halo
    satellite_mask = ~is_central
    return central_mask, satellite_mask


# ── halo 2D shape ─────────────────────────────────────────────────────────────

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


def compute_halo_2d_shape(cat, mask=None):
    """
    Project host halo eigenvectors onto the sky plane and return Ellipse2DParams.
    Only applied to the subset selected by `mask` (defaults to all rows).
    """
    def _vec(base):
        X = cat[f"{base}X"].astype(float)
        Y = cat[f"{base}Y"].astype(float)
        Z = cat[f"{base}Z"].astype(float)
        if mask is not None:
            X, Y, Z = X[mask], Y[mask], Z[mask]
        return np.stack([X, Y, Z], axis=-1)

    hA = _vec("top_host_infall_fof_halo_eigS3")   # major (largest)
    hB = _vec("top_host_infall_fof_halo_eigS2")   # intermediate
    hC = _vec("top_host_infall_fof_halo_eigS1")   # minor

    alpha_h, beta_h, gamma_h = euler_angles_ZXZ(hA, hB, hC)
    return compute_ellipse2d(
        np.sqrt(np.sum(hA**2, axis=-1)),
        np.sqrt(np.sum(hB**2, axis=-1)),
        np.sqrt(np.sum(hC**2, axis=-1)),
        np.cos(beta_h), alpha_h - np.pi / 2.0, gamma_h,
        envelop=True,
    )


# ── treecorr catalog helpers ──────────────────────────────────────────────────

def _make_catalog(ra_deg, dec_deg, r_mpc, g1=None, g2=None, patch_centers=None, npatch=None):
    """
    Build a treecorr.Catalog.

    ra_deg / dec_deg are in degrees.  Shapes (g1, g2) are optional.
    Pass patch_centers (from a previously built catalog) OR npatch to trigger
    the patch formalism; patch_centers takes precedence.
    """
    kwargs = dict(
        ra=np.radians(ra_deg),
        dec=np.radians(dec_deg),
        r=r_mpc,
        ra_units="radians",
        dec_units="radians",
    )
    if g1 is not None:
        kwargs["g1"] = g1
        kwargs["g2"] = g2
    if patch_centers is not None:
        kwargs["patch_centers"] = patch_centers
    elif npatch is not None and npatch > 1:
        kwargs["npatch"] = npatch
    return treecorr.Catalog(**kwargs)


def build_patch_centers(ra_deg, dec_deg, r_mpc, npatch):
    """Compute k-means patch centres from the combined valid-galaxy sample."""
    print(f"  Computing {npatch} patch centres from {len(ra_deg):,} positions...")
    seed_cat = treecorr.Catalog(
        ra=np.radians(ra_deg), dec=np.radians(dec_deg), r=r_mpc,
        ra_units="radians", dec_units="radians",
        npatch=npatch,
    )
    return seed_cat.patch_centers


# ── treecorr measurement helpers ─────────────────────────────────────────────

def _validate_bin_slop(bin_slop, min_sep, max_sep, nbins):
    """
    Validate bin_slop against its documented valid range and accuracy threshold.

    bin_slop controls how much a pair's separation is allowed to differ from
    the true bin edge before it is assigned to an adjacent bin.  Concretely,
    the tolerance distance is  b = bin_size * bin_slop  where
    bin_size = log(max_sep / min_sep) / nbins  (in log-space).

    Valid range  : bin_slop >= 0.  Negative values are rejected here because
                   TreeCorr would silently treat them as a request for its
                   internal auto-default, which hides user intent.
    Accuracy note: TreeCorr emits its own warning when bin_slop exceeds
                   max_good_slop ~ 0.1 / bin_size.  As a rule of thumb,
                   values above 1.0 risk significant inaccuracies.
    """
    if bin_slop < 0.0:
        raise ValueError(
            f"bin_slop must be >= 0.0, got {bin_slop:.4g}.  "
            "Negative values would trigger TreeCorr's internal auto-default, "
            "masking the true bin_slop in use.  Pass 0.0 for exact computation "
            "or a positive value for the desired tolerance fraction."
        )
    import math
    bin_size = math.log(max_sep / min_sep) / nbins   # log-space bin width
    max_good_slop = 0.1 / bin_size
    if bin_slop > max_good_slop:
        print(
            f"[warn] bin_slop={bin_slop:.4g} exceeds max_good_slop={max_good_slop:.4g} "
            f"(= 0.1 / bin_size where bin_size={bin_size:.4g}).  "
            "TreeCorr may produce significant inaccuracies at this setting."
        )


def _treecorr_cfg(args):
    return dict(
        min_sep=args.min_sep,
        max_sep=args.max_sep,
        nbins=args.nbins,
        bin_slop=args.bin_slop,
        metric="Rperp",
        min_rpar=-100.0,
        max_rpar=100.0,
    )


def measure_wgg(cat1, rand, cfg, out_path, num_threads, cat2=None):
    """Landy-Szalay w_gg estimator for auto- or cross-correlations."""
    cat2_eff = cat2 if cat2 is not None else cat1
    dd = treecorr.NNCorrelation(**cfg)
    rr = treecorr.NNCorrelation(**cfg)
    dr = treecorr.NNCorrelation(**cfg)
    rd = treecorr.NNCorrelation(**cfg)
    dd.process(cat1, cat2_eff, num_threads=num_threads)
    rr.process(rand, rand,     num_threads=num_threads)
    dr.process(cat1, rand,     num_threads=num_threads)
    rd.process(rand, cat2_eff, num_threads=num_threads)
    dd.write(str(out_path), rr=rr, dr=dr, rd=rd,
             file_type="ASCII", write_patch_results=False, write_cov=False)


def measure_wgp(pos_cat, rand, shape_cat, cfg, out_path, num_threads):
    """w_g+ estimator: density (pos_cat) cross shape (shape_cat)."""
    dg = treecorr.NGCorrelation(**cfg)
    rg = treecorr.NGCorrelation(**cfg)
    dg.process(pos_cat, shape_cat, num_threads=num_threads)
    rg.process(rand,    shape_cat, num_threads=num_threads)
    dg.write(str(out_path), rg=rg,
             file_type="ASCII", write_patch_results=False, write_cov=False)


def measure_wpp(shape_cat1, cfg, out_path, num_threads, shape_cat2=None):
    """w_++ estimator: shape auto- or cross-correlation."""
    pp = treecorr.GGCorrelation(**cfg)
    if shape_cat2 is not None:
        pp.process(shape_cat1, shape_cat2, num_threads=num_threads)
    else:
        pp.process(shape_cat1, num_threads=num_threads)
    pp.write(str(out_path),
             file_type="ASCII", write_patch_results=False, write_cov=False)


# ── argument parser ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_dir",  required=True,
                   help="Directory containing catalog_z*.hdf5 files from inject_ia.py")
    p.add_argument("--data_dir",   required=True,
                   help="Directory containing the original lc_cores*.hdf5 diffsky mock files "
                        "(used to read the fiducial cosmology)")
    p.add_argument("--output_dir", required=True,
                   help="Directory for output correlation function text files")
    p.add_argument("--z_min", type=float, default=0.0,
                   help="Minimum redshift of galaxies to include (default 0.0)")
    p.add_argument("--z_max", type=float, default=3.0,
                   help="Maximum redshift of galaxies to include (default 3.0)")
    # TreeCorr binning
    p.add_argument("--min_sep", type=float, default=0.1,
                   help="Minimum projected separation [Mpc] (default 0.1)")
    p.add_argument("--max_sep", type=float, default=120.0,
                   help="Maximum projected separation [Mpc] (default 120.0)")
    p.add_argument("--nbins",   type=int,   default=20,
                   help="Number of log-spaced r_p bins (default 20)")
    p.add_argument("--bin_slop", type=float, default=0.0,
                   help="TreeCorr bin_slop: fraction of a bin width by which pair separations "
                        "may be misassigned to adjacent bins.  0 = exact (slowest); larger "
                        "values trade accuracy for speed.  Must be >= 0; values above 1.0 "
                        "risk significant inaccuracies and trigger a warning.  (default 0.0)")
    # Performance
    p.add_argument("--shape", choices=["disk", "bulge"], default="disk",
                   help="Galaxy shape component to use for 2PCF: disk or bulge (default disk)")
    p.add_argument("--num_threads",     type=int, default=8,
                   help="Number of OpenMP threads for TreeCorr (default 8)")
    p.add_argument("--patch_threshold", type=int, default=500_000,
                   help="Switch to patch formalism when N_gal > this value (default 500000)")
    p.add_argument("--n_patches",       type=int, default=50,
                   help="Number of sky patches when patch formalism is active (default 50)")
    return p.parse_args()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    _validate_bin_slop(args.bin_slop, args.min_sep, args.max_sep, args.nbins)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cosmo = load_cosmology(args.data_dir)

    # ── Load catalog ──────────────────────────────────────────────────────────
    print(f"\nLoading catalog from {args.input_dir}  z=[{args.z_min:.3f}, {args.z_max:.3f})")
    t0 = time.time()
    cat = load_catalog(args.input_dir, args.z_min, args.z_max)
    n_total = len(cat["redshift"])
    print(f"Total loaded: {n_total:,} galaxies  ({time.time()-t0:.1f}s)")

    # ── Galaxy selection ──────────────────────────────────────────────────────
    central_mask, satellite_mask = select_galaxies(cat)
    n_cen = int(np.sum(central_mask))
    n_sat = int(np.sum(satellite_mask))
    n_valid = n_cen + n_sat
    print(f"Valid centrals (valid_host_halo_shape): {n_cen:,}")
    print(f"Satellites:                             {n_sat:,}")
    print(f"Total for 2PCF measurement:             {n_valid:,}")

    # ── Comoving distances ────────────────────────────────────────────────────
    r_all = cosmo.comoving_distance(cat["redshift"].astype(float)).value  # Mpc

    shape_tag = args.shape
    print(f"\nGalaxy shape component: {shape_tag}")

    # ── Halo 2D shapes (centrals only — halo catalog) ─────────────────────────
    print("Computing halo 2D shapes for central galaxies...")
    E2D_halo = compute_halo_2d_shape(cat, mask=central_mask)
    g1_halo = np.asarray(E2D_halo.ellipticity) * np.cos(2.0 * np.asarray(E2D_halo.psi))
    g2_halo = np.asarray(E2D_halo.ellipticity) * np.sin(2.0 * np.asarray(E2D_halo.psi))

    # ── Galaxy 2D shapes (disk or bulge) ──────────────────────────────────────
    e_gal   = cat[f"{shape_tag}_2d_ellipticity"].astype(float)
    psi_gal = cat[f"{shape_tag}_2d_psi"].astype(float)
    g1_gal  = e_gal * np.cos(2.0 * psi_gal)
    g2_gal  = e_gal * np.sin(2.0 * psi_gal)

    ra  = cat["ra_obs"].astype(float)   # degrees
    dec = cat["dec_obs"].astype(float)  # degrees

    # ── Patch setup ───────────────────────────────────────────────────────────
    use_patches = n_valid > args.patch_threshold
    patch_centers = None
    if use_patches:
        print(f"\nN_valid={n_valid:,} > threshold={args.patch_threshold:,}; "
              f"using {args.n_patches} patches.")
        valid_mask = central_mask | satellite_mask
        patch_centers = build_patch_centers(
            ra[valid_mask], dec[valid_mask], r_all[valid_mask],
            args.n_patches,
        )
    else:
        print(f"\nN_valid={n_valid:,} ≤ threshold={args.patch_threshold:,}; "
              f"no patches needed.")

    pc = patch_centers  # shorthand

    # ── Build TreeCorr catalogs ───────────────────────────────────────────────
    print("\nBuilding TreeCorr catalogs...")

    # Central galaxies with selected shape component
    cen_cat = _make_catalog(
        ra[central_mask], dec[central_mask], r_all[central_mask],
        g1=g1_gal[central_mask], g2=g2_gal[central_mask],
        patch_centers=pc,
    )
    # Satellite galaxies with selected shape component
    sat_cat = _make_catalog(
        ra[satellite_mask], dec[satellite_mask], r_all[satellite_mask],
        g1=g1_gal[satellite_mask], g2=g2_gal[satellite_mask],
        patch_centers=pc,
    )
    # Central galaxies with halo shape (for wgp / wpp halo-halo)
    halo_cat = _make_catalog(
        ra[central_mask], dec[central_mask], r_all[central_mask],
        g1=g1_halo, g2=g2_halo,
        patch_centers=pc,
    )

    # Random catalog (1× the valid galaxy count, matching survey footprint)
    print("Building random catalog...")
    t0 = time.time()
    valid_mask = central_mask | satellite_mask
    ra_rand, dec_rand, r_rand = make_random_catalog(
        ra[valid_mask], dec[valid_mask], r_all[valid_mask],
        n_randoms=int(n_valid),
        n_ang_bins=20,
        n_r_bins=15,
        unit="deg",
        seed=42,
    )
    print(f"  {len(ra_rand):,} random points  ({time.time()-t0:.1f}s)")
    rand_cat = _make_catalog(ra_rand, dec_rand, r_rand, patch_centers=pc)

    cfg = _treecorr_cfg(args)
    nt  = args.num_threads

    # ── w_gg ──────────────────────────────────────────────────────────────────
    print("\n── w_gg ─────────────────────────────────────────────────────────────")
    for label, c1, c2 in [
        ("cen_cen", cen_cat, None),
        ("cen_sat", cen_cat, sat_cat),
        ("sat_sat", sat_cat, None),
    ]:
        print(f"  -> {label}")
        t0 = time.time()
        measure_wgg(c1, rand_cat, cfg, out_dir / f"wgg_{label}_{shape_tag}.txt", nt, cat2=c2)
        print(f"     {time.time()-t0:.1f}s")

    # ── w_g+ ──────────────────────────────────────────────────────────────────
    print("\n── w_g+ ─────────────────────────────────────────────────────────────")
    for label, pos, shape in [
        ("cen_cen",   cen_cat,  cen_cat),
        ("sat_sat",   sat_cat,  sat_cat),
        ("cen_sat",   cen_cat,  sat_cat),
        ("sat_cen",   sat_cat,  cen_cat),
        ("halo_halo", halo_cat, halo_cat),
    ]:
        print(f"  -> {label}")
        t0 = time.time()
        measure_wgp(pos, rand_cat, shape, cfg, out_dir / f"wgp_{label}_{shape_tag}.txt", nt)
        print(f"     {time.time()-t0:.1f}s")

    # ── w_++ ──────────────────────────────────────────────────────────────────
    print("\n── w_++ ─────────────────────────────────────────────────────────────")
    for label, s1, s2 in [
        ("cen_cen",   cen_cat,  None),
        ("sat_sat",   sat_cat,  None),
        ("cen_sat",   cen_cat,  sat_cat),
        ("halo_halo", halo_cat, None),
    ]:
        print(f"  -> {label}")
        t0 = time.time()
        measure_wpp(s1, cfg, out_dir / f"wpp_{label}_{shape_tag}.txt", nt, shape_cat2=s2)
        print(f"     {time.time()-t0:.1f}s")

    print(f"\nDone. Results written to {out_dir}")


if __name__ == "__main__":
    main()
