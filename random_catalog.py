"""
Generate a randomized catalog that matches the (ra, dec, r) selection function of
an input galaxy catalog.

Assumes the selection function is separable:
    P(ra, dec, r) ≈ P_angular(ra, dec) × P_radial(r)

Angular footprint is estimated via a 2D occupancy grid; radial distribution is
estimated from the data histogram and sampled via inverse CDF.  Angular positions
are drawn uniformly on the sphere (sin(dec) weighting) so that the resulting
randoms are unbiased on the sky.

All angular inputs/outputs are in radians unless `unit='deg'` is passed.
"""

import numpy as np
from scipy.interpolate import interp1d


def make_random_catalog(
    ra,
    dec,
    r,
    n_randoms=None,
    n_ang_bins=100,
    n_r_bins=100,
    unit="rad",
    seed=None,
):
    """
    Generate a random catalog that follows the same selection function as the
    input galaxy catalog.

    Parameters
    ----------
    ra : array_like, shape (N,)
        Right ascension.  In radians [0, 2π) by default; use `unit='deg'` for
        degrees [0, 360).
    dec : array_like, shape (N,)
        Declination.  In radians [-π/2, π/2] by default; use `unit='deg'` for
        degrees [-90, 90].
    r : array_like, shape (N,)
        Comoving distance (any consistent unit, e.g. Mpc).
    n_randoms : int, optional
        Number of random points to generate.  Default: 10 × len(ra).
    n_ang_bins : int
        Resolution of the 2D angular occupancy grid (per axis).  Finer grids
        better capture survey holes at the cost of more rejection draws.
    n_r_bins : int
        Number of bins for the radial n(r) estimate.
    unit : {'rad', 'deg'}
        Angular unit of ra / dec inputs (and outputs).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    ra_rand : ndarray, shape (n_randoms,)
        Random right ascension in the same units as input.
    dec_rand : ndarray, shape (n_randoms,)
        Random declination in the same units as input.
    r_rand : ndarray, shape (n_randoms,)
        Random comoving distance drawn from the estimated n(r).
    """
    rng = np.random.default_rng(seed)

    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    r = np.asarray(r, dtype=float)

    if unit == "deg":
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
    else:
        ra_rad = ra.copy()
        dec_rad = dec.copy()

    n_randoms = n_randoms or 10 * len(ra_rad)

    # ------------------------------------------------------------------
    # Handle RA wrap-around at 0 / 2π
    # If there is a large gap on one side, the field straddles 0/2π and
    # we shift values < the midpoint gap up by 2π so the range is contiguous.
    # ------------------------------------------------------------------
    ra_sorted = np.sort(ra_rad)
    gaps = np.diff(ra_sorted)
    max_gap_idx = np.argmax(gaps)
    max_gap = gaps[max_gap_idx]

    # If the largest gap occupies more than half the observed ra span,
    # the field is likely straddling 0/2π.
    ra_span = ra_sorted[-1] - ra_sorted[0]
    if max_gap > 0.5 * ra_span and max_gap < 2 * np.pi - ra_span:
        wrap_threshold = ra_sorted[max_gap_idx + 1]  # start of the gap
        ra_rad = np.where(ra_rad < wrap_threshold, ra_rad + 2 * np.pi, ra_rad)

    # ------------------------------------------------------------------
    # Angular footprint: 2D occupancy grid in (ra, dec)
    # Add a small buffer (1% of range) so boundary galaxies are interior.
    # ------------------------------------------------------------------
    ra_min, ra_max = ra_rad.min(), ra_rad.max()
    dec_min, dec_max = dec_rad.min(), dec_rad.max()

    buf_ra = 0.01 * (ra_max - ra_min) if ra_max > ra_min else 1e-4
    buf_dec = 0.01 * (dec_max - dec_min) if dec_max > dec_min else 1e-4
    ra_min -= buf_ra
    ra_max += buf_ra
    dec_min -= buf_dec
    dec_max += buf_dec

    hist2d, ra_edges, dec_edges = np.histogram2d(
        ra_rad, dec_rad,
        bins=n_ang_bins,
        range=[[ra_min, ra_max], [dec_min, dec_max]],
    )
    occupied = hist2d > 0  # shape (n_ang_bins, n_ang_bins)

    # ------------------------------------------------------------------
    # Radial selection function: n(r) via histogram → inverse CDF
    # ------------------------------------------------------------------
    r_hist, r_edges = np.histogram(r, bins=n_r_bins)
    cdf = np.cumsum(r_hist.astype(float))
    cdf /= cdf[-1]
    cdf = np.concatenate([[0.0], cdf])

    # Map uniform [0, 1] → r using linear interpolation of the inverse CDF
    inv_cdf_r = interp1d(
        cdf, r_edges,
        kind="linear",
        bounds_error=False,
        fill_value=(r_edges[0], r_edges[-1]),
    )

    # ------------------------------------------------------------------
    # Draw angular positions uniformly on the sphere (via sin(dec) trick)
    # then reject points outside the survey footprint.
    # ------------------------------------------------------------------
    sin_dec_min = np.sin(dec_min)
    sin_dec_max = np.sin(dec_max)

    ra_out, dec_out, r_out = [], [], []
    needed = n_randoms

    while needed > 0:
        # Oversample by ~5× to account for the rejection fraction;
        # never draw fewer than 10 000 at once to keep loop overhead low.
        n_draw = max(needed * 5, 10_000)

        ra_try = rng.uniform(ra_min, ra_max, n_draw)
        dec_try = np.arcsin(rng.uniform(sin_dec_min, sin_dec_max, n_draw))

        # Locate each candidate in the occupancy grid
        ra_idx = np.searchsorted(ra_edges, ra_try, side="right") - 1
        dec_idx = np.searchsorted(dec_edges, dec_try, side="right") - 1
        ra_idx = np.clip(ra_idx, 0, n_ang_bins - 1)
        dec_idx = np.clip(dec_idx, 0, n_ang_bins - 1)

        mask = occupied[ra_idx, dec_idx]
        n_accept = mask.sum()

        ra_accept = ra_try[mask][:needed]
        dec_accept = dec_try[mask][:needed]

        # Draw r values independently from n(r)
        r_accept = inv_cdf_r(rng.uniform(0.0, 1.0, len(ra_accept)))

        ra_out.append(ra_accept)
        dec_out.append(dec_accept)
        r_out.append(r_accept)
        needed -= len(ra_accept)

    ra_rand = np.concatenate(ra_out)[:n_randoms]
    dec_rand = np.concatenate(dec_out)[:n_randoms]
    r_rand = np.concatenate(r_out)[:n_randoms]

    # Wrap ra back to [0, 2π) if we shifted for wrap-around
    ra_rand = ra_rand % (2 * np.pi)

    if unit == "deg":
        ra_rand = np.degrees(ra_rand)
        dec_rand = np.degrees(dec_rand)

    return ra_rand, dec_rand, r_rand
