"""Calculate 2d ellipse from projecting a 3d triaxial ellipsoid

Use Rodrigues rotation formula to align the LoS with the z-axis,
then apply a spin around the LoS by angle gamma, and finally project onto the xy-plane.

Two projection methods are available via the `envelop` flag:
  envelop=True  (default) — correct 2D silhouette via Schur complement of S33
  envelop=False           — z=0 cross-section (sub-matrix method, faster but approximate)

Motivated by Chen+16
https://ui.adsabs.harvard.edu/abs/2016ApJ...830..123C/abstract

"""

from collections import namedtuple
from functools import partial

import jax
from jax import jit as jjit
from jax import vmap as jvmap
from jax import numpy as jnp
from jax import random as jran

Ellipse2DParams = namedtuple(
    "Ellipse2DParams",
    ("alpha", "beta", "psi", "ellipticity", "e_alpha", "e_beta", "A", "B", "C"),
)


def mc_mu_phi(n, ran_key):
    """Monte Carlo realization of random projection angles mu, phi"""
    mu_key, phi_key = jran.split(ran_key, 2)
    mu_ran = jran.uniform(mu_key, minval=-1, maxval=1, shape=(n,))
    phi_ran = jran.uniform(phi_key, minval=0, maxval=2 * jnp.pi, shape=(n,))
    return mu_ran, phi_ran


def mc_mu_phi_gamma(n, ran_key):
    """Monte Carlo realization of random projection angles mu, phi, gamma"""
    mu_key, phi_key, gamma_key = jran.split(ran_key, 3)
    mu_ran = jran.uniform(mu_key, minval=-1, maxval=1, shape=(n,))
    phi_ran = jran.uniform(phi_key, minval=0, maxval=2 * jnp.pi, shape=(n,))
    gamma_ran = jran.uniform(gamma_key, minval=0, maxval=2 * jnp.pi, shape=(n,))
    return mu_ran, phi_ran, gamma_ran


def mc_ellipsoid_params(r50, b_over_a, c_over_a, ran_key):
    """Monte Carlo realization of 2d ellipse with random projection angles mu, phi, gamma"""
    los_key, _ = jran.split(ran_key, 2)
    mu_ran, phi_ran, gamma_ran = mc_mu_phi_gamma(r50.size, los_key)
    a = r50
    b = b_over_a * a
    c = c_over_a * a
    return compute_ellipse2d(a, b, c, mu_ran, phi_ran, gamma_ran)


def _project_ellipsoid_scalar(a, b, c, mu, phi, gamma, envelop=True):
    """Scalar (single-sample) projection. All array inputs must be 0-d JAX scalars.

    Parameters
    ----------
    a, b, c : scalars
        Semi-major, semi-medium, semi-minor axes of the 3D ellipsoid
    mu : scalar
        mu = cos(theta), polar angle between LoS and z-axis
    phi : scalar
        Azimuthal angle of LoS (radians)
    gamma : scalar
        Spin angle around the LoS after aligning it with z (radians)
    envelop : bool, static
        True  — correct 2D projection envelope via Schur complement of S33:
                  S_env = S_2d - (1/S33) s_col s_col^T
                where s_col = S_rotated[:2, 2].
                Derived by requiring the discriminant of the quadratic in z
                (S33 z² + 2(S13 x + S23 y) z + (quad in x,y - 1) = 0) to vanish.
        False — z=0 cross-section: S_2d = S_rotated[:2, :2] (approximate).
    """

    # ── Shape matrix ────────────────────────────────────────────────────────
    S = jnp.diag(jnp.array([1.0 / a**2, 1.0 / b**2, 1.0 / c**2]))

    # ── LoS direction vector ─────────────────────────────────────────────────
    sin_theta = jnp.sqrt(jnp.clip(1.0 - mu**2, 0.0))
    los_vec = jnp.array([sin_theta * jnp.cos(phi), sin_theta * jnp.sin(phi), mu])

    # ── R: Rodrigues rotation that maps LoS → +z ────────────────────────────
    z_axis = jnp.array([0.0, 0.0, 1.0])
    axis = jnp.cross(los_vec, z_axis)
    axis_norm = jnp.linalg.norm(axis)

    cos_angle = jnp.dot(los_vec, z_axis)
    sin_angle = axis_norm

    safe_norm = jnp.where(axis_norm > 1e-10, axis_norm, 1.0)
    axis_normalized = axis / safe_norm

    K = jnp.array([
        [0.0,                  -axis_normalized[2],  axis_normalized[1]],
        [axis_normalized[2],    0.0,                 -axis_normalized[0]],
        [-axis_normalized[1],   axis_normalized[0],   0.0],
    ])

    R_rod = jnp.eye(3) + sin_angle * K + (1.0 - cos_angle) * (K @ K)
    R = jnp.where(axis_norm > 1e-10, R_rod, jnp.eye(3))

    # ── R_gamma: spin around the new z (LoS) by gamma ───────────────────────
    cos_g = jnp.cos(gamma)
    sin_g = jnp.sin(gamma)
    R_gamma = jnp.array([
        [ cos_g, -sin_g, 0.0],
        [ sin_g,  cos_g, 0.0],
        [   0.0,    0.0, 1.0],
    ])

    R_total = R_gamma @ R
    S_rotated = R_total.T @ S @ R_total

    # ── 2D projection ────────────────────────────────────────────────────────
    if envelop:
        # Schur complement of S33: envelope of the full 3D projection
        # Derived from discriminant = 0 of the quadratic in z
        s_col = S_rotated[:2, 2]                        # [S13, S23]
        S33   = S_rotated[2, 2]
        S_2d  = S_rotated[:2, :2] - jnp.outer(s_col, s_col) / S33
    else:
        # z=0 cross-section (sub-matrix method)
        S_2d = S_rotated[:2, :2]

    # ── Eigendecomposition: eigh returns eigenvalues in ascending order ──────
    eigenvalues, eigenvectors = jnp.linalg.eigh(S_2d)

    semi_axes = 1.0 / jnp.sqrt(eigenvalues)
    alpha = semi_axes[0]   # semi-major (smaller eigenvalue → larger semi-axis)
    beta  = semi_axes[1]   # semi-minor

    major_eigenvector = eigenvectors[:, 0]
    psi = jnp.arctan2(major_eigenvector[1], major_eigenvector[0])

    # Wrap to [-pi/2, pi/2): semi-major axis has 180-deg freedom
    psi = jnp.mod(psi + jnp.pi / 2, jnp.pi) - jnp.pi / 2

    # ── Derived quantities ───────────────────────────────────────────────────
    ellipticity = 1.0 - beta / alpha

    cos_psi = jnp.cos(psi)
    sin_psi = jnp.sin(psi)
    e_alpha = jnp.array([cos_psi, sin_psi])
    e_beta  = jnp.array([-sin_psi, cos_psi])

    A = cos_psi**2 / alpha**2 + sin_psi**2 / beta**2
    B = 2.0 * cos_psi * sin_psi * (1.0 / alpha**2 - 1.0 / beta**2)
    C = sin_psi**2 / alpha**2 + cos_psi**2 / beta**2

    return Ellipse2DParams(alpha, beta, psi, ellipticity, e_alpha, e_beta, A, B, C)


@partial(jjit, static_argnums=(6,))
def compute_ellipse2d(a, b, c, mu, phi, gamma, envelop=True):
    """Project a batch of 3D ellipsoids to 2D ellipses.

    The 3D ellipsoid is defined by: (x/a)^2 + (y/b)^2 + (z/c)^2 = 1

    Parameters
    ----------
    a, b, c : arrays, shape (n,)
        Semi-major, semi-medium, and semi-minor axes
    mu : array, shape (n,)
        mu = cos(theta), polar angle between LoS and z-axis
    phi : array, shape (n,)
        Azimuthal angle of LoS (radians)
    gamma : array, shape (n,)
        Spin angle around the LoS after aligning it with z (radians)
    envelop : bool, optional (static, default True)
        True  — correct projection envelope (Schur complement of S33)
        False — z=0 cross-section (sub-matrix, approximate)

    Returns
    -------
    Ellipse2DParams : namedtuple of arrays, shape (n,)
        alpha, beta      — semi-major/minor axes of the projected ellipse
        psi              — position angle in [-pi/2, pi/2) (radians)
        ellipticity      — 1 - beta/alpha
        e_alpha, e_beta  — shape (n, 2), unit vectors along semi-axes
        A, B, C          — coefficients of Ax^2 + Bxy + Cy^2 = 1
    """
    fn = partial(_project_ellipsoid_scalar, envelop=envelop)
    return jvmap(fn)(a, b, c, mu, phi, gamma)


# ── Unit test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """Spinning-top GIF: visual unit test of the full (mu, phi, gamma) projection.

    Active vs Passive rotation
    ──────────────────────────
    PASSIVE rotation rotates the coordinate frame while the ellipsoid stays
    fixed.  Writing  x' = R x  in the usual sense means "express the same
    point in the rotated frame".

    ACTIVE rotation moves the physical ellipsoid while the coordinate frame
    stays fixed.  If R is the passive rotation that maps the LoS → +z, then
    its transpose  M = R^T  is the ACTIVE rotation that carries the
    ellipsoid's c-axis (originally along z) onto the LoS direction.

    This file uses ACTIVE rotations throughout:
      1. _unit_test_rodrigues_R(los) returns R such that  R @ los = z.
         Applied ACTIVELY as  M = R^T,  the c-axis tilts to align with LoS.
      2. R_gamma spins the frame around the new z (= LoS after R).
         Applied ACTIVELY as  R_gamma^T,  the ellipsoid spins by -gamma
         around the LoS (equivalently, gamma in the opposite sense).
      3. The combined active rotation is  M = (R_gamma @ R)^T = R^T @ R_gamma^T,
         so surface points transform as  x_rot = M @ x.
      4. The shape matrix transforms as  S' = R_total^T S R_total  (same M),
         confirming that  S'  describes the actively-rotated ellipsoid.
      5. Projecting onto the xy-plane via the Schur complement of S'33 gives
         the correct 2D envelope silhouette of that rotated ellipsoid.

    The GIF shows:
      LEFT  — 3D panel: actively-rotated ellipsoid, LoS arrow, gamma arc
              (arc sweeps in the plane perpendicular to LoS, centred at the
              LoS tip, indicating how far the ellipsoid has been spun)
      RIGHT — 2D panel: wireframe shadow of the rotated ellipsoid projected
              onto the xy-plane, overlaid with the analytical v2 envelope
              ellipse (solid orange) whose semi-axes should exactly touch the
              silhouette boundary of the wireframe.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Ellipse
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

    # ── Configuration ─────────────────────────────────────────────────────
    A, B, C    = 2.0, 1.5, 0.5    # ellipsoid semi-axes  (a >= b >= c)
    N_FRAMES   = 96                # total animation frames
    N_PHI_SLOW = 2.0               # precession: total phi turns over all frames
    N_GAM_FAST = 7.0               # spin:       total gamma turns over all frames
    FPS        = 12
    DPI        = 90
    SAVE_PATH  = "ellipsoid_projection.gif"

    # ── Trajectory ────────────────────────────────────────────────────────
    _t       = np.linspace(0.0, 1.0, N_FRAMES)
    mu_traj  = np.cos(_t * np.pi / 2)           # 1 → 0  (face-on → edge-on)
    phi_traj = _t * N_PHI_SLOW * 2 * np.pi      # slow precession
    gam_traj = _t * N_GAM_FAST * 2 * np.pi      # fast spin

    # ── Vectorised pre-computation (ALL frames at once via JAX vmap) ──────
    # Calling compute_ellipse2d once for the full trajectory is far cheaper
    # than calling the scalar kernel inside the animation loop.
    _ones = jnp.ones(N_FRAMES)
    _mu   = jnp.array(mu_traj)
    _phi  = jnp.array(phi_traj)
    _gam  = jnp.array(gam_traj)

    res_v2 = compute_ellipse2d(
        _ones * A, _ones * B, _ones * C, _mu, _phi, _gam, envelop=True
    )
    # Cache as plain numpy for fast per-frame scalar indexing
    v2 = {k: np.array(getattr(res_v2, k))
          for k in ("alpha", "beta", "psi", "e_alpha", "e_beta")}

    # ── Geometry helpers (defined here to avoid polluting module namespace) ──

    def _unit_test_rodrigues_R(los):
        """Return R s.t. R @ los = +z (passive rotation; R^T is the active one).

        Uses the Rodrigues formula:  R = I + sin(θ)·K + (1-cos(θ))·K²
        where K is the skew-symmetric matrix of the rotation axis
        k = (los × z) / |los × z|  and  cos(θ) = los · z.
        """
        ax = np.cross(los, [0.0, 0.0, 1.0])
        n  = np.linalg.norm(ax)
        if n < 1e-10:
            return np.eye(3)
        ax /= n
        K  = np.array([[0.0, -ax[2], ax[1]],
                       [ax[2], 0.0, -ax[0]],
                       [-ax[1], ax[0], 0.0]])
        ca = np.dot(los, [0.0, 0.0, 1.0])   # cos(theta) = los_z = mu
        sa = n                               # sin(theta)
        return np.eye(3) + sa * K + (1.0 - ca) * (K @ K)

    def _unit_test_build_active_M(mu, phi, gamma):
        """Return (los, M) where M = (R_gamma @ R)^T is the active rotation.

        Active rotation sequence (coordinate frame fixed throughout):
          Step 1 — R^T tilts the c-axis from z onto the LoS direction.
          Step 2 — R_gamma^T spins the ellipsoid by angle gamma around the LoS.

        The combined active matrix M = R^T @ R_gamma^T = (R_gamma @ R)^T
        matches the shape-matrix transform  S' = M^T S M  in the core code.
        """
        st  = np.sqrt(max(1.0 - mu**2, 0.0))
        los = np.array([st * np.cos(phi), st * np.sin(phi), mu])
        R   = _unit_test_rodrigues_R(los)
        cg, sg = np.cos(gamma), np.sin(gamma)
        Rg  = np.array([[cg, -sg, 0.0], [sg, cg, 0.0], [0.0, 0.0, 1.0]])
        return los, (Rg @ R).T

    def _unit_test_rotate_surface(M, x, y, z):
        """Apply active rotation M to surface arrays; preserve original shape."""
        sh  = x.shape
        pts = np.stack([x.ravel(), y.ravel(), z.ravel()])   # (3, N)
        rot = M @ pts
        return rot[0].reshape(sh), rot[1].reshape(sh), rot[2].reshape(sh)

    def _unit_test_gamma_arc_geometry(los, gamma, radius):
        """Partial ring of angular extent gamma in the plane perpendicular to LoS.

        The arc is centred at the tip of the unit LoS vector, lying in the
        plane spanned by (e1, e2) — two orthonormal vectors perpendicular to
        los.  Sweeping from 0 to gamma traces the spin already applied to the
        ellipsoid, giving an intuitive visual of how far it has been rotated.

        Returns
        -------
        arc : (3, 64) array — 3D arc coordinates
        e1, e2 : (3,) arrays — orthonormal basis in the perpendicular plane
        """
        ref = np.array([0.0, 1.0, 0.0]) if abs(los[0]) > 0.9 \
              else np.array([1.0, 0.0, 0.0])
        e1  = np.cross(los, ref);  e1 /= np.linalg.norm(e1)
        e2  = np.cross(los, e1);   e2 /= np.linalg.norm(e2)
        t   = np.linspace(0.0, gamma, 64)
        arc = los[:, None] + radius * (
            np.cos(t) * e1[:, None] + np.sin(t) * e2[:, None]
        )
        return arc, e1, e2

    def _unit_test_wireframe_shadow(M, a, b, c, n_lat=7, n_lon=10):
        """Project latitude/longitude lines of the actively-rotated ellipsoid.

        Returns a list of (x_proj, y_proj) pairs (xy-plane projection, z dropped).
        These should form a cloud whose outer boundary coincides exactly with
        the analytical v2 envelope ellipse — that is the key visual check.
        """
        segs = []
        for v in np.linspace(0.1 * np.pi, 0.9 * np.pi, n_lat):
            u = np.linspace(0, 2 * np.pi, 150)
            p = M @ np.stack([a * np.cos(u) * np.sin(v),
                               b * np.sin(u) * np.sin(v),
                               c * np.full_like(u, np.cos(v))])
            segs.append((p[0], p[1]))
        for u in np.linspace(0, 2 * np.pi, n_lon + 1)[:-1]:
            v = np.linspace(0, np.pi, 80)
            p = M @ np.stack([a * np.cos(u) * np.sin(v),
                               b * np.sin(u) * np.sin(v),
                               c * np.cos(v)])
            segs.append((p[0], p[1]))
        return segs

    # ── Static surface mesh (pre-generated once) ──────────────────────────
    _Ug, _Vg = np.meshgrid(np.linspace(0, 2 * np.pi, 48),
                            np.linspace(0, np.pi, 24))
    _X0 = A * np.cos(_Ug) * np.sin(_Vg)
    _Y0 = B * np.sin(_Ug) * np.sin(_Vg)
    _Z0 = C * np.cos(_Vg)

    _COORD_CLR = {"x": "#d62728", "y": "#2ca02c", "z": "#1f77b4"}
    LIM = max(A, B, C) * 1.5

    # ── Per-frame drawing ─────────────────────────────────────────────────

    def _unit_test_draw_3d(ax, los, M, mu, phi, gamma):
        """Left panel: actively-rotated 3D ellipsoid, LoS arrow, gamma arc.

        The coordinate axes (x, y, z) remain fixed — only the ellipsoid moves.
        This illustrates the ACTIVE rotation convention: the frame is the
        observer's frame; the object moves within it.
        """
        ax.cla()

        # Actively-rotated ellipsoid surface
        xr, yr, zr = _unit_test_rotate_surface(M, _X0, _Y0, _Z0)
        ax.plot_surface(xr, yr, zr, alpha=0.22, color="steelblue",
                        rstride=2, cstride=2, linewidth=0.3,
                        edgecolor="steelblue")

        # Fixed coordinate axes (unchanged — active rotation, not passive)
        for lbl, vec in zip(["x", "y", "z"], np.eye(3)):
            ax.quiver(0, 0, 0, *(vec * LIM * 0.60),
                      color=_COORD_CLR[lbl], lw=1.0,
                      arrow_length_ratio=0.14, alpha=0.6)

        # LoS direction arrow (always in the fixed frame)
        ax.quiver(0, 0, 0, *(los * LIM * 0.90), color="k", lw=2.5,
                  arrow_length_ratio=0.12, label="LoS")

        # Gamma arc: partial ring in the plane ⊥ LoS, centred at LoS tip.
        # The arc sweeps from 0 to gamma, visualising the spin already applied.
        arc_r = LIM * 0.30
        arc, e1, e2 = _unit_test_gamma_arc_geometry(los, gamma, arc_r)
        ax.plot(*arc, color="darkorange", lw=2.0)

        # Reference spoke at gamma = 0 (dashed)
        ref_tip = los + arc_r * e1
        ax.plot([los[0], ref_tip[0]], [los[1], ref_tip[1]],
                [los[2], ref_tip[2]],
                color="darkorange", lw=1.2, ls="--", alpha=0.55)

        # Arrowhead at arc end
        if abs(gamma % (2 * np.pi)) > 0.05:
            tip = arc[:, -1]
            d   = arc_r * 0.08 * (
                -np.sin(gamma) * e1 + np.cos(gamma) * e2
            )
            ax.quiver(*tip, *d, color="darkorange", lw=1.5,
                      arrow_length_ratio=1.0, normalize=False)

        # Transparent projection plane (z = 0)
        _px, _py = np.meshgrid([-LIM, LIM], [-LIM, LIM])
        ax.plot_surface(_px, _py, np.zeros_like(_px),
                        alpha=0.06, color="gray", linewidth=0)

        ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_zlim(-LIM, LIM)
        ax.set_xlabel("x", labelpad=1); ax.set_ylabel("y", labelpad=1)
        ax.set_zlabel("z", labelpad=1)
        ax.set_title(
            rf"$\mu$={mu:.2f}  $\phi$={np.degrees(phi) % 360:.0f}°"
            rf"  $\gamma$={np.degrees(gamma) % 360:.0f}°",
            fontsize=9, pad=4,
        )
        ax.legend(fontsize=8, loc="upper left")

    def _unit_test_draw_2d(ax, M, i):
        """Right panel: wireframe shadow + v2 Schur-complement envelope.

        Visual check: the solid orange analytical ellipse should exactly
        touch the outermost extent of the blue wireframe shadow at all angles.
        If it does, the Schur-complement projection (envelop=True) is correct.
        """
        ax.cla()
        ax.set_aspect("equal")
        ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)
        ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.3)
        ax.grid(True, alpha=0.15)

        # Wireframe shadow (outer boundary = true projection envelope)
        for xp, yp in _unit_test_wireframe_shadow(M, A, B, C):
            ax.plot(xp, yp, color="steelblue", alpha=0.18, lw=0.7)

        # Analytical v2 envelope ellipse (Schur complement, envelop=True)
        ea = v2["e_alpha"][i]
        eb = v2["e_beta"][i]
        ax.add_patch(Ellipse(
            (0, 0), 2 * v2["alpha"][i], 2 * v2["beta"][i],
            angle=np.degrees(v2["psi"][i]),
            fill=False, edgecolor="darkorange", lw=2.2,
            label=(rf"v2  $\alpha$={v2['alpha'][i]:.2f}"
                   rf"  $\beta$={v2['beta'][i]:.2f}"),
        ))
        ax.annotate("", xy=v2["alpha"][i] * ea, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.8))
        ax.annotate("", xy=v2["beta"][i]  * eb, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="green", lw=1.8))
        ax.text(*(v2["alpha"][i] * ea * 1.12), r"$\alpha$",
                color="darkorange", fontsize=9, ha="center", va="center")
        ax.text(*(v2["beta"][i]  * eb * 1.18), r"$\beta$",
                color="green",      fontsize=9, ha="center", va="center")

        ax.set_title("2D projection (drop z)", fontsize=9, pad=4)
        ax.legend(fontsize=8, loc="upper right")

    def _unit_test_update(i):
        """FuncAnimation callback: redraw both panels for frame i."""
        mu, phi, gamma = mu_traj[i], phi_traj[i], gam_traj[i]
        los, M = _unit_test_build_active_M(mu, phi, gamma)
        _unit_test_draw_3d(ax3d, los, M, mu, phi, gamma)
        _unit_test_draw_2d(ax2d, M, i)
        fig.suptitle(
            rf"Ellipsoid  $a$={A}  $b$={B}  $c$={C}"
            rf"  |  frame {i + 1}/{N_FRAMES}",
            fontsize=10, y=1.0,
        )

    # ── Render and save ───────────────────────────────────────────────────
    fig  = plt.figure(figsize=(13, 6))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    ani = animation.FuncAnimation(
        fig, _unit_test_update, frames=N_FRAMES,
        interval=1000 // FPS, blit=False,
    )
    ani.save(SAVE_PATH, writer="pillow", dpi=DPI, fps=FPS)
    plt.close(fig)
    print(f"Saved → {SAVE_PATH}")
