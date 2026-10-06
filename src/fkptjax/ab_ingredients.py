"""
ab_ingredients.py
------------------
Coarse-grid A(kf,k1,k2)/B(kf,k1,k2) "ingredients" for the fkpt_emulation route:
a SMALL, representative grid of the beyond-EdS kernel building blocks
(fkptjax.MG_kernels.A_B_grid), meant to be emulated (Chebyshev/Taylor) over
the MG parameters, with JAX-native interpolation filling in the actual
(much finer) triangles the P22/P13/I1udd1-family loop integral and the
bispectrum's Z2 kernel need.

Motivation (see this session's validation): a per-triangle A/B fit is smooth
in the MG parameters even for genuinely scale-dependent models (Hu-Sawicki,
BZ_Mass) -- a 5-node Chebyshev fit at one fixed triangle reached ~2-7e-4
relative accuracy across a wide fR0_HS range sweeping the Compton-wavelength
transition clear through the loop's k-range. That is NOT true of the
loop-INTEGRATED P22/P13/bias table (whole-pt emulation), where the
transition's *location* moving with the parameter mixes into every output k
bin's own parameter-dependence. Emulating the raw per-triangle ingredient
instead -- and integrating live, in JAX -- keeps the parameter-dependence at
each grid node simple, while the loop integral (already pure JAX either way)
does not care whether A/B came from a live ODE solve or an interpolated
emulator prediction.

Parametrization -- NATIVE LEG MAGNITUDES, not triangle-shape
--------------------------------------------------------------
Earlier versions of this module gridded A/B over the "triangle-shape"
coordinates ``(kf, r=k1/kf, x)`` (kf log-spaced, r log-spaced, x linear in
``[-1, 1]``). That works well for models with SMOOTH ``mu(k,eta)`` (HDKI/
mu_OmDE, BZ_Mass, Hu-Sawicki), but fails badly for ``PHENOM/binning``
(``scale_bins=True``): its ``mu(k)`` is a near-STEP function (tanh transitions
of width ``k_tw`` as small as 1e-3, at three FIXED locations ``k_TGR``,
``k_c``, ``k_S``), and ``mg_jax.secondOrderAB``'s RHS evaluates ``mu`` at
EACH leg's magnitude independently -- ``mu(kf)``, ``mu(k1)``, ``mu(k2)``. In
triangle-shape coordinates, "leg k1 crosses k_TGR" is the condition
``r * kf = k_TGR``: a DIAGONAL curve through the (log kf, log r) plane, not
aligned with either grid axis -- no amount of per-axis resolution captures it
well, and trilinear interpolation across it rings (Gibbs-phenomenon-like
overshoot/undershoot) at the 5-20% level right at the transition.

Gridding directly in the three independent leg magnitudes ``(kf, k1, k2)``
turns every one of those conditions into an AXIS-ALIGNED plane, which
:func:`_clustered_log_nodes` can then resolve with a handful of extra nodes
placed exactly at the model's own fixed transition points (``k_attractors``)
-- on ALL THREE axes, so ``kf``'s, ``k1``'s AND ``k2``'s own
threshold-crossings are each resolved directly, not just two of the three.

Two intermediate designs were tried and rejected here, and both are worth
recording because the failure modes are non-obvious:

1. An EARLIER version reasoned that ``A_B_grid``'s own opening-angle formula
   (``_x_from_triple``, a ratio of polynomials) stays "well-defined" even
   for a ``(kf,k1,k2)`` triple that does not satisfy the triangle
   inequality, and so left the three axes fully independent with no
   safeguard. That reasoning was WRONG for one of the two source terms:
   ``A``'s own driving term includes ``mg_jax.S2FL``, which contains an
   explicit ``p/k + k/p`` (see its docstring) -- perfectly finite for a
   valid triangle, but genuinely DIVERGING for the wildly mismatched leg
   ratios an unconstrained independent-axis grid inevitably samples (e.g.
   ``kf=0.001`` paired with ``k1=1e-5``, ``k2=8``, which cannot form a
   triangle at all). Measured (HDKI/BZ_Mass): that produced a coarse ``A``
   array with mean ~7e7 (physically ~1), which trilinear interpolation then
   smeared into every nearby VALID query, poisoning the final kP0/kP2 with
   a smooth ~3-12% bias -- while ``B`` (sourced by ``S2b`` alone, no such
   term) stayed fine, a sharp diagnostic signature worth remembering if
   this class of bug recurs elsewhere.
2. The NEXT version reacted by giving up independence altogether: keeping
   ``(kf, k1)`` as clustered axes but deriving ``k2`` from them via the law
   of cosines with an explicit ``x in [-1, 1]`` axis, guaranteeing every
   training triangle is valid by construction. That fixed HDKI/BZ_Mass, but
   cost PHENOM/binning real accuracy (kP0 max 3.5%->4.3%, kP2 6.1%->7.7%):
   ``k2``'s own mu(k) threshold-crossings were no longer axis-aligned at
   all, a real, model-dependent regression, not just a smaller residual.

The actual fix keeps ALL THREE axes independent and clustered (recovering
full axis-alignment everywhere) and instead regularizes the HANDFUL of
invalid ``(kf,k1,k2)`` corners a blind Cartesian product necessarily
contains: :func:`compute_ab_coarse` snaps any corner that violates the
triangle inequality onto the NEAREST valid one (clip its own ``x`` to
``[-1, 1]`` and rebuild ``k2`` from that) before ever calling ``A_B_grid``,
so ``S2FL`` never sees an unbounded leg ratio. A real query is always a
valid triangle already (:func:`interpolate_ab` is untouched, exactly the
original independent-axis lookup), so this only changes what the handful of
otherwise-never-directly-queried invalid corners contribute to nearby
interpolation -- a small, bounded, physically-motivated value instead of a
divergence.

The fused/third-order (R-loop) kernel is gridded the same way, over
``(k, p, x)`` instead of the old ``(k, r=p/k, x)`` -- this axis-aligns the
``mu(k)``/``mu(p)`` conditions exactly as above. Its own third relevant
magnitude, ``|k+p|`` (and, separately, ``|k-p|``), is a function of ALL of
``(k, p, x)`` together (``mg_jax.kpp``) and can't be made a fourth
independent axis (only 3 degrees of freedom exist), so a transition driven
purely by that combined leg is only partially resolved by this grid -- see
:func:`build_fused_coarse_grid`.
"""
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from . import MG_kernels as _mgk


# ---------------------------------------------------------------------------
# Shared node-building and interpolation-index machinery.
# ---------------------------------------------------------------------------

def _clustered_log_nodes(kmin, kmax, n_base, attractors=(), cluster_width=None, n_cluster=8,
                          inner_width=None):
    """Log-uniform base coverage over ``[kmin, kmax]`` PLUS a two-scale
    GEOMETRIC ladder of extra nodes around each of ``attractors`` (e.g. a
    model's own FIXED mu(k) transition points, such as PHENOM/binning's
    k_TGR/k_c/k_S).

    A FLAT ``linspace(k0-cluster_width, k0+cluster_width, n_cluster)`` (the
    original scheme) spends most of its nodes far from ``k0``, where a tanh
    step of half-width ``k_tw`` has already saturated -- e.g. at
    ``cluster_width=0.3*k0``, ``k_tw=1e-3``, the window is ~300 ``k_tw`` wide,
    so a UNIFORM 8-16 node spacing across it is itself several ``k_tw``,
    under-resolving the one region (within a few ``k_tw`` of ``k0``) that
    actually needs it. This instead places nodes GEOMETRICALLY from
    ``inner_width`` (default ``cluster_width / 20`` -- sub-``k_tw`` when
    ``cluster_width`` itself is set to a few ``k_tw``, see
    ``fit_synthetic_data.py``'s ``--fkpt-emu-cluster-width``) out to
    ``cluster_width`` on each side of ``k0``: dense right at the step,
    coarsening outward, covering the SAME total window at the SAME node
    count as the flat scheme but reallocating most of it to where the
    tanh actually moves. Degenerates to a single node at ``k0`` if
    ``inner_width`` ends up ``>= cluster_width`` (e.g. a tiny ``cluster_width``
    passed in).

    Nodes placed in LINEAR k (the transition window is tiny relative to k
    itself, so log vs. linear does not matter there). Attractors outside
    ``[kmin, kmax]`` are skipped; the result is de-duplicated and sorted, so
    calling with no attractors reproduces the old plain ``linspace`` axis
    exactly.
    """
    base = np.linspace(np.log(kmin), np.log(kmax), n_base)
    parts = [base]
    for k0 in attractors:
        if not (kmin < k0 < kmax):
            continue
        width = cluster_width if cluster_width is not None else 0.3 * k0
        inner = inner_width if inner_width is not None else width / 20.0
        inner = min(max(inner, 1e-4 * width), 0.9 * width)
        half_n = max(2, n_cluster // 2)
        offsets = np.geomspace(inner, width, half_n)
        pts = np.concatenate([[k0], k0 - offsets, k0 + offsets])
        pts = np.clip(pts, kmin, kmax)
        parts.append(np.log(pts))
    nodes = np.unique(np.concatenate(parts))
    return jnp.asarray(nodes)


def _frac_index(value, nodes):
    """Fractional index of ``value`` into ascending ``nodes`` via linear
    interpolation of the index itself. Reduces exactly to the old
    uniform-step formula on a ``linspace`` array, and generalizes correctly
    to a clustered/non-uniform node layout (see :func:`_clustered_log_nodes`)
    -- ``jnp.interp`` handles arbitrary node spacing directly. Clipped so a
    query just outside the box (floating-point roundoff, not a real
    extrapolation) does not wrap.
    """
    n = nodes.shape[0]
    idx = jnp.interp(value, nodes, jnp.arange(n, dtype=nodes.dtype))
    return jnp.clip(idx, 0.0, float(n - 1))


def _cubic_weights_local(x, n):
    """4 LOCAL cubic-Lagrange weights + start index for position ``x`` in
    UNIFORM index space ``[0, n)`` -- the same 4-node bracketing scheme as
    ``cosmoprimo.emulators.tools.utils.cardinal_cubic_weights``/FOLPS's own
    ``fog_collocation_weights`` (4th-order accurate, weights sum to one), but
    returning only the local ``(index, weights)`` pair instead of that
    function's DENSE length-``n`` broadcast -- forming the dense vector for
    every one of the real production grid's ~1e5-6e5 query points would
    materialize an (Npoints, n) array per axis for no reason; a `dynamic_slice`
    gather of the local 4-wide block plus a tiny per-point contraction (see
    :func:`_interp_grid`) is exactly equivalent and cheap.
    """
    position = jnp.clip(x, 0.0, n - 1.0)
    index = jnp.clip(jnp.floor(position).astype(jnp.int32) - 1, 0, n - 4)
    offset = position - (index.astype(x.dtype) + 1.0)
    w = jnp.stack([
        -offset * (offset - 1.0) * (offset - 2.0) / 6.0,
        (offset + 1.0) * (offset - 1.0) * (offset - 2.0) / 2.0,
        -(offset + 1.0) * offset * (offset - 2.0) / 2.0,
        (offset + 1.0) * offset * (offset - 1.0) / 6.0,
    ], axis=-1)
    return index, w


def _interp_grid_cubic(coords, arrays):
    """Separable LOCAL tricubic interpolation (see :func:`_cubic_weights_local`)
    -- a strictly more accurate drop-in for :func:`_interp_grid`'s
    ``order=1`` trilinear path, at the SAME node set: a curved function
    between two coarse nodes carries an O(h^2) error under linear
    interpolation regardless of how many more nodes are added elsewhere,
    where h is the LOCAL spacing; cubic reduces that to O(h^4) using the
    same two bracketing nodes plus their two next neighbors, with no extra
    training cost (the coarse grid's own values are reused as-is).
    """
    shape = coords.shape[1:]
    flat = coords.reshape(3, -1)
    n0, n1, n2 = arrays[0].shape

    i0, w0 = _cubic_weights_local(flat[0], n0)
    i1, w1 = _cubic_weights_local(flat[1], n1)
    i2, w2 = _cubic_weights_local(flat[2], n2)

    def interp_one(values):
        def one_point(idx0, idx1, idx2, ww0, ww1, ww2):
            block = jax.lax.dynamic_slice(values, (idx0, idx1, idx2), (4, 4, 4))
            return jnp.einsum('i,j,k,ijk->', ww0, ww1, ww2, block)

        out = jax.vmap(one_point)(i0, i1, i2, w0, w1, w2)
        return out.reshape(shape)

    return tuple(interp_one(a) for a in arrays)


def _interp_grid(coords, arrays, order=1):
    if order == 3:
        return _interp_grid_cubic(coords, arrays)

    shape = coords.shape[1:]

    def interp_one(values):
        out = jax.scipy.ndimage.map_coordinates(values, coords.reshape(3, -1), order=order, mode='nearest')
        return out.reshape(shape)

    return tuple(interp_one(a) for a in arrays)


# ---------------------------------------------------------------------------
# A/B (Q-loop / I1udd1-family) ingredient: native (kf, k1, k2) grid.
# ---------------------------------------------------------------------------

class ABCoarseGrid(NamedTuple):
    """The coarse grid's own node coordinates (log-kf, log-k1, log-k2) --
    fixed, cosmology-independent -- kept alongside the (cosmology/MG
    -dependent) values so :func:`interpolate_ab` needs nothing else. ``k1``
    and ``k2`` share the same physical range and (by default) the same node
    values, since ``A_B_grid`` treats them symmetrically."""
    log_kf_nodes: jnp.ndarray  # (Nkf,)
    log_k1_nodes: jnp.ndarray  # (Nk1,)
    log_k2_nodes: jnp.ndarray  # (Nk2,)


def _x_from_triple(kf, k1, k2):
    """cos(angle) between legs k1, k2 such that ``|k1+k2| = kf``. Mirrors
    ``MG_kernels._x_from_triple`` (kept private there); duplicated here
    (one line) rather than imported, since it's simple enough not to be
    worth coupling the two modules' internals over."""
    return (kf * kf - k1 * k1 - k2 * k2) / (2.0 * k1 * k2)


def _clamp_to_valid_triangle(kf, k1, k2):
    """Snap a (kf, k1, k2) triple that violates the triangle inequality onto
    the NEAREST one that doesn't, by clipping its own ``x`` (via
    :func:`_x_from_triple`) to ``[-1, 1]`` and rebuilding ``k2`` from that --
    leaving genuinely valid triples untouched (their ``x`` is already inside
    ``[-1, 1]``, so the clip is a no-op and this reconstructs the ORIGINAL
    ``k2`` exactly). See :func:`build_coarse_grid`'s docstring for why an
    UNCONSTRAINED independent ``(kf, k1, k2)`` grid needs this: a blind
    Cartesian product over three independently-clustered axes inevitably
    contains combinations no real triangle can have (e.g. ``kf=0.001`` with
    ``k1=1e-5``, ``k2=8``), and ``A_B_grid``'s ``A``-source term
    (``mg_jax.S2FL``) genuinely diverges there, not just extrapolates.

    Must match ``_x_from_triple``'s OWN convention exactly: ``x`` there is
    the angle BETWEEN ``k1`` and ``k2``, with ``kf`` their vector SUM, i.e.
    ``kf^2 = k1^2 + k2^2 + 2*k1*k2*x`` (note the ``+``, not the law of
    cosines' more familiar ``-`` form, which would put the angle between
    ``kf`` and ``k1`` instead). That's a QUADRATIC in ``k2`` with two roots,
    ``k2 = -k1*x +- sqrt(kf^2 - k1^2*(1-x^2))``: two different lengths can
    share the same ``k1``, ``kf`` and angle ``x``, so picking one root
    unconditionally reconstructs the WRONG ``k2`` for roughly a tenth of
    genuinely valid triangles (caught only by checking many random known
    -valid cases against their own unclipped round-trip -- a single
    hand-picked example is not enough to catch a ~10%-of-the-time ambiguity).
    The fix: leave an ALREADY-valid ``k2`` (``|x| <= 1``) untouched --
    trivially exact, no root ambiguity possible -- and only use the ``+``
    -root reconstruction for a genuinely invalid triple, where any nearby
    finite value is an equally arbitrary (and equally fine) regularization.
    """
    x_raw = _x_from_triple(kf, k1, k2)
    is_valid = jnp.abs(x_raw) <= 1.0
    x_clipped = jnp.clip(x_raw, -1.0, 1.0)
    k2_reconstructed = -k1 * x_clipped + jnp.sqrt(
        jnp.clip(kf * kf - k1 * k1 * (1.0 - x_clipped * x_clipped), 1e-30, None))
    return jnp.where(is_valid, k2, k2_reconstructed)


def build_coarse_grid(kf_min, kf_max, kleg_min, kleg_max, Nkf=16, Nkleg=32,
                       k_attractors=(), n_cluster=8, cluster_width=None,
                       inner_width=None) -> ABCoarseGrid:
    """The coarse grid's node coordinates only (no ODE solve yet).

    ``kf_min``/``kf_max``: the physical output-leg (kf) range.
    ``kleg_min``/``kleg_max``: the physical range the k1/k2 input legs can
    reach -- typically much wider than kf's own range (e.g. the Q-loop's own
    quadrature k1 legs run from ``pmin`` to ``pmax``, not just ``[kmin,kmax]``)
    -- match ``setup_kfunctions``'s OWN ``pmin``/``pmax`` (roughly
    ``max(k_in[0], 0.01*kmin)`` to ``min(k_in[-1], 16*kmax)``), not a
    deliberately generous placeholder: this axis has no attractor-clustering
    to fall back on away from a model's fixed mu(k) transitions, so its
    accuracy in a "generic" (non-threshold) region depends entirely on
    ``Nkleg``'s log-uniform base density across whatever range you give it --
    a wider-than-necessary range dilutes that density for no benefit.
    Measured this session (PHENOM/binning, an asymmetric r=3 triangle whose
    legs land in such a generic region): going from a padded
    ``kleg_max=10`` to a realistic ``kleg_max=2`` cut the worst-case error
    from 4.3% to 3.6% on its own, and raising ``Nkleg`` 16->48 on top of that
    (independent of the range) cut it further to 1.9% -- a genuine, still
    shrinking, log-uniform-density effect, not a diagonal-feature problem
    like the one ``k_attractors`` fixes.

    All three axes are independent and (optionally) clustered -- see the
    module docstring for the two designs this one replaced (an unsafeguarded
    independent grid that let ``A``'s ``S2FL`` term diverge on invalid
    triangles; a derived-``k2`` grid that avoided that but lost ``k2``'s own
    axis-alignment). :func:`compute_ab_coarse` regularizes the invalid
    corners this Cartesian product inevitably contains instead, so all three
    of ``kf``'s, ``k1``'s and ``k2``'s own mu(k) transitions stay
    axis-aligned.
    ``k_attractors``: fixed k-values (e.g. a model's own mu(k) transition
    points) to cluster EXTRA nodes around, on all three axes -- see
    :func:`_clustered_log_nodes`. Harmless (a few extra, unused grid points)
    for a model with no such features.
    """
    log_kf_nodes = _clustered_log_nodes(kf_min, kf_max, Nkf, k_attractors, cluster_width,
                                         n_cluster, inner_width)
    log_kleg_nodes = _clustered_log_nodes(kleg_min, kleg_max, Nkleg, k_attractors, cluster_width,
                                           n_cluster, inner_width)
    return ABCoarseGrid(log_kf_nodes, log_kleg_nodes, log_kleg_nodes)


def compute_ab_coarse(grid: ABCoarseGrid, P, xnow, xstop, solver='adaptive', n_steps=None, **solver_kwargs):
    """Solve A_B_grid on the FULL Cartesian product of the coarse grid's
    ``(kf, k1, k2)`` nodes -- one batched ``jax.vmap`` call, cheap because the
    grid is small (typically a few thousand points, not the fine grid's
    hundreds of thousands). Corners that don't form a valid triangle are
    first snapped onto the nearest one that does (:func:`_clamp_to_valid_triangle`)
    -- see :func:`build_coarse_grid` for why.

    Returns ``(A, B, Ap, Bp)``, each of shape ``(Nkf, Nk1, Nk2)``.
    """
    kf_b, k1_b, k2_b = jnp.meshgrid(
        jnp.exp(grid.log_kf_nodes), jnp.exp(grid.log_k1_nodes), jnp.exp(grid.log_k2_nodes),
        indexing='ij',
    )
    k2_b = _clamp_to_valid_triangle(kf_b, k1_b, k2_b)

    kwargs = dict(solver=solver)
    if n_steps is not None:
        kwargs['n_steps'] = n_steps
    kwargs.update(solver_kwargs)
    A, B, Ap, Bp = _mgk.A_B_grid(kf_b, k1_b, k2_b, P, xnow, xstop, **kwargs)
    return A, B, Ap, Bp


def compute_ab_coarse_raw(grid: ABCoarseGrid, P, xnow, xstop, solver='adaptive', n_steps=None, **solver_kwargs):
    """Like :func:`compute_ab_coarse`, but returns the raw ODE states
    ``(Dk1, dDk1, Dk2, dDk2, DA, dDA, DB, dDB)`` instead of the combined
    ``(A, B, Ap, Bp)`` -- see :func:`fkptjax.MG_kernels.A_B_grid_raw`'s docstring
    for why :class:`IngredientsProvider` fits these instead of ``Ap``/``Bp``
    directly (a cancellation-prone quotient-rule difference of two
    comparably-sized terms, versus these individually well-behaved states).
    """
    kf_b, k1_b, k2_b = jnp.meshgrid(
        jnp.exp(grid.log_kf_nodes), jnp.exp(grid.log_k1_nodes), jnp.exp(grid.log_k2_nodes),
        indexing='ij',
    )
    k2_b = _clamp_to_valid_triangle(kf_b, k1_b, k2_b)

    kwargs = dict(solver=solver)
    if n_steps is not None:
        kwargs['n_steps'] = n_steps
    kwargs.update(solver_kwargs)
    return _mgk.A_B_grid_raw(kf_b, k1_b, k2_b, P, xnow, xstop, **kwargs)


def interpolate_ab(kf_query, k1_query, k2_query, grid: ABCoarseGrid, A, B, Ap, Bp, order=1):
    """Trilinear (``order=1``) JAX-native interpolation of the coarse A/B/Ap/Bp
    arrays onto arbitrary (kf, k1, k2) triangles -- e.g. the fine Q-loop/R-loop
    grid, or a bispectrum triangle's three cyclic leg assignments. A real
    query is always a genuinely valid triangle, so -- unlike
    :func:`compute_ab_coarse` -- no clamping is needed here.

    ``kf_query``/``k1_query``/``k2_query``: broadcastable arrays, any shape.
    Returns ``(A, B, Ap, Bp)`` of that broadcast shape.
    """
    kf_b, k1_b, k2_b = jnp.broadcast_arrays(
        jnp.asarray(kf_query, dtype=jnp.float64),
        jnp.asarray(k1_query, dtype=jnp.float64),
        jnp.asarray(k2_query, dtype=jnp.float64),
    )
    i_kf = _frac_index(jnp.log(kf_b), grid.log_kf_nodes)
    i_k1 = _frac_index(jnp.log(k1_b), grid.log_k1_nodes)
    i_k2 = _frac_index(jnp.log(k2_b), grid.log_k2_nodes)
    coords = jnp.stack([i_kf, i_k1, i_k2], axis=0)
    return _interp_grid(coords, (A, B, Ap, Bp), order=order)


# ---------------------------------------------------------------------------
# P13-loop's fused kernel: A_fused = A - B*x^2 (D2_fused_grid) and the
# third-order CFD3/CFD3p (D3_fused_grid) -- a DIFFERENT ODE from A_B_grid
# (fkptjax.MG_kernels.D2_D3_fused_grid), needed for the Gamma2/Gamma3-type
# P13-loop pieces. Native (k, p, x) grid: k and p are the two independent
# input-leg magnitudes ``mu(k)``/``mu(p)`` are evaluated at directly, so
# gridding them independently (rather than the old (k, r=p/k, x)) axis
# -aligns those two transitions exactly as :func:`build_coarse_grid` does
# for A/B. The combined leg ``|k+p|``/``|k-p|`` (``mg_jax.kpp``) a third
# transition can live on is a function of all of (k, p, x) together and
# can't be made a fourth independent axis -- see the module docstring.
# ---------------------------------------------------------------------------

class FusedCoarseGrid(NamedTuple):
    log_k_nodes: jnp.ndarray   # (Nk,)
    log_p_nodes: jnp.ndarray   # (Np,)
    x_nodes: jnp.ndarray       # (Nx,)


def _ang_factor(r, x):
    """The rational (r, x)-structure ``MG_kernels._solve_D3`` bakes into its
    own initial condition (its local name is ``ang``):
    ``1/(1+r^2+2rx) + 1/(1+r^2-2rx)`` -- diverging as the two legs fold
    collinear (``r->1``, ``x->+-1``, i.e. ``|k-p|->0`` or ``|k+p|->0``).
    ``D3_fused_grid``'s ``CFD3``/``CFD3p`` inherit this same sharp structure
    from the ODE's IC, which is what makes them interpolate so much worse
    than ``A_B_grid``'s A/B (whose IC is the smooth ``(3/7)(1-x^2)``):
    validated this session -- at a fixed (k, r) slice, CFD3 sharpens exactly
    where ``ang`` does, and dividing it out before interpolating (then
    multiplying back in afterward, see ``interpolate_fused``) turns the
    residual into a quantity that interpolates about as well as A/B does,
    cutting the worst-case relative error from ~30% to ~0.5% at comparable
    grid cost. ``r`` here is always ``p / k``, computed on the fly -- it is
    no longer a stored grid axis (see the native (k, p, x) parametrization
    above)."""
    return 1.0 / (1.0 + r * r + 2.0 * r * x) + 1.0 / (1.0 + r * r - 2.0 * r * x)


def build_fused_coarse_grid(k_min, k_max, p_min, p_max, Nk=24, Np=40, Nx=64,
                             k_attractors=(), n_cluster=8, cluster_width=None,
                             inner_width=None) -> FusedCoarseGrid:
    """Same idea as :func:`build_coarse_grid`, over the R-loop's own
    ``(k, p, x)`` domain.

    ``k_min``/``k_max``: the physical output-leg (k) range. ``p_min``/``p_max``:
    the physical range the p leg can reach -- typically much wider (match
    whatever the real R-loop grid spans). ``k_attractors``: see
    :func:`build_coarse_grid` -- clusters extra nodes on BOTH the k and p
    axes (harmless for a model with no such features).

    Defaults are higher than :func:`build_coarse_grid`'s, and specifically
    richer in ``x`` (64 vs. A/B's typical 6-16): CFD3's ``ang``-driven
    sharpness (see :func:`_ang_factor`) lives almost entirely along x for
    models with otherwise-smooth mu(k), not k or p -- boosting Nk/Np alone
    barely moves that worst-case error, boosting Nx does. Combined with the
    ``ang`` pre-division in ``interpolate_fused``, this resolution reaches
    ~0.5% worst-case / ~0.03% median relative error against the live ODE
    solve for SMOOTH-mu(k) models, comparable to A/B's own accuracy. Add
    ``k_attractors`` on top of this for a model like PHENOM/binning whose
    ``mu(k)`` itself has sharp features.
    """
    log_k_nodes = _clustered_log_nodes(k_min, k_max, Nk, k_attractors, cluster_width,
                                        n_cluster, inner_width)
    log_p_nodes = _clustered_log_nodes(p_min, p_max, Np, k_attractors, cluster_width,
                                        n_cluster, inner_width)
    # Padded away from the exact endpoints: `_solve_D3`'s own IC has a genuine 1/0
    # singularity at r=p/k=1, x=+-1 (perfectly collinear legs). The OLD (k, r=p/k)
    # grid rarely landed exactly there by luck of an even Nr; independent k/p axes
    # -- especially with k_attractors clustering BOTH toward the same physical
    # values -- make r=1 far more likely to be hit almost exactly, and x=+-1 was
    # always an exact node (`linspace(-1,1,Nx)` includes both endpoints), so the two
    # combined reliably produced NaNs during training. A real query at x=+-1 exactly
    # still gets a well-defined (if very slightly extrapolated) answer via
    # `interpolate_fused`'s `mode='nearest'` clamping to this nearest interior node.
    x_nodes = jnp.linspace(-1.0 + 1e-3, 1.0 - 1e-3, Nx)
    return FusedCoarseGrid(log_k_nodes, log_p_nodes, x_nodes)


def compute_fused_coarse(grid: FusedCoarseGrid, P, xnow, xstop, f0=None, solver='adaptive', n_steps=None, **solver_kwargs):
    """Solve D2_D3_fused_grid on the coarse (k, p, x) grid.

    ``f0`` is ALWAYS solved with the reference value 1.0, regardless of what
    (if anything) is passed here -- ``D2_D3_fused_grid``'s only use of ``f0``
    is dividing ``CFD3p`` by ``3*f0`` at the very end (``A_fused``, ``Ap_fused``
    and ``CFD3`` don't depend on it at all), so training with ``f0=1``
    reproduces the exact per-``f0`` scaling law with no accuracy loss and no
    need to know the (parameter-dependent!) live ``f0`` at TRAINING time.
    The caller supplies the real, live ``f0`` at PREDICT/query time instead
    (see :func:`interpolate_fused`) -- passing anything else here is
    accepted only so old call sites don't break, and is ignored.

    Returns ``(A_fused, Ap_fused, CFD3, CFD3p)`` (the last computed with
    ``f0=1``, i.e. ``CFD3p * 1 == CFD3p_content / 3``), each of shape
    ``(Nk, Np, Nx)``.
    """
    k_b, p_b, x_b = jnp.meshgrid(
        jnp.exp(grid.log_k_nodes), jnp.exp(grid.log_p_nodes), grid.x_nodes,
        indexing='ij',
    )

    kwargs = dict(solver=solver)
    if n_steps is not None:
        kwargs['n_steps'] = n_steps
    kwargs.update(solver_kwargs)
    A_fused, Ap_fused, CFD3, CFD3p = _mgk.D2_D3_fused_grid(x_b, k_b, p_b, P, xnow, xstop, 1.0, **kwargs)
    return A_fused, Ap_fused, CFD3, CFD3p


def compute_fused_coarse_raw(grid: FusedCoarseGrid, P, xnow, xstop, solver='adaptive', n_steps=None,
                              **solver_kwargs):
    """Like :func:`compute_fused_coarse`, but returns the raw ODE states
    ``(Dk, dDk, Dp, dDp, D2p, dD2p, D2m, dD2m, D3, dD3)`` instead of the combined
    ``(A_fused, Ap_fused, CFD3, CFD3p)`` -- see :func:`compute_ab_coarse_raw`'s
    docstring for why :class:`IngredientsProvider` fits these instead of
    ``Ap_fused`` directly.
    """
    k_b, p_b, x_b = jnp.meshgrid(
        jnp.exp(grid.log_k_nodes), jnp.exp(grid.log_p_nodes), grid.x_nodes,
        indexing='ij',
    )
    kwargs = dict(solver=solver)
    if n_steps is not None:
        kwargs['n_steps'] = n_steps
    kwargs.update(solver_kwargs)
    return _mgk.D2_D3_fused_grid_raw(x_b, k_b, p_b, P, xnow, xstop, **kwargs)


def interpolate_fused(k_query, p_query, x_query, grid: FusedCoarseGrid, A_fused, Ap_fused, CFD3, CFD3p,
                       f0=1.0, order=1):
    """Trilinear interpolation of the coarse fused-kernel arrays onto arbitrary
    (k, p, x) input-leg triples -- e.g. the fine R-loop grid.

    ``A_fused``/``Ap_fused`` are interpolated directly (they are about as
    smooth as A/B, and have no f0 dependence at all). ``CFD3``/``CFD3p`` are
    first divided by the analytic ``_ang_factor`` on the coarse grid's own
    (k, p, x) nodes, interpolated as that smoother residual, then multiplied
    back by ``_ang_factor`` at the query points -- see :func:`_ang_factor`
    for why.

    ``f0``: the CURRENT, live growth-rate normalization (``MG_kernels``'s own
    ``f0`` argument, matching ``kfuncs_to_tables.py``'s live call). ``CFD3``
    (input here) was computed by :func:`compute_fused_coarse` with the fixed
    reference ``f0=1`` (see its docstring for why that costs nothing), so the
    interpolated ``CFD3p`` must be divided by the REAL ``f0`` here to recover
    the physically correct, ``kfuncs_to_tables.py``-consistent value -- e.g.
    ``I1udd1_and_P13_grid``'s own ``ApOverf0``/``BpOverf0``/``CFD3p`` outputs
    are likewise normalized by the live ``f0``, not a fixed placeholder.
    Getting this wrong does not show up as noisy/ringing error like a
    spatial-resolution problem -- it is a smooth, systematic bias across the
    whole ``k`` range (checked this session: forgetting it, or using the
    wrong scalar, made an otherwise-smooth model, HDKI/BZ_Mass, look WORSE
    than the crude ``fkpt_approximation=True`` squeezed-limit fallback).

    ``k_query``/``p_query``/``x_query``: broadcastable arrays, any shape.
    Returns ``(A_fused, Ap_fused, CFD3, CFD3p)`` of that broadcast shape.
    """
    k_b, p_b, x_b = jnp.broadcast_arrays(
        jnp.asarray(k_query, dtype=jnp.float64),
        jnp.asarray(p_query, dtype=jnp.float64),
        jnp.asarray(x_query, dtype=jnp.float64),
    )
    i_k = _frac_index(jnp.log(k_b), grid.log_k_nodes)
    i_p = _frac_index(jnp.log(p_b), grid.log_p_nodes)
    i_x = _frac_index(x_b, grid.x_nodes)
    coords = jnp.stack([i_k, i_p, i_x], axis=0)

    k_grid = jnp.exp(grid.log_k_nodes)[:, None, None]
    p_grid = jnp.exp(grid.log_p_nodes)[None, :, None]
    x_grid = grid.x_nodes[None, None, :]
    r_grid = jnp.broadcast_to(p_grid / k_grid, CFD3.shape)
    ang_grid = _ang_factor(r_grid, x_grid)
    CFD3_reduced = CFD3 / ang_grid
    CFD3p_reduced = CFD3p / ang_grid

    A_i, Ap_i, CFD3_red_i, CFD3p_red_i = _interp_grid(
        coords, (A_fused, Ap_fused, CFD3_reduced, CFD3p_reduced), order=order
    )
    ang_q = _ang_factor(p_b / k_b, x_b)
    return A_i, Ap_i, CFD3_red_i * ang_q, (CFD3p_red_i * ang_q) / f0


# ---------------------------------------------------------------------------
# fkpt_emulation route: train Chebyshev emulators for the coarse A/B and
# fused-kernel ingredients over an ARBITRARY MG-parameter box, and bind them
# into a closure matching kfuncs_to_tables.py's `ingredients_fn` hook -- see
# that function's docstring. Generic over the parameter dict on purpose: a
# scale-independent model (mu0) and a 4-parameter binned one (mu1..mu4) use
# the exact same machinery, since only `P_from_params`/`param_bounds` differ.
# ---------------------------------------------------------------------------

class IngredientsProvider:
    """Trained Chebyshev emulators for the coarse A/B (Q-loop) and fused-kernel
    (R-loop) ingredients, plus the coarse grids they were fit on.

    Call :meth:`bind` with the CURRENT MG parameter values to get a zero-extra
    -argument closure suitable for ``kfuncs_to_tables.Kfuncs_to_tables``'s
    ``ingredients_fn=`` hook -- rebuild the binding on every
    parameter-dependent call, mirroring how ``full_shape.py`` rebuilds
    ``mg_kernel_fn`` fresh each time rather than caching it.
    """

    def __init__(self, grid_ab, grid_fused, emu_ab, emu_fused, param_names=(), interp_order=1):
        self.grid_ab = grid_ab
        self.grid_fused = grid_fused
        self.emu_ab = emu_ab
        self.emu_fused = emu_fused
        #: the MG parameter names this provider was trained over -- callers (e.g.
        #: full_shape.py) use this to filter their own full `_mg_kwargs()` dict down to
        #: exactly what `.bind()`/`.predict()` accepts, without needing to know which
        #: subset of names is relevant for a given model/variant.
        self.param_names = tuple(param_names)
        #: 1 (trilinear, default) or 3 (local tricubic, see `_interp_grid_cubic`) --
        #: the k-space interpolation order onto the real fine Q-loop/R-loop grid.
        #: Distinct from the emulator's own MG-parameter-space fit (`emu_ab`/
        #: `emu_fused`'s `budget`/`levels`), which this has no effect on.
        self.interp_order = int(interp_order)

    def predict_coarse(self, **params):
        """Chebyshev-evaluate the coarse A/B and fused-kernel arrays at `params`
        (no interpolation yet) -- exposed mainly for accuracy/timing diagnostics.

        `Ap`/`Bp`/`Apf` are forced to ZERO rather than fit (directly, or reconstructed from
        raw states -- both tried and measured to fail: `Ap = dDA/C - DA*Cp/C^2` is a
        cancellation-prone difference of two comparably-sized terms, ~30x larger than their
        own difference, so even a well-fit ~0.5-2% error on the inputs blows up to 70-80%+
        mean relative error on the combination, however it's computed). An ablation on the
        FULLY LIVE calculation (HDKI/BZ_Mass, no emulator at all) confirmed forcing Ap/Bp/Apf
        to zero moves the final P0/P2 by at most ~0.55%/1.87% (max) -- LESS than the Hybrid
        emulator's own overall error even before this simplification -- so fitting them at
        all was not worth the complexity or the cancellation-driven inaccuracy. `A`/`B`/`Af`
        (well-behaved, no cancellation) and `CFD3`/`CFD3p` (plain ratios) still fit directly.
        """
        ab_pred = self.emu_ab.predict(**params)
        A, B = ab_pred['A'], ab_pred['B']
        ab_pred = {'A': A, 'B': B, 'Ap': jnp.zeros_like(A), 'Bp': jnp.zeros_like(B)}

        fused_pred = self.emu_fused.predict(**params)
        Af = fused_pred['Af']
        fused_pred = {'Af': Af, 'Apf': jnp.zeros_like(Af),
                      'CFD3': fused_pred['CFD3'], 'CFD3p': fused_pred['CFD3p']}

        return ab_pred, fused_pred

    def __call__(self, k_ext_q, q_loop_Q, kminus_Q, x_r, k_r, p_r, f0, **params):
        """The full ingredients computation: predict + interpolate onto the
        (k_ext_q, q_loop_Q, kminus_Q) Q-loop triple (all three cyclic leg
        orderings: Q, N, RQ) and the (k_r, p_r, x_r) R-loop triple, returning
        the same 16-tuple `MG_kernels.I1udd1_and_P13_grid` does.

        ``f0``: the CURRENT, live growth-rate normalization -- REQUIRED (no
        default), because ``kfuncs_to_tables.py``'s own ``I1udd1_and_P13_grid``
        call always has a genuine, just-computed value in scope, and this is
        the single place both families of that normalization (the Q/N/RQ
        family's ``Ap/f0``, ``Bp/f0``, done here; the fused kernel's
        ``CFD3p``, done inside :func:`interpolate_fused`) get applied. Note
        ``A_B_grid`` (the Q/N/RQ family's own live ODE, called by
        :func:`compute_ab_coarse`) has NO f0 concept at all -- ``Ap``/``Bp``
        come back from it, and hence from the coarse grid, completely raw;
        the ``/f0`` has to happen here to match ``I1udd1_and_P13_grid``'s own
        ``_AB`` helper, which divides by the live ``f0`` at the very end.
        """
        ab_pred, fused_pred = self.predict_coarse(**params)
        A_c, B_c, Ap_c, Bp_c = ab_pred['A'], ab_pred['B'], ab_pred['Ap'], ab_pred['Bp']
        Af_c, Apf_c, CFD3_c, CFD3p_c = (fused_pred['Af'], fused_pred['Apf'],
                                        fused_pred['CFD3'], fused_pred['CFD3p'])

        A_Q, B_Q, ApQ_raw, BpQ_raw = interpolate_ab(k_ext_q, q_loop_Q, kminus_Q, self.grid_ab,
                                                     A_c, B_c, Ap_c, Bp_c, order=self.interp_order)
        A_N, B_N, ApN_raw, BpN_raw = interpolate_ab(q_loop_Q, k_ext_q, kminus_Q, self.grid_ab,
                                                     A_c, B_c, Ap_c, Bp_c, order=self.interp_order)
        A_RQ, B_RQ, ApRQ_raw, BpRQ_raw = interpolate_ab(kminus_Q, k_ext_q, q_loop_Q, self.grid_ab,
                                                         A_c, B_c, Ap_c, Bp_c, order=self.interp_order)
        A_fused, Ap_fused, CFD3, CFD3p = interpolate_fused(
            k_r, p_r, x_r, self.grid_fused, Af_c, Apf_c, CFD3_c, CFD3p_c, f0=f0,
            order=self.interp_order)

        return (A_Q, B_Q, ApQ_raw / f0, BpQ_raw / f0,
                A_N, B_N, ApN_raw / f0, BpN_raw / f0,
                A_RQ, B_RQ, ApRQ_raw / f0, BpRQ_raw / f0,
                A_fused, Ap_fused, CFD3, CFD3p)

    def bind(self, **params):
        """A closure over the CURRENT MG parameter values, taking only the
        grid arrays and the live ``f0`` -- for
        ``Kfuncs_to_tables(ingredients_fn=provider.bind(mu1=..., ...))``,
        which supplies ``f0`` itself (see :meth:`__call__`)."""
        def ingredients_fn(k_ext_q, q_loop_Q, kminus_Q, x_r, k_r, p_r, f0):
            return self(k_ext_q, q_loop_Q, kminus_Q, x_r, k_r, p_r, f0, **params)
        return ingredients_fn

    def save(self, path):
        """Write this trained provider to the directory ``path`` (created if
        needed): ``emu_ab.h5``/``emu_fused.h5`` (``cosmoprimo.emulators.tools.
        Emulator``'s own ``write``/``read``, which already handles engine
        state generically), plus ``grid.npz`` (both coarse grids' node
        arrays) and ``meta.json`` (``param_names``, ``interp_order``). Read
        back with :meth:`load`.
        """
        import json
        import os

        os.makedirs(path, exist_ok=True)
        self.emu_ab.write(os.path.join(path, 'emu_ab.h5'))
        self.emu_fused.write(os.path.join(path, 'emu_fused.h5'))
        np.savez(
            os.path.join(path, 'grid.npz'),
            log_kf_nodes=np.asarray(self.grid_ab.log_kf_nodes),
            log_k1_nodes=np.asarray(self.grid_ab.log_k1_nodes),
            log_k2_nodes=np.asarray(self.grid_ab.log_k2_nodes),
            log_k_nodes=np.asarray(self.grid_fused.log_k_nodes),
            log_p_nodes=np.asarray(self.grid_fused.log_p_nodes),
            x_nodes=np.asarray(self.grid_fused.x_nodes),
        )
        with open(os.path.join(path, 'meta.json'), 'w') as f:
            json.dump({'param_names': list(self.param_names), 'interp_order': self.interp_order}, f)

    @classmethod
    def load(cls, path):
        """Read a provider written by :meth:`save`."""
        import json
        import os

        from cosmoprimo.emulators.tools import Emulator as _Emulator

        emu_ab = _Emulator.read(os.path.join(path, 'emu_ab.h5'))
        emu_fused = _Emulator.read(os.path.join(path, 'emu_fused.h5'))
        with np.load(os.path.join(path, 'grid.npz')) as z:
            grid_ab = ABCoarseGrid(
                jnp.asarray(z['log_kf_nodes']), jnp.asarray(z['log_k1_nodes']),
                jnp.asarray(z['log_k2_nodes']),
            )
            grid_fused = FusedCoarseGrid(
                jnp.asarray(z['log_k_nodes']), jnp.asarray(z['log_p_nodes']), jnp.asarray(z['x_nodes']),
            )
        with open(os.path.join(path, 'meta.json')) as f:
            meta = json.load(f)
        return cls(grid_ab, grid_fused, emu_ab, emu_fused,
                    param_names=tuple(meta['param_names']), interp_order=int(meta['interp_order']))


def build_ingredients_provider(kf_min, kf_max, kleg_min, kleg_max, param_bounds, P_from_params,
                                xnow, xstop, budget=4, engine='chebyshev', engine_kwargs=None,
                                space_kwargs=None,
                                ab_grid_kwargs=None, fused_grid_kwargs=None,
                                solver='rk4', n_steps=64, k_attractors=(), interp_order=1,
                                mpicomm=None):
    """Train the A/B + fused-kernel coarse-grid Chebyshev emulators over an
    arbitrary MG-parameter box and return a ready-to-use :class:`IngredientsProvider`.

    Parameters
    ----------
    kf_min, kf_max : float
        Physical range of the output-leg magnitude (A/B's ``kf``, the fused
        kernel's ``k``).
    kleg_min, kleg_max : float
        Physical range the OTHER legs (A/B's ``k1``/``k2``, the fused
        kernel's ``p``) can reach -- typically much wider than
        ``kf_min``/``kf_max`` (match whatever the real Q-loop/R-loop grids
        span, e.g. the quadrature's own ``pmin``/``pmax``).
    param_bounds : dict
        ``{name: (lo, hi)}`` for every MG parameter to emulate over, e.g.
        ``{'mu0': (-1.0, 0.0)}`` or ``{'mu1': (0.5, 1.5), 'mu2': ..., ...}``.
    P_from_params : callable
        ``params (dict of floats) -> mg_jax.MGConstants``, e.g.
        ``lambda params: mg_jax.pack_constants_jnp(om=Om, ol=1-Om, kind=kind, **params)``.
    xnow, xstop : the growth-ODE bounds the coarse solves need (same meaning as
        everywhere else in this module / `MG_kernels.py`). No `f0` here: the
        coarse fused-kernel solve always uses the fixed reference `f0=1`
        (exact, see `compute_fused_coarse`'s docstring) -- the real, live,
        MG-parameter-dependent `f0` is only needed at PREDICT time, which
        `IngredientsProvider.__call__`/`.bind()` require as an explicit
        argument instead.
    solver, n_steps : the coarse-grid ODE solve settings -- default to the
        SAME `solver='rk4', n_steps=64` `kfuncs_to_tables.py` itself uses for
        `I1udd1_and_P13_grid`, so training reproduces the exact live formula
        at the coarse grid's nodes (only the SPATIAL resolution differs, not
        the numerics).
    engine : str, default='chebyshev'
        Which `cosmoprimo.emulators.tools` engine fits the MG-parameter
        dependence: `'chebyshev'` (exact collocation on a tensor-product
        grid; a single bad/NaN node poisons every coefficient) or
        `'polynomial'` (least-squares regression on scattered points; a bad
        node is just a dropped row, at the cost of being exact nowhere).
    engine_kwargs : dict, default=None
        Extra keywords forwarded to the engine on top of `budget` -- e.g.
        `{'order': 3, 'oversampling': 2.0}` for `'polynomial'`. NOT where
        `'chebyshev'`'s own `levels` goes -- see `space_kwargs`.
    space_kwargs : dict, default=None
        Extra keywords forwarded to the `cosmoprimo.emulators.tools.Space`
        that defines the training box -- notably `levels={name: level}`,
        `'chebyshev'`'s PER-PARAMETER resolution (a level ``l`` gives
        ``2**l + 1`` nested nodes; DEFAULTS TO 2 FOR EVERY PARAMETER
        regardless of `budget`, which only caps the Smolyak sum of levels
        across parameters -- raising `budget` alone cannot raise any single
        parameter's own resolution past its `levels` default. Pass e.g.
        `{'levels': {'mu1': 3, 'mu2': 3, 'mu3': 3, 'mu4': 3}}` for a model
        whose A/B ingredients need more than the default per-parameter
        resolution, and raise `budget` to at least that level's own value
        too (the Smolyak admissible set needs `sum(levels) <= budget`, so a
        single axis at level 3 needs `budget >= 3` to ever be reached).
    k_attractors : tuple of float, default=()
        Fixed k-values to cluster extra coarse-grid nodes around on every
        k-type axis (A/B's kf/k1/k2, the fused kernel's k/p) -- pass a
        model's own mu(k) transition points here (e.g. PHENOM/binning's
        ``(k_TGR, k_c, k_S)``) to resolve them directly instead of relying on
        log-uniform density that may or may not land nearby. Harmless
        (a modest number of unused extra points) for a model with smooth
        mu(k).
    interp_order : int, default=1
        1 (trilinear, `jax.scipy.ndimage.map_coordinates` -- JAX supports no
        higher order there) or 3 (local tricubic, `_interp_grid_cubic`): the
        K-SPACE interpolation order onto the real fine Q-loop/R-loop grid at
        QUERY time. Independent of `budget`/`space_kwargs['levels']`, which
        only affect the MG-PARAMETER-space fit -- validated (PHENOM/binning)
        to be the wrong knob for a k-interpolation residual: raising budget
        2->4 and levels 2->4 (properly matched) changed a straddling-
        threshold triangle's error by <0.1% of its own value, while this is
        the one that actually reduces it (3rd order vs 1st, same node set).
    mpicomm : MPI communicator, default=None
        Forwarded to both coarse-grid emulators' own `.train(mpicomm=...)`
        (`cosmoprimo.emulators.tools.Emulator.train` already supports this --
        it scatters the Smolyak/Chebyshev node evaluations, each an
        independent coarse ODE solve, across ranks). Pass e.g.
        `mpicomm=MPI.COMM_WORLD` under `srun --mpi=pmix -n N` for an N-way
        speedup on this function's dominant cost; `None` (default) trains
        single-process.
    """
    from cosmoprimo.emulators.tools import Emulator as _Emulator, Space as _Space

    ab_kwargs = dict(k_attractors=k_attractors)
    ab_kwargs.update(ab_grid_kwargs or {})
    fused_kwargs = dict(k_attractors=k_attractors)
    fused_kwargs.update(fused_grid_kwargs or {})

    grid_ab = build_coarse_grid(kf_min, kf_max, kleg_min, kleg_max, **ab_kwargs)
    grid_fused = build_fused_coarse_grid(kf_min, kf_max, kleg_min, kleg_max, **fused_kwargs)

    class _ABTarget:
        # Fits ONLY A/B directly -- NOT Ap/Bp (fit directly, or reconstructed from raw
        # states -- both tried and measured to fail via cancellation) and NOT the raw states
        # either (only ever needed for that reconstruction). IngredientsProvider.predict_
        # coarse forces Ap/Bp to zero instead -- see its docstring for the ablation that
        # justified this.
        def __call__(self, params):
            params = {name: float(np.atleast_1d(value)[0]) for name, value in params.items()}
            P = P_from_params(params)
            A, B, _Ap, _Bp = compute_ab_coarse(grid_ab, P, xnow, xstop, solver=solver, n_steps=n_steps)
            return {'A': A, 'B': B}

    class _FusedTarget:
        # Same reasoning as _ABTarget above for Apf. CFD3/CFD3p have no such cancellation
        # (plain ratios, not differences) and stay fit directly.
        def __call__(self, params):
            params = {name: float(np.atleast_1d(value)[0]) for name, value in params.items()}
            P = P_from_params(params)
            Af, _Apf, CFD3, CFD3p = compute_fused_coarse(grid_fused, P, xnow, xstop,
                                                            solver=solver, n_steps=n_steps)
            return {'Af': Af, 'CFD3': CFD3, 'CFD3p': CFD3p}

    space = _Space(bounds=param_bounds, **(space_kwargs or {}))
    engine_opts = dict(budget=budget)
    engine_opts.update(engine_kwargs or {})
    emu_ab = _Emulator(_ABTarget(), space, engine=engine, **engine_opts)
    emu_ab.train(mpicomm=mpicomm)
    emu_fused = _Emulator(_FusedTarget(), space, engine=engine, **engine_opts)
    emu_fused.train(mpicomm=mpicomm)

    return IngredientsProvider(grid_ab, grid_fused, emu_ab, emu_fused,
                                param_names=tuple(param_bounds), interp_order=interp_order)
