"""
template_emulation.py
----------------------
Emulates ONLY the genuinely Boltzmann-derived, non-JAX-native pieces of
``desilike.theories.galaxy_clustering.template.DirectSpectrum2Template``
(the linear P(k), P(k)-nowiggle, the AP dilation factors, sigma8, the
background ``Omega_m``, and the growth quantities ``fk``/``f0``) -- the
"emulate_fkpt=True" route's counterpart to ``ab_ingredients.py``'s
beyond-EdS kernel emulator.

Motivation (session notes, ``emulate_fkpt`` design discussion): once the
beyond-EdS kernel itself is emulated (``ab_ingredients.IngredientsProvider``),
fkptjax's OWN loop-integral/RSD/bias-combination code is already pure JAX --
fast and jit+vmap-native given cheap A/B/A_fused/CFD3 and cheap P(k)/f(k)/AP
inputs. The only remaining non-JAX-native piece is the ``template`` itself
(a live Boltzmann/ISiTGR call). Wrapping the WHOLE ``pt`` node in a Taylor
expansion (the older ``create_pt_emulator``/``build_pt_emulator_inmemory``
route) is therefore redundant with the beyond-EdS piece already handled, and
was found (this session, and historically per the user) to not generalize
well across all models anyway -- hence this narrower, template-only route.

Design: rather than trying to produce a drop-in ``DirectSpectrum2Template``
via desilike's own ``Emulator``/``CalculatorEmulator`` machinery (which hits
a real gap -- ``DirectSpectrum2Template.tree_flatten()`` does not include its
own ``cosmo`` dependency, so ``self.template.cosmo['Omega_m']`` in
``full_shape.py``'s own ``FKPTJAXPTSpectrum2Poles.__call__`` would still hit
a live, expensive, freshly-reconstructed ``cosmo`` even after "emulating"
``template`` -- verified this session by tracing ``full_shape.py``), this
module instead reuses ``ab_ingredients.py``'s OWN pattern: a plain Python
callable target passed to ``cosmoprimo.emulators.tools.Emulator``, trained
to output a plain dict of numpy arrays (``pk_dd``, ``pknow_dd``, ``qpar``,
``qper``, ``sigma8``, ``fsigma8``, ``sigma8_fid``, ``f``, ``fk``, ``f0``, and
-- critically -- ``Omega_m`` alongside them, sidestepping the
``tree_flatten()`` gap entirely since ``Omega_m`` just becomes one more thing
this emulator predicts, no hand-derived closed-form formula needed).
:class:`EmulatedTemplate` then wraps a PREDICTION of that dict into a
duck-typed stand-in for ``DirectSpectrum2Template`` itself -- exposing
exactly the attributes/methods ``FKPTJAXPTSpectrum2Poles.__call__`` reads
(verified this session, ``full_shape.py:3141-3226,3303-3305``: ``.qpar``,
``.qper``, ``.z``, ``.k``, ``.pk_dd``, ``.pknow_dd``, ``.sigma8``,
``.fsigma8``, ``.sigma8_fid``, ``.f``, ``.fk``, ``.f0``, ``.ap_k_mu(k, mu)``,
``.cosmo['Omega_m']``, and a no-op ``.update(**kwargs)`` for
``__post_init__``'s one-time ``self.template.update(with_now=...)`` call) --
so it can be passed directly as ``template=`` with NO changes needed to
``full_shape.py``/desilike at all.

Growth source: ``fk``/``f0`` ARE emulated too (as of the ``growth_source=
'template'`` support added below), since ``DirectSpectrum2Template.__call__``
already computes both unconditionally as ``sqrt(P_theta/P_delta)`` (``fk`` on
``self.k``, ``f0`` at ``k0=1e-3``) regardless of which ``growth_source`` ends
up consuming them -- there was nothing extra to compute, only to also capture
into the trained dict. Both ``growth_source`` options remain fully supported:
``'ode'`` (fkptjax's own internal growth ODE -- already JAX-native/fast, and
ignores ``template.fk``/``.f0`` entirely) and ``'template'`` (reads the
emulated ``fk``/``f0`` straight from here). Pick whichever at run time; this
module makes no assumption about which one a given run uses.

Scope note (neutrino corrections still NOT emulated): this route still
REQUIRES ``include_neutrino_corrections=False`` regardless of
``growth_source``: that flag's only effect route (this session's tracing of
``full_shape.py``/``kfuncs_to_tables.py``) is through the internal growth
ODE via a live ``cosmo.get_transfer()`` call that would itself need its own,
separate 2D (k, eta) emulator to stay JAX-native -- a real, deferred followup,
not something this module attempts.
"""
import os
import json

import numpy as np
import jax.numpy as jnp

from desilike.theories.galaxy_clustering.template import _ap_k_mu


class _Cosmo:
    """Duck-typed stand-in for ``template.cosmo`` -- ``FKPTJAXPTSpectrum2Poles``
    only ever reads ``['Omega_m']`` off it (``full_shape.py:3145,3303``)."""

    def __init__(self, Omega_m):
        self.Omega_m = Omega_m

    def __getitem__(self, name):
        if name == 'Omega_m':
            return self.Omega_m
        raise KeyError(
            f"EmulatedTemplate's cosmo stand-in only exposes 'Omega_m', not {name!r} "
            "-- add it to template_emulation.py's emulated quantity list if needed."
        )


class EmulatedTemplate:
    """Duck-typed, fully-JAX-native stand-in for ``DirectSpectrum2Template``,
    populated from a trained :class:`TemplateProvider`'s prediction. Pass
    directly as ``FKPTJAXPTSpectrum2Poles(template=EmulatedTemplate(...),
    growth_source='ode' or 'template', ...)`` -- no other changes needed;
    both growth sources are supported (see module docstring).
    """

    def __init__(self, k, pk_dd, pknow_dd, qpar, qper, sigma8, fsigma8, sigma8_fid, f, z, Omega_m,
                 fk, f0):
        self.k = k
        self.pk_dd = pk_dd
        self.pknow_dd = pknow_dd
        self.qpar = qpar
        self.qper = qper
        self.sigma8 = sigma8
        self.fsigma8 = fsigma8
        self.sigma8_fid = sigma8_fid
        self.f = f
        self.z = z
        self.cosmo = _Cosmo(Omega_m=Omega_m)
        #: only read when growth_source='template' (full_shape.py:3161-3167);
        #: 'ode' ignores these entirely.
        self.fk = fk
        self.f0 = f0

    def update(self, **kwargs):
        """No-op: this stand-in is already fully populated at construction --
        harmless for ``FKPTJAXPTSpectrum2Poles.__post_init__``'s own
        ``self.template.update(with_now='peakaverage')`` call."""
        return self

    def ap_k_mu(self, k, mu):
        """Identical formula desilike's own template classes use
        (``template.py``'s own ``_ap_k_mu``, imported directly here rather
        than re-derived, so this can never drift from the real AP formula)."""
        return _ap_k_mu(k, mu, self.qpar, self.qper)


# The full set of quantities FKPTJAXPTSpectrum2Poles.__call__ can read off
# `template`, covering BOTH growth_source options (fk/f0 only consumed by
# 'template'; harmless to always carry for 'ode' too). Order matters only for
# readability; TemplateProvider.predict looks these up by name.
_TEMPLATE_QUANTITIES = (
    'pk_dd', 'pknow_dd', 'qpar', 'qper', 'sigma8', 'fsigma8', 'sigma8_fid', 'f', 'Omega_m',
    'fk', 'f0',
)


class TemplateProvider:
    """Trained coarse-grid (Chebyshev/Taylor/polynomial) emulator over the
    free COSMOLOGICAL (and any background-affecting MG, e.g. NDGP's r_c)
    parameters, predicting an :class:`EmulatedTemplate` on demand.
    """

    def __init__(self, emu, k, z, param_names=()):
        self.emu = emu
        self.k = k
        self.z = z
        self.param_names = tuple(param_names)

    def predict(self, **params):
        pred = self.emu.predict(**params)
        return EmulatedTemplate(
            k=self.k,
            pk_dd=jnp.asarray(pred['pk_dd']),
            pknow_dd=jnp.asarray(pred['pknow_dd']),
            qpar=jnp.asarray(pred['qpar']).reshape(()),
            qper=jnp.asarray(pred['qper']).reshape(()),
            sigma8=jnp.asarray(pred['sigma8']).reshape(()),
            fsigma8=jnp.asarray(pred['fsigma8']).reshape(()),
            sigma8_fid=jnp.asarray(pred['sigma8_fid']).reshape(()),
            f=jnp.asarray(pred['f']).reshape(()),
            z=self.z,
            Omega_m=jnp.asarray(pred['Omega_m']).reshape(()),
            fk=jnp.asarray(pred['fk']),
            f0=jnp.asarray(pred['f0']).reshape(()),
        )

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.emu.write(os.path.join(path, 'emu_template.h5'))
        np.savez(os.path.join(path, 'meta.npz'), k=np.asarray(self.k), z=np.asarray(self.z))
        with open(os.path.join(path, 'params.json'), 'w') as f:
            json.dump(list(self.param_names), f)

    @classmethod
    def load(cls, path):
        from cosmoprimo.emulators.tools import Emulator as _Emulator

        emu = _Emulator.read(os.path.join(path, 'emu_template.h5'))
        with np.load(os.path.join(path, 'meta.npz')) as z_:
            k = jnp.asarray(z_['k'])
            z = float(z_['z'])
        with open(os.path.join(path, 'params.json')) as f:
            param_names = tuple(json.load(f))
        return cls(emu, k, z, param_names=param_names)


def build_template_provider(param_bounds, build_template_fn, budget=4, engine='chebyshev',
                             engine_kwargs=None):
    """Train a :class:`TemplateProvider`.

    Parameters
    ----------
    param_bounds : dict
        ``{name: (lo, hi)}`` for every free parameter to train over (the
        free cosmological parameters, e.g. h/omega_b/omega_cdm/logA/n_s, plus
        any MG parameter that also affects ISiTGR's own background/AP -- see
        this session's notes: HS's fR0_HS, NDGP's r_c, HDKI/BZ's
        beta_1/lambda_1/exp_s, PHENOM/binning's mu1-4 all route into
        ``isitgr_options``/``build_cosmology``, not purely into
        ``pt._mg_kwargs()``).
    build_template_fn : callable
        ``params (dict of floats) -> a LIVE, ALREADY-EVALUATED
        DirectSpectrum2Template`` (i.e. ``.pk_dd``/``.qpar``/etc already
        readable) -- the caller's own responsibility, since building one
        needs the full cosmology-construction context (fiducial, tracer z,
        engine='isitgr', MG kind, fixed vs. free parameter values, ...) this
        module has no business knowing about. Mirrors
        ``ab_ingredients.build_ingredients_provider``'s own
        ``P_from_params`` callback.
    budget, engine, engine_kwargs : see ``ab_ingredients.build_ingredients_provider``.
    """
    from cosmoprimo.emulators.tools import Emulator as _Emulator, Space as _Space

    k_holder: dict = {}

    class _TemplateTarget:
        def __call__(self, params):
            params = {name: float(np.atleast_1d(value)[0]) for name, value in params.items()}
            template = build_template_fn(params)
            if 'k' not in k_holder:
                k_holder['k'] = np.asarray(template.k)
                k_holder['z'] = float(template.z)
            return {
                'pk_dd': np.asarray(template.pk_dd),
                'pknow_dd': np.asarray(template.pknow_dd),
                'qpar': np.asarray(template.qpar),
                'qper': np.asarray(template.qper),
                'sigma8': np.asarray(template.sigma8),
                'fsigma8': np.asarray(template.fsigma8),
                'sigma8_fid': np.asarray(template.sigma8_fid),
                'f': np.asarray(template.f),
                'Omega_m': np.asarray(template.cosmo['Omega_m']),
                'fk': np.asarray(template.fk),
                'f0': np.asarray(template.f0),
            }

    space = _Space(bounds=param_bounds)
    engine_opts = dict(budget=budget)
    engine_opts.update(engine_kwargs or {})
    emu = _Emulator(_TemplateTarget(), space, engine=engine, **engine_opts)
    emu.train()

    return TemplateProvider(emu, jnp.asarray(k_holder['k']), k_holder['z'],
                             param_names=tuple(param_bounds))
