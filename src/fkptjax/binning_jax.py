"""
binning_jax.py
--------------
DEPRECATED ALIAS for :mod:`fkptjax.mg_jax`.

This module used to hold the PHENOM/binning JAX right-hand sides.  They now live
in :mod:`fkptjax.mg_jax`, which supports HDKI/BZ_Mass as well and selects the
model with a static ``MGConstants.kind`` rather than a packed float.  Everything
that was importable from here still is, and ``pack_constants_jnp`` keeps its old
signature (``kind`` defaults to ``'binning'``), so existing binning code needs no
change.

The one removal: the numpy ``pack_constants`` that built the flat float64 layout.
It had no callers -- the numba path uses :func:`fkptjax.binning_numba.pack_constants`,
which is a separate, unchanged implementation -- and the JAX side no longer has a
flat layout to build.  It is re-exported from ``binning_numba`` here so that any
notebook still reaching for ``binning_jax.pack_constants`` gets the identical
array it always did.

Prefer ``from fkptjax import mg_jax`` in new code.
"""

from .binning_numba import pack_constants                      # noqa: F401
from .mg_jax import (                                          # noqa: F401
    BINNING, BZ_MASS, KINDS, MGConstants, pack_constants_jnp,
    f1, kpp, mu,
    S2a, S2b, S2FL, SD2,
    S3IIplus, S3FLplus, S3I, S3II, S3FL,
    firstOrder, secondOrder, thirdOrder,
)

__all__ = [
    'BINNING', 'BZ_MASS', 'KINDS', 'MGConstants',
    'pack_constants', 'pack_constants_jnp',
    'f1', 'kpp', 'mu',
    'S2a', 'S2b', 'S2FL', 'SD2',
    'S3IIplus', 'S3FLplus', 'S3I', 'S3II', 'S3FL',
    'firstOrder', 'secondOrder', 'thirdOrder',
]
