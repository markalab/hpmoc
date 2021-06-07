"""
``healpy`` wrapper
"""

import importlib
from math import log2, floor
from .healpy_utils import alt_compress, alt_expand


class LazyMod:

    def __init__(self, mod, defaults):
        self._mod = mod
        self._defaults = defaults

    def __getattr__(self, name):
        try:
            return getattr(importlib.import_module(self._mod), name)
        except AttributeError as ae:
            try:
                return self._defaults[name]
            except KeyError:
                raise ae

    def __dir__(self):
        return dir(importlib.import_module(self._mod))+[*self._defaults.keys()]


def nside2order(nside):
    """
    Drop-in replacement for healpy `~healpy.pixelfunc.nside2order`.
    """
    if nside > 0 and nside < 1<<30:
        res = len(f"{nside:b}")-1
        if 1<<res == nside:
            return res
    raise ValueError(f"{nside} is not a valid nside parameter (must be an "
                     "integral power of 2, less than 2**30)")


def pix2xyf(nside, ipix, nest=False):
    "Drop-in replacement for healpy `~healpy.pixelfunc.pix2xyf`."
    import numpy as np

    nside = np.uint64(nside)
    # Check for mistake in ``astropy_healpix`` scalar handling
    scalar = np.isscalar(ipix)
    ipix = np.uint64(ipix) if scalar else ipix.astype(np.uint64)
    # Original healpy expects int64 only; uints cause problems here
    ipix = ipix if nest else healpy\
        .ring2nest(int(nside), ipix.astype(np.int64)).astype(np.uint64)
    if scalar and not np.isscalar(ipix):
        ipix = ipix.ravel()[0]
    ipix = np.uint64(ipix) if scalar else ipix.astype(np.uint64)
    nsq = nside*nside
    f = ipix//nsq
    #f = np.uint64(f) if scalar else f.astype(np.uint64)
    i = ipix-f*nsq
    return alt_compress(i), alt_compress(i>>np.uint64(1), True), f


def xyf2pix(nside, x, y, face, nest=False):
    "Drop-in replacement for healpy `~healpy.pixelfunc.xyf2pix`."
    import numpy as np

    # Check for mistake in ``astropy_healpix`` scalar handling
    scalar = all(map(np.isscalar, [x, y, face]))
    # mixed int type products are cast to float; everything must be uint64
    nside = np.uint64(nside)
    face = np.uint64(face) if np.isscalar(face) else face.astype(np.uint64)
    ipix = alt_expand(x) + (alt_expand(y) << np.uint64(1)) + face*nside*nside
    assert isinstance(ipix, np.uint64) or np.issubdtype(ipix.dtype, np.uint64)
    # Original healpy expects int64 only; uints cause problems here.
    ipix = ipix if nest else healpy\
        .nest2ring(int(nside), ipix.astype(np.int64)).astype(np.uint64)
    return ipix.ravel()[0] if scalar and not np.isscalar(ipix) else ipix


if importlib.util.find_spec("healpy") is None:
    actual_hp = 'astropy_healpix.healpy'
else:
    actual_hp = 'healpy'

HP_DEFAULTS = {
    'UNSEEN': -1.6375e+30,
    'nside2order': nside2order,
    'pix2xyf': pix2xyf,
    'xyf2pix': xyf2pix,
}

healpy = LazyMod(actual_hp, HP_DEFAULTS)
