# Copyright (C) 2019-2023 Columbia Experimental Gravity Group (GECo)

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""
``healpy`` wrapper
"""

import importlib
from .healpy_utils import alt_compress, alt_expand


_ANG2VEC_TRANSPOSED = [None]
_BOUNDARIES_SCALARIZED = [None]
EXCEPTIONS = {}


class LazyMod:

    def __init__(self, mod, defaults, exceptions=None):
        self._mod = mod
        self._defaults = defaults
        self._exceptions = exceptions if exceptions is not None else {}

    def __getattr__(self, name):
        if name in self._exceptions:
            return self._exceptions[name]
        try:
            return getattr(importlib.import_module(self._mod), name)
        except AttributeError as ae:
            try:
                return self._defaults[name]
            except KeyError:
                raise ae

    def __dir__(self):
        return sorted({*dir(importlib.import_module(self._mod)),
                       *self._defaults.keys(), *self._exceptions.keys()})


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
    ACTUAL_HP = 'astropy_healpix.healpy'
else:
    ACTUAL_HP = 'healpy'


def ang2vec(theta, phi, lonlat=False):
    """
    A drop-in replacement for ``healpy.ang2vec`` that preserves the
    ``healpy`` row, column ordering (since ``astropy_healpix.healpy.ang2vec``
    returns a transposed version). Dynamically checks behavior in case
    ``astropy_healpix`` fixes its return value to conform to ``healpy``
    behavior at some point in the future.

    See Also
    --------
    healpy.ang2vec
    astropy_healpix.healpy.ang2vec
    """
    from astropy_healpix.healpy import ang2vec

    # test behavior and set default in case they fix this in the future
    if _ANG2VEC_TRANSPOSED[0] is None:
        transposed = ang2vec([1, 2], [3, 4], lonlat=False).shape == (3, 2)
        _ANG2VEC_TRANSPOSED[0] = transposed
    ans = ang2vec(theta, phi, lonlat=lonlat)
    return ans.T if _ANG2VEC_TRANSPOSED[0] else ans


def boundaries(nside, pix, step=1, nest=False):
    """
    A drop-in replacement for ``healpy.boundaries`` that preserves the shape of
    the returned result. ``astropy_healpix.healpy.boundaries`` will return a
    shape of ``(3, 4)`` when given a length-1 vector of pixel indices, whereas
    the original will return a shape of ``(n, 3, 4)`` where ``n`` is the length
    of the input list, even in the case when ``n=1``. This function conforms to
    the ``healpy.boundaries`` behavior.

    See Also
    --------
    healpy.boundaries
    astropy_healpix.healpy.boundaries
    """
    from astropy_healpix.healpy import boundaries

    # test behavior and set default in case they fix this in the future
    if _BOUNDARIES_SCALARIZED[0] is None:
        shape = boundaries(1, [0]).shape
        _BOUNDARIES_SCALARIZED[0] = shape != (1, 3, 4)
        assert shape in ((1, 3, 4), (3, 4)), "Unexpected shape"
    ans = boundaries(nside, pix, step=step, nest=nest)
    return ans.reshape((-1, 3, 4)) if _BOUNDARIES_SCALARIZED[0] else ans


if ACTUAL_HP == 'astropy_healpix.healpy':
    EXCEPTIONS['ang2vec'] = ang2vec
    EXCEPTIONS['boundaries'] = boundaries


HP_DEFAULTS = {
    'UNSEEN': -1.6375e+30,
    'nside2order': nside2order,
    'pix2xyf': pix2xyf,
    'xyf2pix': xyf2pix,
}

healpy = LazyMod(ACTUAL_HP, HP_DEFAULTS, EXCEPTIONS)
