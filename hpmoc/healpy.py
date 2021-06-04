"""
``healpy`` wrapper
"""

import importlib
from math import log2, floor


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
    Work-alike replacement for ``healpy.nside2order``. Might be a little bit
    more forgiving.
    """
    if nside > 0 and nside < 1<<30:
        res = floor(log2(nside)+.5)
        if 1<<res == nside:
            return res
    raise ValueError(f"{nside} is not a valid nside parameter (must be an "
                     "integral power of 2, less than 2**30)")


if importlib.util.find_spec("healpy") is None:
    actual_hp = 'astropy_healpix.healpy'
else:
    actual_hp = 'healpy'

HP_DEFAULTS = {
    'UNSEEN': -1.6375e+30,
    'nside2order': nside2order,
}

healpy = LazyMod(actual_hp, HP_DEFAULTS)
