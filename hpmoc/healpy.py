"""
``healpy`` wrapper
"""

import importlib


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


if importlib.util.find_spec("healpy") is None:
    actual_hp = 'astropy_healpix.healpy'
else:
    actual_hp = 'healpy'

HP_DEFAULTS = {
    'UNSEEN': -1.6375e+30,
}

# TODO make nside2order shim
healpy = LazyMod(actual_hp, HP_DEFAULTS)
