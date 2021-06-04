"Test ``healpy`` shim."

import importlib


def test_nside2order():
    """
    ``nside2order`` is scalar-only in ``healpy`` for some reason.
    Only run if ``healpy`` is installed.
    """
    if importlib.util.find_spec("healpy") is None:
        return
    import numpy as np
    from hpmoc.healpy import nside2order
    import healpy as real_hp

    ns = real_hp.order2nside(np.arange(30))
    for n in ns:
        assert real_hp.nside2order(n) == nside2order(n)


def test_installed_nside2order():
    "Test ``nside2order``, whether real ``healpy`` or shim version."
    from hpmoc.healpy import healpy as hp

    for o in range(30):
        assert hp.nside2order(1<<o) == o
    for bad in [-1, 256., 33, 1<<30]:
        try:
            hp.nside2order(bad)
            assert False
        except (TypeError, ValueError):
            pass
