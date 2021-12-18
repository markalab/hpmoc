"Test ``healpy`` shim."

from importlib.util import find_spec

MAX = 10000


def test_nside2order():
    """
    ``nside2order`` is scalar-only in ``healpy`` for some reason.
    Only run if ``healpy`` is installed.
    """
    if find_spec("healpy") is None:
        return
    import numpy as np
    from hpmoc.healpy import nside2order
    try:
        import healpy as real_hp
    except ImportError:
        assert False, "healpy seemingly installed but import failed."

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


def test_pix2xyf():
    """
    Test ``pix2xyf`` against ``healpy`` version. Only run if ``healpy`` is
    installed.
    """
    if find_spec("healpy") is None:
        return
    import numpy as np
    from hpmoc.healpy import pix2xyf
    try:
        import healpy as real_hp
    except ImportError:
        assert False, "healpy seemingly installed but import failed."

    np.random.seed(0)
    for order in range(30):
        nside = 1<<order
        npix = 12*nside*nside
        # test collections of pixels
        i = np.arange(npix) if npix < MAX else np.random.randint(0, npix, MAX,
                                                                 np.uint64)
        for nest in [True, False]:
            assert (pix2xyf(nside, i, nest=nest) ==
                    real_hp.pix2xyf(nside, i, nest=nest)).all()
        assert (pix2xyf(nside, i) == real_hp.pix2xyf(nside, i)).all()
        for s in [0, np.random.randint(0, npix), npix-1]:
            for nest in [True, False]:
                assert (pix2xyf(nside, s, nest=nest) ==
                        real_hp.pix2xyf(nside, s, nest=nest))
            assert pix2xyf(nside, s) == real_hp.pix2xyf(nside, s)


def test_xyf2pix():
    """
    Test ``xyf2pix`` against ``healpy`` version. Only run if ``healpy`` is
    installed.
    """
    if find_spec("healpy") is None:
        return
    import numpy as np
    from hpmoc.healpy import xyf2pix
    try:
        import healpy as real_hp
    except ImportError:
        assert False, "healpy seemingly installed but import failed."

    np.random.seed(0)
    for order in range(30):
        nside = 1<<order
        npix = 12*nside*nside
        count = min(MAX, nside)
        # test collections of pixels
        x = np.random.randint(0, nside, count, np.uint64)
        y = np.random.randint(0, nside, count, np.uint64)
        f = np.random.randint(0, 12, count, np.uint64)
        for nest in [True, False]:
            assert (xyf2pix(nside, x, y, f, nest=nest) ==
                    real_hp.xyf2pix(nside, x, y, f, nest=nest)).all()
        assert (xyf2pix(nside, x, y, f) ==
                real_hp.xyf2pix(nside, x, y, f)).all()
        for s in [0, np.random.randint(0, nside, nside, dtype=np.uint64), npix-1]:
            for nest in [True, False]:
                assert (xyf2pix(nside, s, s, 0, nest=nest) ==
                        real_hp.xyf2pix(nside, s, s, 0, nest=nest))
            assert xyf2pix(nside, s, s, 0) == real_hp.xyf2pix(nside, s, s, 0)


def test_installed_pix2xyf_xyf2pix():
    """
    Test ``pix2xyf`` and ``xyf2pix`` shims against each other for
    invertibility. Can run without ``healpy`` installed.
    """
    import numpy as np
    from hpmoc.healpy import pix2xyf, xyf2pix

    np.random.seed(0)
    for order in range(30):
        nside = 1<<order
        npix = 12*nside*nside
        # test collections of pixels
        i = np.arange(npix) if npix < MAX else np.random.randint(0, npix, MAX,
                                                                 np.uint64)
        for nest in [True, False]:
            converted = xyf2pix(nside, *pix2xyf(nside, i, nest=nest), nest=nest)
            assert np.issubdtype(converted.dtype, np.uint64)
            assert np.all(converted == i)
        assert (xyf2pix(nside, *pix2xyf(nside, i)) == i).all()
        for s in [0, np.random.randint(0, npix, dtype=np.uint64), npix-1]:
            for nest in [True, False]:
                assert (xyf2pix(nside, *pix2xyf(nside, s, nest=nest),
                                nest=nest) == s)
            assert xyf2pix(nside, *pix2xyf(nside, s)) == s
