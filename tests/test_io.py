"File read/write tests."

import os
from pathlib import Path
import numpy as np
from astropy.table import Table
from hpmoc.partial import PartialUniqSkymap
from hpmoc.utils import uniq_minimize, uniq_intersection

DATA = Path(__file__).absolute().parent/'data'


def check_load(mocfits, tablehdf5, strategy='ligo_old'):
    """
    Make sure that the skymap loaded from ``mocfits`` is the same as the known
    good table stored in ``tablehdf5``. This is meant to automate checking that
    known good maps still load the right way; *you need to add new test cases
    manually*. Specify an alternative load ``strategy`` if not reading the
    default provided file type.

    You can make a new table in the ``tablehdf5`` expected format by loading a
    fits file into a ``PartialUniqSkymap`` and confirming that it loaded
    correctly, then dumping it into a table with ``.to_table()`` and saving
    that table using ``.write(tablehdf5, serialize_meta=True,
    compression=True)``. Then, add a new test with the original input
    fits/fits.gz file as ``mocfits`` and the manually-confirmed good table
    output as ``tablehdf5``. Both of these files must be stored in ``./data``.
    """
    good = Table.read(DATA/tablehdf5)
    check = PartialUniqSkymap.read(DATA/mocfits, strategy=strategy)
    checktab = check.to_table()
    assert np.all(good == checktab), "Got unexpected pixel/NUNIQ values."
    for m in check.meta, checktab.meta:
        for k, v in check.meta.items():
            if k != 'HISTORY':
                assert good.meta[k] == v, (f"meta key mismatch: {k}: {v} != "
                                           f"{good.meta[k]} in {infile}")


def compare_new_old_ligo_io_read(infile, mask: PartialUniqSkymap = None):
    """
    Make sure ``hpmoc.partial.LigoIo`` and ``hpmoc.partial.OldLigoIo`` both
    read skymaps in the same way.
    """
    old = (mask or PartialUniqSkymap).read(DATA/infile, strategy='ligo_old')
    new = (mask or PartialUniqSkymap).read(DATA/infile, strategy='ligo')
    iold, inew, do = uniq_intersection(old.u, new.u)
    assert (old.s[iold] == new.s[inew]).all(), (f"old ({old}) != new ({new}) "
                                                f"for {infile}")
    cov_old = uniq_minimize(old.u)[0]
    cov_new = uniq_minimize(new.u)[0]
    assert (cov_old == cov_new).all(), \
        f"old cov ({cov_old}) != new cov ({cov_new}) for {infile}"
    for (k, v), (ko, vo) in zip(new.meta.items(), old.meta.items()):
        if k != 'HISTORY':
            assert k == ko, f"meta key mismatch: {k} != {ko} for {infile}"
            assert v == vo, f"meta val mismatch: {v} != {vo} for {infile}"


# TEST AGAINST KNOWN GOOD SKYMAPS
def test_ligo_o3_ligo_skymap_from_samples_nested_fitsgz_512():
    check_load('S200219ac-3-Update.fits.gz', 'S200219ac-3-Update.hdf5')


def test_ligo_o3_bayestar_nuniq_fitsgz_256():
    check_load('S200105ae.fits.gz', 'S200105ae.hdf5')


def test_ligo_o3_bayestar_nuniq_fits_256():
    check_load('S200105ae.fits', 'S200105ae.hdf5')


def test_ligo_o3_bayestar_nested_fitsgz_1024():
    check_load('S200316bj-1-Preliminary.fits.gz',
               'S200316bj-1-Preliminary.hdf5')


def test_ligo_o3_cwb_nested_fitsgz_128():
    check_load('S200114f-3-Initial.fits.gz', 'S200114f-3-Initial.hdf5')


def test_ligo_o3_cwb_ring_fitsgz_128():
    check_load('S200129m-3-Initial.fits.gz', 'S200129m-3-Initial.hdf5')


# TEST OLD/NEW IO
def test_io_ligo_o3_ligo_skymap_from_samples_nested_fitsgz_512():
    compare_new_old_ligo_io_read('S200219ac-3-Update.fits.gz')


def test_io_ligo_o3_bayestar_nuniq_fitsgz_256():
    compare_new_old_ligo_io_read('S200105ae.fits.gz')


def test_io_ligo_o3_bayestar_nuniq_fits_256():
    compare_new_old_ligo_io_read('S200105ae.fits')


def test_io_ligo_o3_bayestar_nested_fitsgz_1024():
    compare_new_old_ligo_io_read('S200316bj-1-Preliminary.fits.gz')


def test_io_ligo_o3_cwb_nested_fitsgz_128():
    compare_new_old_ligo_io_read('S200114f-3-Initial.fits.gz')


def test_io_ligo_o3_cwb_ring_fitsgz_128():
    compare_new_old_ligo_io_read('S200129m-3-Initial.fits.gz')

def test_gracedb():
    local = PartialUniqSkymap.read(DATA/'S200105ae.fits', strategy='ligo')
    remote = PartialUniqSkymap.read('S200105ae', 'bayestar.multiorder.fits,0',
                                    strategy='gracedb')
    assert (local.u == remote.u).all(), "Gracedb indices read failed."
    assert (local.s == remote.s).all(), "Gracedb pixel read failed."
    assert local.meta == remote.meta, "Gracedb meta read failed."
