"Test utility functions."

from pathlib import Path
from hpmoc import PartialUniqSkymap
from hpmoc.utils import *
import numpy as np

DATA = Path(__file__).absolute().parent.parent/"tests"/"data"


def test_uniq_minimize():
    mo = PartialUniqSkymap.read(DATA/"S191216ap.multiorder.fits",
                                strategy="ligo")
    m = PartialUniqSkymap.read(DATA/"S191216ap.fits.gz", strategy="ligo")
    u, s = uniq_minimize(m.u, m.s)
    assert (mo.u == u).all()
    assert np.all(np.isclose(mo.s, s))
