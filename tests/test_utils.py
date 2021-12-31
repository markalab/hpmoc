"Test utility functions."

from pathlib import Path
from operator import mul
import numpy as np
from hpmoc import PartialUniqSkymap
from hpmoc.utils import *

DATA = Path(__file__).absolute().parent.parent/"tests"/"data"


def test_uniq_minimize():
    mo = PartialUniqSkymap.read(DATA/"S191216ap.multiorder.fits",
                                strategy="ligo")
    m = PartialUniqSkymap.read(DATA/"S191216ap.fits.gz", strategy="ligo")
    u, s = uniq_minimize(m.u⃗, m.s⃗)
    assert (mo.u⃗ == u).all()
    assert ((mo.s⃗ - s)/s).max() < 1e-15
