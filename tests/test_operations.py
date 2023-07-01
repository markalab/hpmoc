"Test consistency of MOC-aware diadic operations with flattened versions."

from pathlib import Path
import numpy as np
from hpmoc import PartialUniqSkymap
from hpmoc.utils import *
import mhealpy
import pytest
import operator
import re


DATA = Path(__file__).absolute().parent.parent/"tests"/"data"

GW_SKYMAPS = [
    DATA / "S191216ap.multiorder.fits",
    DATA / "S230629ad.multiorder.fits",
]

GRB_SKYMAPS = [
    DATA / "GRB230512A_IPN_map_hpx_moc_v2.fits.gz",
]

def is_ligo_map(name):
    return bool(re.match(r"M?S[0-9]{6}[a-z]+\.(fits\.gz|multiorder\.fits)", name))

@pytest.fixture(scope="module")
def skymaps_cache():
    skymaps = {}
    for path in (*GW_SKYMAPS, *GRB_SKYMAPS):
        if is_ligo_map(path.name):
            strategy = "ligo"
        else:
            strategy = "basic"
        skymaps[path] = PartialUniqSkymap.read(path, strategy=strategy)
    yield skymaps

@pytest.mark.parametrize('skymap_1,skymap_2', list(zip(GW_SKYMAPS, GW_SKYMAPS)))
@pytest.mark.parametrize('op', [operator.mul, operator.add])
def test_moc_fixed_op_consistency(skymap_1, skymap_2, op, skymaps):
    op_then_fixed = op(skymaps[skymap_1], skymaps[skymap_2]).fixed()
    fixed_then_op = op(skymaps[skymap_1].fixed(), skymaps[skymap_2].fixed())
    assert np.allclose(0., (op_then_fixed - fixed_then_op).s.value)
