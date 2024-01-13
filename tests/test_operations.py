"Test consistency of MOC-aware diadic operations with flattened versions."

from pathlib import Path
import numpy as np
from hpmoc import PartialUniqSkymap
from hpmoc.utils import *
import mhealpy
import pytest
import operator
import re


DATA = Path(__file__).absolute().parent.parent / "tests" / "data"

GW_SKYMAPS = [
    DATA / "S191216ap.multiorder.fits",
    DATA / "S230629ad.multiorder.fits",
]

GRB_SKYMAPS = [
    DATA / "GRB230512A_IPN_map_hpx_moc_v2.fits.gz",
]


def is_ligo_map(name):
    return bool(
        re.match(r"M?S[0-9]{6}[a-z]+\.(fits\.gz|multiorder\.fits)", name)
    )


@pytest.fixture(scope="module")
def skymaps():
    skymaps = {}
    for path in (*GW_SKYMAPS, *GRB_SKYMAPS):
        if is_ligo_map(path.name):
            strategy = "ligo_old"
        else:
            strategy = "basic"
        s = skymaps[path] = PartialUniqSkymap.read(path, strategy=strategy)
        scramble = np.random.shuffle(np.arange(len(s.u)))
        s.s[:] = s.s[scramble]
        s.u[:] = s.u[scramble]
    yield skymaps


@pytest.mark.parametrize(
    "skymap_1,skymap_2", list(zip(GW_SKYMAPS, GW_SKYMAPS))
)
@pytest.mark.parametrize("op", [operator.mul, operator.add])
def test_moc_fixed_op_consistency(skymap_1, skymap_2, op, skymaps):
    op_then_fixed = op(skymaps[skymap_1], skymaps[skymap_2]).fixed()
    fixed_then_op = op(skymaps[skymap_1].fixed(), skymaps[skymap_2].fixed())
    assert np.allclose(0.0, (op_then_fixed - fixed_then_op).s.value)


@pytest.mark.parametrize(
    "skymap_1,skymap_2",
    [
        (
            DATA / "S230629ad.multiorder.fits",
            DATA / "GRB230512A_IPN_map_hpx_moc_v2.fits.gz",
        ),
        (
            DATA / "S191216ap.multiorder.fits",
            DATA / "S230629ad.multiorder.fits",
        ),
    ],
)
def test_moc_mhealpy_op_consistency(skymap_1, skymap_2, skymaps):
    hpmoc_result = skymaps[skymap_1] * skymaps[skymap_2]

    mhealpy_skymap_1 = mhealpy.HealpixMap.read_map(
        skymap_1, density=is_ligo_map(skymap_1.name)
    )
    mhealpy_skymap_2 = mhealpy.HealpixMap.read_map(
        skymap_2, density=is_ligo_map(skymap_2.name)
    )
    mhealpy_result = mhealpy_skymap_1 * mhealpy_skymap_2

    hpmoc_result = hpmoc_result.reraster(mhealpy_result.uniq)

    assert np.allclose(hpmoc_result.s.value, mhealpy_result.data)
