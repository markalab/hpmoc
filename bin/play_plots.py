#!/usr/bin/env python

from pathlib import Path
import numpy as np
from astropy.table import Table

# from ligo.skymap.io import read_sky_map
import matplotlib.pyplot as plt
import hpmoc
from hpmoc.partial import PartialUniqSkymap

ROOT = Path(__file__).absolute().parent.parent


def main():
    m = PartialUniqSkymap.read(
        ROOT / "tests" / "data" / "S191216ap.fits.gz", strategy="ligo"
    )
    m.plot()
    fig = plt.gcf()
    fig.set_facecolor("white")


if __name__ == "__main__":
    main()
