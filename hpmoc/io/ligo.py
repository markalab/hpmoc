# Copyright (C) 2022-2023 Columbia Experimental Gravity Group (GECo)

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""
Strategies for loading LIGO/Virgo/KAGRA skymaps into ``PartialUniqSkymap``
instances.
"""

from typing import Optional, Union, IO, TYPE_CHECKING
from hpmoc.io.abstract import IoStrategy
from hpmoc.io.fits import load_ligo
from hpmoc.partial import PartialUniqSkymap
from hpmoc.utils import uniq_coarsen, uniq_minimize

import numpy as np
from numpy.typing import ArrayLike

class LigoIo(IoStrategy):
    """
    Read/write files in the format used by LIGO/Virgo for their skymaps.
    """

    @classmethod
    def read(
        cls,
        skymap: Optional[Union[PartialUniqSkymap, ArrayLike]],
        file: Union[IO, str],
        *args,
        name: Optional[str] = None,
        coarsen: Optional[int] = None,
        **kwargs
    ):
        """
        Read a file saved in the format used by LIGO/Virgo for their skymaps.

        Parameters
        ----------
        mask : PartialUniqSkymap or array, optional
            Only read in pixels overlapping with ``mask``.
        file : file or str
            The file object or filename to read from. Can be a stream as no
            seeking will be performed.
        name : str, optional
            The column-name of the pixel data. Defaults to 'PROBDENSITY'
        coarsen : int, optional
            If provided, coarsen the ``mask`` by up to this many HEALPix
            orders (up to order 0) to speed up read times. This will select
            a superset of the sky region defined in ``mask``.
        *args, **kwargs
            Arguments to pass on to ``hpmoc.fits.load_ligo``.
        """
        if name is None:
            name = 'PROBDENSITY'

        pt = skymap.point_sources if isinstance(skymap, PartialUniqSkymap) else []
        if skymap is not None:
            mask = skymap.u if isinstance(skymap, PartialUniqSkymap) else skymap
            mask = np.asarray(mask, dtype=np.int64)
            mask = uniq_coarsen(mask, coarsen) if coarsen is not None else mask
            mask = uniq_minimize(mask)[0]
        else:
            mask = None
        [[u, s, meta]] = load_ligo(file, mask=mask, **kwargs)
        return PartialUniqSkymap(s, u, name=name, meta=meta, point_sources=pt)

    @classmethod
    def write(
        cls,
        skymap: PartialUniqSkymap,
        file: Union[IO, str],
        name: Optional[str] = None,
        *args,
        **kwargs
    ):
        """
        Write a skymap to file in the format used by LIGO/Virgo for their
        skymaps. A thin wrapper around ``BasicIo.write``.
        """
        from hpmoc.io import BasicIo

        BasicIo.write(skymap, file, name=name, *args, **kwargs)
