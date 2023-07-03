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
Read/Write methods for ``hpmoc.partial.PartialUniqSkymap``.
"""

from typing import Optional, Union, IO
from .abstract import IoStrategy
from .astroquery import AstroqueryIo
from .gracedb import GracedbIo
from .ligo import LigoIo
from ..points import PointsTuple
from ..partial import PartialUniqSkymap
from ..utils import uniq2order, read_partial_skymap

import numpy as np
from numpy.typing import ArrayLike


class BasicIo(IoStrategy):
    """
    Read/write files saved in the default format used by ``PartialUniqSkymap``.
    """

    #FIXME add mask
    @classmethod
    def read(
        cls,
        skymap: Optional[Union[PartialUniqSkymap, ArrayLike]],
        file: Union[IO, str],
        *args,
        name: Optional[str] = None,
        uname: str = 'UNIQ',
        empty = None,
        **kwargs
    ) -> PartialUniqSkymap:
        """
        Read a file saved in the default format used by ``PartialUniqSkymap``.

        Parameters
        ----------
        mask : PartialUniqSkymap
            Only read in pixels overlapping with ``mask``.
        file : file or str
            The file object or filename to read from.
        name : str, optional
            The column-name of the pixel data. If not specified and if reading
            from a file with only one non-index column, that column will be
            chosen automatically.
        uname : str, optional
            The column-name of the HEALPix NUNIQ pixel data, if different from
            the default value.
        empty : scalar, optional
            ``empty`` argument to pass to ``PartialUniqSkymap`` initializer.
            **Not used when writing.**
        *args, **kwargs
            Arguments to pass on to ``astropy.table.Table.read``.

        Returns
        -------
        m : PartialUniqSkymap
            A new ``PartialUniqSkymap`` instance with the specified data.
        """
        from astropy.table import Table

        t = Table.read(file, **kwargs)
        if not name:
            c = [t for t in t.colnames if t != uname]
            if len(c) != 1:
                raise ValueError(f"Ambiguous colname; pick from {c}")
            name = c[0]
        #from IPython.core.debugger import set_trace; set_trace()
        return PartialUniqSkymap(t[name], t[uname], name=name, empty=empty,
                                 meta=t.meta,
                                 point_sources=PointsTuple.meta_read(t.meta))

    @classmethod
    def write(
        cls,
        skymap: PartialUniqSkymap,
        file: Union[IO, str],
        *args,
        name: Optional[str] = None,
        uname: str = 'UNIQ',
        **kwargs
    ):
        """
        Read a file saved in the default format used by ``PartialUniqSkymap``.

        Parameters
        ----------
        skymap : PartialUniqSkymap
            The skymap to save.
        file : file or str
            The file object or filename to write to.
        name : str, optional
            The column-name of the pixel data in the saved file, if different
            from that specified by the skymap.
        uname : str, optional
            The column-name of the HEALPix NUNIQ pixel data in the saved file,
            if different from the default value.
        *args, **kwargs
            Arguments to pass on to ``astropy.table.Table.write``.
        """
        skymap.to_table(name=name, uname=uname).write(file, *args, **kwargs)


class OldLigoIo(IoStrategy):
    """
    Read/write files in the format used by LIGO/Virgo for their skymaps using
    the old method (pre 0.3.0).
    """

    @classmethod
    def read(
        cls,
        skymap: Optional[Union[PartialUniqSkymap, ArrayLike]],
        file: Union[IO, str],
        *args,
        name: Optional[str] = None,
        memmap: bool = True,
        coarsen: int = 0,
        **kwargs
    ):
        """
        Read a file saved in the format used by LIGO/Virgo for their skymaps.

        Parameters
        ----------
        mask : PartialUniqSkymap
            Only read in pixels overlapping with ``mask``.
        file : file or str
            The file object or filename to read from.
        name : str, optional
            The column-name of the pixel data. Defaults to 'PROBDENSITY'.
        memmap : bool, optional
            Whether to memory-map the input file during read. Useful when
            reading small sky areas from large files to conserve memory.
            The returned skymap will be stored as a copy in memory.
        coarsen : int, optional
            If provided, coarsen the ``mask`` by up to this many HEALPix
            orders (up to order 0) to speed up read times. This will select
            a superset of the sky region defined in ``mask``.
        *args, **kwargs
            Arguments to pass on to ``astropy.table.Table.read``.
        """
        import numpy as np

        if name is None:
            name = "PROBDENSITY"

        if skymap is None:
            pt = []
            m = np.arange(12)+4  # 12 base pixels = whole sky
        elif isinstance(skymap, PartialUniqSkymap):
            pt = skymap.point_sources
            m = skymap.u
        else:
            pt = []
            m = np.asarray(skymap, dtype=np.int64)
        m = np.unique(m >> (2*min(uniq2order(m.min()), coarsen)))
        p = read_partial_skymap(file, m, memmap=memmap)
        return PartialUniqSkymap(np.squeeze(p[name]),
                                 p['UNIQ'],
                                 name=name, meta=p.meta,
                                 point_sources=pt)

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
        BasicIo.write(skymap, file, name=name, *args, **kwargs)


class IoRegistry:
    """
    Handle IO for ``PartialUniqSkymap`` instances.
    """
    basic = BasicIo
    ligo = LigoIo
    ligo_old = OldLigoIo
    astroquery = AstroqueryIo
    gracedb = GracedbIo
