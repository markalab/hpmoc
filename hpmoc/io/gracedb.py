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
An IO interface for downloading LVK skymaps from GraceDB_ using
``ligo.gracedb``. Implemented using the ligo-gracedb_ package, which contains
far more features than are used here.

.. _GraceDB: https://gracedb.ligo.org
.. _ligo-gracedb: https://ligo-gracedb.readthedocs.io/en/latest/quickstart.html
"""

import importlib.util
from io import BytesIO
from typing import Optional, Union, IO, TYPE_CHECKING
from .abstract import StubIo, IoStrategy
from .ligo import LigoIo
from ..partial import PartialUniqSkymap
from numpy.typing import ArrayLike

if not TYPE_CHECKING and (importlib.util.find_spec("ligo") is None or importlib.util.find_spec("ligo.gracedb") is None):

    class GracedbIo(StubIo):
        """
        A stub for accessing data from GraceDB_ using ``ligo.gracedb``. Install
        ``ligo-gracedb`` from ``pip`` or ``conda`` to be able to directly read
        ``hpmoc.partial.PartialUniqSkymap`` instances from GraceDB_.

        .. _GraceDB: https://gracedb.ligo.org
        """
        qualname = "hpmoc.io.gracedb.GracedbIo"
        requirements = "ligo-gracedb"

else:

    if TYPE_CHECKING:
        import ligo.gracedb.rest

    class GracedbIo(IoStrategy):
        """
        Use ``ligo.gracedb.rest.GraceDb.files`` to download skymaps from
        GraceDB_ and automatically parse, compress, and convert them into
        ``hpmoc.partial.PartialUniqSkymap`` instances.

        .. _GraceDB: https://gracedb.ligo.org
        """

        @classmethod
        def read(
            cls,
            skymap: Optional[Union[PartialUniqSkymap, ArrayLike]],
            file: Union[IO, str],
            *args,
            graceid: str,
            client: Optional['ligo.gracedb.rest.GraceDb'] = None,
            **kwargs
        ):
            """
            Load a file from GraceDB_ in the format used by LIGO/Virgo/KAGRA
            for their skymaps. Just a shortcut for passing the skymap data
            fetched by ``ligo.gracedb.GraceDb.files`` into ``hpmoc.io.LigoIo``,
            but with the nice property that it will try to identify the latest
            skymap and download it for you if you don't specify one (useful for
            prototyping).

            .. _GraceDB: https://gracedb.ligo.org

            Parameters
            ----------
            mask : PartialUniqSkymap or array, optional
                Only read in pixels overlapping with ``mask``.
            graceid : str
                The GraceID (either event or superevent) for which a skymap is
                desired.
            file : str
                The name of the skymap. Append the version followed by a comma
                if you would like to specify a specific version, e.g.
                ``"bayestar.multiorder.fits"`` will just be the most recent
                version whereas ``"bayestar.multiorder.fits,0"`` will be the
                first version of that file (regardless of whether newer
                versions have been uploaded).
            client : ligo.gracedb.rest.GraceDb, optional
                The GraceDB_ client to use. If not provided, a new one will be
                instantiated. Pass a custom one if you need to handle
                authentication or the like.
            *args, **kwargs
                Arguments to pass on to ``hpmoc.io.LigoIo.read``.
            """
            from ligo.gracedb.rest import GraceDb

            if not isinstance(file, str):
                raise TypeError("file must be a str")

            if client is None:
                client = GraceDb()
            # Unfortunately GraceDb responses don't work for streaming reads.
            # Not sure why, but defaulting to a single read operation seems
            # safest.
            buf = BytesIO(client.files(graceid, file).read()) # type: ignore
            return LigoIo.read(skymap, buf, *args, **kwargs)

        @classmethod
        def write(cls, *args, **kwargs):
            raise NotImplementedError("Not yet implemented. Might never be "
                                      "implemented. Requires auth.")
