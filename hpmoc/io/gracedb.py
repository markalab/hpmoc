"""
An IO interface for downloading LVK skymaps from GraceDB_ using
``ligo.gracedb``. Implemented using the ligo-gracedb_ package, which contains
far more features than are used here.

.. _GraceDB: https://gracedb.ligo.org
.. _ligo-gracedb: https://ligo-gracedb.readthedocs.io/en/latest/quickstart.html
"""

import importlib
from io import BytesIO
from typing import Optional, Union
from nptyping import NDArray
from .abstract import StubIo, IoStrategy
from .ligo import LigoIo
from ..partial import PartialUniqSkymap

if importlib.util.find_spec("ligo.gracedb") is None:

    class GracedbIo(StubIo):
        """
        A stub for accessing data from GraceDB_ using ``ligo.gracedb``. Install
        ``ligo-gracedb`` from ``pip`` or ``conda`` to be able to directly read
        ``hpmoc.partial.PartialUniqSkymap`` instances from GraceDB_.

        .. _GraceDB: https://gracedb.ligo.org
        """
        qualname: "hpmoc.io.gracedb.GracedbIo"
        requirements: "ligo-gracedb"

else:

    class GracedbIo(IoStrategy):
        """
        Use ``ligo.gracedb.rest.GraceDb.files`` to download skymaps from
        GraceDB_ and automatically parse, compress, and convert them into
        ``hpmoc.partial.PartialUniqSkymap`` instances.

        .. _GraceDB: https://gracedb.ligo.org
        """

        @staticmethod
        def read(
                mask: Optional[Union[PartialUniqSkymap, NDArray]],
                graceid: str,
                file: str,
                *args,
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

            if client is None:
                client = GraceDb()
            # Unfortunately GraceDb responses don't work for streaming reads.
            # Not sure why, but defaulting to a single read operation seems
            # safest.
            buf = BytesIO(client.files(graceid, file).read())
            return LigoIo.read(mask, buf, *args, **kwargs)

        @staticmethod
        def write(*args, **kwargs):
            raise NotImplementedError("Not yet implemented. Might never be "
                                      "implemented. Requires auth.")
