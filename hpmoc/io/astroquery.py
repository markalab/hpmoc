"""
An IO interface for ``astroquery`` for quickly loading and rasterizing data
from the wealth of observatories whose data is accessible through that package.
Can be only be used if ``astroquery`` is already locally installed, which can
be accomplished automatically by adding the ``astroquery`` dependencies on
installation.
"""

import importlib
from typing import Optional
from .abstract import StubIo, ReadonlyIo
from ..partial import PartialUniqSkymap

# TODO think through these features further
if importlib.util.find_spec("astroquery") is None:

    class AstroqueryIo(ReadonlyIo, StubIo):
        """
        A stub for accessing data from ``astroquery``. Install
        ``astroquery`` if you want to be able to directly read
        ``hpmoc.partial.PartialUniqSkymap`` instances from ``astroquery``.
        """
        qualname: "hpmoc.io.astroquery.AstroqueryIo"
        requirements: "astroquery"

else:

    class AstroqueryIo(ReadonlyIo):
        """
        Read ``hpmoc.partial.PartialUniqSkymap`` instances directly from
        astroquery. Use this for rapid plotting, prototyping, or offline
        analyses.
        """

        @staticmethod
        def read(
            # FIXME implement masking.
            _skymap: Optional[PartialUniqSkymap],
            *args,
            **kwargs,
        ) -> PartialUniqSkymap:
            """
            Use ``astroquery.skyview.SkyView.get_images`` to load FITS data,
            which is then rasterized to a HEALPix grid. All arguments are
            passed to that function.

            See Also
            --------
            astroquery.skyview.SkyView
            """
            from astropy.wcs import WCS
            from astroquery.skyview import SkyView

            # TODO maybe use ``get_image_list`` which returns a generator over
            # filenames (which are lazily downloaded).
            hdu = SkyView.get_images(*args, **kwargs)
            # TODO handle multiple returned results and multiple constituent
            # skymaps more gracefully, ideally by making IoStrategy.read return
            # a generator over loaded skymaps.
            return PartialUniqSkymap(hdu.data[0][0], WCS(hdu.header))
