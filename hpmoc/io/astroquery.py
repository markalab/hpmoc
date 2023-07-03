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
An IO interface for ``astroquery`` for quickly loading and rasterizing data
from the wealth of observatories whose data is accessible through that package.
Can be only be used if ``astroquery`` is already locally installed, which can
be accomplished automatically by adding the ``astroquery`` dependencies on
installation.
"""

import importlib.util
from typing import Optional, Union
from .abstract import StubIo, ReadonlyIo
from ..partial import PartialUniqSkymap
from numpy.typing import ArrayLike

# TODO think through these features further
if importlib.util.find_spec("astroquery") is None:

    class AstroqueryIo(ReadonlyIo, StubIo):
        """
        A stub for accessing data from ``astroquery``. Install
        ``astroquery`` if you want to be able to directly read
        ``hpmoc.partial.PartialUniqSkymap`` instances from ``astroquery``.
        """
        qualname = "hpmoc.io.astroquery.AstroqueryIo"
        requirements = "astroquery"

else:

    class AstroqueryIo(ReadonlyIo):
        """
        Read ``hpmoc.partial.PartialUniqSkymap`` instances directly from
        astroquery. Use this for rapid plotting, prototyping, or offline
        analyses.
        """

        @classmethod
        def read(
            cls,
            # FIXME implement masking.
            skymap: Optional[Union[PartialUniqSkymap, ArrayLike]],
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
            from astropy.io.fits.hdu import PrimaryHDU

            # TODO maybe use ``get_image_list`` which returns a generator over
            # filenames (which are lazily downloaded).
            hdulistlist = SkyView.get_images(*args, **kwargs)
            if len(hdulistlist) != 1:
                raise ValueError("Got more than one image")
            hdu: PrimaryHDU = hdulistlist[0][0] # type: ignore
            # TODO handle multiple returned results and multiple constituent
            # skymaps more gracefully, ideally by making IoStrategy.read return
            # a generator over loaded skymaps.
            return PartialUniqSkymap(hdu.data[0][0], WCS(hdu.header)) # type: ignore
