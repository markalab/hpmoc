# Copyright (C) 2019-2023 Columbia Experimental Gravity Group (GECo)

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

# pylint: disable=line-too-long,invalid-name,bad-continuation
# flake8: noqa

"""
Functions for working with point sources, applying point spread functions
(PSFs) thereto, and making those PSFs compatible with other HEALPix skymaps.
"""

from .utils import nest2uniq, resol2nside, nest2dangle
from .points import PointsTuple
from .healpy import healpy as hp


def psf_gaussian(ra, dec, σ, cutoff=5, pt_label=None, nside=None,
                 nside_factor=1, **kwargs):
    """
    Create a new Gaussian point-spread-function (PSF) ``PartialUniqSkymap``
    from the right-ascension ``ra``, declination ``dec``, and standard
    deviation ``σ`` of a point-source. ``ra, dec, σ`` can be angular
    ``astropy.units.Quantity`` instances (radians or degrees) or scalar
    values in degrees.

    Parameters
    ----------
    ra : float or astropy.units.Quantity
        Right Ascension of the center of the distribution.
    dec : float or astropy.units.Quantity
        Declination of the center of the distribution.
    σ : float or astropy.units.Quantity
        Standard deviation of the distribution.
    cutoff : float, optional
        How large of a disk to query in multiples of σ, i.e. the support of the
        PSF in units of σ.
    pt_label : str, optional
        A string label for this point when plotting (e.g. the event ID).
    nside : int, optional
        NSIDE to use. If not specified, calculated automatically using
        ``nside_factor``.
    nside_factor : float, optional
        If ``nside`` is not provided, calculate the NSIDE value giving pixel
        widths smaller than ``σ * nside_factor``. In other words, set the
        resolution in units of ``σ``.
    **kwargs
        Keyword arguments to pass to ``PointsTuple``.
    """
    import numpy as np
    from astropy.units import deg, Quantity as Qnt  # pylint: disable=E0611
    from .partial import PartialUniqSkymap

    Ω = [θ.to(deg) if isinstance(θ, Qnt) else θ*deg for θ in (ra, dec, σ)]
    ra, dec, σ = Ω                              # store with dimensions

    nˢ = nside or resol2nside(σ * nside_factor) # target NSIDE resolution
    n⃗ = hp.query_disc(nˢ, hp.ang2vec(ra.value, dec.value, True),
                      cutoff*σ.to('rad').value, nest=True)
    inv2σsq = σ.to('rad')**-2/2                 # 1/2σ² factor
    s⃗ = nest2dangle(n⃗, nˢ, ra, dec)             # distances from center
    s⃗ *= s⃗                                      # in-place square exponent,
    s⃗ *= -inv2σsq                               #   then const factor
    np.exp(s⃗, out=s⃗)                            # in-place exponentiation
    s⃗ *= inv2σsq/np.pi                          # final factor
    pts = PointsTuple([(ra.value, dec.value, σ.value, pt_label)], **kwargs)
    return PartialUniqSkymap(s⃗.to('sr-1'), nest2uniq(n⃗, nˢ, in_place=True),
                             copy=False, point_sources=[pts])
