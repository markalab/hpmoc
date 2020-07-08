# pylint: disable=line-too-long,invalid-name,bad-continuation
# flake8: noqa
# (c) Stefan Countryman 2019

"""
Functions for working with point sources, applying point spread functions
(PSFs) thereto, and making those PSFs compatible with other HEALPix skymaps.
"""

from .utils import nest2uniq, resol2nside, nest2dangle
from .partial import PartialUniqSkymap
from .plotters import PointsTuple


# class PsfGaussian(PartialUniqSkymap):
#     """
#     A Point Spread Function (PSF) for a Gaussian distribution. Calculates a
#     ``PartialUniqSkymap`` based on source location, storing the init parameters
#     as attributes.
#     """
# 
#     def __init__(self, ra, dec, σ):
#         """
#         Create a new Gaussian point-spread-function (PSF) ``PartialUniqSkymap``
#         from the right-ascension ``ra``, declination ``dec``, and standard
#         deviation ``σ`` of a point-source. ``ra, dec, σ`` can be angular
#         ``astropy.units.Quantity`` instances (radians or degrees) or scalar
#         values in degrees.
# 
#         Parameters
#         ----------
#         ra : float or astropy.units.Quantity
#             Right Ascension of the center of the distribution.
#         dec : float or astropy.units.Quantity
#             Declination of the center of the distribution.
#         σ : float or astropy.units.Quantity
#             Standard deviation of the distribution.
#         """
#         import numpy as np
#         import healpy as hp
#         from astropy.units import deg, Quantity as Qnt  # pylint: disable=E0611
# 
#         Ω = [θ.to(deg) if isinstance(θ, Qnt) else θ*deg for θ in (ra, dec, σ)]
#         self.ra, self.dec, self.σ = Ω               # store with dimensions
# 
#         nˢ = resol2nside(σ/5)                       # target NSIDE resolution
#         n⃗ = hp.query_disc(nˢ, hp.ang2vec(self.ra, self.dec, True),  # indices
#                           5*self.σ.to('rad'), nest=True)
#         inv2σsq = self.σ.to('rad')**-2/2            # 1/2σ² factor
#         s⃗ = nest2dangle(n⃗, nˢ, self.ra, self.dec)   # distances from center
#         s⃗ *= s⃗                                      # in-place square exponent,
#         s⃗ *= -inv2σsq                               #   then const factor
#         np.exp(s⃗, out=s⃗)                            # in-place exponentiation
#         s⃗ *= inv2σsq/np.pi                          # final factor
#         super().__init__(s⃗, nest2uniq(n⃗, nˢ, in_place=True), copy=False)


def psf_gaussian(ra, dec, σ, cutoff=5, pt_label=None, **kwargs):
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
        How large of a disk to query in multiples of σ.
    pt_label : str, optional
        A string label for this point when plotting (e.g. the event ID).
    **kwargs
        Keyword arguments to pass to ``PointsTuple``.
    """
    import numpy as np
    import healpy as hp
    from astropy.units import deg, Quantity as Qnt  # pylint: disable=E0611

    Ω = [θ.to(deg) if isinstance(θ, Qnt) else θ*deg for θ in (ra, dec, σ)]
    ra, dec, σ = Ω                              # store with dimensions

    nˢ = resol2nside(σ/5)                       # target NSIDE resolution
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
