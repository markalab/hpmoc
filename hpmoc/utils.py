# pylint: disable=line-too-long,invalid-name,bad-continuation
# flake8: noqa
# (c) Stefan Countryman 2019
#
# - Interpolation on a sphere:
#   https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.interpolate.SmoothSphereBivariateSpline.html

"""
Utility functions used across healpix_skymap classes.
"""

import os
from operator import eq
from numbers import Integral
from typing import Union, IO  # possible removal for older pythons
import functools
from dataclasses import dataclass  # possible removal for older pythons
from math import pi
from datetime import datetime
from textwrap import wrap, dedent
import logging
import gzip
from tempfile import NamedTemporaryFile
import binascii
from .healpy_utils import alt_compress, alt_expand
from .healpy import healpy as hp

LOGGER = logging.getLogger(__name__)
GZIP_BUFFSIZE = 10**5
PIX_READ = 4**8
OP_CHUNKSIZE = 10**6
PIXEL_CONSISTENCY_ERROR = 1e-9
MAX_ORDER = 30
UINT_RANGES = {(0, 2**(8*2**i)): f'uint{8*2**i}' for i in range(4)}
INT_RANGES = {(-i//2, i//2): v[1:] for [_, i], v in UINT_RANGES.items()}
OUTLINE_STROKE = 1.2  # thickness of outline surrounding text
OUTLINE_COLOR = (1, 1, 1, 1)  # outlines of text and neutrino markers
FONT_SIZE = 14  # matplotlib font size
N_X_OFFSET = 0.08  # [inches]
N_Y_OFFSET = 0.08  # [inches]


def max_uint_type(largest):
    import numpy as np

    if largest < 0:
        raise ValueError(f"Positive values only: {largest}")
    for dt in ('u1', 'u2', 'u4', 'u8'):
        if ~np.array(0, dt) > largest:
            return dt
    raise ValueError("You didn't pass an integer representable in less than "
                     "64 bits: {largest}")


class EmptyStream(OSError):
    "Raised when a file stream returns no further content."


# TODO test this much more
def uniq2xyf_nside(u⃗):
    """
    Examples
    --------
    >>> uniq2xyf_nside(8)
    (0, 0, 4, 1)

    See Also
    --------
    xyf_nside2uniq
    """
    i, nˢ = uniq2nest_and_nside(u⃗)
    nˢˢ = nˢ*nˢ
    f = (u⃗//nˢˢ)-4
    i -= f*nˢˢ
    return alt_compress(i), alt_compress(i>>1, True), f, nˢ


# TODO test this much more
def xyf_nside2uniq(x, y, f, nˢ):
    """
    Examples
    --------
    >>> import numpy as np
    >>> (xyf_nside2uniq(*uniq2xyf_nside(np.arange(4, 10000))) == np.arange(4, 10000)).all()
    True

    See Also
    --------
    uniq2xyf_nside
    """
    return alt_expand(x) + (alt_expand(y)<<1) + (f+4)*nˢ*nˢ


def min_int_dtype(vmin, vmax):
    """
    Find the smallest integer datatype that can represent a given range of
    values.
    """
    import numpy as np

    if vmax < vmin:
        raise ValueError(f"vmax must be larger than vmin. got: {vmin, vmax}")
    ranges = INT_RANGES if vmin < 0 else UINT_RANGES
    for [dmin, dmax], dtype in INT_RANGES:
        if vmin > dmin and vmax < dmax:
            return np.dtype(dtype)
    raise ValueError("Could not find a type to represent {vmin, vmax}.")


def sky_area():
    """Get the area of the entire sky as an ``Astropy.units.Quantity``."""
    from astropy.units import Quantity
    return Quantity(4*pi, "rad2")


def sky_area_deg():
    """Get the area of the entire sky in square degrees."""
    return sky_area().to("deg**2").value


def nest2ang(n⃗, N⃗ˢ):
    """
    Get the angles corresponding to these nested indices ``n⃗`` and NSIDE
    values ``N⃗ˢ``.

    Parameters
    ----------
    n⃗ : array-like
        HEALPix NESTED indices of pixels of interest.
    N⃗ˢ : int or array-like
        The NSIDE values corresponding to each pixel. Can be a scalar if all
        pixels are at the same NSIDE.

    Returns
    -------
    ra_dec : astropy.units.Quantity
        2D array whose first row is the right-ascension and second row is
        the declination (in degrees) of each pixel. You can get each of these
        individually with ``ra, dec = nest2ang(...)``.
    """
    import numpy as np
    from astropy.units import degree  # pylint: disable=no-name-in-module

    N⃗ˢ = np.full(n⃗.shape, N⃗ˢ) if isinstance(N⃗ˢ, Integral) else N⃗ˢ
    ra_dec = np.ndarray((2, len(n⃗)))                    # pre-allocate results
    for Nˢ in np.unique(N⃗ˢ):                            # iterate NSIDE values
        n⃗̇ = np.nonzero(N⃗ˢ == Nˢ)[0]
        ra_dec[:, n⃗̇] = np.array(hp.pix2ang(Nˢ, n⃗[n⃗̇], nest=True, lonlat=True))
    return ra_dec*degree


def resol2nside(res, coarse=False, degrees=True):
    """Get the HEALPix NSIDE value corresponding to a given resolution.

    Parameters
    ----------
    res : float or array-like
        The required resolution, measured as the angular width of a pixel.
    coarse : bool, optional
        If ``False``, pick a higher resolution than the one specified. If
        ``True``, pick a lower resolution (necessary since HEALPix resolutions
        increment in discrete steps).
    degrees : bool, optional
        Whether ``res`` is specified in degrees. If ``False``, radians
        are assumed.
    """
    import numpy as np

    r = hp.nside2resol(2**np.arange(MAX_ORDER))[::-1]
    r = np.degrees(r) if degrees else r
    side = 'left' if coarse else 'right'
    return 1 << MAX_ORDER - np.searchsorted(r, res, side=side)


def nest2dangle(n⃗, nˢ, ra, dec, degrees=True, in_place=False):
    """
    Get the angular distance between the pixels defined in ``n⃗, nˢ`` and the
    point located at ``ra, dec``.

    Parameters
    ----------
    n⃗ : array-like
        HEALPix NEST indices
    nˢ : int
        HEALPix NSIDE value
    ra : float or astropy.units.Quantity
        Right ascension of the point; assumed degrees if no unit given.
    dec : float or astropy.units.Quantity
        Declination of the point; assumed degrees if no unit given.
    degrees : bool, optional
        If ``True``, and assumed degrees if ``ra`` and/or ``dec`` are not
        ``astropy.units.Quantity`` instances with angular unit defined. If
        ``False``, assume radians. Ignored if a unit is already specified.
    in_place : bool, optional
        If ``True``, store the result in ``n⃗`` to reduce memory usage.
        Requires ``n⃗.dtype == np.float64``.

    Returns
    -------
    Δθ⃗ : astropy.units.Quantity
        Angular distance in radians between each pixel in ``n⃗`` and the point
        at ``ra, dec``.

    Examples
    --------
    The 12 base healpix pixels' distances from north pole should all be equal
    to 90 minus their declinations:

    >>> import numpy as np
    >>> from hpmoc.healpy import healpy as hp
    >>> Δθ⃗ = nest2dangle(range(12), 1, 32, 90).to('deg')
    >>> Δθ⃗
    <Quantity [ 48.1896851,  48.1896851,  48.1896851,  48.1896851,  90.       ,
                90.       ,  90.       ,  90.       , 131.8103149, 131.8103149,
               131.8103149, 131.8103149] deg>
    >>> np.all(abs(Δθ⃗.value - (90 - hp.pix2ang(1, np.arange(12), nest=True,
    ...                                        lonlat=True)[1])) < 1e-13)
    True

    You can run the same check for larger skymaps, too (though note that
    precision drops for very nearby pixels due to the O(x²) behavior of cos()
    for small angles and the fact that a dot-product and arccos are used to
    compute the result):

    >>> nside = 2**10
    >>> nest = np.arange(12*nside**2)
    >>> np.around(
    ...     (
    ...         nest2dangle(nest, nside, 32, 90).to('deg').value
    ...         - (90 - hp.pix2ang(nside, nest, nest=True, lonlat=True)[1])
    ...     ),
    ...     11
    ... ).ptp()
    0.0
    """
    import numpy as np
    from astropy.units import rad, deg, Quantity  # pylint: disable=E0611

    n⃗ = np.array(n⃗, copy=False)
    if in_place and n⃗.dtype != np.float64:
        raise ValueError("Can't operate in-place on a non-float array: %s" % n⃗)
    Ω = [θ.to(deg).value if isinstance(θ, Quantity) else
         (θ if degrees else np.degrees(θ))
         for θ in (ra, dec)]
    x0, y0, z0 = hp.ang2vec(*Ω, lonlat=True)
    x, y, z = [np.ndarray((min(len(n⃗), OP_CHUNKSIZE),)) for _ in range(3)]
    dots = n⃗ if in_place else np.ndarray(n⃗.shape)
    for i in range(0, len(n⃗), OP_CHUNKSIZE):
        n⃗ⁱ = n⃗[i:i+OP_CHUNKSIZE].astype(int, copy=False)
        N = len(n⃗ⁱ)
        x[:N], y[:N], z[:N] = hp.pix2vec(nˢ, n⃗ⁱ, nest=True)
        x *= x0
        y *= y0
        z *= z0
        x += y
        x += z
        dots[i:i+N] = x[:N]
    return Quantity(np.arccos(dots, out=dots), rad, copy=False)


def uniq2dangle(u⃗, ra, dec, degrees=True):
    """
    Like ``nest2dangle``, but takes HEALPix NUNIQ indices as input ``u⃗``.

    Examples
    --------
    The 12 base healpix pixels' distances from north pole should all be equal
    to 90 minus their declinations:

    >>> import numpy as np
    >>> from hpmoc.healpy import healpy as hp
    >>> Δθ⃗ = uniq2dangle(range(4, 16), 32, 90).to('deg')
    >>> Δθ⃗
    <Quantity [ 48.1896851,  48.1896851,  48.1896851,  48.1896851,  90.       ,
                90.       ,  90.       ,  90.       , 131.8103149, 131.8103149,
               131.8103149, 131.8103149] deg>
    >>> np.all(abs(Δθ⃗.value - (90 - hp.pix2ang(1, np.arange(12), nest=True,
    ...                                    lonlat=True)[1])) < 1e-13)
    True

    See Also
    --------
    nest2dangle
    """
    [u⃗ˢ], _, o⃗, _, [v⃗], [u⃗̇ˢ] = nside_slices(u⃗, return_inverse=True,
                                            dtype=float)
    n⃗ˢ = 1 << o⃗                                     # NSIDE for each view
    for nˢ, v⃗ⁱ in zip(n⃗ˢ, v⃗):                       # views into u⃗ˢ
        v⃗ⁱ -= 4*nˢ**2                               # convert to nest in-place
        u = nest2dangle(v⃗ⁱ, nˢ, ra, dec, degrees=degrees, in_place=True).unit
    return u⃗ˢ[u⃗̇ˢ]*u                                 # unsort results in u⃗ˢ*


def dangle_rad(ra, dec, mapra, mapdec):  # pylint: disable=invalid-name
    """Get an array of angular distances in radians between the point specified
    in ``ra``, ``dec`` and the points specified in ``mapra``, ``mapdec``. All
    angles, including the return value, are specified in radians.

    sin(DEC1)*sin(DEC2) + cos(DEC1)*cos(DEC2)*cos(RA2-RA1)

    So we get the angle by taking the arccos of this function.

    Parameters
    ----------
    ra : float
        Right ascension of the single point [radians]
    dec : float
        Declination of the single point [radians]
    mapra : array
        Right ascensions whose angular distances to the single point will be
        calculated; must be same length as ``mapdec``. [radians]
    mapdec : array
        Declinations whose angular distances to the single point will be
        calculated; must be same length as ``mapra``. [radians]

    Returns
    -------
    Δθ⃗ : array
        Angular distances between the single point provided and the arrays of
        points [radians].

    Examples
    --------
    Simple examples of distances to north pole:

    >>> from math import pi
    >>> import numpy as np
    >>> dangle_rad(0, pi/2, np.array([0, 0, 0, 2, 3]),
    ...            np.array([pi/2, -pi/2, 0, 0, 0]))/pi
    array([0. , 1. , 0.5, 0.5, 0.5])
    """
    import numpy as np
    delta_ra = mapra - ra
    dot_prod = (np.sin(mapdec)*np.sin(dec) +
                np.cos(mapdec)*np.cos(dec)*np.cos(delta_ra))
    return np.arccos(dot_prod)


def nest2uniq(indices, nside, in_place=False):
    """Return the NUNIQ pixel indices for nested ``indices`` with
    NSIDE=``nside``.

    Parameters
    ----------
    indices : array
        Indices of HEALPix pixels in NEST ordering.
    nside : int
        A valid NSIDE value not greater than any of the NSIDE values of the
        ``indices``.
    in_place : bool, optional
        If ``True``, perform the conversion in-place and return the converted
        ``indices`` object.

    Returns
    -------
    uniq : array
        An array of HEALPix pixels in NEST ordering corresponding to the input
        ``indices`` and ``nside``.

    Raises
    ------
    ValueError
        If ``nside`` is an invalid NSIDE value, i.e. not a power of 2.

    Examples
    --------
    >>> import numpy as np
    >>> nest2uniq(np.array([284, 286, 287,  10,   8,   2, 279, 278, 285]), 8)
    array([540, 542, 543, 266, 264, 258, 535, 534, 541])
    """
    check_valid_nside(nside)
    add = 4*nside**2
    if in_place:
        indices += add
    else:
        indices = indices + add
    return indices


def check_valid_nside(nside):
    """Checks whether ``nside`` is a valid HEALPix NSIDE value. Raises a
    ValueError if it is not.

    Parameters
    ----------
    nside : int or array
        An integer or array of integers representing HEALPix NSIDE values.

    Raises
    ------
    ValueError
        If any of the values are not valid HEALPix NSIDE values.

    Examples
    --------
    Does nothing if you provide a valid NSIDE value (floats are accepted as
    long as their exact values are valid):

    >>> check_valid_nside([16, 4])
    >>> check_valid_nside(4)
    >>> check_valid_nside(1024)
    >>> check_valid_nside([1024., 4096])

    A descriptive value error is raised if invalid values are provided:

    >>> try:
    ...     check_valid_nside([4.1, 4])
    ... except ValueError as err:
    ...     print("Caught exception:", err)
    Caught exception: Not a valid NSIDE value: [4.1, 4]
    >>> try:
    ...     check_valid_nside(17)
    ... except ValueError as err:
    ...     print("Caught exception:", err)
    Caught exception: Not a valid NSIDE value: [17]
    """
    import numpy as np
    if not np.iterable(nside):
        nside = np.array([nside])
    # this should be okay precision-wise since powers of 2 are exact in
    # floating-point for numbers that aren't too huge; more likely to fail
    # open, fortunately
    if np.any(2**np.floor(np.log2(nside)) != nside):
        raise ValueError("Not a valid NSIDE value: {}".format(nside))


def nside2pixarea(nside, degrees=False):
    """
    Get the area per-pixel at ``nside``. ``nside`` can also be a HEALPix array
    here.

    Parameters
    ----------
    nside : int or array-like
        The NSIDE value or values you would like areas for.
    degrees : bool, optional
        If ``True``, return areas in inverse square degrees. Otherwise, return
        areas in inverse steradians.

    Returns
    -------
    pixarea : float or array-like
        The area-per-pixel in the specified units. If ``nside`` was a scalar,
        ``pixarea`` is given as a scalar; if it was an array, ``pixarea`` is
        returned as an array whose values correspond to those in ``nside``.

    Raises
    ------
    ValueError
        If any ``nside`` values are not valid.

    Examples
    --------
    At NSIDE = 1, we should get 1/12 of the sky, or about 1/steradian:

    >>> from math import pi
    >>> allsky = 4*pi
    >>> nside2pixarea(1) == allsky / 12
    True

    This should work for a list of NSIDES as well.

    >>> import numpy as np
    >>> nsides = np.array([1, 1, 2, 4])
    >>> areas = np.array([allsky/12, allsky/12, allsky/48, allsky/192])
    >>> np.all(nside2pixarea(nsides) == areas)
    True
    """
    import numpy as np

    check_valid_nside(nside)
    result = pi/3/nside**2
    if degrees:
        np.degrees(result, out=result)
        np.degrees(result, out=result)
    return result


def check_valid_nuniq(indices):
    """Checks that ``indices`` are valid NUNIQ indices.

    Raises
    ------
    ValueError
        If ``indices`` are not valid NUNIQ indices, i.e. they are not integers
        greater than 3.
    """
    import numpy as np
    if not np.iterable(indices):
        indices = np.array([indices])
    if not len(indices):  # can't go wrong with zero indices
        return
    if not np.all(np.mod(indices, 1) == 0):
        raise ValueError("NUNIQ indices must be integers.")
    if not indices.min() > 3:
        raise ValueError("NUNIQ indices start at 4; found smaller values.")


def uniq2order(indices):
    """
    Get the HEALPix order of the given NUNIQ-ordered indices.

    Raises
    ------
    ValueError
        If ``indices`` are not valid NUNIQ indices, i.e. they are not integers
        greater than 3.
    """
    import numpy as np
    check_valid_nuniq(indices)
    return np.array(np.floor(np.log2(indices/4.)/2.), dtype=int)


def uniq2nside(indices):
    """
    Get the NSIDE value of the given NUNIQ-ordered indices.

    Raises
    ------
    ValueError
        If ``indices`` are not valid NUNIQ indices, i.e. they are not integers
        greater than 3.
    """
    import numpy as np
    return 1 << uniq2order(indices)


def uniq2nest_and_nside(indices, in_place=False):
    """
    Parameters
    ----------
    indices : array
        A scalar or numpy array of NUNIQ indices
    in_place : bool, optional
        If ``True``, perform the conversion in-place and return the converted
        ``indices`` object along with the calculated ``nside``.

    Returns
    -------
    nest_indices : array
        The indices in nest ordering at their respective resolutions
    nside : array
        The resolution expressed as NSIDE of the indices given in
        ``nest_indices``

    Examples
    --------
    >>> import numpy as np
    >>> from pprint import pprint
    >>> nuniq = np.array([540, 542, 543, 266, 264, 258, 535, 534, 541])
    >>> pprint([u.astype(np.int) for u in uniq2nest_and_nside(nuniq)])
    [array([284, 286, 287,  10,   8,   2, 279, 278, 285]),
     array([8, 8, 8, 8, 8, 8, 8, 8, 8])]

    Confirm that this is the inverse of nest2uniq:

    >>> all(nest2uniq(*uniq2nest_and_nside(nuniq)) == nuniq)
    True
    """
    # uniq2nside implicitly checks whether the indices are valid NUNIQ indices
    nside = uniq2nside(indices)
    sub = 4*nside**2
    if in_place:
        indices -= sub
    else:
        indices = indices - sub
    return indices, nside


def nside_quantile_indices(nside, skymap, quantiles):
    """
    Get the indices and cumulative values of pixels falling between the given
    ``quantiles`` (expressed as a fraction of unity) for the skymap value
    ``skymap``. Also known as confidence/credible region (CR) or percentiles
    (when using percentages rather than fractions of unity).

    Pixels will be sorted by value (density), but quantiles will be taken by
    total integrated value (i.e. density times area, i.e. in proportion to
    density over the square of NSIDE). This puts the highest density regions in
    the upper quantiles while still normalizing for area in a multi-resolution
    skymap.

    Note that this will work perfectly well for partial skymaps, though in that
    case (as one would expect) the quantiles will be taken with respect to the
    remaining pixels.

    Parameters
    ----------
    nside : int or array
        NSIDE value (or values for a multi-order skymap) for the skymap.
    skymap : array
        Skymap values. If ``nside`` is an array, values will correspond to
        those NSIDEs.
    quantiles : array
        Quantiles from which to select pixels. Must be in ascending order with
        values in the interval ``[0, 1]``. These will form endpoints for
        partitions of the ``skymap``. For example, ``[0.1, 0.9]`` will omit the
        lowest and highest value pixels, giving the intermediate pixels
        accounting for 80% of the integrated skymap.
        Note that quantiles returned by this function are non-intersecting and
        half-open on the right (as with python indices), with the exception of
        ``1`` for the last pixel; for example, ``[0, 1]`` will include all
        pixels, ``[0.5, 1]`` will include the highest density pixels accounting
        for 50% of the integrated skymap value, ``[0, 0.5, 1]`` will partition
        the skymap into non-intersecting sets of pixels accounting for the
        high- and low-density partitions of the skymap by integrated value,
        etc.

    Returns
    -------
    indices : generator
        An iterator of arrays of indices suitable for selecting the values from
        ``skymap`` corresponing to the selected quantiles, sorted in order of
        increasing values of ``skymap``. Use these to select the values from
        ``skymap`` corresponding to each of the partitions defined in
        ``quantiles``. For example, ``quantiles=[0, 0.5, 1]`` will return a
        generator yielding two arrays of indices for accessing the values in
        the ``[0, 0.5]`` and ``[0.5, 1]`` quantiles, respectively. For
        applications where you want to just mask ``skymap`` and preserve sort
        order, you will want to sort this returned quantity before using it.
    levels: astropy.units.Quantity or array
        Values of the skymap at each quantile. Useful for e.g. contour
        plots (though ``PartialUniqSkymap.plot`` will handle this
        automatically).
    norm : array
        The total integral of ``skymap``. The integrated region in a partition
        defined by quantiles ``[0.1, 0.2]``, for example, will be
        ``(0.2-0.1)*norm``.

    Raises
    ------
    ValueError
        If ``quantiles`` has length less than 2; if its values are not in order
        and contained in the interval ``[0, 1]``; if ``nside`` and ``skymap``
        cannot be broadcast together; if any values in ``skymap`` are negative;
        or if the total integrated skymap equals zero, in which case quantiles
        are undefined.

    Examples
    --------
    Get the pixel indices for pixels between the 0.1 (10%) and 0.9 (90%)
    quantiles on a fixed-resolution full skymap:

    >>> import numpy as np
    >>> skymap = np.array([ 9, 10, 11,  0,  1,  2,  3,  4,  5,  6,  7,  8])
    >>> i, levels, norm = nside_quantile_indices(1, skymap, [0.1, 0.9])
    >>> [ii.astype(np.int) for ii in i]
    [array([ 7,  8,  9, 10, 11,  0,  1])]

    The levels will be the values of the array at the 0.1 and 0.9 quantiles

    >>> print(levels)
    [ 4 11]

    The norm in this case is just the average pixel value times the total solid
    angle of the sky:

    >>> np.abs(4*np.pi*skymap.mean() - norm) < 1e-13
    True

    Get the pixel indices for the same interval on a mixed resolution partial
    skymap (where the last six pixels contribute far less area despite having
    high density):

    >>> nside = np.array(6*[1]+6*[64])
    >>> [*nside_quantile_indices(nside, skymap, [0.1, 0.9])[0]
    ...  ][0].astype(np.int)
    array([0, 1])

    Equal lower and upper bounds give empty quantiles:

    >>> [*nside_quantile_indices(nside, skymap, [0.5, 0.5])[0]
    ...  ][0].astype(np.int64)
    array([], dtype=int64)

    Recover all indices (sorted by density):

    >>> [*nside_quantile_indices(nside, skymap, [0, 1])[0]][0].astype(np.int)
    array([ 3,  4,  5,  6,  7,  8,  9, 10, 11,  0,  1,  2])

    Pick the 90% CR:

    >>> [*nside_quantile_indices(nside, skymap, [0.1, 1])[0]][0].astype(np.int)
    array([0, 1, 2])

    Get the four top 20% quantiles:

    >>> for q in nside_quantile_indices(nside, skymap,
    ...                                 np.arange(0.2, 1.1, 0.2))[0]:
    ...     print(q)
    [0]
    []
    [1]
    [2]
    """
    import numpy as np

    q = np.array(quantiles, dtype=float)
    if not np.iterable(q):
        raise ValueError(f"quantiles ({quantiles}) must be a list of "
                         "partition boundaries")
    if len(q) < 2:
        raise ValueError(f"must provide at least 2 quantiles ({quantiles})")
    if not (q[1:] >= q[:-1]).all():
        raise ValueError(f"quantiles ({quantiles}) must be in ascending order")
    if (q[0] < 0) or (q[-1] > 1):
        raise ValueError(f"quantiles ({quantiles}) must be in [0, 1]")
    if (skymap < 0).any():
        raise ValueError(f"skymap ({skymap}) must be strictly nonnegative")
    q[q == 1] += 0.1  # make it closed on the right
    i = skymap.argsort()
    r = (skymap/(nside*nside))[i].cumsum()
    norm = r[-1]
    if norm == 0:
        raise ValueError(f"skymap ({skymap}) has total integral of 0")
    r /= norm
    j = r.searchsorted(q)
    l = j.copy()
    l[l >= len(i)] = len(i) - 1
    return (i[l:u] for l, u in zip(j[:-1], j[1:])), skymap[i[l]], norm*np.pi/3


def uniq_intersection(u⃗1, u⃗2):
    """Downselect the pixel indices given in ``u⃗1`` to the set that
    overlaps with pixels in ``u⃗2`` and return pairs of indices into
    both of the input index lists showing which pixel in ``u⃗2`` each
    downselected pixel from ``u⃗1`` overlaps with. Use this to get rid of
    pixels outside of a given region, or alternatively use it as part of a
    calculation involving two multi-resolution skymaps whose pixel sizes are
    non-identical. Written to perform efficiently on arbitrary inputs with
    O(MlogN+NlogM) performance (where M, N are the lengths of the input
    arrays).

    Parameters
    ----------
    u⃗1 : array
        Indices of HEALPix pixels in NUNIQ ordering. Pixels corresponding to
        these indices **MUST NOT OVERLAP**.
    u⃗2 : array
        Indices of HEALPix pixels in NUNIQ ordering. Pixels corresponding to
        these indices **MUST NOT OVERLAP**.

    Returns
    -------
    u⃗̇1 : array
        Indices *into* ``u⃗1`` that overlap with ``u⃗2``.
    u⃗̇2 : array
        Corresponding indices *into* ``u⃗2`` that overlap with ``u⃗1``.
    δo⃗ : array
        Corresponding differences in order between the indices, e.g. if the
        first entry of ``u⃗̇1`` has NSIDE 16 (order 4) and the
        corresponding entry of ``u⃗̇2`` has NSIDE 1024 (order 10), then the
        first entry of ``δo⃗`` will be (10 - 4) = 6.

    Raises
    ------
    ValueError
        If either ``u⃗1`` or ``u⃗2`` contain indices referring to
        overlapping pixels (note that this may happen even if the inputs do not
        contain repeating values since different pixel sizes can overlap).

    Examples
    --------
    Some pixels with NSIDE of 16, 32, and 64, respectively:

    >>> from pprint import pprint
    >>> import numpy as np
    >>> u⃗1 = np.array([1024, 4100, 1027, 1026, 44096])

    Pixels at NSIDE = 32 that overlap with only the first and last pixels of
    ``u⃗1``:

    >>> u⃗2 = np.array([4096, 4097, 1025, 1026, 11024])

    We should see correspondence between index 0 of ``u⃗1`` and indices 0,
    1 of ``u⃗1``; and correspondence between index 2 of ``u⃗1`` and
    index 2 of ``u⃗2``:

    >>> pprint(tuple(a.astype(np.int) for a in uniq_intersection(u⃗1, u⃗2)),
    ...        width=60)
    (array([4, 3, 0, 0, 1]),
     array([4, 3, 0, 1, 2]),
     array([-1,  0,  1,  1, -1]))
    """
    import numpy as np

    u⃗ˢ, s⃗, o⃗, _, v⃗, u⃗̇ = nside_slices(u⃗1, u⃗2, return_index=True)
    if len(u⃗ˢ[0]) != len(u⃗1):
        raise ValueError("`u⃗1` must be unique and non-overlapping.")
    if len(u⃗ˢ[1]) != len(u⃗2):
        raise ValueError("`u⃗2` must be unique and non-overlapping.")

    ζ = 0
    i⃗ᶠ = [np.ndarray((len(u⃗ˢ[0])+len(u⃗ˢ[1]),), dtype=int) for _ in [0, 1]]
    δo⃗ = np.zeros_like(i⃗ᶠ[0], dtype=int)

    for s in reversed(range(len(s⃗[0]))):  # pylint: disable=invalid-name
        ϵ⃗ = list(np.intersect1d(v⃗[0][s], v⃗[1][s], return_indices=True)[1:])
        for i in [0, 1]:
            ϵ⃗[i] += s⃗[i][s].start                       # offset by slice start
            i⃗ᶠ[i][ζ:ζ+len(ϵ⃗[i])] = ϵ⃗[i]                 # put in result array
            if s < len(s⃗[0])-1:                         # coarsen high res
                u⃗ˢ[i][s⃗[i][s].stop:] >>= 2*(o⃗[s+1]-o⃗[s])
        ζ += len(ϵ⃗[0])  # offset for array insertions
        for i, j in [(0, 1), (1, 0)]:
            ρ⃗ = np.intersect1d(v⃗[i][s], u⃗ˢ[i][s⃗[i][s].stop:])
            if len(ρ⃗):  # pylint: disable=len-as-condition
                raise ValueError(f"`i⃗{i}` has pixels overlapping with "
                                 f"themselves at NUNIQ pixel indices {ρ⃗}")
            if len(v⃗[i][s]) == 0:  # pylint: disable=len-as-condition
                continue                                # skip empty same res
            for sⱼ in reversed(range(s+1, len(s⃗[0]))):
                ϵ⃗ʲ = np.searchsorted(v⃗[i][s], v⃗[j][sⱼ])
                i⃗ᵋ = np.nonzero(np.take(v⃗[i][s], ϵ⃗ʲ, mode='clip')==v⃗[j][sⱼ])[0]
                𝓁ᵋ = len(i⃗ᵋ)
                i⃗ᶠ[i][ζ:ζ+𝓁ᵋ] = s⃗[i][s].start
                i⃗ᶠ[j][ζ:ζ+𝓁ᵋ] = s⃗[j][sⱼ].start
                i⃗ᶠ[i][ζ:ζ+𝓁ᵋ] += ϵ⃗ʲ[i⃗ᵋ]
                i⃗ᶠ[j][ζ:ζ+𝓁ᵋ] += i⃗ᵋ
                δo⃗[ζ:ζ+𝓁ᵋ] = (j-i)*(o⃗[sⱼ]-o⃗[s])  # pylint: disable=E1137
                ζ += 𝓁ᵋ

    return u⃗̇[0][i⃗ᶠ[0][:ζ]], u⃗̇[1][i⃗ᶠ[1][:ζ]], δo⃗[:ζ]


# pylint: disable=no-member
# def uniq_intersection_fine(u⃗1, u⃗2):
#     """
#     Like ``uniq_intersection``, but it also returns indices into subpixels not
#     covered by only one of the output index lists, along with a boolean array
#     indicating which pixels are missing. Use this if you need to
#     precisely find the overlap between two index arrays, e.g. for
#     numerically-precise operations on pixel values of two skymaps.

#     Parameters
#     ----------
#     Same as ``uniq_intersection``.

#     Returns
#     -------
#     Iᵢ⃗ⁱ⃗ᵒ : Tuple[array, array, array]
#         Return tuple of ``uniq_intersection``.
#     (u⃗ᵐ1, u⃗̇2, δo⃗1) : Tuple[array, array, array]
#         A tuple containing: NUNIQ indices ``u⃗ᵐ1`` that are *not* contained in
#         ``u⃗1`` but which *are* contained in ``u⃗2[u⃗̇2]``; the indices ``u⃗̇2`` into
#         the NUNIQ indices in ``u⃗2`` which correspond with the NUNIQ indices in
#         ``u⃗ᵐ1``; and the increase in order into each missing pixel in ``u⃗ᵐ1``.
#     (u⃗ᵐ2, u⃗̇1, δo⃗2) : Tuple[array, array, array]
#         Same, but for the missing pixels ``u⃗ᵐ2`` in ``u⃗1[u⃗̇1]`` and their
#         corresponding indices ``u⃗̇1`` into ``u⃗1``.

#     See Also
#     --------
#     uniq_intersection
#     """
#     from operator import gt, lt
#     import numpy as np

#     u⃗ = u⃗1, u⃗2
#     *u⃗̇, δo⃗ = uniq_intersection(*u⃗)                      # initial intersection
#     missing = []                                        # result storage
#     for i, c in enumerate(lt, gt):
#         δo⃗̇ᵢ = c(δo⃗, 0)                                  # where i are subpixels
#         δo⃗ᵢ = δo⃗[δo⃗̇ᵢ]                                   # subpix order increase
#         np.abs(δo⃗ᵢ, out=δo⃗ᵢ)
#         u⃗ᵢ = u⃗[i][u⃗̇[i][δo⃗̇ᵢ]]                            # subpix NUNIQ indices
#         u⃗̇ⱼ = u⃗̇[i-1][δo⃗̇ᵢ]                                # superpix ind indices
#         u⃗ⱼ = u⃗[i-1][u⃗̇ⱼ]                                 # superpix NUNIQ inds
#         u⃗ⱼᵘ, u⃗ⱼᵘ̇ = np.unique(u⃗ⱼ, return_inverse=True)   # unique containing pix
#         u⃗ⱼᵟ = np.zeros(u⃗ⱼᵘ.shape, dtype=int)            # how much smaller is
#         np.maximum.at(u⃗ⱼᵟ, u⃗ⱼᵘ̇, δo⃗ᵢ)                    #   smallest subpixel?
#         u⃗ⱼᶜ = 1 << 2*u⃗ⱼᵟ                                # superpix area
#         np.subtract.at(u⃗ⱼᶜ, u⃗ⱼᵘ̇, 1 << 2*(u⃗ⱼᵟ[u⃗ⱼᵘ̇] - δo⃗ᵢ))   # - area covered
#         u⃗ⱼᵐᵘ̇ = u⃗ⱼᶜ != 0                                 # u⃗̇ⱼ w uncovered area
#         u⃗ⱼᵐ = u⃗ⱼᵘ[u⃗ⱼᵐᵘ̇]                                 # u⃗ⱼ w uncovered area
#         u⃗ⱼˢ = 4*u⃗ⱼᵐ.reshape((-1, 1))+np.arange(4)       # split u⃗ⱼᵐ into subpix
#         u⃗̇ᵢᵐ = u⃗ⱼᵐᵘ̇[u⃗ⱼᵘ̇]                                 # u⃗̇ᵢ w uncovered area
#         u⃗ᵢᵐ = u⃗ᵢ[u⃗̇ᵢᵐ]                                   # u⃗ᵢ w uncovered area
#         [u⃗̇ᵢˢ, u⃗̇ⱼˢ, δo⃗ˢ], [u⃗ᵢˢᵐ, u⃗̇ⱼˢᵐ, δo⃗ᵢˢ], o̸⃗ = uniq_intersection_fine(
#             u⃗ᵢᵐ, u⃗ⱼˢ.ravel())
#         δu⃗̇ⱼˢᵐ = np.setdiff1d(np.arange(u⃗ⱼˢ.size), u⃗̇ⱼˢ)  # in j not in i NSIDE*2
#         δu⃗ᵢˢᵐ = u⃗ⱼˢ[δu⃗̇ⱼˢᵐ]                              #   missing NUNIQ inds
#         δu⃗̇ⱼˢᵐ = (δu⃗̇ⱼˢᵐ//4)  # TODO recover index into u⃗̇ⱼ
#         u⃗̈ⱼˢᵐ = np.isin(np.arange(u⃗ⱼˢ.size), u⃗̇ⱼˢ)
#         assert(all(len(o̸) == 0 for o̸ in o̸⃗))


def uniq2nest(u⃗, nˢ, nest=True):
    """
    Take a set of HEALPix NUNIQ-ordered indices at arbitrary resolution
    covering an arbitrary portion of the sky and convert them to
    non-overlapping pixels at a fixed NSIDE (resolution), returning the indices
    of the resulting skymap in NUNIQ or NEST ordering.

    Parameters
    ----------
    u⃗ : array
        Indices of HEALPix pixels in NUNIQ ordering.
    nˢ : int
        HEALPix NSIDE value of the output map.
    nest : bool, optional
        Whether to return the fixed-resolution indices in NEST ordering. If
        ``False``, leave them in NUNIQ ordering (though they will still be at
        the fixed resolution specified as ``nˢ``).

    Returns
    -------
    u⃗ᵒ : array
        Indices covering the same sky region as ``u⃗`` (possibly a
        larger region if resolution is reduced) at resolution ``nˢ`` in
        either NEST order (if ``nest`` is ``True``) or NUNIQ order (if ``nest``
        is ``False``).

    Raises
    ------
    ValueError
        If ``nˢ`` is not a valid NSIDE value or ``u⃗`` are not valid
        NUNIQ indices, or if the requested resolution is too high too represent
        with int64.

    Examples
    --------
    Let's convert this NUNIQ sky region to an NSIDE=32 nested indices (this
    will split the first, third, and fourth pixels, which are coarser--and
    select the pixel containing the last pixel, which is smaller than--than the
    target pixel size):

    >>> import numpy as np
    >>> u⃗ = np.array([1024, 4100, 1027, 1026, 44096])
    >>> uniq2nest(u⃗, 32).astype(np.int)
    array([   0,    1,    2,    3,    4,    8,    9,   10,   11,   12,   13,
             14,   15, 6928])

    Same pixel indices, but keep them in NUNIQ format:

    >>> uniq2nest(u⃗, 32, nest=False).astype(np.int)
    array([ 4096,  4097,  4098,  4099,  4100,  4104,  4105,  4106,  4107,
            4108,  4109,  4110,  4111, 11024])

    Coarsen the pixels to NSIDE=16:

    >>> uniq2nest(u⃗, 16).astype(np.int)
    array([   0,    1,    2,    3, 1732])
    >>> uniq2nest(u⃗, 16, False).astype(np.int)
    array([1024, 1025, 1026, 1027, 2756])

    Increase resolution of all pixels to NSIDE=64:

    >>> uniq2nest(u⃗, 64).astype(np.int)
    array([    0,     1,     2,     3,     4,     5,     6,     7,     8,
               9,    10,    11,    12,    13,    14,    15,    16,    17,
              18,    19,    32,    33,    34,    35,    36,    37,    38,
              39,    40,    41,    42,    43,    44,    45,    46,    47,
              48,    49,    50,    51,    52,    53,    54,    55,    56,
              57,    58,    59,    60,    61,    62,    63, 27712])
    """
    import numpy as np

    check_valid_nuniq(u⃗)

    o⃗, [𝓁⃗], [v⃗] = nside_slices(u⃗)[2:5]
    δo⃗ = hp.nside2order(nˢ) - o⃗                 # change in order -> final
    ρ⃗ = np.ceil(4.**δo⃗).astype(int)             # repititions for each pixel

    i⃗ˢᶠ = np.cumsum(np.concatenate(([0], ρ⃗*𝓁⃗)))         # output slices
    u⃗ᵒ = np.ndarray((i⃗ˢᶠ[-1],), dtype=np.int64)         # result array
    v⃗ᶠ = [u⃗ᵒ[i⃗ˢᶠ[i]:i⃗ˢᶠ[i+1]] for i in range(len(v⃗))]   # output slice views

    for i in range(max(0, δo⃗[0]), len(v⃗)):      # decimate highres and sameres
        v⃗ᶠ[i][:] = v⃗[i]
        v⃗ᶠ[i] >>= 2*-δo⃗[i]  # in-place, avoid and extra array-copy

    for i in range(min(len(v⃗), δo⃗[0])):     # split lowres pix into higher res
        v⃗ᶠ[i][:] = np.repeat(v⃗[i], ρ⃗[i])
        v⃗ᶠ[i] <<= 2*δo⃗[i]  # in-place
        vᶠⁱ = v⃗ᶠ[i].reshape((-1, ρ⃗[i]))
        vᶠⁱ += np.arange(ρ⃗[i]).reshape((1, ρ⃗[i]))

    u⃗ᵒ = np.unique(u⃗ᵒ)
    return uniq2nest_and_nside(u⃗ᵒ, in_place=True)[0] if nest else u⃗ᵒ


def fill(u⃗, x⃗, nˢ, pad=None):
    """
    Rasterize a HEALPix multi-order skymap into a fixed-res full-sky HEALPix
    nested skymap, filling in missing values with a ``pad`` values.

    Parameters
    ----------
    u⃗ : array-like
        NUNIQ pixel indices of the input skymap
    x⃗ : array-like
        Pixel values of the input skymap
    nˢ : int
        NSIDE of the output skymap
    pad : int or float, optional
        Value to pad missing indices in the output skymap with. If not
        provided, use ``healpy.UNSEEN``, which will render as blank space in
        ``healpy.visufunc`` plots.

    See Also
    --------
    reraster : Similar, but for outputs that are also UNIQ-indexed.
    render : Similar, but for outputs with repreating UNIQ indices.
    nest_reres :
        For changing the resolution of NEST indices.

    Returns
    -------
    x⃗ⁿᵉˢᵗ : np.ndarray
        Fixed-res full-sky version of the input skymap in NEST ordering with
        missing values filled by ``pad``.
    """
    import numpy as np

    u⃗ᵒ0 = 4*nˢ**2                               # output offset
    u⃗ᵒ = np.arange(u⃗ᵒ0, 4*u⃗ᵒ0)                  # output NUNIQ indices
    pad = hp.UNSEEN if pad is None else pad        # default pad value
    return reraster(u⃗, x⃗, u⃗ᵒ, pad=pad)


def nest_reres(nest, nside_in, nside_out):
    """
    Change the NSIDE of nest indices. If decreasing resolution,
    partially-filled pixels will be included but marked in ``full``.

    Parameters
    ----------
    nest : array-like
        HEALPix NEST indices.
    nside_in : int
        The NSIDE of the provided HEALPix indices.
    nside_out : int
        The NSIDE of the output indices.

    Returns
    -------
    reres_nest : array-like
        The smallest set of HEALPix NEST indices at NSIDE = ``nside_out``
        fully covering ``nest``.
    full : bool or array
        If resolution is not decreased, equals ``True``. If resolution is
        decreased, a boolean array that is ``True`` for all indices in
        ``reres_nest`` whose subpixels are all included in ``nest``.

    See Also
    --------
    fill :
        For converting skymap *values* (not indices) from *UNIQ* (not NEST)
        skymaps.

    Examples
    --------
    Double the resolution:

    >>> n, full = nest_reres([0, 1], 2, 4)
    >>> full
    True
    >>> print(n)
    [0 1 2 3 4 5 6 7]

    No effect:

    >>> n, full = nest_reres([0, 1], 2, 2)
    >>> full
    True
    >>> print(n)
    [0 1]

    Halve the resolution:
    >>> n, full = nest_reres([0, 1], 2, 1)
    >>> print(full)
    [False]
    >>> print(n)
    [0]
    """
    import numpy as np

    nest = np.array(nest, copy=True)
    d = hp.nside2order(nside_out) - hp.nside2order(nside_in)
    if d == 0:
        return nest, True
    if d > 0:
        return (4**d*nest.reshape((-1, 1))+np.arange(4**d)).ravel(), True
    i, ct = np.unique(nest//4**(-d), return_counts=True)
    return i, ct == 4


def wcs2nest(wcs, nside=None, order_delta=None):
    """
    Get NEST pixels at ``nside`` resolution covering an ``astropy.wcs.WCS``
    instance's pixels as well as their ``x`` and ``y`` pixel coordinates.
    All returned pixels and coordinates will be within the image boundaries
    and the indices will be non-repeating. If ``order_delta`` is provided, then
    the NEST resolution is doubled the number of times specified thereby. Can
    only pass one of ``nside`` or ``order_delta`` or else a ``ValueError`` is
    raised.

    Returns
    -------
    nside : int
        The NSIDE of the output indices.
    nest : NDArray[(Any,), int]
        The HEALPix NEST indices.
    x : NDArray[(Any,), float]
        The pixel-space x-coordinates of the points in ``nest``.
    y : NDArray[(Any,), float]
        The pixel-space y-coordinates of the points in ``nest``.
    """
    import numpy as np
    from astropy.coordinates.sky_coordinate import SkyCoord
    from astropy.units import deg

    if nside is not None and order_delta is not None:
        raise ValueError("Can only specify one of nside or order_delta.")
    valid, ra, dec = wcs2ang(wcs)
    # coarse search to not miss pixels
    ns = resol2nside(wcs2resol(wcs).to('rad').value,
                     degrees=False, coarse=True) >> 1
    nest = np.unique(hp.ang2pix(ns, ra.to('deg').value, dec.to('deg').value,
                                lonlat=True, nest=True))
    del ra, dec
    nside = nside or ns
    if order_delta is not None:
        d = order_delta  # undo search coarsening
        nside = nside << d if d > 0 else nside >> -d
    nest = nest_reres(nest, ns, nside)[0]
    co = SkyCoord(*(a*deg for a in hp.pix2ang(nside, nest, nest=True,
                                              lonlat=True)), frame='icrs')
    x, y = wcs.world_to_pixel(co)
    xm, ym = wcs.pixel_shape
    include = (~np.isnan(x)) & ((x<xm-.5) & (x>-.5) & (y<ym-.5) & (y>-.5))
    return nside, nest[include], x[include], y[include]


def wcs2resol(wcs):
    """
    Get the resolution of an ``astropy.wcs.WCS`` coordinate system, i.e. the
    smallest inter-pixel distance, as an ``astropy.units.Quantity`` with
    angular unit.
    """
    from astropy.units import Unit

    return min(abs(d)*Unit(u) for d, u in zip(wcs.wcs.cdelt, wcs.wcs.cunit))


def wcs2ang(wcs: 'astropy.wcs.WCS', lonlat=True):
    """
    Convert an ``astropy.wcs.WCS`` world coordinate system's pixels into ICRS
    coordinate angles.

    Parameters
    ----------
    wcs: astropy.wcs.WCS
        The world coordinate system for whose pixels you want coordinate
        values.
    lonlat: bool, optional
        If ``True``, return right ascension/declination. If ``False``, return
        (phi, theta) angles.

    Returns
    -------
    valid: array
        Boolean mask indicating which values from the ``WCS`` are valid. The
        rest can be padded with a fill value by the user (most likely
        ``np.nan``.
    ra_or_theta: astropy.units.Quantity
        The right-ascension/longitude if ``lonlat=True``, otherwise the
        zenith/theta angle.
    dec_or_phi: astropy.units.Quantity
        The declination/latitude angle if ``lonlat=True``, otherwise the
        azimuthal/phi angle.
    """
    import numpy as np
    from astropy.units import deg, Quantity

    sk = wcs.pixel_to_world(*np.meshgrid(*map(np.arange, wcs.pixel_shape),
                                         sparse=True)).icrs
    ra = Quantity(sk.ra)
    dec = Quantity(sk.dec)
    del sk
    valid = ~np.isnan(ra)
    assert np.all(valid == ~np.isnan(dec))
    if lonlat:
        return valid, ra[valid], dec[valid]
    return valid, 90*deg-dec[valid], ra[valid]


def wcs2mask_and_uniq(wcs):
    """
    Convert an ``astropy.wcs.WCS`` world coordinate system's pixels into NUNIQ
    indices for HEALPix pixels of approximately the same size.
    """
    valid, ra, dec = wcs2ang(wcs, lonlat=True)
    nˢ = resol2nside(wcs2resol(wcs).to('rad').value, degrees=False)
    return valid, nest2uniq(
        hp.ang2pix(nˢ, ra.to('deg').value, dec.to('deg').value,
                   lonlat=True, nest=True),
        nˢ,
        in_place=True
    )


def outline_effect():
    """Get a ``matplotlib.patheffects.withStroke`` effect that outlines text
    nicely to improve plot readability."""
    from matplotlib.patheffects import withStroke

    return withStroke(
        linewidth=OUTLINE_STROKE,
        foreground=OUTLINE_COLOR,
    )


def monochrome_opacity_colormap(name, color):
    """
    Get a monochrome ``matplotlib.colors.LinearSegmentedColormap`` with color
    defined by ``rgba`` (values between zero and one). Opacity will range from
    full transparency for the minimum value to the alpha value set in ``rgba``.
    """
    from matplotlib.colors import to_rgba, LinearSegmentedColormap

    *rgb, a = to_rgba(color)
    m = LinearSegmentedColormap.from_list(name, [[*rgb, 0], [*rgb, a]])
    m.set_under((0, 0, 0, 0))  # transparent background (-np.inf imgshow)
    return m


def render(u⃗, x⃗, u⃗ᵒ, pad=None, valid=None, mask_missing=False, Iᵢ⃗ⁱ⃗ᵒ=None):
    """
    Like ``reraster``, but allows you to map to a partially-covered ``u⃗ᵒ``
    skymap, e.g. for rendering a plot, thanks to a call to
    ``np.unique(u⃗ᵒ, return_inverse=True)`` wrapping the whole thing (to take
    care of scattering values to repeated pixels).

    Parameters
    ----------
    u⃗: array
        The indices of the skymap.
    x⃗: array
        The values of the skymap.
    u⃗ᵒ: array or astropy.wcs.WCS
        If ``u⃗ᵒ`` is an ``astropy.wcs.WCS`` world coordinate system, then
        ``wcs2mask_and_uniq`` will be used to get the indices. Non-valid pixels
        (i.e. pixels outside the projection area) will take on ``np.nan`` values,
        while valid pixels will be rendered as usual.
    pad: float, optional
        Pad value for missing pixels. If not provided, will raise an error if
        missing parts of the skymap fall in ``u⃗ᵒ``. To render a ``healpy``
        plot with missing pixels, pass ``pad=healpy.UNSEEN``.
    valid: array, optional
        If provided, results will be scattered into an array of the same shape
        as ``valid``, filling the indices where ``valid==True``. The number of
        ``True`` values in ``valid`` must therefore equal the length of ``u⃗ᵒ``.
        This argument only makes sense if ``u⃗ᵒ`` is an array of NUNIQ indices;
        if it is a ``WCS`` instance and ``valid`` is provided, an error is
        raised. Use ``valid`` to produce plots or to reuse indices produced by
        ``wcs2mask_and_uniq`` in several ``render`` invocations. See note on
        how ``mask_missing`` affects the result.
    mask_missing : bool
        If ``mask_missing=True``, return a ``np.ma.core.MaskedArray``. Missing
        values are tolerated and are marked as ``True`` in the
        ``mask_missing``. They will be set to ``pad or 0`` in the ``data``
        field. If ``valid`` is also provided, then the output will still be a
        ``np.ma.core.MaskedArray``, but will be set to ``True`` wherever
        ``valid == False`` in addition to wherever pixels are missing (and will
        still take on masked values of ``np.nan`` in the invalid regions).
    Iᵢ⃗ⁱ⃗ᵒ : Tuple[array, array, array]
        Return tuple of ``uniq_intersection``. Use this to save time in
        repeated invocations.

    Returns
    -------
    s⃗ₒ : array-like
        The pixel values at locations specified by u⃗ᵒ. If
        ``mask_missing=True``, will be a ``np.ma.core.MaskedArray`` set to
        ``True`` at the missing values in the ``valid`` field with missing
        ``data`` field values set to ``pad or None``.

    Raises
    ------
    ValueError
        If ``u⃗ᵒ`` is a ``WCS`` instance and ``valid`` is not ``None``.

    See Also
    --------
    reraster
    hpmoc.partial.PartialUniqSkymap.render
    hpmoc.points.PointsTuple.render
    np.ma.core.MaskedArray
    """
    import numpy as np
    from astropy.wcs import WCS

    if isinstance(u⃗ᵒ, WCS):
        if valid != None:
            raise ValueError("valid must be None if u⃗ᵒ is WCS.")
        valid, u⃗ᵒ = wcs2mask_and_uniq(u⃗ᵒ)
    u⃗ᵘ, u⃗̇ᵘ = np.unique(u⃗ᵒ, return_inverse=True)
    s⃗ = reraster(u⃗, x⃗, u⃗ᵘ, pad=pad, mask_missing=mask_missing, Iᵢ⃗ⁱ⃗ᵒ=Iᵢ⃗ⁱ⃗ᵒ)[u⃗̇ᵘ]
    if valid is None:  # for both mask_missing True and False
        return s⃗
    s⃗ₒ = np.full(valid.shape, np.nan)
    if mask_missing:
        s⃗ₒ = np.ma.MaskedArray(s⃗ₒ, True)
    s⃗ₒ[valid] = s⃗
    return s⃗ₒ


# pylint: disable=unsupported-assignment-operation,invalid-unary-operand-type
def reraster(u⃗, x⃗, u⃗ᵒ, pad=None, mask_missing=False, Iᵢ⃗ⁱ⃗ᵒ=None):
    """
    Rasterize skymap pixel values ``x⃗`` with NUNIQ indices ``u⃗`` to match
    pixels ``u⃗ᵒ``, discarding sky areas excluded by ``u⃗ᵒ`` and (optionally)
    padding missing values with ``pad``.

    Parameters
    ----------
    u⃗ : array-like
        NUNIQ indices of the skymap.
    x⃗ : array-like
        Pixel values. Must be the same length as u⃗.
    u⃗ᵒ : array-like
        NUNIQ indices of the output skymap.
    pad : float or int, optional
        A pad value to use for pixels missing from the input skymap. Only used
        if ``u⃗`` does not fully cover ``u⃗ᵒ``. Use ``healpy.UNSEEN`` for this
        value if you want to mark pixels as not-observed for HEALPy plots etc.
    mask_missing : bool
        If ``mask_missing=True``, return a ``np.ma.core.MaskedArray``. Missing
        values are tolerated and are marked as ``True`` in the
        ``mask_missing``. They will be set to ``pad or 0`` in the ``data``
        field.
    Iᵢ⃗ⁱ⃗ᵒ : Tuple[np.ndarray, np.ndarray, np.ndarray], optional
        If you've already computed ``uniq_intersection(u⃗, u⃗ᵒ)``, you can pass
        it as this argument to avoid recomputing it. No checks will be made for
        correctness if provided.

    Returns
    -------
    x⃗ᵒ : array-like
        Pixel values of the rasterized skymap corresponding to the indices
        given in ``u⃗ᵒ``. ``x⃗ᵒ`` values are pixel-area-weighted averages of the
        input pixel values, even if some pixels in ``u⃗ᵒ`` are not fully covered
        by pixels from ``u⃗``. Any parts of the sky defined in ``u⃗`` that are
        not covered by ``u⃗ᵒ`` are omitted, so this function can also be used to
        mask a skymap in a single step. If ``mask_missing=True``, is a
        ``np.ma.core.MaskedArray``.

    Raises
    ------
    ValueError
        If ``pad`` is not provided but ``u⃗`` does not cover all pixels in
        ``u⃗ᵒ``.

    See Also
    --------
    render
    hpmoc.partial.PartialUniqSkymap.reraster
    np.ma.core.MaskedArray

    Examples
    --------
    Create a small partial skymap with example pixel values:

    >>> import numpy as np
    >>> u⃗ = np.array([1024, 4100, 1027, 1026, 44096])
    >>> x⃗ = np.array([1.,   2.,   3.,   4.,   5.])

    We will rerasterize this skymap to these sky areas:

    >>> u⃗ᵒ = np.array([4096, 4097, 1025, 1026, 11024])
    >>> reraster(u⃗, x⃗, u⃗ᵒ)
    array([1., 1., 2., 4., 5.])

    The third pixel in ``u⃗`` is not present in ``u⃗ᵒ``, so we will need to
    provide a default pad value for it when rasterizing in the other direction.
    Note that the first pixel of the result is the average of the first and
    second pixels in the input map, since both of these have equal area and
    overlap with the first pixel:

    >>> reraster(u⃗ᵒ, x⃗, u⃗, pad=0.)
    array([1.5, 3. , 0. , 4. , 5. ])

    We can also simply mask that value by passing ``mask_missing=True``, in
    which case the result will be a ``np.ma.core.MaskedArray`` which is ``True`` for
    values which were missing (the missing/masked values themselves will be set
    to zero or ``pad`` if provided):

    >>> m = reraster(u⃗ᵒ, x⃗, u⃗, mask_missing=True)
    >>> print(m.data)
    [1.5 3.  0.  4.  5. ]
    >>> print(m.mask)
    [False False  True False False]

    Note that the values are averages of input pixels; in cases where only one
    input pixel is sampled from, the value remains unchanged. This makes
    ``reraster`` good for working with densities and other intensive spatial
    values; extensive values should have their pixel areas divided out before
    being rasterized.

    If you've already got the ``uniq_intersection`` of ``u⃗`` and ``u⃗ᵒ`` from a
    previous calculation, you can avoid recomputing it during rasterization by
    passing it as the ``Iᵢ⃗ⁱ⃗ᵒ`` argument, though beware it will not be checked
    for correctness:

    >>> reraster(u⃗, x⃗, u⃗ᵒ, Iᵢ⃗ⁱ⃗ᵒ=uniq_intersection(u⃗, u⃗ᵒ))
    array([1., 1., 2., 4., 5.])
    """
    import numpy as np
    from astropy.units import Quantity as Qty

    u⃗̇, u⃗̇ᵒ, δo⃗ = Iᵢ⃗ⁱ⃗ᵒ or uniq_intersection(u⃗, u⃗ᵒ)    # indices into u⃗, u⃗ᵒ
    u⃗̇ₘᵒ = np.setdiff1d(np.arange(len(u⃗ᵒ)), u⃗̇ᵒ)      # u⃗ᵒ pixels missing from u⃗
    if u⃗̇ₘᵒ.size != 0:
        if mask_missing:
            m = np.zeros(u⃗ᵒ.shape, dtype=bool)
            m[u⃗̇ₘᵒ] = True
        elif pad is None:
            raise ValueError(f"u⃗ ({u⃗}) missing pixels in u⃗ᵒ ({u⃗ᵒ}): {u⃗̇ₘᵒ}")
    else:
        m = False

    δ⃗ = 4.**-δo⃗                                     # NUNIQ slice offset tmp
    N⃗ᵒ = np.zeros(u⃗ᵒ.shape, dtype=float)            # normalization for pix avg
    np.add.at(N⃗ᵒ, u⃗̇ᵒ, δ⃗)
    np.add.at(N⃗ᵒ, u⃗̇ₘᵒ, 1.)                          # pad missing if u⃗̇ₘᵒ

    x⃗ᵒ = np.zeros(u⃗ᵒ.shape, dtype=float)            # pixel values
    if isinstance(x⃗, Qty):                          # include units for
        x⃗ᵒ = x⃗ᵒ*x⃗.unit                              #   astropy.Quantity
        δ⃗ = Qty(δ⃗, copy=False)
    δ⃗ *= x⃗[u⃗̇]                                       # subpixel contributions
    np.add.at(x⃗ᵒ, u⃗̇ᵒ, δ⃗)
    np.add.at(x⃗ᵒ, u⃗̇ₘᵒ, pad or 0.)                   # pad missing if u⃗̇ₘᵒ

    x⃗ᵒ /= N⃗ᵒ                                        # normalize pixel values
    if mask_missing:
        return np.ma.MaskedArray(x⃗ᵒ, m)
    return x⃗ᵒ


def uniq_coarsen(u, orders):
    """
    Coarsen the pixel indices in ``u`` to reduce storage and computation
    requirements.

    Parameters
    ----------
    u : array
        UNIQ indices to coarsen.
    orders : int
        How many times the resolution of the smallest pixels will be halved.

    Returns
    -------
    uc : array
        Unique coarsened pixel values in ascending order. All pixels will have
        a HEALPix order capped at the maximum order present in ``u`` minus
        ``orders``, unless this value is negative, in which case the output
        will only consist of base pixels. *If pixels in* ``u`` *overlap, it
        is possible that there will also be overlapping pixels in the output;
        no check is made for this.*

    Raises
    ------
    ValueError
        If ``orders < 0``.

    Examples
    --------
    Pixels 353, 354, and 355 lie within pixels 88, 22, and 5; pixels 80 and
    81 lie within pixels 20 and 5; pixel 21 lies within pixel 5.

    >>> u = [4, 21, 80, 81, 353, 354, 355]

    Coarsening by zero will have no effect:

    >>> print(uniq_coarsen(u, 0))
    [  4  21  80  81 353 354 355]

    Coarsening by one will only combine the very smallest pixels:

    >>> print(uniq_coarsen(u, 1))
    [ 4 21 80 81 88]

    Coarsening by larger numbers will combine so many higher orders:

    >>> print(uniq_coarsen(u, 2))
    [ 4 20 21 22]
    >>> print(uniq_coarsen(u, 3))
    [4 5]

    Coarsening by a value greater than the largest order will have no
    further effect, since the 12 base pixels cannot be combined:

    >>> print(uniq_coarsen([4, 21, 80, 81, 353, 354, 355], 4))
    [4 5]
    """
    import numpy as np

    if orders < 0:
        raise ValueError(f"orders must be > 0, instead got: {orders}")
    u = np.array(u, copy=True)
    o = uniq2order(u)
    omax = o.max()
    oout = max(omax - orders, 0)
    i = o > oout
    u[i] >>= 2 * (o[i] - oout)
    return np.unique(u)


def uniq_minimize(u, *x, test=eq, combine=lambda x, i: x[i]):
    """
    Take a set of HEALPix NUNIQ indices ``u⃗`` (and, optionally, pixel values
    ``x⃗``) and find the shortest equivalent multi-order pixelation by combining
    pixels. If ``x⃗`` is provided, only combine pixels whose values are equal.
    This can also be used if a canonical pixelization is needed for a given
    mask or skymap.

    Parameters
    ----------
    u: array-like
        HEALPix NUNIQ indices of the skymap in question.
    x: array-like, optional
        Pixel values of the array. If included, sub-pixels will only be
        combined if their pixel values are equal accordint to ``test`` for all
        ``x`` values provided.
    test: func, optional
        An equality test for determining whether adjacent pixels can be
        combined. Defaults to a standard equality check, ``operator.eq``.
        Override this if you want, e.g., approximately equal floating point
        values to be combined, small values rounded to zero, etc.
    combine: func, optional
        A function for combining pixels. Expects an argument ``x``, of the form
        of one of the arrays passed in for ``x``, as well as a boolean mask of
        pixels ``i`` which select the first pixel of each four to be combined.
        By default, simply selects this first pixel; you could alternatively
        provide a function which, e.g., averages the combined pixels.

    Returns
    -------
    u⃗ᵐ : numpy.ndarray
        The shortest equivalent NUNIQ indexing that can describe ``u``.
    *x⃗ᵐ : numpy.ndarray, optional
        Corresponding pixel values in ``x``, combined according to
        ``combined``.

    Examples
    --------
    Make a set of pixels corresponding to the first four base pixels as well as
    the first pixel at NSIDE = 2 lying in the fifth base pixel:

    >>> import numpy as np
    >>> u = np.concatenate([nest2uniq(np.arange(2), 1),
    ...                     nest2uniq(np.arange(8, 17), 2)])
    >>> print(u)
    [ 4  5 24 25 26 27 28 29 30 31 32]

    UNIQ indices 24-31 cover the same area as 6-7; ``uniq_minimize`` will
    detect this:

    >>> um, = uniq_minimize(u)
    >>> print(um)
    [ 4  5  6  7 32]

    This makes no difference against a constant skymap, with, e.g., values of
    ``1`` everywhere:

    >>> um, xm = uniq_minimize(u, np.full_like(u, 1))
    >>> print(um)
    [ 4  5  6  7 32]
    >>> print(xm)
    [1 1 1 1 1]

    If, however, the 4 pixels in the range 28-31 do *not* have equal values,
    they will not be combined with the default choice of ``test``:

    >>> um, xm = uniq_minimize(u, np.array([1, 2, 3, 3, 3, 3, 4, 5, 6, 7, 8]))
    >>> print(um)
    [ 4  5  6 28 29 30 31 32]
    >>> print(xm)
    [1 2 3 4 5 6 7 8]

    This can be very effective for combining pixels in skymaps made by
    algorithms like CWB,
    which often produce adjacent pixels with zero probability, or BAYESTAR,
    which is natively MOC but is often distributed at fixed resolution. For
    example, an NSIDE=128 pixel in a BAYESTAR skymap whose smallest pixel
    has NSIDE=1024 will be split into 64 NSIDE=1024 subpixels to bring the
    entire skymap to a single resolution without loss of precision; we can
    represent such a single pixel using its NUNIQ indices ``u2`` and a
    constant skymap value ``x2``:

    >>> u2 = np.arange(4194304, 4194368)
    >>> x2 = np.ones_like(u2)
    >>> um2, xm2 = uniq_minimize(u2, x2)

    The pixel indices will be combined to give the NUNIQ index of the
    original pixel at NSIDE=128:

    >>> print(um2)
    [65536]

    The pixel values will remain the same:

    >>> print(xm2)
    [1]
    """
    import numpy as np

    isort = np.argsort(u)
    u = u[isort]
    x = [xx[isort] for xx in x]
    us = []
    xs = [[] for _ in x]
    orders = np.arange(uniq2order(u[-1])+1, -1, -1)
    bounds = np.searchsorted(u, nest2uniq(np.zeros_like(orders),
                                             hp.order2nside(orders)))
    for xx in x:
        if len(xx) != len(u):
            raise ValueError("Indices and values must have same length.")
    for i in range(len(bounds)-2):
        last, first = bounds[i:i+2]
        # there can be no redundancy with fewer than 4 pixels
        if last - first < 4:
            us.append(u[first:last])
            for xxs, xx in zip(xs, x):
                xxs.append(xx[first:last])
            continue
        uu = u[first:last-3].copy()
        # find 0th pixels from matching quartets
        m = np.zeros((last-first,), dtype=bool)
        m[:-3] = uu % 4 == 0
        for j in range(3):
            uu += 1
            m[:-3] &= uu == u[first+1+j:last-2+j]
            for xx in x:
                m[:-3] &= test(xx[first:last-3], xx[first+1+j:last-2+j])
        # now find skipped indices
        s = m.copy()
        # do not mutate and compare simultaneously
        s[1:-2] = s[1:-2] | s[:-3]
        s[2:] = s[2:] | s[:-2]
        np.logical_not(s, out=s)
        # store the skipped indices for later concatenation
        us.append(u[first:last][s])
        for xxs, xx in zip(xs, x):
            xxs.append(xx[first:last][s])
        # combine pixels and put into existing buf adjacent next lowest order
        combined = u[first:last][m] // 4
        nc = len(combined)
        u[first:first+nc] = combined
        del combined
        for xx in x:
            xx[first:first+nc] = combine(xx[first:last], m)
        assert 4 * nc + len(us[-1]) == last - first
        # absorb combined pixels by modifying the next lowest order's bounds
        bounds[i+1] += nc
        # now re-sort the next lowest order
        next_last, next_first = bounds[i+1:i+3]
        isort = u[next_first:next_last].argsort()
        u[next_first:next_last] = u[next_first:next_last][isort]
        for xx in x:
            xx[next_first:next_last] = xx[next_first:next_last][isort]
    return [np.concatenate([y[:bounds[-2]], *ys[::-1]])
            for y, ys in zip([u, *x], [us, *xs])]


def uniq_diadic(Ω, u⃗ⁱ, x⃗ⁱ, pad=None, coarse=True):
    """
    Apply a diadic function ``Ω(x⃗ᶠ1, x⃗ᶠ2) -> y⃗ᶠ`` that operates on skymap pixel
    values of the same resolution to skymaps with arbitrary
    pixelization/resolution schemes and pixel orders, returning the indices and
    pixel values of the resulting skymap. Useful for binary operations between
    arbitrary partial skymaps.

    Parameters
    ----------
    Ω : Callable[[np.ndarray, np.ndarray], np.ndarray]
        A binary function operating on two sets of skymap pixel values
        corresponding elementwise to the same sky locations. **Must be a
        skymap-resolution independent operation for the results to make
        sense.**
    u⃗ⁱ : Tuple[np.ndarray, np.ndarray]
        NUNIQ indices of the two skymaps to be passed to ``Ω``.
    x⃗ⁱ : List[np.ndarray, np.ndarray]
        Pixel values (corresponding to the locations specified by ``u⃗ⁱ``) of
        the skymaps to be passed to Ω. Must have same lengths as the arrays in
        ``u⃗ⁱ``.
    pad : float or int, optional
        A pad value to use for parts of the sky contained in *either* ``u⃗ⁱ[0]``
        or ``u⃗ⁱ[1]`` but *not* in both (since ``Ω`` will be undefined in these
        regions). If not provided, the returned skymap will only contain the
        intersection of the sky areas defined in ``u⃗ⁱ``.
    coarse : bool, optional
        If ``True``, for sky areas where ``u⃗ⁱ[0]`` and ``u⃗ⁱ[1]`` have different
        resolutions, pick the lower resolution for the output map (using
        pixel-area-weighted averages to decimate the higher-res regions). This
        produces shorter output arrays. If ``False``, split coarse pixels into
        the higher resolution of those specified in ``u⃗ⁱ[0]`` and ``u⃗ⁱ[1]`` for
        a given sky region; use this if you need to maintain resolution for
        subsequent calculations, but be aware that this may impact performance
        without improving accuracy, e.g. if you're planning to integrate the
        result of this operation. This can also be useful if you need to cover
        the *exact* area defined by the input skymaps.

    Returns
    -------
    u⃗ʸ : np.ndarray
        Sorted NUNIQ indices of the result of ``Ω``. In general, will be
        different from *both* ``u⃗ⁱ`` inputs.
    y⃗ : np.ndarray
        Pixel values of the result of ``Ω`` corresponding to indices ``u⃗ʸ``.

    Examples
    --------
    Take the product of two heterogeneous multi-order skymaps:

    >>> from pprint import pprint
    >>> from operator import mul
    >>> import numpy as np
    >>> u⃗1 = np.array([1024, 4100, 1027, 1026, 44096])
    >>> x⃗1 = np.array([1.,   2.,   3.,   4.,   5.])
    >>> u⃗2 = np.array([4096, 4097, 1025, 1026, 11024])
    >>> x⃗2 = np.array([0.,   10.,  1.,   100., 1000.])
    >>> pprint(uniq_diadic(mul, (u⃗1, u⃗2), (x⃗1, x⃗2)), width=60)
    (array([ 1024,  1025,  1026, 11024]),
     array([5.e+00, 2.e+00, 4.e+02, 5.e+03]))

    Provide a default pad value for indices non-overlapping parts of the input
    skymaps:

    >>> pprint(uniq_diadic(mul, (u⃗1, u⃗2), (x⃗1, x⃗2), pad=0.), width=60)
    (array([ 1024,  1025,  1026,  1027, 11024]),
     array([5.e+00, 2.e+00, 4.e+02, 0.e+00, 5.e+03]))

    Increase resolution as necessary (do not combine pixels):

    >>> pprint(uniq_diadic(mul, (u⃗1, u⃗2), (x⃗1, x⃗2), coarse=False), width=60)
    (array([ 1026,  4096,  4097,  4100, 44096]),
     array([4.e+02, 0.e+00, 1.e+01, 2.e+00, 5.e+03]))
    """
    import numpy as np

    tmp = np.array(uniq_intersection(*u⃗ⁱ))          # inds into u⃗ⁱ & changes in
    *u⃗̇ᵢ, δo⃗ = tmp[:, tmp[2].argsort()]              # order δo⃗, sorted on δo⃗
    del tmp                                         # mark for GC

    sᵒ⃗ = 0, *δo⃗.searchsorted([0, 1]), len(δo⃗)       # slice starts
    o̸⃗ = [slice(sᵒ⃗[j], sᵒ⃗[j+1]) for j in (0, 2, 1)]  # slice by δo⃗

    u⃗ʸⁱ = [u⃗ⁱ[0][u⃗̇ᵢ[0][o̸⃗[2]]]]                      # calc, store same-res
    y⃗ⁱ = [Ω(*(x⃗ⁱ[i][u⃗̇ᵢ[i][o̸⃗[2]]] for i in (0, 1)))] # in results list
    assert np.all(u⃗ʸⁱ[0] == u⃗ⁱ[1][u⃗̇ᵢ[1][o̸⃗[2]]]), f'indices do not correspond'

    for j in range(2):                              # j=0: downres; j=1: upres
        i = (j+coarse) % 2                          # target pixelization ind
        u⃗̇ᵁ, u⃗̇ᵁ̇ = np.unique(u⃗̇ᵢ[i][o̸⃗[j]], return_inverse=True)        # target
        u⃗̇ˈᵁ, u⃗̇ˈᵁ̇ = np.unique(u⃗̇ᵢ[i-1][o̸⃗[j]], return_inverse=True)    # reraster
        u⃗ʸⁱ.append(u⃗ⁱ[i][u⃗̇ᵁ])                       # put NUNIQ inds in result
        δx⃗ⁱ = [reraster(u⃗ⁱ[i-1][u⃗̇ˈᵁ], x⃗ⁱ[i-1][u⃗̇ˈᵁ], u⃗ʸⁱ[-1],        # same res
                        Iᵢ⃗ⁱ⃗ᵒ=(u⃗̇ˈᵁ̇, u⃗̇ᵁ̇, (2*j-1)*δo⃗[o̸⃗[j]])), x⃗ⁱ[i][u⃗̇ᵁ]][::2*j-1]
        y⃗ⁱ.append(Ω(*δx⃗ⁱ))                          # calculate result

    if pad is not None:                             # include non-overlapping
        for j in (0, 1):                            # regions if pad provided
            u⃗ʸⁱ.append(u⃗ⁱ[j][np.setdiff1d(np.arange(len(u⃗ⁱ[j])), u⃗̇ᵢ[j])])
            y⃗ⁱ.append(np.full(u⃗ʸⁱ[-1].shape, pad))

    u⃗ʸ = np.concatenate(u⃗ʸⁱ)                        # concatenate result lists
    y⃗ = np.concatenate(y⃗ⁱ)
    u⃗ʸ, i⃗ᵢᵒ = np.unique(u⃗ʸ, return_index=True)      # sort by NUNIQ index
    assert len(i⃗ᵢᵒ) == len(y⃗)                       # inds and values same len
    return u⃗ʸ, y⃗[i⃗ᵢᵒ]


@dataclass
class TmpGunzipFits:
    """
    A context manager that unzips a Gzip file to a temporary ``.fits`` file,
    returning a ``NamedTemporaryFile`` pointing to the tempfile, and deletes
    the tempfile when you're done with it. Can only do binary reads, so mode
    cannot be specified. Pass it a file name or a gzip file object that's
    already been opened in binary read ('rb') mode.
    """
    infile: Union[IO, str]
    filename = None

    @staticmethod
    def _gunzip(infile, outfile):
        buf = infile.read(GZIP_BUFFSIZE)
        while buf:
            outfile.write(buf)
            buf = infile.read(GZIP_BUFFSIZE)

    def __enter__(self):
        with NamedTemporaryFile('wb', suffix='.fits', delete=False) as tmp:
            if isinstance(self.infile, gzip.GzipFile):
                self._gunzip(self.infile, tmp)
                self.infile.seek(0)
            else:
                with gzip.open(self.infile, "rb") as infile:
                    self._gunzip(infile, tmp)
            self.filename = tmp.name
            return self.filename

    def __exit__(self, _type, _value, _traceback):
        if os.path.isfile(self.filename):
            os.unlink(self.filename)


def density_from_table(table, indices, nside, degrees=False):
    """
    Read probability density from ``table`` at the specified ``indices``.
    Will try to read PROBDENSITY, and if it fails, will try to convert PROB
    to a probability density by dividing by area-per-pixel. Probability
    can be provided in a density format (usually for MOC skymaps) or a
    probability-per-pixel format (fixed-res skymaps).

    Parameters
    ----------
    table : astropy.table.Table
        The table to read from. Must have either a ``PROBDENSITY`` or a
        ``PROB`` column.
    indices : array-like
        Indices of the pixels (i.e. table rows) to read.
    nside : int or array-like
        The HEALPix NSIDE parameter for the pixels. If each pixel has a
        different value, you can specify ``nside`` as an array of NSIDE
        values corresponding to the indices in ``indices``.
    degrees : bool, optional
        If ``True``, return densities in inverse square degrees. Otherwise,
        return densities in inverse steradians.

    Returns
    -------
    values : np.array
        A view onto the probability density values corresponding to
        ``indices``.

    Raises
    ------
    ValueError
        If ``table`` does not contain ``PROBDENSITY`` or ``PROB`` as columns or
        if any ``nside`` values are not valid.
    """
    from astropy.units import Unit
    if 'PROBDENSITY' in table.colnames:
        return table['PROBDENSITY'][indices].copy()
    LOGGER.debug("PROBDENSITY not found in %s, trying to calculate it from "
                 "PROB column and NSIDE %s", table, nside)
    # from ligo.skymap.io.read_sky_map
    if 'PROBABILITY' in table.colnames:
        LOGGER.debug("Column named PROBABILITY found. Renaming to PROB "
                     "(Fermi GBM convention?)")
        table.rename_column('PROBABILITY', 'PROB')
    return (table['PROB'].ravel()[indices] /
            nside2pixarea(nside, degrees=degrees) * Unit("pix / sr"))
    raise ValueError(f"No PROBDENSITY or PROB column in skymap {infile}")


def is_gz(infile: Union[IO, str]):
    """
    Check whether ``infile`` is a GZip file, in which case it should be
    unzipped.
    """
    if isinstance(infile, gzip.GzipFile):
        return True
    # from https://stackoverflow.com/a/47080739/3601493
    if hasattr(infile, 'seek') and hasattr(infile, 'read'):
        magic_number = binascii.hexlify(infile.read(2))
        infile.seek(-2, 1)  # seek back before magic number read
    else:
        with open(infile, 'rb') as test_f:
            magic_number = binascii.hexlify(test_f.read(2))
    return magic_number == b'1f8b'


def set_partial_skymap_metadata(meta, mask, caller):
    """
    Write metadata to a partial skymap.
    """
    meta['MOC'] = True
    meta['PARTIAL'] = True
    meta['ORDERING'] = 'NUNIQ'
    history = meta.get('HISTORY', [])
    if not isinstance(history, list):
        history = [history]
    history += [''] + wrap(
        dedent(
            f"""
            Pixels were downselected by the HPMOC library using {caller} to
            overlap with the sky regions specified in the following NUNIQ mask
            indices:

            {{mask}}.

            This is a PARTIAL skymap; not all pixels are included. You can read
            this file with astropy.table.Table.read(); the included indices in
            NUNIQ ordering can be found in the UNIQ column, and their
            corresponding pixel probability densities can be found in the
            PROBDENSITY column.
            """
        ).format(mask=mask),
        width=70
    )
    meta['HISTORY'] = history


def handle_compressed_infile(func):
    """
    Wrap ``read_partial_skymap`` so that it opens zipped fits files by
    temporarily decompressing them.
    """

    @functools.wraps(func)
    def wrapper(infile, *args, **kwargs):
        if is_gz(infile):
            try:
                with TmpGunzipFits(infile) as tmp:
                    LOGGER.debug("zipped skymap %s detected, attempting "
                                 "to decompress to tempfile %s", infile, tmp)
                    return func(tmp, *args, **kwargs)
            except OSError as err:
                if 'gzip' in str(err):
                    LOGGER.error("OSError while attempting to decompress, "
                                 "letting astropy try to read it: %s", err)
                else:
                    raise err
        return func(infile, *args, **kwargs)

    return wrapper


@handle_compressed_infile
def read_partial_skymap(infile: Union[IO, str], u⃗, memmap=True):
    """
    Read in pixels from a FITS skymap (or a gzip-compressed FITS skymap) that
    lie in a specific sky region.  Attempts to minimize memory usage by
    memory-mapping pixels in the input file and only loading those specified in
    ``u⃗``.

    Parameters
    ----------
    infile : str or file
        A FITS HEALPix skymap file path or file object opened in binary read
        mode 'rb' (optionally compressed; see note under ``memmap``)
    u⃗ : array-like
        HEALPix pixel indices in NUNIQ ordering specifying the
        region of the skymap that should be loaded.
    memmap : bool, optional
        If ``True``, use memory mapping when reading in files. This can VASTLY
        reduce memory required for high-resolution skymaps. If ``infile`` is
        gzip-compressed and ``memmap`` is ``True``, then ``infile`` will be
        decompressed to a temporary file and data will be read from it
        (necessary to constrain memory usage); for high-resolution skymaps,
        this can require the availability of several gigabytes of tempfile
        storage. You will need to make use of ``TmpGunzipFits`` when working
        with zipped files in order to be able to use ``memmap=True``.

    Returns
    -------
    partial_skymap : astropy.table.Table
        A partial skymap table in ``nested`` ordering. Has two columns:
        ``UNIQ`` and ``PROBDENSITY``.  If the resolution of the original
        HEALPix skymap file is lower than that of the u⃗, then any pixels
        overlapping with those in ``u⃗`` will be used; this might
        result in a larger portion of the skymap being used than that
        specified in ``u⃗``. The resolution of this skymap will be
        the resolution of the smallest pixel loaded from the input file (in
        the case of ``ring`` or ``nested`` ordering, this is just the
        resolution of the input skymap).

    See Also
    --------
    uniq_minimize
    TmpGunzipFits
    """
    from astropy.table import Table
    import numpy as np

    T = Table.read(infile, format='fits', memmap=memmap)    # read skymap table
    meta = T.meta.copy()
    nˢ = T.meta.get('NSIDE', None)
    ordering = T.meta['ORDERING']
    set_partial_skymap_metadata(meta, u⃗, read_partial_skymap.__qualname__)

    if ordering == 'NUNIQ':
        s⃗̇ = np.concatenate([uniq_intersection(T['UNIQ'][i:i+PIX_READ], u⃗)[0]+i
                            for i in range(0, len(T), PIX_READ)])
        u⃗, s⃗̈ˢ = np.unique(T['UNIQ'][s⃗̇], return_index=True)
        nˢ = uniq2nside(u⃗)
    elif nˢ is None:
        raise ValueError(f"No NSIDE defined in header {meta} for {infile}")
    else:
        u⃗ = np.sort(uniq2nest(u⃗, nˢ, nest=False))           # rasterize
        s⃗̇ = uniq2nest_and_nside(u⃗)[0]                       # nest still sorted
        if ordering == 'RING':                              # sorted RING inds
            s⃗̇, s⃗̈ˢ = np.unique(hp.nest2ring(nˢ, s⃗̇), return_inverse=True)
        elif ordering == 'NESTED':                           # keep nest inds
            s⃗̈ˢ = slice(None)                              # keep order
        else:
            raise ValueError(f"Unexpected ORDERING in {infile}: {ordering}")
    #return Table(np.array([u⃗, density_from_table(T, s⃗̇, nˢ)[s⃗̈ˢ]]).T,
    #             names=['UNIQ', 'PROBDENSITY'], meta=meta)
    return Table({'UNIQ': u⃗,
                  'PROBDENSITY': density_from_table(T, s⃗̇, nˢ)[s⃗̈ˢ]},
                 meta=meta)


def nside_slices(*u⃗, include_empty=False, return_index=False,
                 return_inverse=False, dtype=None):
    """
    Sort and slice up a list of NUNIQ pixel index arrays, returning the sorted
    arrays as well as slice information for chunking them by NSIDE (pixel
    resolution), accessing the original array data, and the NSIDE orders of
    each chunk.

    This is just a wrapper around ``group_slices`` using HEALPix NSIDE order as
    the grouping function.

    Parameters
    ----------
    *u⃗, array-like
        ``np.array`` instances containing NUNIQ HEALPix indices
    include_empty : bool, optional
        If ``True``, also include NSIDE orders not included in the input
        indices. Affects all return values.
    return_index : bool, optional
        Whether to return ``u⃗̇``. Only returned if ``True``.
    return_inverse : bool, optional
        Whether to return ``u⃗̇ˢ``. Only returned if ``True``.
    dtype : int or numpy.dtype, optional
        If provided, cast the returned array to this data type. Useful for
        pre-allocating output arrays that only depend on spatial information.

    Returns
    -------
    u⃗ˢ : List[array]
        Sorted versions of each input array
    s⃗ : List[List[slice]]
        Slices into each ``u⃗ˢ`` chunked by NSIDE order
    o⃗ : array
        An array of HEALPix NSIDE orders included in the input indices
    𝓁⃗ : List[array]
        The lengths of each slice in ``slice_starts``
    v⃗ : List[List[array]]
        Lists of array views into each ``u⃗ˢ`` corresponding to the slices given
        in ``slices``
    u⃗̇ : List[array], optional
        Indices into the original array that give ``u⃗ˢ``
    u⃗̇ˢ : List[Array], optional
        Indices into each ``u⃗ˢ`` that give the original arrays

    See Also
    --------
    group_slices

    Examples
    --------
    >>> import numpy as np
    >>> from pprint import pprint
    >>> u⃗1 = np.array([1024, 4100, 1027, 263168, 263169, 1026, 44096])
    >>> u⃗2 = np.array([4096, 4097, 1025, 16842752, 1026, 11024])
    >>> us, s, o, l, v, ius = nside_slices(u⃗1, u⃗2, return_index=True)
    >>> pprint([uu.astype(np.int) for uu in us])
    [array([  1024,   1026,   1027,   4100,  44096, 263168, 263169]),
     array([    1025,     1026,     4096,     4097,    11024, 16842752])]
    >>> pprint(s)
    [[slice(0, 3, None),
      slice(3, 4, None),
      slice(4, 5, None),
      slice(5, 7, None),
      slice(7, 7, None)],
     [slice(0, 2, None),
      slice(2, 5, None),
      slice(5, 5, None),
      slice(5, 5, None),
      slice(5, 6, None)]]
    >>> o.astype(np.int)
    array([ 4,  5,  6,  8, 11])
    >>> [ll.astype(np.int) for ll in l]
    [array([3, 1, 1, 2, 0]), array([2, 3, 0, 0, 1])]
    >>> for vv in v[0]:
    ...     print(vv)
    [1024 1026 1027]
    [4100]
    [44096]
    [263168 263169]
    []
    >>> for vv in v[1]:
    ...     print(vv)
    [1025 1026]
    [ 4096  4097 11024]
    []
    []
    [16842752]
    >>> [ii.astype(np.int) for ii in ius]
    [array([0, 5, 2, 1, 6, 3, 4]), array([2, 4, 0, 1, 5, 3])]
    """
    return group_slices(*u⃗, f=uniq2order,
                        fⁱ=lambda x: nest2uniq(0, hp.order2nside(x)),
                        include_empty=include_empty, return_index=return_index,
                        return_inverse=return_inverse, dtype=dtype)


def group_slices(*u⃗, f=lambda x: x, fⁱ=lambda x: x, include_empty=False,
                 return_index=False, return_inverse=False, dtype=None):
    """
    Group elements of ``u⃗`` inputs using some sort of monotonic step function
    ``f: u⃗.dtype -> int`` codomain and a pseudo-inverse ``fⁱ`` mapping to the
    smallest element of the input domain giving that output value (both
    identity by default) and return a variety of views and slices into these
    groups. See ``nside_slices`` for documentation and an implementation that
    groups by HEALPix NSIDE order; this function is the same, but with ``o⃗``
    replaced by the result of ``f`` on elements of ``u⃗``.  You can use this
    function with the default grouping functions to group integers by value,
    e.g. for working with ``δo⃗`` arrays from ``uniq_intersection``.

    See Also
    --------
    nside_slices
    """
    import numpy as np

    u⃗ = [np.array(u, dtype=dtype, copy=False) for u in u⃗ ]
    u⃗ˢ, u⃗̇_u⃗̇ˢ = [np.unique(u, return_index=return_index,
                          return_inverse=return_inverse) for u in u⃗], []
    if return_index or return_inverse:
        u⃗ˢ, *u⃗̇_u⃗̇ˢ = zip(*u⃗ˢ)
    s, e = [[uⁱ[i] for uⁱ in u⃗ˢ if len(uⁱ)] for i in (0, -1)]
    if not s and not e:
        return tuple([u⃗ˢ, [[]]*len(u⃗), np.array([]), [], [[]]*len(u⃗)]+u⃗̇_u⃗̇ˢ)
    o⃗ = np.arange(f(min(s)), f(max(e))+2)
    # o⃗ = np.arange(f(min(uⁱ[0] for uⁱ in u⃗ˢ)), f(max(uⁱ[-1] for uⁱ in u⃗ˢ))+2)
    i⃗ₛ = [np.searchsorted(uⁱ, fⁱ(o⃗)) for uⁱ in u⃗ˢ]
    𝓁⃗ = [i⃗ₛⁱ[1:]-i⃗ₛⁱ[:-1] for i⃗ₛⁱ in i⃗ₛ]
    i⃗ᴱ̸ = np.arange(len(𝓁⃗[0])) if include_empty else np.nonzero(sum(𝓁⃗))[0]
    s⃗ = [[slice(i⃗ₛⁱ[j], i⃗ₛⁱ[j+1]) for j in i⃗ᴱ̸] for i⃗ₛⁱ in i⃗ₛ]
    return tuple([u⃗ˢ, s⃗, o⃗[i⃗ᴱ̸], [𝓁[i⃗ᴱ̸] for 𝓁 in 𝓁⃗],
                  [[u⃗ˢ[i][s⃗ᵢⱼ] for s⃗ᵢⱼ in s⃗ᵢ] for i, s⃗ᵢ in enumerate(s⃗)]]+u⃗̇_u⃗̇ˢ)
