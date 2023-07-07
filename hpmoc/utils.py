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
#
# - Interpolation on a sphere:
#   https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.interpolate.SmoothSphereBivariateSpline.html

"""
Utility functions used across healpix_skymap classes.
"""

from __future__ import annotations

import os
from operator import eq
from numbers import Integral
from typing import (
    Optional, TypeVar, Union, IO, Any,
    Tuple, Callable, Protocol, TYPE_CHECKING, overload
)
from typing_extensions import TypeGuard
import functools
from dataclasses import dataclass  # possible removal for older pythons
from math import pi
from textwrap import wrap, dedent
import logging
import gzip
from tempfile import NamedTemporaryFile
import binascii
from .healpy_utils import alt_compress, alt_expand
from .healpy import healpy as hp
from .arraysetops import intersect1d

import numpy as np
from numpy.typing import NDArray, ArrayLike, DTypeLike

Int = Union[int, 'np.integer[Any]']
IntArray = NDArray['np.integer[Any]']

if TYPE_CHECKING:
    from astropy.wcs import wcs
    from astropy.units import Quantity

LOGGER = logging.getLogger(__name__)
GZIP_BUFFSIZE = 10**5
PIX_READ = 4**8
OP_CHUNKSIZE = 10**6
PIXEL_CONSISTENCY_ERROR = 1e-9
MAX_ORDER = 30
UINT_RANGES = {(0, 2**(8*2**i)): f'uint{8*2**i}' for i in range(4)}
INT_RANGES = {(-i//2, i//2): v[1:] for [_, i], v in UINT_RANGES.items()}
OUTLINE_STROKE = 2.2  # thickness of outline surrounding text
OUTLINE_COLOR = (1, 1, 1, 1)  # outlines of text and neutrino markers
FONT_SIZE = 14  # matplotlib font size
N_X_OFFSET = 0.08  # [inches]
N_Y_OFFSET = 0.08  # [inches]


def max_uint_type(largest: Int) -> 'np.dtype':

    if largest < 0:
        raise ValueError(f"Positive values only: {largest}")
    for dt in ('u1', 'u2', 'u4', 'u8'):
        if ~np.array(0, dt) > largest:
            return np.dtype(dt)
    raise ValueError("You didn't pass an integer representable in less than "
                     "64 bits: {largest}")


class EmptyStream(OSError):
    "Raised when a file stream returns no further content."

_IntDType = TypeVar('_IntDType', bound='np.integer[Any]')

# TODO test this much more
def uniq2xyf_nside(u):
    """
    Examples
    --------
    >>> uniq2xyf_nside(8)
    (0, 0, 4, 1)

    See Also
    --------
    xyf_nside2uniq
    """
    i, nside = uniq2nest_and_nside(u)
    nside2 = nside*nside
    f = (u//nside2)-4
    i -= f*nside2
    return alt_compress(i), alt_compress(i>>1, True), f, nside


# TODO test this much more
def xyf_nside2uniq(x, y, f, nside):
    """
    Examples
    --------
    >>> (xyf_nside2uniq(*uniq2xyf_nside(np.arange(4, 10000))) == np.arange(4, 10000)).all()
    True

    See Also
    --------
    uniq2xyf_nside
    """
    return alt_expand(x) + (alt_expand(y)<<1) + (f+4)*nside*nside


def min_int_dtype(vmin, vmax):
    """
    Find the smallest integer datatype that can represent a given range of
    values.
    """

    if vmax < vmin:
        raise ValueError(f"vmax must be larger than vmin. got: {vmin, vmax}")
    ranges = INT_RANGES if vmin < 0 else UINT_RANGES
    for [dmin, dmax], dtype in ranges.items():
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


def nest2ang(n, nside):
    """
    Get the angles corresponding to these nested indices ``n`` and NSIDE
    values ``nside``.

    Parameters
    ----------
    n : array-like
        HEALPix NESTED indices of pixels of interest.
    nside : int or array-like
        The NSIDE values corresponding to each pixel. Can be a scalar if all
        pixels are at the same NSIDE.

    Returns
    -------
    ra_dec : astropy.units.Quantity
        2D array whose first row is the right-ascension and second row is
        the declination (in degrees) of each pixel. You can get each of these
        individually with ``ra, dec = nest2ang(...)``.
    """
    from astropy.units import degree  # pylint: disable=no-name-in-module

    nside = np.full(n.shape, nside) if isinstance(nside, Integral) else nside
    ra_dec = np.ndarray((2, len(n)))                    # pre-allocate results
    for nside in np.unique(nside):                            # iterate NSIDE values
        i = np.nonzero(nside == nside)[0]
        ra_dec[:, i] = np.array(hp.pix2ang(nside, n[i], nest=True, lonlat=True))
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

    r = hp.nside2resol(2**np.arange(MAX_ORDER))[::-1]
    r = np.degrees(r) if degrees else r
    side = 'left' if coarse else 'right'
    return 1 << MAX_ORDER - np.searchsorted(r, res, side=side)


def nest2dangle(n, nside, ra, dec, degrees=True, in_place=False):
    """
    Get the angular distance between the pixels defined in ``n, nside`` and the
    point located at ``ra, dec``.

    Parameters
    ----------
    n : array-like
        HEALPix NEST indices
    nside : int
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
        If ``True``, store the result in ``n`` to reduce memory usage.
        Requires ``n.dtype == np.float64``.

    Returns
    -------
    Δθ⃗ : astropy.units.Quantity
        Angular distance in radians between each pixel in ``n`` and the point
        at ``ra, dec``.

    Examples
    --------
    The 12 base healpix pixels' distances from north pole should all be equal
    to 90 minus their declinations:

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
    from astropy.units import rad, deg, Quantity  # pylint: disable=E0611

    n = np.array(n, copy=False)
    if in_place and n.dtype != np.float64:
        raise ValueError("Can't operate in-place on a non-float array: %s" % n)
    Ω = [θ.to(deg).value if isinstance(θ, Quantity) else
         (θ if degrees else np.degrees(θ))
         for θ in (ra, dec)]
    x0, y0, z0 = hp.ang2vec(*Ω, lonlat=True)
    x, y, z = [np.ndarray((min(len(n), OP_CHUNKSIZE),)) for _ in range(3)]
    dots = n if in_place else np.ndarray(n.shape)
    for i in range(0, len(n), OP_CHUNKSIZE):
        nⁱ = n[i:i+OP_CHUNKSIZE].astype(int, copy=False)
        N = len(nⁱ)
        x[:N], y[:N], z[:N] = hp.pix2vec(nside, nⁱ, nest=True)
        x *= x0
        y *= y0
        z *= z0
        x += y
        x += z
        dots[i:i+N] = x[:N]
    return Quantity(np.arccos(dots, out=dots), rad, copy=False)


def uniq2dangle(u, ra, dec, degrees=True):
    """
    Like ``nest2dangle``, but takes HEALPix NUNIQ indices as input ``u``.

    Examples
    --------
    The 12 base healpix pixels' distances from north pole should all be equal
    to 90 minus their declinations:

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
    [uˢ], _, o⃗, _, [v⃗], [u̇ˢ] = nside_slices(u, return_inverse=True,
                                            dtype=float)
    nside = 1 << o⃗                                     # NSIDE for each view
    for nside, v⃗ⁱ in zip(nside, v⃗):                       # views into uˢ
        v⃗ⁱ -= 4*nside**2                               # convert to nest in-place
        u = nest2dangle(v⃗ⁱ, nside, ra, dec, degrees=degrees, in_place=True).unit
    return uˢ[u̇ˢ]*u                                 # unsort results in uˢ*


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
    >>> dangle_rad(0, pi/2, np.array([0, 0, 0, 2, 3]),
    ...            np.array([pi/2, -pi/2, 0, 0, 0]))/pi
    array([0. , 1. , 0.5, 0.5, 0.5])
    """
    delta_ra = mapra - ra
    dot_prod = (np.sin(mapdec)*np.sin(dec) +
                np.cos(mapdec)*np.cos(dec)*np.cos(delta_ra))
    return np.arccos(dot_prod)


@overload
def nest2uniq(indices: Int, nside: Int, in_place: bool = ...) -> Int: ...

@overload
def nest2uniq(indices: Union[Int, IntArray], nside: Union[Int, IntArray], in_place: bool = ...) -> IntArray: ...

def nest2uniq(indices: Union[Int, IntArray], nside: Union[Int, IntArray], in_place: bool = False): # type: ignore
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
    >>> nest2uniq(np.array([284, 286, 287,  10,   8,   2, 279, 278, 285]), 8)
    array([540, 542, 543, 266, 264, 258, 535, 534, 541])
    """

    check_valid_nside(nside)
    add = 4*nside*nside
    if in_place:
        indices += add # type: ignore
    else:
        indices = indices + add
    return np.int64(indices)


def is_int_array(a: 'NDArray[Any]') -> TypeGuard[IntArray]:
    return issubclass(a.dtype.type, np.integer)

def check_valid_nside(nside: 'ArrayLike'):
    """Checks whether ``nside`` is a valid HEALPix NSIDE value.

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
    Does nothing if you provide a valid NSIDE value

    >>> check_valid_nside([16, 4])
    >>> check_valid_nside(4)
    >>> check_valid_nside(1024)

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

    nside = np.atleast_1d(nside)
    if not is_int_array(nside):
        raise TypeError(
            f"NSIDE must be an integer type! Got '{nside.dtype.name}'")

    if np.any(nside & (nside-1)):
        raise ValueError("Not a valid NSIDE value: {}".format(nside))


RAD_TO_DEG = 180 * pi
SR_TO_DEG2 = RAD_TO_DEG * RAD_TO_DEG

@overload
def nside2pixarea(nside: Int, degrees: bool = ...) -> 'np.floating[Any]': ...

@overload
def nside2pixarea(nside: IntArray, degrees: bool = ...) -> 'NDArray[np.floating[Any]]': ...

def nside2pixarea(nside: Union[Int, IntArray], degrees = False):
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

    >>> nsides = np.array([1, 1, 2, 4])
    >>> areas = np.array([allsky/12, allsky/12, allsky/48, allsky/192])
    >>> np.all(nside2pixarea(nsides) == areas)
    True
    """

    check_valid_nside(nside)
    result = pi/3/nside/nside

    if degrees:
        return result * SR_TO_DEG2
    return result


def check_valid_nuniq(indices: 'ArrayLike') -> None:
    """Checks that ``indices`` are valid NUNIQ indices.

    Raises
    ------
    ValueError
        If ``indices`` are not valid NUNIQ indices, i.e. they are not integers
        greater than 3.
    """

    indices = np.atleast_1d(indices)
    if not len(indices):  # can't go wrong with zero indices
        return
    if not is_int_array(indices):
        raise ValueError("NUNIQ indices must be integers.")
    if not indices.min() > 3:
        raise ValueError("NUNIQ indices start at 4; found smaller values.")


@overload
def uniq2order(indices: 'np.number') -> 'np.int64': ...

@overload
def uniq2order(indices: 'NDArray[np.number]') -> 'NDArray[np.int64]': ...

def uniq2order(indices): # type: ignore
    """
    Get the HEALPix order of the given NUNIQ-ordered indices.

    Raises
    ------
    ValueError
        If ``indices`` are not valid NUNIQ indices, i.e. they are not integers
        greater than 3.
    """
    check_valid_nuniq(indices)
    return np.int64(np.log2(indices)/2 - 1)


@overload
def uniq2nside(indices: 'np.number') -> 'np.int64': ...

@overload
def uniq2nside(indices: 'NDArray[np.number]') -> 'NDArray[np.int64]': ...

def uniq2nside(indices):
    """
    Get the NSIDE value of the given NUNIQ-ordered indices.

    Raises
    ------
    ValueError
        If ``indices`` are not valid NUNIQ indices, i.e. they are not integers
        greater than 3.
    """

    return np.int64(1) << uniq2order(indices)


@overload
def uniq2nest_and_nside(indices: '_IntDType', in_place: bool = ...) -> Tuple['_IntDType', 'np.int64']: ...

@overload
def uniq2nest_and_nside(indices: 'NDArray[_IntDType]', in_place: bool = ...) -> Tuple['NDArray[_IntDType]', 'np.int64']: ...

def uniq2nest_and_nside(indices: Union['_IntDType', 'NDArray[_IntDType]'], in_place: bool = False):
    """
    Parameters
    ----------
    indices : array
        A scalar or numpy array of NUNIQ indices
    in_place : bool, optional
        If ``True``, perform the conversion in-place and return the converted
        ``indices`` object along with the calculated ``nside``. Has no effect
        if scalar is passed

    Returns
    -------
    nest_indices : array
        The indices in nest ordering at their respective resolutions
    nside : array
        The resolution expressed as NSIDE of the indices given in
        ``nest_indices``

    Examples
    --------
    >>> from pprint import pprint
    >>> nuniq = np.array([540, 542, 543, 266, 264, 258, 535, 534, 541])
    >>> pprint([u.astype(int) for u in uniq2nest_and_nside(nuniq)])
    [array([284, 286, 287,  10,   8,   2, 279, 278, 285]),
     array([8, 8, 8, 8, 8, 8, 8, 8, 8])]

    Confirm that this is the inverse of nest2uniq:

    >>> all(nest2uniq(*uniq2nest_and_nside(nuniq)) == nuniq)
    True
    """
    
    # uniq2nside implicitly checks whether the indices are valid NUNIQ indices
    nside = uniq2nside(indices)
    sub = 4*nside*nside

    # We can safely ignore type checking here because every element of sub
    # will be less than indices
    if in_place:
        indices -= sub # type: ignore
    else:
        indices = indices - sub # type: ignore
    return indices, nside


# def uniq_sample(uniq, skymap, quantiles):
    # """
    # Do a probability-weighted random sample of locations in a skymap based on
    # quantile. Follows *ascending* density order, just like
    # ``nside_quantile_indices``.

    # Parameters
    # ----------
    # uniq : array-like
    #     NUNIQ indices of the skymap to be sampled.
    # skymap : array-like
    #     Probability density at each of the pixels in ``uniq``.
    # quantiles : array-like
    #     Values in the interval ``[0, 1]`` specifying which quantile to select
    #     from the skymap. For example, ``[0.5, 0.9]`` will select locations from
    #     within the pixels for which the integral of all
    #     higher-probability-density regions are 0.5 and 0.1, respectively.

    # Returns
    # -------
    # locations : array
    #     Max-resolution (order = 30) NUNIQ indices drawn from the PDF defined by
    #     ``uniq`` and ``skymap`` at the specified ``quantiles`` assuming that
    #     each pixel in ``skymap`` has a uniform PDF within its borders. The
    #     difference between successive values in the density-ordered CDF of
    #     ``skymap`` will be used to select subpixels in NEST ordering from the
    #     pixel in ``uniq`` with the greatest CDF value lower than the
    #     corresponding quantile, allowing for reproducible results. The returned
    #     values have the same order as the input ``quantiles``. You can use
    #     ``healpy.pix2ang(*uniq2nest_and_nside(locations)[::-1], nest=True)`` to
    #     recover the angles corresponding to the recovered locations.

    # See Also
    # --------
    # nside_quantile_indices
    # uniq2nest_and_nside
    # healpy.pix2ang
    # """

    # orders = uniq2order(uniq)
    # nside = hp.order2nside(orders)
    # [inds], levels, norm = nside_quantile_indices(nside, skymap, [0, 1])
    # cdf = hp.nside2pixarea(nside[inds])
    # cdf *= skymap[inds]
    # np.cumsum(cdf, out=cdf)
    # rough_loc = cdf.searchsorted(quantiles)
    # i_loc = rough_loc.argsort()
    # TODO finish


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

    >>> skymap = np.array([ 9, 10, 11,  0,  1,  2,  3,  4,  5,  6,  7,  8])
    >>> i, levels, norm = nside_quantile_indices(1, skymap, [0.1, 0.9])
    >>> [ii.astype(int) for ii in i]
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
    ...  ][0].astype(int)
    array([0, 1])

    Equal lower and upper bounds give empty quantiles:

    >>> [*nside_quantile_indices(nside, skymap, [0.5, 0.5])[0]
    ...  ][0].astype(np.int64)
    array([], dtype=int64)

    Recover all indices (sorted by density):

    >>> [*nside_quantile_indices(nside, skymap, [0, 1])[0]][0].astype(int)
    array([ 3,  4,  5,  6,  7,  8,  9, 10, 11,  0,  1,  2])

    Pick the 90% CR:

    >>> [*nside_quantile_indices(nside, skymap, [0.1, 1])[0]][0].astype(int)
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

    q = np.asarray(quantiles, dtype=float)
    if np.ndim(q) != 1:
        raise ValueError(f"quantiles ({quantiles}) must be a 1-D list of "
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
    return ((i[ll:u] for ll, u in zip(j[:-1], j[1:])), skymap[i[l]],
            norm*np.pi/3)

def uniq_intersection(u1: IntArray, u2: IntArray) -> Tuple[IntArray, IntArray, IntArray]:
    """Downselect the pixel indices given in ``u1`` to the set that
    overlaps with pixels in ``u2`` and return pairs of indices into
    both of the input index lists showing which pixel in ``u2`` each
    downselected pixel from ``u1`` overlaps with. Use this to get rid of
    pixels outside of a given region, or alternatively use it as part of a
    calculation involving two multi-resolution skymaps whose pixel sizes are
    non-identical. Written to perform efficiently on arbitrary inputs with
    O(MlogN+NlogM) performance (where M, N are the lengths of the input
    arrays).

    Parameters
    ----------
    u1 : array
        Indices of HEALPix pixels in NUNIQ ordering. Pixels corresponding to
        these indices **MUST NOT OVERLAP**.
    u2 : array
        Indices of HEALPix pixels in NUNIQ ordering. Pixels corresponding to
        these indices **MUST NOT OVERLAP**.

    Returns
    -------
    u1_idx : array
        Indices *into* ``u1`` that overlap with ``u2``.
    u2_idx : array
        Corresponding indices *into* ``u2`` that overlap with ``u1``.
    δo : array
        Corresponding differences in order between the indices, e.g. if the
        first entry of ``u1_idx`` has NSIDE 16 (order 4) and the
        corresponding entry of ``u2_idx`` has NSIDE 1024 (order 10), then the
        first entry of ``δo`` will be (10 - 4) = 6.

    Raises
    ------
    ValueError
        If either ``u1`` or ``u2`` contain indices referring to
        overlapping pixels (note that this may happen even if the inputs do not
        contain repeating values since different pixel sizes can overlap).

    Examples
    --------
    Some pixels with NSIDE of 16, 32, and 64, respectively:

    >>> from pprint import pprint
    >>> u1 = np.array([1024, 4100, 1027, 1026, 44096])

    Pixels at NSIDE = 32 that overlap with only the first and last pixels of
    ``u1``:

    >>> u2 = np.array([4096, 4097, 1025, 1026, 11024])

    We should see correspondence between index 0 of ``u1`` and indices 0,
    1 of ``u1``; and correspondence between index 2 of ``u1`` and
    index 2 of ``u2``:

    >>> pprint(tuple(a.astype(int) for a in uniq_intersection(u1, u2)),
    ...        width=60)
    (array([4, 3, 0, 0, 1]),
     array([4, 3, 0, 1, 2]),
     array([-1,  0,  1,  1, -1]))
    """

    uˢ, s⃗, o⃗, _, v⃗, u̇ = nside_slices(u1, u2, return_index=True)
    if len(uˢ[0]) != len(u1):
        raise ValueError("`u1` must be unique and non-overlapping.")
    if len(uˢ[1]) != len(u2):
        raise ValueError("`u2` must be unique and non-overlapping.")

    ζ = 0
    i⃗ᶠ = [np.ndarray((len(uˢ[0])+len(uˢ[1]),), dtype=int) for _ in [0, 1]]
    δo = np.zeros_like(i⃗ᶠ[0], dtype=int)

    for s in reversed(range(len(s⃗[0]))):  # pylint: disable=invalid-name
        _, *e = intersect1d(v⃗[0][s], v⃗[1][s], return_indices=True, assume_unique=(True, True))
        for i in (0, 1):                 
            i⃗ᶠ[i][ζ:ζ+len(e[i])] = e[i] + s⃗[i][s].start # offset by slice start and put in result array
            if s < len(s⃗[0])-1:                          # coarsen high res
                uˢ[i][s⃗[i][s].stop:] >>= 2*(o⃗[s+1]-o⃗[s])
        ζ += len(e[0])  # offset for array insertions
        for i, j in [(0, 1), (1, 0)]:
            ρ⃗ = intersect1d(v⃗[i][s], uˢ[i][s⃗[i][s].stop:], assume_unique=(True, False))
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
                δo[ζ:ζ+𝓁ᵋ] = (j-i)*(o⃗[sⱼ]-o⃗[s])  # pylint: disable=E1137
                ζ += 𝓁ᵋ

    return u̇[0][i⃗ᶠ[0][:ζ]], u̇[1][i⃗ᶠ[1][:ζ]], δo[:ζ]


# pylint: disable=no-member
# def uniq_intersection_fine(u1, u2):
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
#     (uᵐ1, u̇2, δo⃗1) : Tuple[array, array, array]
#         A tuple containing: NUNIQ indices ``uᵐ1`` that are *not* contained in
#         ``u1`` but which *are* contained in ``u2[u̇2]``; the indices ``u̇2`` into
#         the NUNIQ indices in ``u2`` which correspond with the NUNIQ indices in
#         ``uᵐ1``; and the increase in order into each missing pixel in ``uᵐ1``.
#     (uᵐ2, u̇1, δo⃗2) : Tuple[array, array, array]
#         Same, but for the missing pixels ``uᵐ2`` in ``u1[u̇1]`` and their
#         corresponding indices ``u̇1`` into ``u1``.

#     See Also
#     --------
#     uniq_intersection
#     """
#     from operator import gt, lt

#     u = u1, u2
#     *u̇, δo⃗ = uniq_intersection(*u)                      # initial intersection
#     missing = []                                        # result storage
#     for i, c in enumerate(lt, gt):
#         δo⃗̇ᵢ = c(δo⃗, 0)                                  # where i are subpixels
#         δo⃗ᵢ = δo⃗[δo⃗̇ᵢ]                                   # subpix order increase
#         np.abs(δo⃗ᵢ, out=δo⃗ᵢ)
#         uᵢ = u[i][u̇[i][δo⃗̇ᵢ]]                            # subpix NUNIQ indices
#         u̇ⱼ = u̇[i-1][δo⃗̇ᵢ]                                # superpix ind indices
#         uⱼ = u[i-1][u̇ⱼ]                                 # superpix NUNIQ inds
#         uⱼᵘ, uⱼᵘ̇ = np.unique(uⱼ, return_inverse=True)   # unique containing pix
#         uⱼᵟ = np.zeros(uⱼᵘ.shape, dtype=int)            # how much smaller is
#         np.maximum.at(uⱼᵟ, uⱼᵘ̇, δo⃗ᵢ)                    #   smallest subpixel?
#         uⱼᶜ = 1 << 2*uⱼᵟ                                # superpix area
#         np.subtract.at(uⱼᶜ, uⱼᵘ̇, 1 << 2*(uⱼᵟ[uⱼᵘ̇] - δo⃗ᵢ))   # - area covered
#         uⱼᵐᵘ̇ = uⱼᶜ != 0                                 # u̇ⱼ w uncovered area
#         uⱼᵐ = uⱼᵘ[uⱼᵐᵘ̇]                                 # uⱼ w uncovered area
#         uⱼˢ = 4*uⱼᵐ.reshape((-1, 1))+np.arange(4)       # split uⱼᵐ into subpix
#         u̇ᵢᵐ = uⱼᵐᵘ̇[uⱼᵘ̇]                                 # u̇ᵢ w uncovered area
#         uᵢᵐ = uᵢ[u̇ᵢᵐ]                                   # uᵢ w uncovered area
#         [u̇ᵢˢ, u̇ⱼˢ, δo⃗ˢ], [uᵢˢᵐ, u̇ⱼˢᵐ, δo⃗ᵢˢ], o̸⃗ = uniq_intersection_fine(
#             uᵢᵐ, uⱼˢ.ravel())
#         δu̇ⱼˢᵐ = np.setdiff1d(np.arange(uⱼˢ.size), u̇ⱼˢ)  # in j not in i NSIDE*2
#         δuᵢˢᵐ = uⱼˢ[δu̇ⱼˢᵐ]                              #   missing NUNIQ inds
#         δu̇ⱼˢᵐ = (δu̇ⱼˢᵐ//4)  # TODO recover index into u̇ⱼ
#         üⱼˢᵐ = np.isin(np.arange(uⱼˢ.size), u̇ⱼˢ)
#         assert(all(len(o̸) == 0 for o̸ in o̸⃗))


def uniq2nest(u, nside, nest=True):
    """
    Take a set of HEALPix NUNIQ-ordered indices at arbitrary resolution
    covering an arbitrary portion of the sky and convert them to
    non-overlapping pixels at a fixed NSIDE (resolution), returning the indices
    of the resulting skymap in NUNIQ or NEST ordering.

    Parameters
    ----------
    u : array
        Indices of HEALPix pixels in NUNIQ ordering.
    nside : int
        HEALPix NSIDE value of the output map.
    nest : bool, optional
        Whether to return the fixed-resolution indices in NEST ordering. If
        ``False``, leave them in NUNIQ ordering (though they will still be at
        the fixed resolution specified as ``nside``).

    Returns
    -------
    u_out : array
        Indices covering the same sky region as ``u`` (possibly a
        larger region if resolution is reduced) at resolution ``nside`` in
        either NEST order (if ``nest`` is ``True``) or NUNIQ order (if ``nest``
        is ``False``).

    Raises
    ------
    ValueError
        If ``nside`` is not a valid NSIDE value or ``u`` are not valid
        NUNIQ indices, or if the requested resolution is too high too represent
        with int64.

    Examples
    --------
    Let's convert this NUNIQ sky region to an NSIDE=32 nested indices (this
    will split the first, third, and fourth pixels, which are coarser--and
    select the pixel containing the last pixel, which is smaller than--than the
    target pixel size):

    >>> u = np.array([1024, 4100, 1027, 1026, 44096])
    >>> uniq2nest(u, 32).astype(int)
    array([   0,    1,    2,    3,    4,    8,    9,   10,   11,   12,   13,
             14,   15, 6928])

    Same pixel indices, but keep them in NUNIQ format:

    >>> uniq2nest(u, 32, nest=False).astype(int)
    array([ 4096,  4097,  4098,  4099,  4100,  4104,  4105,  4106,  4107,
            4108,  4109,  4110,  4111, 11024])

    Coarsen the pixels to NSIDE=16:

    >>> uniq2nest(u, 16).astype(int)
    array([   0,    1,    2,    3, 1732])
    >>> uniq2nest(u, 16, False).astype(int)
    array([1024, 1025, 1026, 1027, 2756])

    Increase resolution of all pixels to NSIDE=64:

    >>> uniq2nest(u, 64).astype(int)
    array([    0,     1,     2,     3,     4,     5,     6,     7,     8,
               9,    10,    11,    12,    13,    14,    15,    16,    17,
              18,    19,    32,    33,    34,    35,    36,    37,    38,
              39,    40,    41,    42,    43,    44,    45,    46,    47,
              48,    49,    50,    51,    52,    53,    54,    55,    56,
              57,    58,    59,    60,    61,    62,    63, 27712])
    """

    check_valid_nuniq(u)

    o⃗, [𝓁⃗], [v⃗] = nside_slices(u)[2:5]
    δo⃗ = hp.nside2order(nside) - o⃗                 # change in order -> final
    ρ⃗ = np.ceil(4.**δo⃗).astype(int)             # repititions for each pixel

    i⃗ˢᶠ = np.cumsum(np.concatenate(([0], ρ⃗*𝓁⃗)))         # output slices
    u_out = np.ndarray((i⃗ˢᶠ[-1],), dtype=np.int64)         # result array
    v⃗ᶠ = [u_out[i⃗ˢᶠ[i]:i⃗ˢᶠ[i+1]] for i in range(len(v⃗))]   # output slice views

    for i in range(max(0, δo⃗[0]), len(v⃗)):      # decimate highres and sameres
        v⃗ᶠ[i][:] = v⃗[i]
        v⃗ᶠ[i] >>= 2*-δo⃗[i]  # in-place, avoid and extra array-copy

    for i in range(min(len(v⃗), δo⃗[0])):     # split lowres pix into higher res
        v⃗ᶠ[i][:] = np.repeat(v⃗[i], ρ⃗[i])
        v⃗ᶠ[i] <<= 2*δo⃗[i]  # in-place
        vᶠⁱ = v⃗ᶠ[i].reshape((-1, ρ⃗[i]))
        vᶠⁱ += np.arange(ρ⃗[i]).reshape((1, ρ⃗[i]))

    u_out = np.unique(u_out)
    return uniq2nest_and_nside(u_out, in_place=True)[0] if nest else u_out


def fill(u, x, nside, pad=None):
    """
    Rasterize a HEALPix multi-order skymap into a fixed-res full-sky HEALPix
    nested skymap, filling in missing values with a ``pad`` values.

    Parameters
    ----------
    u : array-like
        NUNIQ pixel indices of the input skymap
    x : array-like
        Pixel values of the input skymap
    nside : int
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
    xⁿᵉˢᵗ : np.ndarray
        Fixed-res full-sky version of the input skymap in NEST ordering with
        missing values filled by ``pad``.
    """

    u_out0 = 4*nside**2                               # output offset
    u_out = np.arange(u_out0, 4*u_out0)                  # output NUNIQ indices
    pad = hp.UNSEEN if pad is None else pad        # default pad value
    return reraster(u, x, u_out, pad=pad)


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
    nest : NDArray[Any, Int]
        The HEALPix NEST indices.
    x : NDArray[Any, Float]
        The pixel-space x-coordinates of the points in ``nest``.
    y : NDArray[Any, Float]
        The pixel-space y-coordinates of the points in ``nest``.
    """
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


def wcs2resol(wcs: 'wcs.WCS'):
    """
    Get the resolution of an ``astropy.wcs.WCS`` coordinate system, i.e. the
    smallest inter-pixel distance, as an ``astropy.units.Quantity`` with
    angular unit.
    """
    from astropy.units import Unit

    return min(abs(d)*Unit(u) for d, u in zip(wcs.wcs.cdelt, wcs.wcs.cunit))


def wcs2ang(wcs: 'wcs.WCS', lonlat=True) -> tuple[
    'NDArray[np.bool_]',
    'Quantity',
    'Quantity'
]:
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


def wcs2mask_and_uniq(wcs: 'wcs.WCS'):
    """
    Convert an ``astropy.wcs.WCS`` world coordinate system's pixels into NUNIQ
    indices for HEALPix pixels of approximately the same size.
    """
    valid, ra, dec = wcs2ang(wcs, lonlat=True)
    nside = resol2nside(wcs2resol(wcs).to('rad').value, degrees=False)
    return valid, nest2uniq(
        hp.ang2pix(nside, ra.to('deg').value, dec.to('deg').value,
                   lonlat=True, nest=True),
        nside,
        in_place=True
    )

_DType = TypeVar('_DType', covariant=True, bound='np.generic')
def interp_wcs_nn(
        wcs: 'wcs.WCS',
        data: 'NDArray[_DType]',
) -> Tuple[IntArray, 'NDArray[_DType]']:
    """
    Do a nearest-neighbor interpolation of ``data`` with coordinates
    specified by ``wcs`` FITS world coordinate system.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        The world coordinate system defining pixel locations. If loading
        a FITS file as an HDU called ``hdu``, you can get this argument
        as ``astropy.wcs.WCS(hdu.header)``. *Note that you will need to
        manually include units for dimensionful quantities.*
    data : array-like
        The data corresponding to ``WCS``. Available from an HDU as
        ``hdu.data``.

    Returns
    -------
    u : array
        The corresponding NUNIQ HEALPix indices of the input skymap.
    s : array-like
        The pixel-values of the input skymap interpolated at the locations of
        the pixels in ``u``.

    See Also
    --------
    interp_wcs
    hpmoc.partial.PartialUniqSkymap
    astropy.wcs.WCS
    """

    nside, nest, x, y = wcs2nest(wcs, order_delta=2)
    interp = data[np.round(x).astype(int), np.round(y).astype(int)]
    return nest2uniq(nest, nside), interp


def interp_wcs(
        wcs: 'wcs.WCS',
        data: 'NDArray[_DType]',
        interp: Union[
            str,
            Tuple[
                int,
                Callable[
                    [
                        'NDArray[np.floating]',
                        'NDArray[np.floating]',
                        'NDArray[_DType]'
                    ],
                    'NDArray[_DType]'
                ]
            ],
        ] = 'nearest'
) -> Tuple[IntArray, 'NDArray[_DType]']:
    """
    Interpolate ``data`` with coordinates specified by ``wcs`` FITS
    world coordinate system into a HEALPix NUNIQ skymap.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        The world coordinate system defining pixel locations. If loading
        a FITS file as an HDU called ``hdu``, you can get this argument
        as ``astropy.wcs.WCS(hdu.header)``. *Note that you will need to
        manually include units for dimensionful quantities.*
    data : array-like
        The data corresponding to ``WCS``. Available from an HDU as
        ``hdu.data``.
    interp : str or (int, func), optional
        The interpolation strategy to use. Can be a string specifying one
        of the following pre-defined strategies:

        - "nearest" for nearest-neighbor
        - "bilinear" for bicubic

        or else a tuple whose first element is the number of orders by
        which the pixels covering the ``WCS`` should have their resolution
        increased ("nearest" uses a value of 2, "bilinear" a value of 1;
        heuristically, a more sophisticated interpolation scheme can probably
        get away with 1), while the second element is a function taking the
        x, y coordinates of the pixels followed by the pixel values in ``data``
        and returning the interpolated pixel values (which will form the return
        value ``s`` of this function).

    Returns
    -------
    u : array
        The corresponding NUNIQ HEALPix indices of the input skymap.
    s : array-like
        The pixel-values of the input skymap interpolated at the locations of
        the pixels in ``u``.

    See Also
    --------
    interp_wcs_nn
    hpmoc.partial.PartialUniqSkymap
    astropy.wcs.WCS
    """
    if interp == 'nearest':
        return interp_wcs_nn(wcs, data)
    if interp == 'bilinear':
        raise NotImplementedError()
    if isinstance(interp, str):
        raise ValueError(f"Unrecognized interpolation strategy: {interp}")
    nside, nest, x, y = wcs2nest(wcs, order_delta=interp[0])
    return nest2uniq(nest, nside), interp[1](x, y, data)


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


def render(u, x, u_out, pad=None, valid=None, mask_missing=False, Iᵢ⃗ⁱ⃗ᵒ=None):
    """
    Like ``reraster``, but allows you to map to a partially-covered ``u_out``
    skymap, e.g. for rendering a plot, thanks to a call to
    ``np.unique(u_out, return_inverse=True)`` wrapping the whole thing (to take
    care of scattering values to repeated pixels).

    Parameters
    ----------
    u: array
        The indices of the skymap.
    x: array
        The values of the skymap.
    u_out: array or astropy.wcs.WCS
        If ``u_out`` is an ``astropy.wcs.WCS`` world coordinate system, then
        ``wcs2mask_and_uniq`` will be used to get the indices. Non-valid pixels
        (i.e. pixels outside the projection area) will take on ``np.nan`` values,
        while valid pixels will be rendered as usual.
    pad: float, optional
        Pad value for missing pixels. If not provided, will raise an error if
        missing parts of the skymap fall in ``u_out``. To render a ``healpy``
        plot with missing pixels, pass ``pad=healpy.UNSEEN``.
    valid: array, optional
        If provided, results will be scattered into an array of the same shape
        as ``valid``, filling the indices where ``valid==True``. The number of
        ``True`` values in ``valid`` must therefore equal the length of ``u_out``.
        This argument only makes sense if ``u_out`` is an array of NUNIQ indices;
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
        The pixel values at locations specified by u_out. If
        ``mask_missing=True``, will be a ``np.ma.core.MaskedArray`` set to
        ``True`` at the missing values in the ``valid`` field with missing
        ``data`` field values set to ``pad or None``.

    Raises
    ------
    ValueError
        If ``u_out`` is a ``WCS`` instance and ``valid`` is not ``None``.

    See Also
    --------
    reraster
    hpmoc.partial.PartialUniqSkymap.render
    hpmoc.points.PointsTuple.render
    np.ma.core.MaskedArray
    """
    from astropy.wcs import WCS

    if isinstance(u_out, WCS):
        if valid != None:
            raise ValueError("valid must be None if u_out is WCS.")
        valid, u_out = wcs2mask_and_uniq(u_out)
    uᵘ, u̇ᵘ = np.unique(u_out, return_inverse=True)
    s⃗ = reraster(u, x, uᵘ, pad=pad, mask_missing=mask_missing, intersection=Iᵢ⃗ⁱ⃗ᵒ)[u̇ᵘ]
    if valid is None:  # for both mask_missing True and False
        return s⃗
    s⃗ₒ = np.full(valid.shape, np.nan)
    if mask_missing:
        s⃗ₒ = np.ma.MaskedArray(s⃗ₒ, True)
    s⃗ₒ[valid] = s⃗
    return s⃗ₒ


def reraster(u, x, u_out,
             check_missing=True, pad=None, mask_missing=False,
             intersection=None, method='average'):
    """
    Rasterize skymap pixel values ``x`` with NUNIQ indices ``u`` to match
    pixels ``u_out``, discarding sky areas excluded by ``u_out`` and (optionally)
    padding missing values with ``pad``.

    Parameters
    ----------
    u : array-like
        NUNIQ indices of the skymap.
    x : array-like
        Pixel values. Must be the same length as u.
    u_out : array-like
        NUNIQ indices of the output skymap.
    check_missing : bool, optional
        If ``True`` (default), perform a check for pixels in ``u_out``
        that have no overlapping pixels in ``u``. If ``False``, skip this
        check. check_missing must be True if either ``pad`` or
        ``mask_missing=True`` are specified. You want to leave this as the
        default unless you are sure that no pixels are missing.
    pad : float or int, optional
        A pad value to use for pixels missing from the input skymap. Only used
        if ``u`` does not fully cover ``u_out``. Use ``healpy.UNSEEN`` for this
        value if you want to mark pixels as not-observed for HEALPy plots etc.
    mask_missing : bool
        If ``mask_missing=True``, return a ``np.ma.core.MaskedArray``. Missing
        values are tolerated and are marked as ``True`` in the
        ``mask_missing``. They will be set to ``pad or 0`` in the ``data``
        field.
    intersection : Tuple[np.ndarray, np.ndarray, np.ndarray], optional
        If you've already computed ``uniq_intersection(u, u_out)``, you can pass
        it as this argument to avoid recomputing it. No checks will be made for
        correctness if provided.
    method : "average", "sum", or "copy"
        "average" will compute the average of constituent pixels when converting
        to a lower resolution. Use "average" when the quantity represented by `x`
        is intensive. If `x` has an integer dtype, it will automatically be
        converted to `float`.
        "sum" will compute the sum of constituent pixels when converting to
        a lower resolution pixel. Use this when the quantity represented in x is
        extensive.
        "copy" will directly copy, with unreliable results when converting to a
        lower resolution. This method provides a significant speedup when you
        know no pixels will be converted to a lower resolution.

    Returns
    -------
    x_out : array-like
        Pixel values of the rasterized skymap corresponding to the indices
        given in ``u_out``. ``x_out`` values are pixel-area-weighted averages of the
        input pixel values, even if some pixels in ``u_out`` are not fully covered
        by pixels from ``u``. Any parts of the sky defined in ``u`` that are
        not covered by ``u_out`` are omitted, so this function can also be used to
        mask a skymap in a single step. If ``mask_missing=True``, is a
        ``np.ma.core.MaskedArray``.

    Raises
    ------
    ValueError
        If ``pad`` is not provided but ``u`` does not cover all pixels in
        ``u_out``.

    See Also
    --------
    render
    hpmoc.partial.PartialUniqSkymap.reraster
    np.ma.core.MaskedArray

    Examples
    --------
    Create a small partial skymap with example pixel values:

    >>> u = np.array([1024, 4100, 1027, 1026, 44096])
    >>> x = np.array([1.,   2.,   3.,   4.,   5.])

    We will rerasterize this skymap to these sky areas:

    >>> u_out = np.array([4096, 4097, 1025, 1026, 11024])
    >>> reraster(u, x, u_out)
    array([1., 1., 2., 4., 5.])

    The third pixel in ``u`` is not present in ``u_out``, so we will need to
    provide a default pad value for it when rasterizing in the other direction.
    Note that the first pixel of the result is the average of the first and
    second pixels in the input map, since both of these have equal area and
    overlap with the first pixel:

    >>> reraster(u_out, x, u, pad=0.)
    array([1.5, 3. , 0. , 4. , 5. ])

    We can also simply mask that value by passing ``mask_missing=True``, in
    which case the result will be a ``np.ma.core.MaskedArray`` which is ``True`` for
    values which were missing (the missing/masked values themselves will be set
    to zero or ``pad`` if provided):

    >>> m = reraster(u_out, x, u, mask_missing=True)
    >>> print(m.data)
    [1.5 3.  0.  4.  5. ]
    >>> print(m.mask)
    [False False  True False False]

    Note that the values are averages of input pixels; in cases where only one
    input pixel is sampled from, the value remains unchanged. This makes
    ``reraster`` good for working with densities and other intensive spatial
    values; extensive values should have their pixel areas divided out before
    being rasterized.

    If you've already got the ``uniq_intersection`` of ``u`` and ``u_out`` from a
    previous calculation, you can avoid recomputing it during rasterization by
    passing it as the ``intersection`` argument, though beware it will not be checked
    for correctness:

    >>> reraster(u, x, u_out, intersection=uniq_intersection(u, u_out))
    array([1., 1., 2., 4., 5.])
    """
    from astropy.units import Quantity as Qty

    u̇, u̇ᵒ, δo = intersection or uniq_intersection(u, u_out)  # indices into u, u_out

    m = None
    u̇ₘᵒ = None
    if check_missing:
        u̇ₘᵒ = np.setdiff1d(np.arange(len(u_out)), u̇ᵒ)  # u_out pixels missing from u

        if mask_missing:
            m = np.zeros(u_out.shape, dtype=bool)
            if u̇ₘᵒ.size != 0:
                m[u̇ₘᵒ] = True
        elif u̇ₘᵒ.size != 0 and pad is None:
            raise ValueError(f"u ({u}) missing pixels in u_out ({u_out}): {u̇ₘᵒ}")
    elif pad is not None or mask_missing:
        raise ValueError("check_missing=False is not compatible with pad or mask_missing")


    dtype_out = np.float64
    if method == "average" and not np.issubdtype(dtype_out, np.floating):
        dtype_out = np.float64

    x_out = x_out_raw = np.zeros(u_out.shape, dtype=dtype_out)  # pixel values
    unit = None
    if isinstance(x, Qty):                          # include units for
        unit = x.unit
        x = x.value                                 #   astropy.Quantity
        x_out = Qty(x_out, copy=False, unit=unit, dtype=dtype_out)

    if u̇ₘᵒ is not None and pad is not None:
        x_out_raw[u̇ₘᵒ] = pad                        # pad missing if u̇ₘᵒ

    if method == 'average':
        δ⃗ = 4.**-δo                                    # NUNIQ slice offset tmp
        n_out = np.zeros(u_out.shape, dtype=np.float64) # normalization for pix avg
        np.add.at(n_out, u̇ᵒ, δ⃗)
        if u̇ₘᵒ is not None:
            n_out[u̇ₘᵒ] = 1.                             # pad missing if u̇ₘᵒ

        δ⃗ *= x[u̇]                                      # subpixel contributions
        np.add.at(x_out_raw, u̇ᵒ, δ⃗)
        x_out_raw /= n_out                              # normalize pixel values
    elif method == 'sum':
        np.add.at(x_out_raw, u̇ᵒ, x[u̇])
    elif method == 'copy':
        x_out_raw[u̇ᵒ] = x[u̇]
    else:
        raise ValueError(f"Unknown method '{method}'.")

    if m is not None:
        return np.ma.MaskedArray(x_out, m)
    return x_out


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

    if orders < 0:
        raise ValueError(f"orders must be > 0, instead got: {orders}")
    u = np.array(u, copy=True)
    o = uniq2order(u)
    omax = o.max()
    oout = max(omax - orders, 0)
    i = o > oout
    u[i] >>= 2 * (o[i] - oout)
    return np.unique(u)

class _Test(Protocol):
    def __call__(self, x: Any, y: Any, /) -> Any: ...

class _Combine(Protocol):
    def __call__(self, x: 'NDArray[_DType]', i: 'NDArray[np.bool_]') -> 'NDArray[_DType]': ...

# TODO: Consider changing return signature to return
# Tuple[IntArray, Tuple[NDArray[Any], ...]]
# instead

@overload
def uniq_minimize(
    u: IntArray,
    x: 'NDArray[_DType]', /, *,
    test: _Test = ...,
    combine: _Combine = ...
) -> Tuple[IntArray, 'NDArray[_DType]']: ...

@overload
def uniq_minimize(
    u: IntArray,
    *x: 'NDArray[Any]',
    test: _Test = ...,
    combine: _Combine = ...
) -> Tuple['NDArray[Any]', ...]: ...
    
def uniq_minimize(
    u: IntArray,
    *x: 'NDArray[Any]',
    test: _Test = eq,
    combine: _Combine = lambda x, i: x[i]
) -> Tuple['NDArray[Any]', ...]:
    """
    Take a set of HEALPix NUNIQ indices ``u`` (and, optionally, pixel values
    ``x``) and find the shortest equivalent multi-order pixelation by combining
    pixels. If ``x`` is provided, only combine pixels whose values are equal.
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
    u_min : numpy.ndarray
        The shortest equivalent NUNIQ indexing that can describe ``u``.
    *x_min : numpy.ndarray, optional
        Corresponding pixel values in ``x``, combined according to
        ``combined``.

    Examples
    --------
    Make a set of pixels corresponding to the first four base pixels as well as
    the first pixel at NSIDE = 2 lying in the fifth base pixel:

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

    isort = np.argsort(u)
    u = u[isort]
    x = tuple(xx[isort] for xx in x)
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
            uu += 1 # type: ignore causes issues with numpy number precision handling
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
    return tuple(np.concatenate([y[:bounds[-2]], *ys[::-1]])
            for y, ys in zip([u, *x], [us, *xs]))


def uniq_diadic(f, us, xs, pad=None, coarse=False, reraster_method=None):
    """
    Apply a diadic function ``f(x1, x2) -> y`` that operates on skymap pixel
    values of the same resolution to skymaps with arbitrary
    pixelization/resolution schemes and pixel orders, returning the indices and
    pixel values of the resulting skymap. Useful for binary operations between
    arbitrary partial skymaps.

    Parameters
    ----------
    f : Callable[[np.ndarray, np.ndarray], np.ndarray]
        A binary function operating on two sets of skymap pixel values
        corresponding elementwise to the same sky locations. **Must be a
        skymap-resolution independent operation for the results to make
        sense.**
    us : Tuple[np.ndarray, np.ndarray]
        NUNIQ indices of the two skymaps to be passed to ``f``.
    xs : List[np.ndarray, np.ndarray]
        Pixel values (corresponding to the locations specified by ``us``) of
        the skymaps to be passed to ``f``. Must have same lengths as the arrays
        in ``us``.
    pad : float or int, optional
        A pad value to use for parts of the sky contained in *either*
        ``us[0]`` or ``us[1]`` but *not* in both (since ``f`` will be undefined
        in these regions). If not provided, the returned skymap will
        only contain the intersection of the sky areas defined in ``us``.
    coarse : bool, optional
        If ``True``, for sky areas where ``us[0]`` and ``us[1]`` have
        different resolutions, pick the lower resolution for the output map
        (using pixel-area-weighted averages to decimate the higher-res
        regions). This produces shorter output arrays. If ``False`` (default),
        split coarse pixels into the higher resolution of those specified in
        ``us[0]``and ``us[1]`` for a given sky region; use this if you need to
        maintain resolution for subsequent calculations, but be aware that this
        may impact performance without improving accuracy, e.g. if you're
        planning to integrate the result of this operation. This can
        also be useful if you need to cover the *exact* area defined by the
        input skymaps.
    reraster_method : 'average', 'sum', or 'copy', optional
        the method to use for rerastering pixel values. If unspecified,
        defaults to 'copy' if coarse=False and 'average' if ``coarse=True``.
        'copy' should be used for intensive quantities if ``coarse=False`` for
        faster computation.

    Returns
    -------
    uniq_y : np.ndarray
        Sorted NUNIQ indices of the result of ``f``. In general, will be
        different from *both* ``us`` inputs.
    y : np.ndarray
        Pixel values of the result of ``f`` corresponding to indices ``u_y``.

    Examples
    --------
    Take the product of two heterogeneous multi-order skymaps:

    >>> from pprint import pprint
    >>> from operator import mul
    >>> u1 = np.array([1024, 4100, 1027, 1026, 44096])
    >>> x1 = np.array([1.,   2.,   3.,   4.,   5.])
    >>> u2 = np.array([4096, 4097, 1025, 1026, 11024])
    >>> x2 = np.array([0.,   10.,  1.,   100., 1000.])
    >>> pprint(uniq_diadic(mul, (u1, u2), (x1, x2)), width=60)
    (array([ 1024,  1025,  1026, 11024]),
     array([5.e+00, 2.e+00, 4.e+02, 5.e+03]))

    Provide a default pad value for indices non-overlapping parts of the input
    skymaps:

    >>> pprint(uniq_diadic(mul, (u1, u2), (x1, x2), pad=0.), width=60)
    (array([ 1024,  1025,  1026,  1027, 11024]),
     array([5.e+00, 2.e+00, 4.e+02, 0.e+00, 5.e+03]))

    Increase resolution as necessary (do not combine pixels):

    >>> pprint(uniq_diadic(mul, (u1, u2), (x1, x2), coarse=False), width=60)
    (array([ 1026,  4096,  4097,  4100, 44096]),
     array([4.e+02, 0.e+00, 1.e+01, 2.e+00, 5.e+03]))
    """

    if reraster_method is None:
        reraster_method = 'average' if coarse else 'copy'

    tmp = np.array(uniq_intersection(*us))          # inds into uⁱ & changes in
    *u̇ᵢ, δo = tmp[:, tmp[2].argsort()]              # order δo, sorted on δo
    del tmp                                         # mark for GC

    bound_down, bound_up = δo.searchsorted([0, 1]) 
    slice_down = slice(0, bound_down)
    slice_same = slice(bound_down, bound_up)
    slice_up   = slice(bound_up, len(δo))

    uʸⁱ = [us[0][u̇ᵢ[0][slice_same]]]                       # calc, store same-res
    y⃗ⁱ = [f(xs[0][u̇ᵢ[0][slice_same]], xs[1][u̇ᵢ[1][slice_same]])] # in results list
    assert np.all(uʸⁱ[0] == us[1][u̇ᵢ[1][slice_same]]), f'indices do not correspond'

    for j, sl in ((0, slice_down), (1, slice_up)):  # j=0: downres; j=1: upres
        if sl.start == sl.stop:
            continue # empty slice

        i = (j+coarse) % 2                          # target pixelization ind
        u̇ᵁ, u̇ᵁ̇ = np.unique(u̇ᵢ[i][sl], return_inverse=True)        # target
        u̇ˈᵁ, u̇ˈᵁ̇ = np.unique(u̇ᵢ[i-1][sl], return_inverse=True)    # reraster
        u_out = us[i][u̇ᵁ]                       # put NUNIQ inds in result
        uʸⁱ.append(u_out)

        t = xs[i][u̇ᵁ]
        r = reraster(us[i-1][u̇ˈᵁ], xs[i-1][u̇ˈᵁ], u_out,
                        intersection=(u̇ˈᵁ̇, u̇ᵁ̇, (2*j-1)*δo[sl]),
                        check_missing=False,
                        method=reraster_method)

        # calculate result, order depending on if down or up res
        y⃗ⁱ.append(f(t, r) if j else f(r, t))
        

    if pad is not None:                             # include non-overlapping
        for j in (0, 1):                            # regions if pad provided
            uʸⁱ.append(us[j][np.setdiff1d(np.arange(len(us[j])), u̇ᵢ[j])])
            y⃗ⁱ.append(np.full(uʸⁱ[-1].shape, pad))

    uʸ = np.concatenate(uʸⁱ)                        # concatenate result lists
    y⃗ = np.concatenate(y⃗ⁱ)
    uʸ, i⃗ᵢᵒ = np.unique(uʸ, return_index=True)      # sort by NUNIQ index
    assert len(i⃗ᵢᵒ) == len(y⃗)                       # inds and values same len
    return uʸ, y⃗[i⃗ᵢᵒ]


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
        if self.filename is None:
            raise RuntimeError("self.filename is None. Was __exit__() called before __enter__()?")
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


StrOrPath = Union[str, os.PathLike]

def is_gz(infile: Union[IO, StrOrPath]):
    """
    Check whether ``infile`` is a GZip file, in which case it should be
    unzipped.
    """
    if isinstance(infile, gzip.GzipFile):
        return True
    # from https://stackoverflow.com/a/47080739/3601493
    if isinstance(infile, str) or isinstance(infile, os.PathLike):
        with open(infile, 'rb') as test_f:
            magic_number = binascii.hexlify(test_f.read(2))
    elif hasattr(infile, "read") and hasattr(infile, "seek"):
        magic_number = binascii.hexlify(infile.read(2))
        infile.seek(-2, 1)  # seek back before magic number read
    else:
        raise TypeError("infile must be file-like object or str")
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
def read_partial_skymap(infile: Union[IO, str], uniq, memmap=True):
    """
    Read in pixels from a FITS skymap (or a gzip-compressed FITS skymap) that
    lie in a specific sky region.  Attempts to minimize memory usage by
    memory-mapping pixels in the input file and only loading those specified in
    ``uniq``.

    Parameters
    ----------
    infile : str or file
        A FITS HEALPix skymap file path or file object opened in binary read
        mode 'rb' (optionally compressed; see note under ``memmap``)
    uniq : array-like
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
        HEALPix skymap file is lower than that of the u, then any pixels
        overlapping with those in ``u`` will be used; this might
        result in a larger portion of the skymap being used than that
        specified in ``u``. The resolution of this skymap will be
        the resolution of the smallest pixel loaded from the input file (in
        the case of ``ring`` or ``nested`` ordering, this is just the
        resolution of the input skymap).

    See Also
    --------
    uniq_minimize
    TmpGunzipFits
    """
    from astropy.table import Table

    T = Table.read(infile, format='fits', memmap=memmap)    # read skymap table
    meta = T.meta.copy()
    nside = T.meta.get('NSIDE', None)
    ordering = T.meta['ORDERING']
    set_partial_skymap_metadata(meta, uniq, read_partial_skymap.__qualname__)

    if ordering == 'NUNIQ':
        s⃗̇ = np.concatenate([uniq_intersection(T['UNIQ'][i:i+PIX_READ], uniq)[0]+i
                            for i in range(0, len(T), PIX_READ)])
        uniq, s⃗̈ˢ = np.unique(T['UNIQ'][s⃗̇], return_index=True)
        nside = uniq2nside(uniq)
    elif nside is None:
        raise ValueError(f"No NSIDE defined in header {meta} for {infile}")
    else:
        uniq = np.sort(uniq2nest(uniq, nside, nest=False))           # rasterize
        s⃗̇ = uniq2nest_and_nside(uniq)[0]                       # nest still sorted
        if ordering == 'RING':                              # sorted RING inds
            s⃗̇, s⃗̈ˢ = np.unique(hp.nest2ring(nside, s⃗̇), return_inverse=True)
        elif ordering == 'NESTED':                           # keep nest inds
            s⃗̈ˢ = slice(None)                              # keep order
        else:
            raise ValueError(f"Unexpected ORDERING in {infile}: {ordering}")
    #return Table(np.array([u, density_from_table(T, s⃗̇, nside)[s⃗̈ˢ]]).T,
    #             names=['UNIQ', 'PROBDENSITY'], meta=meta)
    return Table({'UNIQ': uniq,
                  'PROBDENSITY': density_from_table(T, s⃗̇, nside)[s⃗̈ˢ]},
                 meta=meta)


def nside_slices(*u: IntArray, include_empty: bool = False,
                 return_index: bool = False,
                 return_inverse: bool = False,
                 dtype: Optional[DTypeLike] = None):
    """
    Sort and slice up a list of NUNIQ pixel index arrays, returning the sorted
    arrays as well as slice information for chunking them by NSIDE (pixel
    resolution), accessing the original array data, and the NSIDE orders of
    each chunk.

    This is just a wrapper around ``group_slices`` using HEALPix NSIDE order as
    the grouping function.

    Parameters
    ----------
    *u, array-like
        ``np.array`` instances containing NUNIQ HEALPix indices
    include_empty : bool, optional
        If ``True``, also include NSIDE orders not included in the input
        indices. Affects all return values.
    return_index : bool, optional
        Whether to return ``index``. Only returned if ``True``.
    return_inverse : bool, optional
        Whether to return ``inverse``. Only returned if ``True``.
    dtype : DTypeLike, optional
        If provided, cast the returned array to this data type. Useful for
        pre-allocating output arrays that only depend on spatial information.
        Can be a dtype or a value with .dtype

    Returns
    -------
    u_sorted : List[array]
        Sorted versions of each input array
    slices : List[List[slice]]
        Slices into each ``u_sorted`` chunked by NSIDE order
    orders : array
        An array of HEALPix NSIDE orders included in the input indices
    lengths : List[array]
        The lengths of each slice in ``slice_starts``
    views : List[List[array]]
        Lists of array views into each ``u_sorted`` corresponding to the
        slices given in ``slices``
    index: List[array], optional
        Indices into the original array that give ``u_sorted``
    inverse : List[Array], optional
        Indices into each ``us_sorted`` that give the original arrays

    See Also
    --------
    group_slices

    Examples
    --------
    >>> from pprint import pprint
    >>> u1 = np.array([1024, 4100, 1027, 263168, 263169, 1026, 44096])
    >>> u2 = np.array([4096, 4097, 1025, 16842752, 1026, 11024])
    >>> us, s, o, l, v, ius = nside_slices(u1, u2, return_index=True)
    >>> pprint([uu.astype(int) for uu in us])
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
    >>> o.astype(int)
    array([ 4,  5,  6,  8, 11])
    >>> [ll.astype(int) for ll in l]
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
    >>> [ii.astype(int) for ii in ius]
    [array([0, 5, 2, 1, 6, 3, 4]), array([2, 4, 0, 1, 5, 3])]
    """
    return group_slices(*u, f=uniq2order,
                        inv=lambda x: nest2uniq(0, hp.order2nside(x)),
                        include_empty=include_empty, return_index=return_index,
                        return_inverse=return_inverse, dtype=dtype)


def group_slices(*u: 'NDArray[_DType]',
                 f: Callable[['NDArray[_DType]'], IntArray]=lambda x: x, # type: ignore
                 inv: Callable[[IntArray], 'NDArray[_DType]']=lambda x: x,
                 include_empty: bool = False,
                 return_index: bool = False,
                 return_inverse: bool = False,
                 dtype: Any = None):
    """
    Group elements of ``u`` inputs using some sort of monotonic step function
    ``f: u.dtype -> int`` codomain and a pseudo-inverse ``inv`` mapping to the
    smallest element of the input domain giving that output value (both
    identity by default) and return a variety of views and slices into these
    groups. See ``nside_slices`` for documentation and an implementation that
    groups by HEALPix NSIDE order; this function is the same, but with ``o⃗``
    replaced by the result of ``f`` on elements of ``u``.  You can use this
    function with the default grouping functions to group integers by value,
    e.g. for working with ``intersection`` arrays from ``uniq_intersection``.

    See Also
    --------
    nside_slices
    """

    u = tuple(np.array(u, dtype=dtype, copy=False) for u in u)
    uˢ, u̇_u̇ˢ = [np.unique(u, return_index=return_index, # type: ignore
                          return_inverse=return_inverse) for u in u], [] # type: ignore
    if return_index or return_inverse:
        uˢ, *u̇_u̇ˢ = zip(*uˢ)
    s, e = [[uⁱ[i] for uⁱ in uˢ if len(uⁱ)] for i in (0, -1)]
    if not s and not e:
        return tuple([uˢ, [[]]*len(u), np.array([]), [], [[]]*len(u)]+u̇_u̇ˢ)
    o⃗ = np.arange(f(min(s)), f(max(e))+2)
    # o⃗ = np.arange(f(min(uⁱ[0] for uⁱ in uˢ)), f(max(uⁱ[-1] for uⁱ in uˢ))+2)
    i⃗ₛ = [np.searchsorted(uⁱ, inv(o⃗)) for uⁱ in uˢ]
    𝓁⃗ = [i⃗ₛⁱ[1:]-i⃗ₛⁱ[:-1] for i⃗ₛⁱ in i⃗ₛ]
    i⃗ᴱ̸ = np.arange(len(𝓁⃗[0])) if include_empty else np.nonzero(sum(𝓁⃗))[0]
    s⃗ = [[slice(i⃗ₛⁱ[j], i⃗ₛⁱ[j+1]) for j in i⃗ᴱ̸] for i⃗ₛⁱ in i⃗ₛ]
    return tuple([uˢ, s⃗, o⃗[i⃗ᴱ̸], [𝓁[i⃗ᴱ̸] for 𝓁 in 𝓁⃗],
                  [[uˢ[i][s⃗ᵢⱼ] for s⃗ᵢⱼ in s⃗ᵢ] for i, s⃗ᵢ in enumerate(s⃗)]]+u̇_u̇ˢ)
