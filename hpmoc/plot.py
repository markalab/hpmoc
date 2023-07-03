# Copyright (C) 2021-2023 Columbia Experimental Gravity Group (GECo)

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
Plotting commands implemented using Astropy (rather than HEALPy) for greater
flexibility, robustness, and cross-platform compatibility. Meant to replace
the plotters in ``plotters``. These provide similar functionality to the
plotters provided by the ``ligo.skymap`` package as well as ``healpy``; those
packages are not included as dependencies, but the high-performance rendering
tools included in ``hpmoc`` are totally compatible with those plotting tools
through the ``PartialUniqSkymap.render`` interface (in fact, the previous
version of these plotting scripts used ``healpy`` for its included projection
axes and coordinate transforms).

Available projections are drawn from the FITS standard. You can specify your
own projections using the appropriate FITS headers plugged into an
``astropy.wcs.WCS`` world coordinate system instance as the ``projection``
argument to a subplot or ``astropy.visualization.wcsaxes.WCSAxes`` instance.
The projection code follows the "4-3" form defined in the
`FITS Definition`_ document, i.e. by specifying the coordinate type
(e.g. ``RA--`` for right ascension) right-padded by enough dashes to fill
4 characters, followed by an additional ``-`` and the 3-character projection
code (one of those listed below) to specify the projection. This information is
then stored in the ``CTYPEi`` headers. Pixel-scaling through the ``CDELTi``
headers has a projection-dependent normalization and further depends on pixel
resolution of the final image. Details on WCS header fields are available in
the `FITS Definition`_, and the normalization factors can be extracted from
each projection's definition in
`Representations of celestial coordinates in FITS`_.

.. _`FITS Definition`: https://fits.gsfc.nasa.gov/fits_standard.html
.. _`Representations of celestial coordinates in FITS`: https://ui.adsabs.harvard.edu/abs/2002A&A...395.1077C

The complete list of available projections can be found in the
`FITS Definition`_, with concrete transformation definitions given in
the `Representations of Celestial Coordinates in FITS`_. The table of
available projections is reproduced below, with sections linking to the
corresponding section in `Representations of Celestial Coordinates in FITS`_.
Note that not all valid WCS projections can be displayed by astropy at time of
writing; in particular, the HEALPIX ``HPX`` projection does not work out of the
box, which is one of the (many) motivations for this plotting library.

.. list-table:: Available Projections
   :widths: 25 25 25 50 75
   :header-rows: 1

   * - Code
     - φ_0
     - θ_0
     - Properties1
     - Projection name
   * - AZP
     - 0◦
     - 90◦
     - Sect. 5.1.1
     - Zenithal perspective
   * - SZP
     - 0◦
     - 90◦
     - Sect. 5.1.2
     - Slant zenithal perspective
   * - TAN
     - 0◦
     - 90◦
     - Sect. 5.1.3
     - Gnomonic
   * - STG
     - 0◦
     - 90◦
     - Sect. 5.1.4
     - Stereographic
   * - SIN
     - 0◦
     - 90◦
     - Sect. 5.1.5
     - Slant orthographic
   * - ARC
     - 0◦
     - 90◦
     - Sect. 5.1.6
     - Zenithal equidistant
   * - ZPN
     - 0◦
     - 90◦
     - Sect. 5.1.7
     - Zenithal polynomial
   * - ZEA
     - 0◦
     - 90◦
     - Sect. 5.1.8
     - Zenithal equal-area
   * - AIR
     - 0◦
     - 90◦
     - Sect. 5.1.9
     - Airy
   * - CYP
     - 0◦
     - 0◦
     - Sect. 5.2.1
     - Cylindrical perspective
   * - CEA
     - 0◦
     - 0◦
     - Sect. 5.2.2
     - Cylindrical equal area
   * - CAR
     - 0◦
     - 0◦
     - Sect. 5.2.3
     - Plate carrée
   * - MER
     - 0◦
     - 0◦
     - Sect. 5.2.4
     - Mercator
   * - SFL
     - 0◦
     - 0◦
     - Sect. 5.3.1
     - Samson-Flamsteed
   * - PAR
     - 0◦
     - 0◦
     - Sect. 5.3.2
     - Parabolic
   * - MOL
     - 0◦
     - 0◦
     - Sect. 5.3.3
     - Mollweide
   * - AIT
     - 0◦
     - 0◦
     - Sect. 5.3.4
     - Hammer-Aitoff
   * - COP
     - 0◦
     - θa
     - Sect. 5.4.1
     - Conic perspective
   * - COE
     - 0◦
     - θa
     - Sect. 5.4.2
     - Conic equal-area
   * - COD
     - 0◦
     - θa
     - Sect. 5.4.3
     - Conic equidistant
   * - COO
     - 0◦
     - θa
     - Sect. 5.4.4
     - Conic orthomorphic
   * - BON
     - 0◦
     - 0◦
     - Sect. 5.5.1
     - Bonne’s equal area
   * - PCO
     - 0◦
     - 0◦
     - Sect. 5.5.2
     - Polyconic
   * - TSC
     - 0◦
     - 0◦
     - Sect. 5.6.1
     - Tangential spherical cube
   * - CSC
     - 0◦
     - 0◦
     - Sect. 5.6.2
     - COBE quadrilateralized spherical cube
   * - QSC
     - 0◦
     - 0◦
     - Sect. 5.6.3
     - Quadrilateralized spherical cube
   * - HPX
     - 0◦
     - 0◦
     - Sect. 6 2
     - HEALPix grid

See Also
--------
hpmoc.partial.PartialUniqSkymap
astropy.wcs.WCS
astropy.visualization.wcsaxes
matplotlib.axes.Axes
matplotlib.figure.Figure
ligo.skymap
healpy
"""

from __future__ import annotations

import math
from copy import deepcopy
from functools import partial
from warnings import warn
from typing import (
    Optional, Union, Tuple, Iterable, Callable,
    Sequence, Mapping, Sized, Any, TYPE_CHECKING
)
from textwrap import indent, wrap
from .healpy import healpy as hp
from .utils import (
    outline_effect,
    N_X_OFFSET,
    N_Y_OFFSET,
    wcs2ang,
    wcs2mask_and_uniq,
    monochrome_opacity_colormap,
)
from .points import PointsTuple

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import astropy.io.fits
    from astropy.wcs.wcs import WCS
    import astropy.visualization.wcsaxes.frame

    import matplotlib.colors
    import matplotlib.gridspec
    import matplotlib.figure

    from .partial import PartialUniqSkymap

DEFAULT_CBAR_KWARGS = {
    'orientation': 'horizontal',
    'aspect': 40,
}
EDGE_EXCLUSION = 80
EXCLUSIONS_INITIALIZED = [False]
AIT_MOL_CDELT_BASE = math.sqrt(8) / math.pi
CEA_CDELT2_BASE = 2 / math.pi
TAN_CDELT_BASE = 0.025
SIN_CDELT_BASE = 2 / math.pi
ARC_CDELT_BASE = 2
ZEA_CDELT_BASE = 4 / math.pi
DEFAULT_WIDTH = 360
DEFAULT_HEIGHT = 180
DEFAULT_ROT = (180, 0, 0)
DEFAULT_FACING = True
DEFAULT_DPI = 200
DEFAULT_GRID_ROW_HEIGHT = 4
DEFAULT_HSPACE = 0.2
DEFAULT_WSPACE = 0.2
DEFAULT_NCOLS = 2
BASE_FONT_SIZE = 10
DEFAULT_C_KWARGS = {}
DEFAULT_CLABEL_KWARGS = {
    'inline': False,
    'fontsize': BASE_FONT_SIZE,
}
_DELTA_HIGHRES = tuple(v*.1**i for i in range(20) for v in (5, 2, 1))
DELTA_PARALLELS = (15.,)+_DELTA_HIGHRES     # space btwn graticule parallels
DELTA_MERIDIANS = (30., 10.)+_DELTA_HIGHRES #   and meridians [deg]
MIN_GRAT = 3
_WCS_HEADERS = dict()
_WCS_FRAMES = dict()
_SHARED = {
    "NAXIS": 2,
    "NAXIS2": 180,
    "CRVAL1": 180.0,
    "CTYPE1": 'RA---',
    "CUNIT1": 'deg',
    "CRVAL2": 0.0,
    "CTYPE2": 'DEC--',
    "CUNIT2": 'deg',
    "COORDSYS": 'icrs'
}
_ALL_SKY = _SHARED.copy()
_ALL_SKY["NAXIS1"] = 360.0
_ZENITHAL = _SHARED.copy()
_ZENITHAL["NAXIS1"] = 180.0
_ZENITHAL["LONPOLE"] = 180.0
_docs = {
    "allsky": "\n",
    "zenithal": "\n",
}
_PROJ_SETTINGS = {
    "allsky": (
        _ALL_SKY,
        {
            'MOL': (('Mollweide', 'mollview', 'Homolographic', 'Homalographic',
                     'Babinet', 'Elliptical'),
                    {'CDELT1': AIT_MOL_CDELT_BASE,
                     'CDELT2': AIT_MOL_CDELT_BASE}, 'e'),
            'AIT': (('Hammer-Aitoff', 'Aitoff', 'Hammer', 'Aitov',
                     'Hammer equal-area'),
                    {'CDELT1': AIT_MOL_CDELT_BASE,
                     'CDELT2': AIT_MOL_CDELT_BASE}, 'e'),
            'CAR': (('Carée', 'Plate carée', 'Caree', 'Plate caree',
                     'Cartesian', 'Tyre', 'cartview',
                     'Equidistant cylindrical'),
                    {'CDELT1': 1, 'CDELT2': 1}, 'r'),
            'CEA': (('Cylindrical equal-area', 'Lambert equal area'),
                    {'CDELT1': 1, 'CDELT2': CEA_CDELT2_BASE}, 'r'),
            'SFL': (('Sanson-Flamsteed',),
                    {'CDELT1': 1, 'CDELT2': 1}, 'p'),
            'PAR': (('Parabolic', 'Craster'),
                    {'CDELT1': 1, 'CDELT2': 1}, 'p'),
        },
    ),
    "zenithal": (
        _ZENITHAL,
        {
            # NB: TAN is just AZP (zenithal perspective) with mu set to zero.
            'TAN': (('gnomonic', 'gnomview', 'Central', 'zoom'),
                    {'CDELT1': TAN_CDELT_BASE, 'CDELT2': TAN_CDELT_BASE}, 'r'),
            'SIN': (('Slant Orthographic', 'Orthographic', 'Globe',
                     'orthview'),
                    {'CDELT1': SIN_CDELT_BASE, 'CDELT2': SIN_CDELT_BASE}, 'e'),
            'ARC': (('Zenithal Equidistant', 'azimuthal equidistant',
                     'azeqview', 'Postel', 'Equidistant', 'Globular'),
                    {'CDELT1': ARC_CDELT_BASE, 'CDELT2': ARC_CDELT_BASE}, 'e'),
            'ZEA': (('Zenithal Equal-area', 'Azimuthal equivalent',
                     'polar azimuthal', 'Lambert azimuthal equivalent',
                     'Lambert azimuthal equal-area',
                     'Lambert polar azimuthal', 'Lambert'),
                    {'CDELT1': ZEA_CDELT_BASE, 'CDELT2': ZEA_CDELT_BASE}, 'e'),
        },
    ),
}
# Add all-sky projections; aliases drawn from common usage, healpy.visufunc,
# WCS standard, and aliases as given in astro-ph/0207413
for docname, (defaults, config) in _PROJ_SETTINGS.items():
    for proj, (aliases, projdefaults, frame) in config.items():
        _docs[docname] += '        - ' + indent('\n'.join(
            wrap(f"{proj}: *{', '.join(aliases)}*", 69)), ' '*10)[10:] + '\n'
        _WCS_FRAMES[proj] = frame
        _WCS_HEADERS[proj] = defaults.copy()
        _WCS_HEADERS[proj]['CTYPE1'] += proj
        _WCS_HEADERS[proj]['CTYPE2'] += proj
        _WCS_HEADERS[proj].update(projdefaults)
        for alias in aliases:
            _WCS_HEADERS[alias.upper()] = _WCS_HEADERS[proj]
            _WCS_FRAMES[alias.upper()] = frame
    _docs[docname] = _docs[docname].rstrip('\n')


def _one_pt(scatter, rot):
    return len(scatter) == 1 and len(scatter[0].points) == 1 and rot is None


def _set_delts(projection, hdelta, vdelta, scatter, sigmas):
    return (projection in ['ARC', *_PROJ_SETTINGS['zenithal'][1]['ARC'][0]]
            and hdelta is None and vdelta is None and sigmas
            and len(scatter[0].points[0]) > 2)


def get_frame_class(
        projection: Union[
            str,
            'WCS',
            'astropy.io.fits.Header',
        ] = 'MOL',
        frame_class: Optional[
            Union[
                str,
                'astropy.visualization.wcsaxes.frame.BaseFrame',
            ]
        ] = None,
        vdelta: Optional[float] = None,
        hdelta: Optional[float] = None,
        rot: Optional[
            Union[
                Tuple[float, float, float],
                Tuple[float, float],
            ]
        ] = None,
        scatter: Sequence[PointsTuple] = tuple(),
        sigmas: Sequence[float] = (1,),
) -> 'astropy.visualization.wcsaxes.frame.BaseFrame':
    """
    Get the frame class associated with a given projection. If already given a
    ``frame_class``, returns it immediately, making this function idempotent.
    Will otherwise try to determine the best frame choice from the given
    arguments.

    Parameters
    ----------
    projection : str, WCS, or Header, optional
        The projection to use. See ``get_wcs`` for details. Only used if a
        ``str``.
    frame_class : str or BaseFrame
        The frame class; optionally pass it to ensure that an argument is an
        instance of ``BaseFrame``.
    vdelta : float, optional
        Manual vdelta override. Give up if set and return ``RectangularFrame``.
    hdelta : float, optional
        Manual hdelta override. Give up if set and return ``RectangularFrame``.
    rot, scatter, sigmas
        See the note in ``get_wcs`` on these arguments. If both rotation and
        window are set for ARC (zenithal equidistant), switch to
        ``RectangularFrame``.

    Returns
    -------
    frame_class: astropy.visualization.wcsaxes.frame.BaseFrame
        The frame class best-suited to this type of projection.

    Raises
    ------
    IndexError
        If the specified projection could not be found.
    """
    from astropy.visualization.wcsaxes import frame

    if isinstance(frame_class, type) and \
            issubclass(frame_class, frame.BaseFrame):
        return frame_class
    if isinstance(frame_class, str):
        return {
            'ELLIPTICAL': frame.EllipticalFrame,
            'RECTANGULAR': frame.RectangularFrame,
        }[frame_class.upper()]
    if {vdelta, hdelta} != {None} or not isinstance(projection, str):
        return frame.RectangularFrame
    if _one_pt(scatter, rot) and _set_delts(projection, hdelta, vdelta,
                                            scatter, sigmas):
        return frame.RectangularFrame
    f = _WCS_FRAMES[projection.upper()]
    if f == 'e':
        return frame.EllipticalFrame
    if f == 'r':
        return frame.RectangularFrame
    warn(f"Frame class for {projection} not yet available. Using default for "
         "now; specify your own for a better-looking plot.", UserWarning)
    return frame.RectangularFrame


GET_FRAME_CLASS_KWARG_KEYS = ('frame_class', 'vdelta', 'hdelta', 'rot',
                              'scatter', 'sigmas')
GET_WCS_KWARG_KEYS = ('width', 'height', 'hdelta', 'vdelta', 'rot',
                      'facing_sky', 'scatter', 'sigmas')


# TODO allow ICRS override
def get_wcs(
        projection: str,
        width: Optional[int] = None,
        height: int = DEFAULT_HEIGHT,
        hdelta: Optional[float] = None,
        vdelta: Optional[float] = None,
        rot: Optional[
            Union[
                Tuple[float, float, float],
                Tuple[float, float],
            ]
        ] = None,
        facing_sky: bool = DEFAULT_FACING,
        scatter: Sequence[PointsTuple] = tuple(),
        sigmas: Sequence[float] = (),
) -> 'WCS':
    """
    Get a ``WCS`` instance by name to match the given parameters.

    Parameters
    ----------
    projection: str
        The following projections are available by default. See ``hpmoc.plot``
        documentation for instructions on constructing your own WCS headers for
        other plot styles, which can be passed in instead using a ``WCS``
        instance. You can also use this approach to plot the skymap over an
        existing ``WCS`` taken from another fits file, making it easy to plot
        skymaps over other astrophysical data. This function will return
        ``WCS`` instances for the following projection types:

        **Cylindrical/Pseudo-cylindrical** (For all-sky plots)
        {allsky}

        **Zenithal**
        {zenithal}
    width: int, optional
        Width of the image in pixels. If not provided, default to the height
        for zenithal projections and twice the height for azimuthal
        projections.
    height: int
        Height of the image in pixels.
    hdelta: float, optional
        The CDELT1 value, if you wish to override the default. Note that the
        actual angular width of a pixel at the reference point depends on this
        value *as well as* the projection used. *Ignored if* ``ax`` *is given.*
    vdelta: float, optional
        The CDELT2 value, if you wish to override the default. Note that the
        actual angular height of a pixel at the reference point depends on this
        value *as well as* the projection used. *Ignored if* ``ax`` *is given.*
    rot: float, float, float) or (float, float), optional
        Euler angles for rotations about the Z, X, Z axes. These are
        immediately translated to ``CRVAL1, CRVAL2, LONPOLE`` in the returned
        ``WCS``; that is, the first two angles specify the angle of the center
        of the image, while the last specifies an additional rotation of the
        reference pole about the Z-axis. See ``hpmoc.plot`` documentation for
        further references. Defaults to centering on RA = 180, dec = 0 (the
        convention used in most plots produced by LIGO). If only two angles are
        provided, ``LONPOLE`` will be determined automatically such that the
        center of the skymap is given by the `(RA, dec)` given and the
        orientation of the plot is otherwise unchanged.
    facing_sky: bool
        Whether the projection is outward- or inward-facing. Equivalent to
        reversing the direction of the longitude.
    scatter : Sequence[PointsTuple]
        A list of collections of point sources that will be plotted. If only
        one collection containing one point source is provided and ``rot=None``
        (the default), then ``rot`` will be set to the location of that single
        point source.
    sigmas : Sequence[float]
        See ``plot`` for meaning. If ``sigmas`` is non-empty; ``rot`` is set by
        a single point source in ``scatter`` as described above; ``hdelta``
        and ``vdelta`` are both ``None``; and the ``projection`` is an alias
        for ARC (zenithal equidistant), then ``hdelta`` and ``vdelta`` will be
        set so that the returned frame is 1.5x the size of the largest error
        region plotted in the smallest axis. This provides an easy way to
        visually emphasize singular points of interest.

    Returns
    -------
    wcs: astropy.wcs.WCS
        A ``WCS`` instance that can be used for rendering and plotting.

    Raises
    ------
    IndexError
        If the specified projection could not be found or if an argument is
        specified incorrectly.
    """
    from astropy.io.fits import Header
    from astropy.wcs import WCS

    dec_dir = -1 if facing_sky else 1
    header = Header(_WCS_HEADERS[projection.upper()].copy())
    if width is not None:
        header['CDELT1'] *= header['NAXIS1'] / width # type: ignore
        header['NAXIS1'] = width # type: ignore
    header['CRPIX1'] = header['NAXIS1'] / 2 + 0.5 # type: ignore
    header['CDELT2'] *= header['NAXIS2'] / height # type: ignore
    header['NAXIS2'] = height # type: ignore
    header['CRPIX2'] = header['NAXIS2'] / 2 + 0.5 # type: ignore
    # rot set to point location if only one point present
    if _one_pt(scatter, rot):
        pt = scatter[0].points[0]
        rot = tuple(pt[:2])
        if _set_delts(projection, hdelta, vdelta, scatter, sigmas):
            sig = 1.5 * max(sigmas) * pt[2] # type: ignore
            header['CDELT1'] *= sig / 180
            header['CDELT2'] *= sig / 180
    if rot:
        header['CRVAL1'] = rot[0]
        header['CRVAL2'] = rot[1]
        if len(rot) == 3:
            header['LONPOLE'] = rot[2]
        elif len(rot) == 2:
            if rot[1] < 0:
                for name, (aliases, *_) in _PROJ_SETTINGS['allsky'][1].items():
                    if projection.upper() in [n.upper()
                                              for n in [name, *aliases]]:
                        header['LONPOLE'] = 180
                        break
        else:
            raise ValueError(f"rot must have len 2 or 3; got {rot}")
    if hdelta is not None:
        header['CDELT1'] = hdelta
    if vdelta is not None:
        header['CDELT2'] = vdelta
    header['CDELT1'] *= dec_dir # type: ignore
    return WCS(header)


get_wcs.__doc__ = get_wcs.__doc__.format(allsky=_docs['allsky'], # type: ignore
                                         zenithal=_docs['zenithal'])


def get_colormap(cmap, bad=None):
    from matplotlib.cm import get_cmap
    from matplotlib.colors import Colormap

    if not isinstance(cmap, Colormap):
        cmap = deepcopy(get_cmap(cmap))
    if bad is not None:
        cmap.set_bad(color=bad)
    return cmap


def get_projection(projection, *args, **kwargs):
    """
    If ``projection`` is already an ``astropy.wcs.WCS`` instance, return it; if
    it is an ``astropy.io.fits.Header``, return ``WCS(projection)``. If it is a
    string, try to create a new ``WCS`` using ``get_wcs`` with ``*args`` and
    ``**kwargs`` passed to that function; failing that, try to interpret
    ``projection`` as a raw FITS header and instantiate a ``WCS`` therefrom.
    """
    from astropy.wcs import WCS
    from astropy.io.fits import Header

    if isinstance(projection, WCS):
        return projection
    if isinstance(projection, Header):
        return WCS(projection)
    try:
        return get_wcs(projection, *args, **kwargs)
    except IndexError:
        return WCS(Header.fromstring(projection))


# TODO allow ICRS override
def plot(
        skymap: Union[
            'PartialUniqSkymap',
            NDArray[Any],
            Tuple[
                NDArray[Any],
                Optional[
                    Union[
                        NDArray['np.integer[Any]'],
                        'WCS',
                        str,
                    ]
                ],
            ],
        ],
        *scatter: PointsTuple,
        scatter_marker_size: float = BASE_FONT_SIZE*4,
        scatter_label_size: float = BASE_FONT_SIZE,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: Optional[
            Union[str, 'matplotlib.colors.Colormap']
        ] = 'gist_heat_r',
        missing_color: Optional[Union[str, Tuple[int, int, int]]] = None,
        nan_color: Optional[Union[str, Tuple[int, int, int]]] = None,
        alpha: float = 1.,
        sigmas: Sequence[float] = [],
        outline_sigmas: Sequence[float] = [],
        scatter_labels: Union[bool, dict] = True,
        ax: Optional[Union[
            'astropy.visualization.wcsaxes.WCSAxes',
            'astropy.visualization.wcsaxes.WCSAxesSubplot'
        ]] = None,
        projection: Union[
            str,
            'WCS',
            'astropy.io.fits.Header',
        ] = 'Mollweide',
        frame_class: Optional[
            Union[
                'astropy.visualization.wcsaxes.frame.BaseFrame',
                str
            ]
        ] = None,
        width: Optional[int] = None,
        height: int = DEFAULT_HEIGHT,
        hdelta: Optional[float] = None,
        vdelta: Optional[float] = None,
        rot: Optional[
            Union[
                Tuple[float, float, float],
                Tuple[float, float],
            ]
        ] = None,
        facing_sky: bool = DEFAULT_FACING,
        fig: Optional[Union['matplotlib.figure.Figure', dict]] = None,
        subplot: Optional[
            Union[
                Tuple[int, int, int],
                'matplotlib.gridspec.SubplotSpec',
            ]
        ] = None,
        cr: Iterable[float] = tuple(),
        cr_format: Optional[Callable[[float, float], str]] = None,
        cr_filled: bool = False,
        cr_kwargs: Optional[dict] = None,
        cr_label_kwargs: Optional[dict] = None,
        pixels: Union[bool, dict] = False,
        pixels_format: Optional[
            Callable[['PartialUniqSkymap'], Iterable[str]]
        ] = None,
        cbar: Union[bool, dict] = False,
) -> Union[
    'astropy.visualization.wcsaxes.WCSAxes',
    'astropy.visualization.wcsaxes.WCSAxesSubplot'
]:
    """
    Parameters
    ----------
    skymap : 'PartialUniqSkymap', array, or (array, array)
        The skymap to plot. Can be a ``PartialUniqSkymap``, a single-resolution
        HEALPix skymap *in NEST ordering only*, or a tuple of (pixel values,
        NUNIQ indices) accepted as the first two arguments to
        ``PartialUniqSkymap``.
    scatter: PointsTuple
        Point-sources to plot as a scatter-map, with disks showing their error
        regions. Provide multiple (ideally with different colors) to plot many
        populations at once.
    scatter_marker_size: float, optional
        Point-source marker font size.
    scatter_label_size: float, optional
        Point-source label font size.
    vmin: float, optional
        The smallest value in the color map used to plot ``skymap``. Set
        ``None`` to have it calculated automatically.
    vmax: float, optional
        The largest value in the color map used to plot ``skymap``. Set
        ``None`` to have it calculated automatically.
    cmap: str or matplotlib.colors.Colormap
        The color map to use to plot ``skymap``. Note that the colors for the
        point sources in ``scatter`` are set using the ``rgba`` parameter in
        ``PointsTuple`` and will not be affected by this value. If ``None``,
        the skymap itself will not be plotted; this can be useful if overlaying
        multiple skymaps.
    missing_color : str or (int, int, int), optional
        The color to use for missing parts of the skymap. If not provided, they
        simply will not be shown.
    nan_color : str or (int, int, int), optional
        The color to use for parts of the map that are included but equal to
        ``np.nan``. If not provided, will be the same as ``missing_color``
        (meaning transparent if that argument is not provided), i.e. they will
        be plotted as if the pixels were missing. **Will not use the**
        ``alpha`` **argument (to allow for highlighting NaN values more
        clearly;** to use the same ``alpha`` value, you can create an RGBA
        tuple including the desired color and your shared alpha value, e.g.
        ``nan_color=[*matplotlib.colors.to_rgb('pink'), alpha]``.
    alpha: float, optional
        The opacity of the plotted skymap image.
    sigmas: Iterable[float], optional
        The size of the error region about each point source to plot in units
        of its error parameter sigma.
    outline_sigmas: Iterable[float], optional
        The size of the error region about each point source to plot as an
        unfilled circle in units of its error parameter sigma.
    scatter_labels: bool, optional
        Whether to show labels for the scattered points. If ``True``, display
        either their labels (if defined) or their indices within the
        ``PointsTuple.points`` list. If given as a ``dict``, will be
        interpreted as keyword arguments to pass to ``WCSAxes.text``, which can
        be used to control the appearance of the point source labels.
    ax: WCSAxes or WCSAxesSubplot, optional
        Axes to plot to. If provided, all other arguments pertaining to
        creating a ``WCS`` and ``WCSAxes`` instance are ignored, and these axes
        are used instead.
    projection: str, WCS, or Header, optional
        Either provide the name of the projection (see ``get_wcs`` docstring
        for valid names) to create a new ``WCS`` for this plot, or provide
        a ready-made ``WCS`` instance or FITS ``Header`` from which such an
        instance can be crafted. In the first case, you will need to specify
        other parameters needed to fully define the world coordinate system.
        In the latter cases, you might need to customize ``frame``.
        *Ignored if* ``ax`` *is given.*
    frame_class: BaseFrame or str, optional
        The frame type to use for this plot, e.g. a ``RectangularFrame`` (for
        a plate carée/Cartesian plot) or an ``EllipticalFrame`` for a
        Mollweide plot. Selected automatically when ``projection`` is specified
        by name, otherwise defaults to ``RectangularFrame``. You can also
        specify `frame_class='rectangular'` or `frame_class='elliptical'` to
        choose one of these two frames.
    width: int, optional
        Width of the image in pixels. If not provided, default to the height
        for zenithal projections and twice the height for azimuthal
        projections. *Ignored if* ``ax`` *is given or if* ``projection`` *is a*
        ``WCS`` *instance.*
    height: int, optional
        The height of the plot in pixels. *Ignored if* ``ax`` *is given or if*
        ``projection`` *is a* ``WCS`` *instance.*
    hdelta: float, optional
        The CDELT1 value, if you wish to override the default. Note that the
        actual angular width of a pixel at the reference point depends on this
        value *as well as* the projection used. *Ignored if* ``ax`` *is given
        or if* ``projection`` *is a* ``WCS`` *instance.*
    vdelta: float, optional
        The CDELT2 value, if you wish to override the default. Note that the
        actual angular height of a pixel at the reference point depends on this
        value *as well as* the projection used. *Ignored if* ``ax`` *is given
        or if* ``projection`` *is a* ``WCS`` *instance.*
    rot: (float, float, float) or (float, float), optional
        Euler angles for rotations about the Z, X, Z axes. These are
        immediately translated to ``CRVAL1, CRVAL2, LONPOLE`` in the returned
        ``WCS``; that is, the first two angles specify the angle of the center
        of the image, while the last specifies an additional rotation of the
        reference pole about the Z-axis. See ``hpmoc.plot`` documentation for
        further references. Defaults to centering on RA = 180, dec = 0 (the
        convention used in most plots produced by LIGO). If only two angles are
        provided, ``LONPOLE`` will be determined automatically such that the
        center of the skymap is given by the `(RA, dec)` given and the
        orientation of the plot is otherwise unchanged. *Ignored if* ``ax``
        *is given or if* ``projection`` *is a* ``WCS`` *instance.*
    facing_sky: bool, optional
        Whether the projection is outward- or inward-facing. Equivalent to
        reversing the direction of the longitude. *Ignored if* ``ax`` *is given
        or if* ``projection`` *is a* ``WCS`` *instance.*
    fig: matplotlib.figure.Figure or dict, optional
        The figure to plot to. If not provided, a new figure will be created.
        If a dictonary is provided, it will be passed as keyword arguments to
        create a new figure. *Ignored if* ``ax`` *is given.*
    subplot: (int, int, int) or SubplotSpec, optional
        If provided, initialize the plot as a subplot using the standard
        ``Figure.subplot`` matplotlib interface, returning a ``WCSAxesSubplot``
        instance rather than a ``WCSAxes`` instance.
        *Ignored if* ``ax`` *is given.*
    cr: Iterable[float], optional
        If provided, plot contour lines around the credible regions
        specified in this list. For example, ``cr=[0.9]`` will plot contours
        around the smallest region containing 90% of the skymap's integrated
        value.
    cr_format: Callable[[float, float], str], optional
        A function taking the CR level (e.g. ``0.9`` for 90% CR) and the actual
        value of the skymap on that contour and returning a string to be used
        to label each contour. If not provided, contours will not be labeled.
        *Ignored if* ``cr`` *is empty.*
    cr_filled: bool, optional
        Whether to fill the contours using ``contourf`` rather than
        ``contour``. *Ignored if* ``cr`` *is empty.*
    cr_kwargs: dict, optional
        Additional arguments to pass to either ``contour`` or ``contourf``.
        *Ignored if* ``cr`` *is empty.*
    cr_label_kwargs: dict, optional
        Arguments to pass to ``clabel`` governing the display format of
        contour labels.
    pixels: bool or dict, optional
        Whether to plot pixel borders. *You should probably keep this*
        ``False`` *if you are doing an all-sky plot, since border plotting is
        slow for a large number of pixels, and the boundaries will not be
        visible anyway unless each visible pixel's size is comparable to the
        overall size of the plot window.* If you just want to see information
        about e.g. the size of pixels across the whole sky, consider plotting
        ``skymap.orders(as_skymap=True)`` to see a color map of pixel sizes
        instead. If given as a ``dict``, will be interpreted as keyword
        arguments to pass to ``WCSAxes.plot``, which can be used to control the
        appearance of the pixel borders.
    pixels_format: Callable[[PartialUniqSkymap], Iterable[str]], optional
        A function that takes a skymap and returns a string for each pixel.
        Will be called on the selection of pixels overlapping with the visible
        area, so don't worry about optimizing it, since just plotting the
        borders will be slow enough for a large number of visible pixels.
        The returned string will be plotted over the center of each pixel,
        which again is only useful if the pixels are large with respect to the
        size of the plot window.
    cbar: bool or dict, optional
        If ``True``, add a colorbar for the plotted skymap. If a ``dict`` is
        provided, pass it as the keyword arguments to
        ``matplotlib.pyplot.colorbar``. *Ignored if* ``cmap=None``.

    Returns
    -------
    ax: WCSAxes or WCSAxesSubplot
        The axes that were just plotted to.

    See Also
    --------
    hpmoc.partial.PartialUniqSkymap
    matplotlib.figure.Figure
    matplotlib.axes.Axes
    matplotlib.pyplot.colorbar
    matplotlib.gridspec.GridSpec
    matplotlib.gridspec.SubplotSpec
    astropy.io.fits.Header
    astropy.wcs.WCS
    astropy.visualization.wcsaxes.WCSAxes
    """
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.transforms import ScaledTranslation
    from matplotlib.gridspec import SubplotSpec
    from astropy.coordinates.sky_coordinate import SkyCoord
    from astropy.visualization.wcsaxes import WCSAxes, SphericalCircle
    from astropy.visualization.wcsaxes.frame import (
        RectangularFrame,
        EllipticalFrame,
    )
    from astropy.units import Quantity, deg
    from .partial import PartialUniqSkymap

    # initialize arguments
    cr = np.unique([*cr, 1])
    if not isinstance(skymap, PartialUniqSkymap):
        if isinstance(skymap, np.ndarray):
            skymap = (skymap, None)
        skymap = PartialUniqSkymap(*skymap)
    ## rot set to point location if only one point present
    #if len(scatter) == 1 and len(scatter[0].points) == 1 and rot is None:
    #    rot = tuple(scatter[0].points[0][:2])

    # initialize our axes if needed
    if ax is None:
        # initialize figure (just to get our axes)
        if not isinstance(fig, Figure):
            fig = plt.figure(**(fig or {}))
        # initialize frame class
        frame_class = get_frame_class(projection, frame_class, vdelta, hdelta,
                                      rot, scatter, sigmas)
        # initialize projection
        projection = get_projection(projection, width=width, height=height,
                                    hdelta=hdelta, vdelta=vdelta, rot=rot,
                                    facing_sky=facing_sky, scatter=scatter,
                                    sigmas=sigmas)
        # initialize axes
        if subplot is None:
            ax = WCSAxes(fig, rect=[0.1, 0.1, 0.9, 0.9], wcs=projection,
                         frame_class=frame_class)
            fig.add_axes(ax)
        else:
            _subplot = subplot
            if isinstance(_subplot, SubplotSpec):
                _subplot = (subplot,)
            ax = fig.add_subplot(*_subplot, projection=projection, # type: ignore
                                 frame_class=frame_class)
    else:
        # initialize projection
        projection = ax.wcs

    assert ax is not None
    # get the pixel coordinates
    _, ra, dec = wcs2ang(ax.wcs)

    # set default coordinate ticks and style
    outline = [outline_effect()]
    co_ra, co_dec = ax.coords
    ra_ticks = get_ticks(ra.value+3, ra_exclusions(ax)+3,
                         np.arange(30, 390, 30), 3, lambda x, d: x+d%360)
    co_ra.set_major_formatter("dd")
    if len(ra_ticks) >= MIN_GRAT:
        co_ra.set_ticks(ra_ticks * deg)
        #co_ra.set_ticks(spacing=30 * deg)
    else:
        # use degree formatting by default
        filled_arcmin = (np.histogram(ra, bins=360*120)[0] != 0).sum()
        if filled_arcmin < 10:
            co_ra.set_major_formatter("dd:mm:ss")
        elif filled_arcmin < 60 * 10:
            co_ra.set_major_formatter("dd:mm")
    dec_ticks = get_ticks(dec.value, dec_exclusions(ax),
                          np.arange(-75, 90, 15), 2, lambda x, _: x)
    if len(dec_ticks) >= MIN_GRAT:
        co_dec.set_major_formatter("dd")
        co_dec.set_ticks(dec_ticks * deg)
    # ticks can look bad on EllipticalFrame
    if issubclass(ax.frame_class, EllipticalFrame):
        co_ra.set_ticks_visible(False)
        co_dec.set_ticks_visible(False)
    co_ra.set_ticklabel(size=BASE_FONT_SIZE, path_effects=outline)
    co_dec.set_ticklabel(size=BASE_FONT_SIZE, path_effects=outline)
    # TODO override if not ICRS
    co_ra.set_axislabel("Right ascension (ICRS) [deg]",
                        size=BASE_FONT_SIZE, path_effects=outline)
    co_dec.set_axislabel("Declination (ICRS) [deg]",
                         size=BASE_FONT_SIZE, path_effects=outline)
    ax.grid(True)

    # plot skymap
    if (cmap is not None) or (cr.size > 1):
        mask, u = wcs2mask_and_uniq(projection)
        render = skymap.render(projection, pad=np.nan, mask_missing=True)
    if cmap is not None:
        cmap = get_colormap(cmap, bad=missing_color)
        img = ax.imshow(render, vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha)
        if nan_color is not None:
            ncmap = monochrome_opacity_colormap("NanCMap", nan_color)
            if missing_color is not None:
                ncmap.set_bad(missing_color)
            nans = np.isnan(render)
            ax.imshow(nans, vmin=0, vmax=1, cmap=ncmap, alpha=alpha)
        if cbar is not False:
            kw = DEFAULT_CBAR_KWARGS.copy()
            label = skymap.name or ''
            if skymap.unit is not None:
                label += (' [{}]' if label else '{}').format(skymap.unit)
            if label:
                kw['label'] = label
            if cbar is not True:
                kw.update(cbar)
            plt.colorbar(img, ax=ax, **kw)

    # plot scatterplots, layering sigma regions first
    # TODO do not scatter plot or label points that are off of the plot axes
    transform = ax.get_transform('world')
    label_transform = transform + ScaledTranslation(N_X_OFFSET/2, N_Y_OFFSET/2,
                                                    ax.figure.dpi_scale_trans)
    for pts in scatter:
        # Skip pts if it is empty, since you cannot have empty SkyCoords
        if not pts.points:
            continue

        # only include points in the visible region
        pts_x, pts_y = ax.wcs.world_to_pixel(SkyCoord(
            *[*zip(*((r, d) for r, d, *_ in pts.points))]*deg, frame='icrs'
        ))
        include = (pts_x < ax.wcs.pixel_shape[0]+.5) & (pts_x > .5)
        include &= (pts_y < ax.wcs.pixel_shape[1]+.5) & (pts_y > .5)
        col = pts.rgba.to_hex(False)
        ax.scatter(pts_x[include], pts_y[include], c=col, marker=pts.marker,
                   s=scatter_marker_size, label=pts.label, path_effects=outline,
                   zorder=10)

        # plot the shaded error circles, if needed
        cm = pts.cmap()
        for sigma in sigmas:
            ax.imshow(pts.render(projection, extent=sigma), vmin=0, vmax=1,
                      cmap=cm, zorder=5)

        # plot the error circle outlines, if needed
        for outline_sigma in outline_sigmas:
            for ra, dec, sigma, _ in pts.points:
                coord = SkyCoord(ra * deg, dec * deg)
                s = SphericalCircle(coord, sigma * outline_sigma * deg,
                    edgecolor=col, facecolor='none',
                    transform=ax.get_transform('icrs'), zorder=5)
                ax.add_patch(s)

        # plot the scatter labels, if needed
        if scatter_labels:
            for i, (r, d, *sl) in enumerate(pts.points):
                if not include[i]:
                    continue
                if len(sl) == 2:
                    pt_label = sl[1]
                else:
                    pt_label = str(i+1)
                kw = {
                    'fontsize': scatter_label_size,
                    'path_effects': outline,
                    'color': col,
                    'va': 'bottom',
                    'ha': 'left',
                    'zorder': 15
                }
                if not isinstance(scatter_labels, bool):
                    kw.update(scatter_labels)
                ax.text(r, d, pt_label, transform=label_transform, **kw)

    # plot contours
    if cr.size > 1:
        q = (1-cr)[::-1]
        _, levels, _ = skymap.quantiles((1-cr[::-1]))
        levels = levels[1:]
        cr_lut = dict(
            zip(
                levels.value if isinstance(levels, Quantity) else levels,
                cr[-2::-1]
            )
        )
        ptrans = ax.get_transform(projection)
        # TODO handle contourf needing two levels and reconcile it with
        # disposing of the 1.0 quantile
        contour = ax.contourf if cr_filled else ax.contour
        ckw = DEFAULT_C_KWARGS.copy()
        if cr_kwargs is not None:
            ckw.update(cr_kwargs)
        cntrs = contour(render, transform=ptrans, levels=levels, **ckw)
        if cr_format is not None:
            clabelkw = DEFAULT_CLABEL_KWARGS.copy()
            if cr_label_kwargs is not None:
                clabelkw.update(cr_label_kwargs)
            ax.clabel(cntrs, cntrs.levels, **clabelkw,
                      fmt=lambda v: cr_format(cr_lut[v], v))

    return ax


def gridplot(
        *skymaps: Union[
            'PartialUniqSkymap',
            NDArray[Any],
            Tuple[
                NDArray[Any],
                Optional[
                    Union[
                        NDArray['np.integer[Any]'],
                        'WCS',
                        str,
                    ]
                ],
            ],
        ],
        fig: Optional[
            Union[
                'matplotlib.figure.Figure',
                'matplotlib.gridspec.GridSpec',
                Mapping,
            ]
        ] = None,
        projections: Sequence[
            Union[
                str,
                'WCS',
                'astropy.io.fits.Header',
                Sequence[
                    Union[
                        str,
                        'WCS',
                        'astropy.io.fits.Header',
                        'astropy.visualization.wcsaxes.WCSAxes',
                    ]
                ],
            ]
        ]= ('MOL',),
        scatters: Optional[Sequence[Sequence[PointsTuple]]] = None,
        # fig args
        subplot_height: float = DEFAULT_GRID_ROW_HEIGHT,
        # fig.add_gridspec args
        ncols: int = DEFAULT_NCOLS,
        hspace: float = DEFAULT_HSPACE,
        wspace: float = DEFAULT_WSPACE,
        wshrink: Union[float, Sequence[float]] = 1.,
        # plotter args
        subplot_kwargs: Optional[Sequence[Optional[Sequence[dict]]]] = None,
        left=0.,
        right=1.,
        bottom=0.,
        top=1.,
        **kwargs
) -> Tuple[
        'matplotlib.gridspec.GridSpec',
        Sequence[Sequence['astropy.visualization.wcsaxes.WCSAxes']]
]:
    """
    Make a grid plot of multiple skymaps (optionally with scatterplots for
    each).

    Parameters
    ----------
    *skymaps : 'PartialUniqSkymap', array, or (array, array)
        The skymaps to plot. Can be a ``PartialUniqSkymap``, a
        single-resolution HEALPix skymap *in NEST ordering only*, or a tuple of
        (pixel values, NUNIQ indices) accepted as the first two arguments to
        ``PartialUniqSkymap``.
    fig: matplotlib.figure.Figure or dict, optional
        The figure to plot to. If not provided, a new figure will be created.
        If a dictonary is provided, it will be passed as keyword arguments to
        create a new figure. If a ``GridSpec`` is provided, then the figure
        to which it is attached will be used, and that ``GridSpec`` will be used
        to define the layout.
    projections : Sequence[Union[str, WCS, fits.Header, Sequence[Union[str, WCS, fits.Header, WCSAxes]]]], optional
        A list of projections (see the ``projection`` argument of ``plot``) to
        use for each skymap in ``skymaps``. If multiple projections are
        specified, they will be plotted alongside each other; if this makes the
        figure too wide, change the number of columns in the grid with
        ``ncols``. You can also pass a list of lists of axes of the type
        returned by this function (which you should do while also passing a
        ``GridSpec`` as ``fig``), allowing you to plot multiple layers of data
        to the same grid plot.
    scatters : Sequence[Sequence[PointsTuple]], optional
        Scatterplots to use, one list for each skymap containing the sets of
        points to plot for that skymap. For any of the skymaps which are
        ``PartialUniqSkymap`` instances, you can default to plotting that
        instance's ``point_sources`` by passing ``None`` instead of a list of
        point sources; this is also the default behavior if ``scatters=None``.
    subplot_height : float, optional
        The height of each subplot (in inches). Width is automatically
        determined. Overall figure height will be this times ``ncols``.
        *Ignored if* ``fig`` *is a pre-existing figure, or if* ``figsize`` *is
        specified as a keyword argument in* ``fig`` (though this is not
        recommended usage and should be avoided unless you know what you're
        doing).
    ncols : int, optional
        How many columns of *skymaps* to include in the grid. **NB: All
        subplots for a single skymap are counted as a single column** (in
        contrast to ``matplotlib.gridspec.GridSpec``, which counts each subplot
        in a single column). The number of rows is determined automatically
        from the number of skymaps provided combined with ``ncols``.
    hspace : float, optional
        How much vertical space (height) to reserve between subplots.
    wspace : float, optional
        How much horizontal space (width) to reserve between subplots.
    wshrink : float or list of floats, optional
        Scale the plot widths by this much. Useful if you are adding color
        bars to preserve spacing. If a list, each element corresponds to a
        projection.
    subplot_kwargs : Sequence[Optional[Sequence[dict]]], optional
        Lists of keyword argument dictionaries that will be used for each
        subplot. The first index specifies the plotter, and the second index
        specifies the skymap from ``skymaps``. This behavior allows you to
        easily specify lists of keyword arguments for specific projections
        (since often) only one of the ``projections`` requires skymap-specific
        parameters).  **NB: These subplot-specific keyword arguments take
        precedence over** ``**kwargs`` **for their respective subplots.
        Projection-related keyword arguments are ignored if** ``projections``
        **is a list of lists of axes; see the** ``plot`` **documentation
        for further details.**
    left, right, bottom, top : float
        Bounds for the GridSpec within the plot. Move ``top`` down, for
        example, if you want more space for a super-title with
        ``plt.suptitle``.
    **kwargs
        Keyword arguments applied to all projections; **again, see** ``plot``
        **for usages and caveats.**

    Returns
    -------
    gs : matplotlib.gridspec.GridSpec
        The ``GridSpec`` defining the shape of the subplots. Access the plotted
        figure as ``gs.figure``. Reuse this layout by passing ``gs`` to another
        invocation of ``gridplot``.
    axs : Sequence[Sequence['astropy.visualization.wcsaxes.WCSAxes']]
        The axes which were plotted. The first index (outer list) corresponds
        to the projection used, and the second index (inner lists) correspond
        to the skymap. Reuse the axes and projections by passing
        ``projections=axs`` to another invocation of ``gridplot``.

    Raises
    ------
    ValueError
        If ``scatters`` is provided and is ill-formatted or not of the same
        length as ``skymaps``; if ``subplot_kwargs`` is included and is not of
        the same length as ``projections``, or if its elements are neither
        ``None`` nor lists of the same length as ``skymaps``; if ``fig`` is
        passed as a ``GridSpec`` which is not compatible with the rest of the
        arguments given; or if one of the
        ``projections`` is specified as a string but cannot be found in this
        module.

    See Also
    --------
    plot
    """

    from math import ceil
    from matplotlib.pyplot import figure
    from matplotlib.figure import Figure
    from matplotlib.gridspec import GridSpec
    from astropy.io.fits import Header
    from astropy.wcs import WCS

    from .partial import PartialUniqSkymap

    gs = fig if isinstance(fig, GridSpec) else None
    fig = fig if gs is None else gs.figure # type: ignore
    projections = list(projections)
    nᵖ = len(projections)  # number of projections per skymap
    nᶜ = len(skymaps)       # number of cells, i.e. skymaps
    nˢ = nᶜ*nᵖ          # number of true subplots in fig
    if gs is None:
        nʳ = nᵖ*ncols       # number of true subplots per row
        nrows = int(ceil(nᶜ/ncols))
    else:
        nʳ = gs.ncols
        if nʳ % nᵖ != 0:
            raise ValueError(f"Cannot evenly fit {nᵖ} projections in "
                             f"{nʳ} columns.")
        ncols = int(nʳ / nᵖ)
        nrows = gs.nrows
        if nrows * nʳ < nˢ:
            raise ValueError(f"Not enough rows {nrows} for {nˢ} subplots.")

    if scatters is None:
        scatters = [
            s.point_sources if isinstance(s, PartialUniqSkymap) else []
            for s in skymaps
        ]
    subplot_kwargs = subplot_kwargs or [None]*nᵖ
    if isinstance(wshrink, Sized):
        if len(wshrink) != nᵖ:
            raise ValueError(f"Must provide one wshrink {wshrink} for "
                             f"each projection {projections}")
    else:
        wshrink = [wshrink]*nᵖ

    if len(scatters) != nᶜ:
        raise ValueError(f"scatters {scatters} must have same len as skymaps "
                         f"{skymaps}")
    if len(subplot_kwargs) != nᵖ:
        raise ValueError(f"subplot_kwargs {subplot_kwargs} must have same "
                         f"len as projections {projections}")

    # initialize kwargs
    for i in range(len(subplot_kwargs)):
        kw = subplot_kwargs[i]
        if kw is None:
            subplot_kwargs[i] = [{}]*nᶜ # type: ignore
        elif len(kw) != nᶜ:
            raise ValueError(f"{i}-th element of subplot_kwargs {kw} must "
                             f"have same len as skymaps {skymaps} or else be "
                             "omitted.")
        del kw
        for j in range(nᶜ):
            subplot_kwargs[i][j].update(kwargs) # type: ignore

    # initialize frame classes and projections
    frames = []
    for iᵖ in range(len(projections)):
        p = projections[iᵖ]
        fa = []
        pa = []
        if isinstance(p, (str, WCS, Header)):
            p = [p]*nᶜ
        for iᶜ, pp in enumerate(p):
            kw = subplot_kwargs[iᵖ][iᶜ] # type: ignore
            if isinstance(pp, (str, WCS, Header)):
                fa.append(get_frame_class(
                    pp, scatter=scatters[iᶜ],
                    **{k: v for k, v in kw.items()
                       if k in GET_FRAME_CLASS_KWARG_KEYS}
                ))
                pa.append(get_projection(
                    pp, scatter=scatters[iᶜ],
                    **{k: v for k, v in kw.items()
                       if k in GET_WCS_KWARG_KEYS}
                ))
            else:
                fa.append(pp.frame_class)
                pa.append(pp.wcs)
        projections[iᵖ] = pa
        frames.append(fa)

    # from https://matplotlib.org/tutorials/intermediate/gridspec.html
    widths = [w * p[0].pixel_shape[0] / p[0].pixel_shape[1]
              for w, p in zip(wshrink, projections)]
    width_ratios = widths*ncols
    nat_width, nat_height = (subplot_height*sum(width_ratios),
                             subplot_height*nrows)
    if not isinstance(fig, Figure):
        fkw = {
            'figsize': (nat_width, nat_height),
            'facecolor': 'w',
            'edgecolor': 'k',
        }
        fkw.update(**(fig or {})) # type: ignore
        fig = figure(**fkw)
    if gs is None:
        gs = fig.add_gridspec(
            nrows=nrows,
            ncols=nᵖ*ncols,
            width_ratios=width_ratios,
            hspace=hspace,
            wspace=wspace,
            left=left,
            right=right,
            bottom=bottom,
            top=top,
        )

    axs = []
    for iᵖ, (p, f, k) in enumerate(zip(projections, frames, subplot_kwargs)):
        axr = []
        for iᶜ, s in enumerate(skymaps):
            row, col = divmod(iᶜ*nᵖ+iᵖ, nʳ)
            ax = plot(s, *scatters[iᶜ], projection=p[iᶜ], fig=fig,
                      subplot=gs[row, col], frame_class=f[iᶜ], **k[iᶜ])
            # hide axis labels
            co_ra, co_dec = ax.coords
            co_ra.set_axislabel_visibility_rule('labels')
            co_dec.set_axislabel_visibility_rule('labels')
            #co_ra.set_ticklabel_visible(row + 1 == nrows)
            #co_dec.set_ticklabel_visible(False)
            axr.append(ax)
        axs.append(axr)

    return gs, axs


def _parametric_ra_exclusion(ra_height, w):
    import numpy as np

    return w.pixel_to_world(
        *np.meshgrid(*map(np.arange, w.pixel_shape), sparse=True)
    ).icrs.ra.value[
        ra_height(w),
        np.concatenate((
            np.arange(int(w.pixel_shape[0])//EDGE_EXCLUSION),
            -np.arange(1, int(w.pixel_shape[0])//EDGE_EXCLUSION),
        ))
    ]


def _dec_exclusion(w):
    import numpy as np

    dec = w.pixel_to_world(
            *np.meshgrid(*map(np.arange, w.pixel_shape), sparse=True)
        ).icrs.dec.value[
        np.concatenate((
            np.arange(int(w.pixel_shape[1])//EDGE_EXCLUSION),
            np.arange(1, int(w.pixel_shape[1])//EDGE_EXCLUSION),
        ))
    ]
    return dec[~np.isnan(dec)]


def _exclusions(registry, ax):
    _init_exclusions()
    w = ax.wcs
    frame_class = ax.coords.frame.__class__
    for c in frame_class.mro():
        if c in registry:
            return registry[c](w)


def register_ra_exclusion(cls, func):
    _init_exclusions()
    _RA_EXCLUSIONS[cls] = func


def register_dec_exclusion(cls, func):
    _init_exclusions()
    _DEC_EXCLUSIONS[cls] = func


# to allow lazy-loading
def _init_exclusions():
    from astropy.visualization.wcsaxes.frame import EllipticalFrame

    if not EXCLUSIONS_INITIALIZED[0]:
        _RA_EXCLUSIONS[EllipticalFrame] = \
            partial(_parametric_ra_exclusion,
                    lambda w: int(w.pixel_shape[1])//2)
        EXCLUSIONS_INITIALIZED[0] = True


_RA_EXCLUSIONS = {
    object: partial(_parametric_ra_exclusion, lambda _: -1,),
}
_DEC_EXCLUSIONS = {
    object: _dec_exclusion,
}
ra_exclusions = partial(_exclusions, _RA_EXCLUSIONS)
ra_exclusions.__doc__ = "Get RA values to exclude from ticks."
dec_exclusions = partial(_exclusions, _DEC_EXCLUSIONS)
dec_exclusions.__doc__ = "Get declination values to exclude from ticks."


def get_ticks(include, exclude, ticks, delta, transform):
    import numpy as np

    bins = np.repeat(ticks, 2).reshape((-1, 2))
    bins += np.array([-1, 1]) * delta
    bins = bins.ravel()
    ihist, _ = np.histogram(transform(include, delta), bins=bins)
    ehist, _ = np.histogram(transform(exclude, delta), bins=bins)
    matches = (ihist != 0) & (ehist == 0)
    return ticks[matches[::2]]
