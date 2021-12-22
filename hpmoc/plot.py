# (c) Stefan Countryman 2021

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
.. _`Representations of celestial coordinates in FITS`: https://ui.adsabs.harvard.edu/abs/2007MNRAS.381..865C

The complete list of available projections can be found in the
`FITS Definition`_, with concrete transformation definitions given in
the `Representations of Celestial Coordinates in FITS`_. The table of
available projections is reproduced below, with sections linking to the
appropriate table in `FITSWorld`_. Note that not all valid WCS projections
can be displayed by astropy at time of writing; in particular, the HEALPIX
``HPX`` projection does not work out of the box, which is one of the
(many) motivations for this plotting library.

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
   * - Cylindrical projections
     - CYP
     - 0◦
     - 0◦
     - Sect. 5.2.1
   * - Cylindrical perspective
     - CEA
     - 0◦
     - 0◦
     - Sect. 5.2.2
   * - Cylindrical equal area
     - CAR
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
hpmoc.PartialUniqSkymap
astropy.wcs.WCS
astropy.visualization.wcsaxes
hpmoc.plotters
ligo.skymap
healpy
"""

from typing import Optional, Union, Tuple, Iterable, Callable
from textwrap import indent, wrap
from .points import PointsTuple

import hpmoc

_ALLSKY = """
NAXIS   =                    2
NAXIS1  =                  360
NAXIS2  =                  180
CTYPE1  = 'RA---{proj}'
CRVAL1  =                180.0
CUNIT1  = 'deg     '
CTYPE2  = 'DEC--{proj}'
CRVAL2  =                  0.0
CUNIT2  = 'deg     '
COORDSYS= 'icrs    '
"""
_WCS_HEADERS = dict()
_PROJ_DOC = """
    All-sky:
"""
for proj, aliases in {
    'MOL': ('Mollweide',),
    'AIT': ('Hammer-Aitoff', 'Aitoff', 'Hammer'),
    'CAR': ('Carée', 'Plate-carée', 'Caree', 'Plate-caree', 'Cartesian',
            'Tyre'),
    'SFL': ('Sanson-Flamsteed',),
    'PAR': ('Parabolic', 'Craster'),
}.items():
    _PROJ_DOC += '    - ' + indent('\n'.join(
        wrap(f"{proj}: *{', '.join(aliases)}*", 73)), ' '*6)[6:] + '\n'
    _WCS_HEADERS[proj] = _ALLSKY.format(proj=proj)
    for alias in aliases:
        _WCS_HEADERS[alias.upper()] = _WCS_HEADERS[proj]


def get_wcs(
        projection: str,
        width: float,
        height: float,
        rot: Tuple[float, float, float],
) -> 'astropy.wcs.WCS':
    from astropy.io.fits import Header
    from astropy.wcs import WCS

    header = Header.fromstring(_WCS_HEADERS[projection.upper()])


def plot(
        skymap: 'hpmoc.PartialUniqSkymap',
        *scatter: PointsTuple,
        projection: Union[str, 'astropy.wcs.WCS'] = 'Mollweide',
        frame: Optional['astropy.visualization.wcsaxes.frame.BaseFrame'] = None,
        width: float = 360,
        height: float = 180,
        rot: Tuple[float, float, float] = (180, 0, 0),
        facing_sky: bool = True,
        fig: Optional['matplotlib.figure.Figure'] = None,
        subplot: Optional[Tuple[int, int, int]] = None,
        cr: Optional[Iterable[float]] = None,
        cr_format: Callable[[float, float], str] = None,
        cr_filled: bool = False,
        cr_kwargs: Optional[dict] = None,
) -> Union[
    'astropy.visualization.wcsaxes.WCSAxes',
    'astropy.visualization.wcsaxes.WCSAxesSubplot'
]:
    """
    skymap: hpmoc.PartialUniqSkymap
        The HEALPix skymap to plot. You can overlay further plots using ``
    projection: str or astropy.wcs.WCS, optional
        The following projections are available by default. See ``hpmoc.plot``
        documentation for instructions on constructing your own WCS headers for
        other plot styles, which can be passed in instead using a ``WCS``
        instance. You can also use this approach to plot the skymap over an
        existing ``WCS`` taken from another fits file, making it easy to plot
        skymaps over other astrophysical data. You can select from any of the
        projections available in ``get_wcs``.
    """
CRPIX1  =                180.5
CRPIX2  =                 90.5
LONPOLE =               0
