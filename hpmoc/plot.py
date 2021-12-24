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

from warnings import warn
from typing import Optional, Union, Tuple, Iterable, Callable
from textwrap import indent, wrap
from .utils import outline_effect, N_X_OFFSET, N_Y_OFFSET
from .points import PointsTuple

import hpmoc

DEFAULT_WIDTH = 360
DEFAULT_HEIGHT = 180
DEFAULT_ROT = (180, 0, 0)
DEFAULT_FACING = True
BASE_FONT_SIZE = 10
DEFAULT_C_KWARGS = {}
DEFAULT_CLABEL_KWARGS = {
    'inline': False,
    'fontsize': BASE_FONT_SIZE,
}
_ALLSKY = {
    "NAXIS": 2,
    "NAXIS1": 360,
    "NAXIS2": 180,
    "CTYPE1": 'RA---',
    "CRVAL1": 180.0,
    "CUNIT1": 'deg',
    "CTYPE2": 'DEC--',
    "CRVAL2": 0.0,
    "CUNIT2": 'deg',
    "COORDSYS": 'icrs'
}
_WCS_HEADERS = dict()
_WCS_FRAMES = dict()
_ALL_SKY_DOC = """
"""
for proj, (frame, aliases) in {
    'MOL': ('e', ('Mollweide',)),
    'AIT': ('e', ('Hammer-Aitoff', 'Aitoff', 'Hammer')),
    'CAR': ('r', ('Carée', 'Plate-carée', 'Caree', 'Plate-caree', 'Cartesian',
                  'Tyre')),
    'SFL': ('p', ('Sanson-Flamsteed',)),
    'PAR': ('p', ('Parabolic', 'Craster')),
}.items():
    _ALL_SKY_DOC += '    - ' + indent('\n'.join(
        wrap(f"{proj}: *{', '.join(aliases)}*", 73)), ' '*6)[6:] + '\n'
    _WCS_FRAMES[proj] = frame
    _WCS_HEADERS[proj] = _ALLSKY.copy()
    _WCS_HEADERS[proj]['CTYPE1'] += proj
    _WCS_HEADERS[proj]['CTYPE2'] += proj
    for alias in aliases:
        _WCS_HEADERS[alias.upper()] = _WCS_HEADERS[proj]
        _WCS_FRAMES[alias.upper()] = frame


def get_frame_class(
        projection: str
) -> 'astropy.visualization.wcsaxes.frame.BaseFrame':
    """
    Get the frame class associated with a given projection.
    Parameters
    ----------
    projection: str
        The projection to use. See ``get_wcs`` for details.

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

    f = _WCS_FRAMES[projection.upper()]
    if f == 'e':
        return frame.EllipticalFrame
    if f == 'r':
        return frame.RectangularFrame
    warn(f"Frame class for {projection} not yet available. Using default for "
         "now; specify your own for a better-looking plot.", UserWarning)
    return frame.RectangularFrame


def get_wcs(
        projection: str,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        rot: Tuple[float, float, float] = DEFAULT_ROT,
        facing_sky: bool = DEFAULT_FACING,
) -> 'astropy.wcs.WCS':
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

        All-sky:{_ALL_SKY_DOC}
    width: int
        Width of the image in pixels.
    height: int
        Height of the image in pixels.
    rot: Tuple[float, float, float]
        Euler angles for rotations about the Z, X, Z axes. These are
        immediately translated to ``CRVAL1, CRVAL2, LONPOLE`` in the returned
        ``WCS``; that is, the first two angles specify the angle of the center
        of the image, while the last specifies an additional rotation of the
        reference pole about the Z-axis. See ``hpmoc.plot`` documentation for
        further references.
    facing_sky: bool
        Whether the projection is outward- or inward-facing. Equivalent to
        reversing the direction of the longitude.

    Returns
    -------
    wcs: astropy.wcs.WCS
        A ``WCS`` instance that can be used for rendering and plotting.

    Raises
    ------
    IndexError
        If the specified projection could not be found.
    """
    from astropy.io.fits import Header
    from astropy.wcs import WCS
    import numpy as np

    dec_dir = -1 if facing_sky else 1
    header = Header(_WCS_HEADERS[projection.upper()].copy())
    header['NAXIS1'] = width
    header['CRPIX1'] = width / 2 + 0.5
    header['CDELT1'] = dec_dir * np.sqrt(8) / np.pi * 360 / width
    header['CRVAL1'] = rot[0]
    header['NAXIS2'] = height
    header['CRPIX2'] = height / 2 + 0.5
    header['CDELT2'] = np.sqrt(8) / np.pi * 180 / height
    header['CRVAL2'] = rot[1]
    header['LONPOLE'] = rot[2]
    return WCS(header)


def plot(
        skymap: 'hpmoc.PartialUniqSkymap',
        *scatter: PointsTuple,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: Optional[
            Union[str, 'matplotlib.colors.Colormap']
        ] = 'gist_heat_r',
        alpha: float = 1.,
        sigmas: Iterable[float] = (1,),
        scatter_labels: bool = True,
        ax: Optional[Union[
            'astropy.visualization.wcsaxes.WCSAxes',
            'astropy.visualization.wcsaxes.WCSAxesSubplot'
        ]] = None,
        projection: Union[
            str,
            'astropy.wcs.WCS',
            'astropy.io.fits.Header',
        ] = 'Mollweide',
        frame_class: Optional[
            'astropy.visualization.wcsaxes.frame.BaseFrame'
        ] = None,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        rot: Tuple[float, float, float] = DEFAULT_ROT,
        facing_sky: bool = DEFAULT_FACING,
        fig: Optional['matplotlib.figure.Figure'] = None,
        subplot: Optional[Tuple[int, int, int]] = None,
        cr: Iterable[float] = tuple(),
        cr_format: Callable[[float, float], str] = None,
        cr_filled: bool = False,
        cr_kwargs: Optional[dict] = None,
        cr_label_kwargs: Optional[dict] = None,
        pixels: bool = False,
        pixels_format: Optional[
            Callable[['hpmoc.PartialUniqSkymap'], Iterable[str]]
        ] = None,
        pixels_kwargs: Optional[dict] = None,
) -> Union[
    'astropy.visualization.wcsaxes.WCSAxes',
    'astropy.visualization.wcsaxes.WCSAxesSubplot'
]:
    """
    Parameters
    ----------
    skymap: hpmoc.PartialUniqSkymap
        The skymap to plot.
    scatter: PointsTuple
        Point-sources to plot as a scatter-map, with disks showing their error
        regions. Provide multiple (ideally with different colors) to plot many
        populations at once.
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
    alpha: float, optional
        The opacity of the plotted skymap image.
    sigmas: Iterable[float], optional
        The size of the error region about each point source to plot in units
        of its error parameter sigma.
    scatter_labels: bool, optional
        Whether to show labels for the scattered points. If ``True``, display
        either their labels (if defined) or their indices within the
        ``PointsTuple.points`` list.
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
    frame_class: BaseFrame, optional
        The frame type to use for this plot, e.g. a ``RectangularFrame`` (for
        a Plate-carée/Cartesian plot) or an ``EllipticalFrame`` for a
        Mollweide plot. Selected automatically when ``projection`` is specified
        by name, otherwise defaults to ``RectangularFrame``.
        *Ignored if* ``ax`` *is given.*
    width: int, optional
        The width of the plot in pixels. *Ignored if* ``ax`` *is given.*
    height: int, optional
        The height of the plot in pixels. *Ignored if* ``ax`` *is given.*
    rot: Tuple[float, float, float], optional
        Euler angles for rotations about the Z, X, Z axes. These are
        immediately translated to ``CRVAL1, CRVAL2, LONPOLE`` in the returned
        ``WCS``; that is, the first two angles specify the angle of the center
        of the image, while the last specifies an additional rotation of the
        reference pole about the Z-axis. See ``hpmoc.plot`` documentation for
        further references. *Ignored if* ``ax`` *is given.*
    facing_sky: bool, optional
        Whether the projection is outward- or inward-facing. Equivalent to
        reversing the direction of the longitude. *Ignored if* ``ax`` *is
        given.*
    fig: matplotlib.figure.Figure, optional
        The figure to plot to. If not provided, a new figure will be created.
        *Ignored if* ``ax`` *is given.*
    subplot: Tuple[int, int, int], optional
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
    pixels: bool, optional
        Whether to plot pixel borders. *You should probably keep this*
        ``False`` *if you are doing an all-sky plot, since border plotting is
        slow for a large number of pixels, and the boundaries will not be
        visible anyway unless each visible pixel's size is comparable to the
        overall size of the plot window.* If you just want to see information
        about e.g. the size of pixels across the whole sky, consider plotting
        ``skymap.o⃗(as_skymap=True)`` to see a color map of pixel sizes
        instead.
    pixels_format: Callable[[PartialUniqSkymap], Iterable[str]], optional
        A function that takes a skymap and returns a string for each pixel.
        Will be called on the selection of pixels overlapping with the visible
        area, so don't worry about optimizing it, since just plotting the
        borders will be slow enough for a large number of visible pixels.
        The returned string will be plotted over the center of each pixel,
        which again is only useful if the pixels are large with respect to the
        size of the plot window.
    pixels_kwargs: dict, optional
        Keyword arguments to pass to ``WCSAxes.plot``, which can be used to
        control the appearance of the pixel borders.

    Returns
    -------

    """
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.transforms import ScaledTranslation
    from astropy.wcs import WCS
    from astropy.visualization.wcsaxes import WCSAxes
    from astropy.visualization.wcsaxes.frame import RectangularFrame
    from astropy.io.fits import Header
    from astropy.units import Quantity, deg

    # initialize arguments
    cr = np.unique([*cr, 1])

    # initialize our axes
    if ax is None:
        if fig is None:
            fig = plt.figure()
        if frame_class is None:
            if isinstance(projection, str):
                frame_class = get_frame_class(projection)
            else:
                frame_class = RectangularFrame
        if not isinstance(projection, WCS):
            if not isinstance(projection, Header):
                try:
                    projection = get_wcs(projection, width, height, rot,
                                         facing_sky)
                except IndexError:
                    projection = WCS(Header.fromstring(projection))
            else:
                projection = WCS(projection)
        if subplot is None:
            ax = WCSAxes(fig, rect=[0.1, 0.1, 0.9, 0.9], wcs=projection,
                         frame_class=frame_class)
            fig.add_axes(ax)
        else:
            ax = fig.add_subplot(*subplot, projection=projection,
                                 frame_class=frame_class)

    # set default coordinate ticks and style
    outline = [outline_effect()]
    co_ra, co_dec = ax.coords
    co_ra.set_major_formatter("dd")
    co_dec.set_major_formatter("dd")
    co_ra.set_ticks(np.arange(30, 360, 30) * deg)
    co_dec.set_ticks(np.arange(-75, 90, 15) * deg)
    co_ra.set_ticks_visible(False)
    co_dec.set_ticks_visible(False)
    co_ra.set_ticklabel(size=BASE_FONT_SIZE, path_effects=outline)
    co_dec.set_ticklabel(size=BASE_FONT_SIZE, path_effects=outline)
    co_ra.set_axislabel("Right ascension (ICRS) [deg]",
                        size=BASE_FONT_SIZE, path_effects=outline)
    co_dec.set_axislabel("Declination (ICRS) [deg]",
                         size=BASE_FONT_SIZE, path_effects=outline)
    ax.grid(True)

    # plot skymap
    if (cmap is None) or (cr.size > 1):
        render = skymap.render(projection, pad=np.nan)
    if cmap is not None:
        ax.imshow(render, vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha)

    # plot scatterplots, layering sigma regions first
    # TODO see if z level need be specified
    transform = ax.get_transform('world')
    label_transform = transform + ScaledTranslation(N_X_OFFSET/2, N_Y_OFFSET/2,
                                                    ax.figure.dpi_scale_trans)
    for pts in scatter:
        cm = pts.cmap()
        for sigma in sigmas:
            ax.imshow(pts.render(projection, sigma), vmin=0, vmax=1, cmap=cm)
    for pts in scatter:
        col = pts.rgba.to_hex(False)
        ax.scatter(*np.array([(r, d) for (r, d, *_) in pts.points]).T,
                   c=col, marker=pts.marker, transform=transform,
                   s=BASE_FONT_SIZE*2, label=pts.label)
        if scatter_labels:
            for i, (r, d, *sl) in enumerate(pts.points):
                if len(sl) == 2:
                    pt_label = sl[1]
                else:
                    pt_label = str(i)
                ax.text(r, d, pt_label, va='bottom', ha='left',
                        path_effects=outline, color=col,
                        fontsize=BASE_FONT_SIZE,
                        transform=label_transform)

    # plot contours
    if cr.size > 1:
        q = (1-cr)[::-1]
        _, levels, _ = skymap.quantiles((1-cr[::-1]))
        levels = levels[1:]
        cr_lut = dict(
            zip(
                levels.value if isinstance(levels, Quantity) else levels,
                cr[:-1]
            )
        )
        ptrans = ax.get_transform(projection)
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