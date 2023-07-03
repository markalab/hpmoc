# Copyright (C) 2016-2023 Columbia Experimental Gravity Group (GECo)

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
More complex plotting commands for joint events in more human-readable formats,
e.g. the GW skymap plots from O2 with their overlaid crosses marking neutrino
positions.
"""

import re
import functools
from colorsys import rgb_to_hsv, hsv_to_rgb
from typing import List, Tuple, Callable, Optional, Union
from textwrap import dedent
from .points import Rgba, PointsTuple
from .utils import (
    resol2nside,
    nest2uniq,
    monochrome_opacity_colormap,
    outline_effect,
    OUTLINE_STROKE,
    N_X_OFFSET,
    N_Y_OFFSET,
    FONT_SIZE,
)
from .abstract import AbstractPartialUniqSkymap

MAX_NSIDE = 512  # maximum NSIDE to use for plots (to conserve memory)
DPI = 200
WIDTH = 6  # [inches]
HEIGHT = 4  # [inches]
HSPACE = 0.2
WSPACE = 0.04
NCOLS = 2
TITLE_OUTLINE_STROKE = 2  # thickness of outline surrounding plot titles
LEFT_SHIFT_COEFF = 1/20.  # quadratic curve dec labels away from meridian
LEFT_SHIFT_BASE = -20  # baseline shift from leftmost meridian [deg]
XMARGIN = 0.4  # additional margin in x (make sure text fits) [inches]
TOPMARGIN = -0.0  # additional margin at top [inches]
BOTTOMMARGIN = -0.0  # additional margin at bottom [inches]
NEUTRINO_COLOR = 'green'


DEC_X_OFFSET = -0.37  # [inches]
DEC_Y_OFFSET = -0.02  # [inches]
RA_X_OFFSET = 0  # [inches]
RA_Y_OFFSET = 0.06  # [inches]
MIN_GRAT = 3
_DELTA_HIGHRES = tuple(v*.1**i for i in range(20) for v in (5, 2, 1))
DELTA_PARALLELS = (15.,)+_DELTA_HIGHRES     # space btwn graticule parallels
DELTA_MERIDIANS = (30., 10.)+_DELTA_HIGHRES #   and meridians [deg]
GRATICULE_COLOR = "#B0B0B0"
GRATICULE_LABEL_COLOR = (0.2, 0.2, 0.2)
TITLES = (
    '1-detector Skymap (LHO)',
    '2-detector Skymap (LHO, LLO)',
    '3-detector Skymap (LHO, LLO, Virgo)',
)
DEFAULT_SCATTER_MARKER = "$\\bigodot$"  # bullseye hides less GW density
MERIDIAN_FONT_SIZE = 11
UNCERTAINTY_ALPHA = 0.4  # opacity in [0,1] for scatterplot uncertainty discs
DEFAULT_ROT = (180, 0, 0)


def _proj_grat_label(ax, ras, decs, vals, trans, δ, **kwargs):
    import numpy as np

    for ra, dec, val in zip(ras, decs, vals):
        if val%1:
            fmt = f'{{:1.{int(-np.floor(np.log10(δ)))}f}}'
        else:
            fmt = '{}'
            val = int(val)
        ax.projtext(
            ra,
            dec,
            fmt.format(val)+r'$\degree$',
            #f'{val}{prec}$\\degree$',
            transform=trans,
            **kwargs
        )


def default_cmap():
    """Get the matplotlib colormap."""
    from matplotlib import cm

    # other colors: magma_r, BuPu, bone_r, YlGnBu, gnuplot2_r, CMRmap_r
    cmap = cm.gist_heat_r
    cmap.set_under((0, 0, 0, 0))  # transparent background (-np.inf imgshow)
    return cmap


def get_hp_ax_img_shape(ax):
    """
    Get the shape of a ``healpy.projaxes.SphericalProjAxes`` instance's
    plotted images.
    """
    from matplotlib.image import AxesImage

    for i in ax.get_children():
        if isinstance(i, AxesImage):
            return i.get_array().shape
    raise ValueError('`ax` contains no `AxesImage` instances. '
                     f'Children: {ax.get_children()}')


def layer_plot(s⃗, ax=None, vmin=None, vmax=None, cmap=None, zorder=0,
               shape=None, nest=True, θϕm=None):
    """
    Show a new plot on top of an existing ``healpy`` axes instance.

    Parameters
    ----------
    s⃗ : PartialUniqSkymap or array
        The skymap to plot. If a ``np.ndarray`` is given, assume it is a full
        skymap.
    ax : healpy.projaxes.SphericalProjAxes, optional
        The ``healpy`` axes instance on which to plot ``s⃗``. Uses current axes
        if not provided, i.e. ``plt.gca()``.
    vmin : float, optional
        Minimum value of the colormap.
    vmax : float, optional
        Maximum value of the colormap.
    cmap : str, optional
        ``matplotlib`` colormap to use.
    zorder : float, optional
        The z-order of the image. Make it higher to place it over
        previously-plotted images.
    shape : Tuple[int, int], optional
        The shape of the projected image returned by one of the
        ``healpy.visufunc`` plotters when ``return_projected_map=True``, i.e.
        the result of a call to ``ax.projmap``. This determines the pixel
        locations.
    nest : bool, optional
        Whether ``s⃗`` has NEST HEALPix pixel ordering. This argument is ignored
        if ``s⃗`` is a ``PartialUniqSkymap`` (implied ``UNIQ`` ordering).
    θϕm : Tuple[array, array, array]
        The return result of ``get_ax_angles``. If you are making multiple
        plots, it might be slightly faster to reuse this value.
    """
    import healpy as hp
    from matplotlib import pyplot as plt
    from astropy.units import Quantity as Qty

    ax = ax or plt.gca()
    shape = shape or get_hp_ax_img_shape(ax)
    if isinstance(s⃗, AbstractPartialUniqSkymap):
        nest = True
        nˢ = get_ax_nside(ax, shape=None)
    else:
        nˢ = hp.npix2nside(len(s⃗))
    θ, ϕ, m = θϕm or get_ax_angles(ax, shape=shape)
    s⃗̇ = hp.ang2pix(nˢ, θ, ϕ, nest=nest)
    del θ, ϕ

    if isinstance(s⃗, AbstractPartialUniqSkymap):
        s⃗ = s⃗.value.render(nest2uniq(s⃗̇, nˢ, in_place=True),
                           pad=hp.UNSEEN)
    else:
        if isinstance(s⃗, Qty):
            s⃗ = s⃗.value
        s⃗ = s⃗[s⃗̇]
    del s⃗̇

    img = unmask_partial_image(s⃗, m, shape)
    del m
    ax.imshow(img, vmin=vmin, vmax=vmax, origin='lower', cmap=cmap,
              extent=ax.proj.get_extent(), zorder=zorder)


def add_disks(disks, rgba, ax=None, sigma=1, degrees=True, zorder=1,
              shape=None, θϕm=None):
    """
    disks : Iterable[Tuple[float, float, float]]
        A list of ``(ra, dec, σ)`` tuples where ``ra`` is the right ascension,
        ``dec`` the declination, and ``σ`` the radius of the disk to be
        plotted.
    rgba : Tuple[float, float, float, float]
        A tuple of red, green, blue, and alpha values between 0 and 1 to use
        for the disk color. If the disks overlap, the alpha value will
        gradually increase to reflect their overlap.
    ax : healpy.projaxes.SphericalProjAxes, optional
        The ``healpy`` axes instance on which to plot ``s⃗``. Uses current axes
        if not provided, i.e. ``plt.gca()``.
    sigma : float, optional
        The size of the error region in terms of disk σ.
    degrees : bool, optional
        Whether the ``disks`` angles are defined in ``degrees``. If ``False``,
        radians are assumed instead.
    zorder : float, optional
        The z-order of the image. Make it higher to place it over
        previously-plotted images.
    shape : Tuple[int, int], optional
        The shape of the projected image returned by one of the
        ``healpy.visufunc`` plotters when ``return_projected_map=True``, i.e.
        the result of a call to ``ax.projmap``. This determines the pixel
        locations.
    θϕm : Tuple[array, array, array]
        The return result of ``get_ax_angles``. If you are making multiple
        plots, it might be slightly faster to reuse this value.
    """
    import numpy as np
    import healpy as hp
    from matplotlib import pyplot as plt

    *rgb, a = rgba
    rgbhex = Rgba(*rgb, a).to_hex()[:-2]
    cmap = monochrome_opacity_colormap(rgbhex, rgb)     # set color
    ax = ax or plt.gca()
    shape = shape or get_hp_ax_img_shape(ax)
    d = np.float32 if len(disks) < 16777216 else np.float64

    # get angles of the disks as well as the max dot-product via σ
    ϕ̃, θ̃, σ = (np.radians(disks) if degrees else np.array(disks, copy=True)).T
    np.multiply(σ, sigma, out=σ)                        # scale disk by sigma
    np.cos(σ, out=σ)                                    # min dot product
    np.subtract(np.radians(90), θ̃, out=θ̃)               # ra/dec (rad) -> ϕ̃/θ̃
    disks = hp.ang2vec(θ̃, ϕ̃).T                          # rows are r̂ᵢ = x, y, z

    # get the angles of the plot pixels; overwrite θ/ϕ
    θ, ϕ, m = θϕm or get_ax_angles(ax, shape=shape)     # θ/ϕ of pixels
    plot = hp.ang2vec(θ, ϕ)                             # cols are x, y, z
    del θ, ϕ

    # take dot products and sum overlaps of each pixel
    s⃗ = (plot@disks>σ).sum(axis=1, dtype=d)             # slow for many disks
    del plot, disks
    np.power(1-a, s⃗, out=s⃗)                             # compute layered alpha
    np.subtract(1, s⃗, out=s⃗)

    img = unmask_partial_image(s⃗, m, shape)
    del m, s⃗
    ax.imshow(img, vmin=0, vmax=1, origin='lower', cmap=cmap,
              extent=ax.proj.get_extent(), zorder=zorder)


def get_ax_nside(ax, shape=None):
    shape = shape or get_hp_ax_img_shape(ax)
    return resol2nside(ax.proj.get_fov()/max(shape), degrees=False)


def get_ax_angles(ax, shape=None, lonlat=False):
    import numpy as np
    from healpy.projaxes import (
        HpxMollweideAxes,
    )

    shape = shape or get_hp_ax_img_shape(ax)
    p = ax.proj
    if isinstance(ax, HpxMollweideAxes):
        xm, ym = p.ij2xy()
    else:
        i, j = [a.ravel() for a in np.indices(shape, dtype=np.int16)]
        xm, ym = p.ij2xy(i, j)
    if any(isinstance(a, np.ma.MaskedArray) for a in (xm, ym)):
        assert np.all(xm.mask==ym.mask)
        m = ~xm.mask
        if np.iterable(m):
            return (*p.xy2ang(xm.data[m], ym.data[m], lonlat=lonlat), m)
        elif m:
            xm, ym = xm.data, ym.data
        else:
            raise ValueError("Full skymap masked?")
    return (*p.xy2ang(xm, ym, lonlat=lonlat), None)


def unmask_partial_image(partial_image, mask, shape):
    import numpy as np
    import healpy as hp

    if mask is not None:
        #img = np.zeros(mask.shape, dtype=partial_image.dtype)
        img = np.full(mask.shape, -np.inf, dtype=partial_image.dtype)
        img[mask] = partial_image
    else:
        img = partial_image
    return hp.ma(img.reshape(shape), copy=False)


def label_graticule(ax, ras, decs, δr, δd, rot, ra_window=None, dec_window=None,
                    **kwargs):
    """
    Add text labels to a graticule.
    """
    import numpy as np
    from matplotlib.transforms import ScaledTranslation
    from healpy.projaxes import (
        HpxMollweideAxes,
        HpxGnomonicAxes,
        HpxOrthographicAxes,
        HpxCartesianAxes,
        HpxAzimuthalAxes,
    )

    # https://matplotlib.org/3.1.1/tutorials/advanced/transforms_tutorial.html
    dectrans = ax.transData + ScaledTranslation(DEC_X_OFFSET, DEC_Y_OFFSET,
                                                ax.figure.dpi_scale_trans)
    ratrans = ax.transData + ScaledTranslation(RA_X_OFFSET, RA_Y_OFFSET,
                                               ax.figure.dpi_scale_trans)
    # common kwargs
    ra_kwargs = {
        'path_effects': [outline_effect()],
        'color': GRATICULE_LABEL_COLOR,
        'lonlat': True,
    }
    dec_kwargs = ra_kwargs.copy()

    # base ra kwargs
    ra_kwargs['fontsize'] = MERIDIAN_FONT_SIZE
    ra_kwargs['horizontalalignment'] = 'center'
    ra_kwargs['verticalalignment'] = 'bottom'

    # base dec kwargs
    dec_kwargs['horizontalalignment'] = 'right'
    dec_kwargs['verticalalignment'] = 'center'

    ra4dec = np.full_like(decs, LEFT_SHIFT_BASE)
    dec4dec = decs
    ra4ra = ras
    dec4ra = np.zeros_like(ras)

    if isinstance(ax, HpxMollweideAxes):
        # curve dec labels away from leftmost meridian
        ra4dec = (rot[0]+165)%360 + (decs*LEFT_SHIFT_COEFF)**2
    elif isinstance(ax, (HpxCartesianAxes, HpxAzimuthalAxes)):
        [xax_ra, xax_dec], [yax_ra, yax_dec] = kwargs['ac']
        yax_pos = (yax_dec > np.roll(yax_dec, 1))[1:]
        yax = np.floor(yax_dec/δd)
        yax_change = (yax != np.roll(yax, 1))[1:]
        yax_loc = (yax_change&yax_pos)|np.roll(yax_change&(~yax_pos), -1)
        decs = δd*yax[1:][yax_loc]
        ra4dec = yax_ra[1:][yax_loc]
        dec4dec = yax_dec[1:][yax_loc]
        if isinstance(ax, HpxAzimuthalAxes):
            xax = xax_ra//δr
            xax_loc = (xax != np.roll(xax, -1))[:-1]
            ras = δr*xax[:-1][xax_loc]
            ra4ra = xax_ra[:-1][xax_loc]
            dec4ra = xax_dec[:-1][xax_loc]
            ratrans = ax.transData
            dectrans = ax.transData
            #if kwargs['lat'] >= 0:
            ra_kwargs['verticalalignment'] = 'top'
        else:
            dectrans = ax.transData + ScaledTranslation(
                DEC_X_OFFSET/3, DEC_Y_OFFSET, ax.figure.dpi_scale_trans)
    elif isinstance(ax, HpxOrthographicAxes):
        # ra4dec -= 100
        # ra4dec += (decs*LEFT_SHIFT_COEFF)**2
        decs = dec4dec = ra4dec = np.array([])
        ra_kwargs['rotation'] = 'vertical'
    elif isinstance(ax, HpxGnomonicAxes):
        return
    else:
        raise ValueError(f"Unrecognized axes type: {type(ax)}")

    _proj_grat_label(ax, ra4dec, dec4dec, decs, dectrans, δd, **dec_kwargs)
    _proj_grat_label(ax, ra4ra, dec4ra, ras, ratrans, δr, **ra_kwargs)


def graticule(ax, **kwargs):
    """
    Add labels and a graticule to the given axes ``ax`` (with placement
    tailored to the plot type and window). Pass this to ``visufunc``
    plotter functions.
    """
    import numpy as np
    from healpy.projaxes import (
        HpxMollweideAxes,
        HpxGnomonicAxes,
        HpxOrthographicAxes,
        HpxCartesianAxes,
        HpxAzimuthalAxes,
    )

    d = (-90, 90)       # declination range
    r = (-180, 180)     # right-ascension range
    kw = {}             # keyword args passed to label plotter

    if isinstance(ax, HpxMollweideAxes):
        r = (0, 360)
    elif isinstance(ax, HpxGnomonicAxes):
        Δθ = ax.proj.arrayinfo['reso']/2/60
        xsize = ax.proj.arrayinfo['xsize']
        ysize = ax.proj.arrayinfo['ysize']
        lon, lat = np.around(ax.proj.get_center(lonlat=True),
                             ax._coordprec)
        d = (lat-ysize*Δθ, lat+ysize*Δθ)
        r = (lon-xsize*Δθ, lon+xsize*Δθ)
    elif isinstance(ax, (HpxCartesianAxes, HpxAzimuthalAxes)):
        axinds = [np.arange(ax.proj.arrayinfo[i]) for i in ['xsize', 'ysize']]
        xaxi = 0
        kw['lon'], kw['lat'] = ax.proj.get_center(lonlat=True)
        if isinstance(ax, HpxAzimuthalAxes):
            if kw['lat'] < 0:
                xaxi = axinds[1][-1]
        ac = [ax.proj.xy2ang(ax.proj.ij2xy(*ii), lonlat=True)
              for ii in ((np.full_like(axinds[0], xaxi), axinds[0]),
                         (axinds[1], np.full_like(axinds[1], 1)))]
        ac[0][0] %= 360
        ac[1][0] %= 360
        kw['ac'] = ac
        if isinstance(ax, HpxAzimuthalAxes):
            r, d = [(a.min(), a.max()) for a in (ac[0][0], ac[1][1])]
        else:
            r, d = [ax.proj.arrayinfo[i] for i in ('lonra', 'latra')]
            r = r+kw['lon']  # DO NOT DO IN-PLACE; mutates ax state
    elif isinstance(ax, HpxOrthographicAxes):
        r = r+ax.proj.get_center(lonlat=True)[0]+90  # DO NOT DO IN-PLACE

    δd = [δ for δ in DELTA_PARALLELS if δ<(d[1]-d[0])/MIN_GRAT][0]
    δr = [δ for δ in DELTA_MERIDIANS if δ<(r[1]-r[0])/MIN_GRAT][0]
    di = δd*(d[0]//δd)  # rounded initial dec window
    ri = δr*(r[0]//δr)  # rounded initial ra window
    decs = di+δd*np.arange(1, np.ceil((d[1]-di)/δd))
    ras = (ri+δr*np.arange(1, np.ceil((r[1]-ri)/δr)))%360  # ras -> (0, 360)
    decs = decs[(decs>-90)&(decs<90)]  # keep decs in range
    ax.graticule(δd, δr, color=GRATICULE_COLOR)

    label_graticule(ax, ras, decs, δr, δd, kwargs['rot'], ra_window=r,
                    dec_window=d, **kw)


def visufunc(**default_kwargs):
    """
    A decorator that adds LLAMA styling and overlaid scatterplots to
    ``healpy.visufunc`` plotting functions. Automatically calls the
    ``hp.visufunc`` function of the same name as ``func``, leaving the wrapped
    function body to do apply post-plot styling defaults specific to that
    plotter. The current figure is passed to the wrapped function as the
    keyword argument ``fig``.

    If only a single point source is specified in ``*scatter``, then the
    wrapped plotter will use the location of that point as the default ``rot``
    value. If ``xsize`` is also provided to the decorator, then
    ``reso`` will also take a default value that fits the sky area contained
    in ``2*max(sigmas)``.

    Provide additional ``**default_kwargs`` to add to the wrapper if needed.
    """

    def decorator(func):
        "The actual visufunc decorator with ``graticule`` as a closure."
        preamble = dedent(f"""
            Make a ``healpy.visufunc.{func.__name__}`` plot using LLAMA default
            styling with point sources optionally plotted on the sky.
        """)
        doc = dedent(f"""
            Parameters
            ----------
            s⃗ : array-like or astropy.units.Quantity
                The full HEALPix skymap in NEST or RING ordering to plot.
            *scatter : PointsTuple
                Point-sources to add as a scatterplot over ``s⃗``. Each
                collection of points consists of a tuple of the form
                ``(points, rgba, marker, name)``, where each tuple is meant to
                represent a specific type of point-source; for example, if
                plotting both neutrino and catalog point sources, you'd provide
                a ``tuple`` for each. ``points`` is a list of point source
                tuples of the form ``(ra, dec)`` or ``(ra, dec, σ)``, where
                ``ra`` is right ascension, ``dec`` declination, and ``σ``,
                optionally, the radial size of the point-source, all expressed
                **in degrees**.  For each point where ``σ`` is specified, a
                translucent disk of radius sigma in the color specified by RGBA
                tuple ``rgba``, where each value is in the range ``[0, 1]``,
                will be plotted over the skymap. The central location of each
                point source will be indicated by a marker, specified
                (optionally using LaTeX) for a collection of points as the
                string ``marker``. The name of this collection of sources is
                given as the string ``name``.
            unit : str or astropy.units.Unit, optional
                The unit of the skymap values. If ``s⃗`` is an
                ``astropy.units.Quantity`` instance and ``unit`` is not
                provided, it will be read from ``s⃗``.
            cmap : matplotlib.colors.Colormap
                The colormap for the plotted skymap. If not defined, the
                default LLAMA colormap returned by ``default_cmap`` is used.
            rot : Tuple[float, float, float], optional
                The rotation to apply to the skymap; define the central
                longitude, central latitude, and rotation about center. The
                default LIGO skymap orientation is used by default, *unless* a
                single point is specified in ``*scatter``; in this case, the
                plot is centered at that point by default.
            sigmas : Union[float, List[float]]
                Plot error region disks for point sources of size ``sigmas``.
                Provide a single sigma or a list of sigmas for multiple layers
                of disks, or omit error region disks from the plot by setting
                to an empty list.
            graticule : Callable, optional
                A function called on a figure for rendering a graticule on the
                given figures.
            **kwargs
                Keyword args to pass to ``healpy.visufunc.{func.__name__}``.

            Returns
            -------
            fig : matplotlib.axes.Axes
                The axes instance that was plotted to.

            See Also
            --------
            healpy.visufunc.{func.__name__}
        """)

        @functools.wraps(func)
        def wrapper(s⃗, *scatter, sigmas=1, graticule=graticule, **kwargs):
            import numpy as np
            from healpy import visufunc as vf, UNSEEN
            from matplotlib import pyplot as plt

            s⃗, scatter, kwargs = visufunc_defaults(s⃗, default_kwargs, *scatter,
                                                   sigmas=sigmas,
                                                   graticule=graticule,
                                                   **kwargs)
            sigmas = kwargs.pop('sigmas')
            graticule = kwargs.pop('graticule')
            vmin, vmax = kwargs['min'], kwargs['max']
            figsize, dpi = kwargs.pop('figsize'), kwargs.pop('dpi')

            plt.rcParams['font.size'] = FONT_SIZE
            pltr = functools.partial(vars(vf)[func.__name__], **kwargs)
            disks = [(c, [p[:3] for p in p⃗ if p[2]]) for p⃗, c, *_ in scatter]

            if not kwargs.get('hold'):
                plt.figure(figsize=figsize, dpi=dpi, facecolor='w',
                           edgecolor='k')
                kwargs['hold'] = True

            pltr(np.full((12,), UNSEEN), min=vmin, max=vmax)
            ax = plt.gca()                          # plotted figure
            shape = get_hp_ax_img_shape(ax)
            θϕm = get_ax_angles(ax, shape=shape)
            layer_plot(s⃗, ax, vmin, vmax, cmap=kwargs['cmap'], θϕm=θϕm,
                       zorder=0.5, nest=kwargs['nest'])

            for σ in sigmas:                            # plot source areas
                for c, p⃗ in disks:                      #   w/ σ defined
                    add_disks(p⃗, c, ax, sigma=σ, degrees=True, zorder=1,
                              θϕm=θϕm)
            del θϕm

            if graticule is not None:
                graticule(ax, **kwargs)                 # graticule (if given)

            for p⃗, c, mrk, _ in scatter:                # point source centers
                *p, _, l = zip(*[[*p]+[None]*(4-len(p)) for p in p⃗])
                if mrk is not None:
                    *hs, v = rgb_to_hsv(*c[:3])
                    add_scatterplot(ax, *p, marker=mrk, pt_labels=l,
                                    color=hsv_to_rgb(*hs, v*0.4))

            func(s⃗, *scatter, **kwargs)                 # finalize in wrapped

            return ax

        wrapper.__doc__ = f"{preamble}\n{dedent(wrapper.__doc__ or '')}\n{doc}"
        wrapper.__doc__ = re.sub(r'\n\n\n*', '\n\n', wrapper.__doc__)

        return wrapper

    return decorator


# pylint: disable=unused-argument,missing-docstring
@visufunc(xsize=800)
def azeqview(s⃗, *scatter, unit=None, cmap=None, rot=DEFAULT_ROT, **kwargs):
    pass


# pylint: disable=unused-argument,missing-docstring
@visufunc()
def cartview(s⃗, *scatter, unit=None, cmap=None, rot=DEFAULT_ROT, **kwargs):
    pass


# pylint: disable=unused-argument,missing-docstring
@visufunc(xsize=200)
def gnomview(s⃗, *scatter, unit=None, cmap=None, rot=DEFAULT_ROT, **kwargs):
    pass


# pylint: disable=unused-argument,missing-docstring
@visufunc()
def orthview(s⃗, *scatter, unit=None, cmap=None, rot=DEFAULT_ROT, **kwargs):
    pass


# pylint: disable=unused-argument,missing-docstring
@visufunc()
def mollview(s⃗, *scatter, unit=None, cmap=None, rot=DEFAULT_ROT, **kwargs):
    pass


def multiplot(
        *s⃗ₗ,
        transform: Optional[Callable] = None,
        plotters: List[Union[Callable, str]] = (mollview,),
        scatters: Optional[List[List[PointsTuple]]] = None,
        # fig args
        subplot_height: float = HEIGHT,
        suptitle: Optional[str] = None,
        dpi: float = DPI,
        # fig.add_gridspec args
        ncols: int = NCOLS,
        widths: Optional[List[float]] = None,
        hspace: float = HSPACE,
        wspace: float = WSPACE,
        # plotter args
        subplot_kwargs: Optional[List[Optional[List[dict]]]] = None,
        **kwargs
):
    """
    Make a grid plot of multiple skymaps (optionally with scatterplots for
    each).

    Parameters
    ----------
    *s⃗ₗ : List[np.ndarray]
        A list of skymaps to plot. Can either be a single-resolution full-sky
        HEALPix array or another format, though in the latter case a suitable
        ``transform`` function must be provided.
    transform : Callable, optional
        A function taking a
        single element of ``s⃗ₗ`` and transforming it into a single-resolution
        full-sky HEALPix array. Only necessary if the elements of ``s⃗ₗ`` are
        not already in such a format. ``transform`` is called on each
        skymap directly before plotting, making it an effective way to manage
        memory usage when plotting compressed skymap formats.
    plotters : List[Union[Callable, str]], optional
        A list of plotters from this module to use for each plot. If multiple
        plotters are specified, they will be plotted alongside each other; if
        this makes the figure too wide, change the number of columns in the
        grid with ``ncols``. Plotters contained in this module can also be
        specified using their function names.
    scatters : List[List[PointsTuple]], optional
        Scatterplots to use, one list for each skymap containing the sets of
        points to plot for that skymap. See plotter dosctrings for ``scatter``,
        e.g. the ``mollview`` docstring for the format of each scatterplot
        specifier. Scatterplots for a given skymap will be added to each
        plotter.
    subplot_height : float, optional
        The height of each subplot (in inches). Width is automatically
        determined. Overall figure height will be this times ``ncols``.
    suptitle : str, optional
        If provided, add a title for the entire ``multiplot`` at the top. You
        can manually call ``suptitle`` on the returned figure if you need to
        customize beyond the default title styling.
    dpi : float, optional
        DPI of the output figure. Modify this to change the figure's
        resolution.
    ncols : int, optional
        How many columns of *skymaps* to include in the grid. **NB: All
        subplots for a single skymap are counted as a single column** (in
        contrast to ``matplotlib.gridspec.GridSpec``, which counts each subplot
        in a single column). The number of rows is determined automatically
        from the number of skymaps provided combined with ``ncols``.
    widths : List[float], optional
        Ratios of widths of the plots in ``plotters`` as a multiple of
        ``subplot_height``. Must have the same length as ``plotters``.
    hspace : float, optional
        How much vertical space (height) to reserve between subplots.
    wspace : float, optional
        How much horizontal space (width) to reserve between subplots.
    subplot_kwargs : List[Optional[List[dict]]], optional
        Lists of keyword argument dictionaries that will be used for each
        subplot. The first index specifies the plotter, and the second index
        specifies the skymap from ``s⃗ₗ``. This behavior allows you to easily
        specify lists of keyword arguments for specific plotters (since often)
        only one of the ``plotters`` requires skymap-specific parameters).
        **NB: These subplot-specific keyword arguments take precedence over**
        ``**kwargs`` **for their respective subplots.**
    **kwargs
        Keyword arguments applied to all plotters. **NB:** ``hold=True`` **is
        set for all subplots by default since a new figure is always created.**

    Returns
    -------
    fig : matplotlib.figure.Figure
        A new ``matplotlib`` figure containing the specified subplots.

    Raises
    ------
    ValueError
        If ``scatters`` is provided and is ill-formatted or not of the same
        length as ``s⃗ₗ``; if ``widths`` is included and is not of the
        same length as ``plotters``; if ``subplot_kwargs`` is included
        and is not of the same length as ``plotters``, or if its elements
        are neither ``None`` nor lists of the same length as ``s⃗ₗ``; or if one
        of the ``plotters`` is specified as a string but cannot be found in
        this module.

    See Also
    --------
    mollview
    orthview
    gnomview
    cartview
    azeqview
    """
    from math import ceil
    from matplotlib.pyplot import figure

    nᵖ = len(plotters)  # number of plotters per skymap
    nᶜ = len(s⃗ₗ)       # number of cells, i.e. skymaps
    nˢ = nᶜ*nᵖ          # number of true subplots in fig
    nʳ = nᵖ*ncols       # number of true subplots per row
    widths = widths or [WIDTH/HEIGHT]*nᵖ
    scatters = scatters or [[]]*nᶜ
    subplot_kwargs = subplot_kwargs or [None]*nᵖ
    plotters = [globals()[p] if isinstance(p, str) else p for p in plotters]
    transform = transform or (lambda x: x)
    nrows = ceil(nᶜ/ncols)

    if len(widths) != nᵖ:
        raise ValueError(f"widths {widths} must have same len as "
                         f"plotters {plotters}")
    if len(scatters) != nᶜ:
        raise ValueError(f"scatters {scatters} must have same len as s⃗ₗ "
                         f"{s⃗ₗ}")
    if len(subplot_kwargs) != nᵖ:
        raise ValueError(f"subplot_kwargs {subplot_kwargs} must have same "
                         f"len as plotters{plotters}")
    for i in range(len(subplot_kwargs)):
        kw = subplot_kwargs[i]
        if kw is None:
            subplot_kwargs[i] = [{}]*nᶜ
        elif len(kw) !=nᶜ:
            raise ValueError(f"{i}-th element of subplot_kwargs {kw} must have"
                             f" same len as s⃗ₗ {s⃗ₗ} or else be omitted.")

    # from https://matplotlib.org/tutorials/intermediate/gridspec.html
    width_ratios = widths*ncols
    fig = figure(
        #constrained_layout=True,
        figsize=(subplot_height*sum(width_ratios), subplot_height*nrows),
        dpi=dpi,
        facecolor='w',
        edgecolor='k',
    )

    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=nᵖ*ncols,
        width_ratios=width_ratios,
        hspace=hspace,
        wspace=wspace,
        left=0,
        right=1,
        bottom=0,
        top=1,
    )

    for iᶜ, s in enumerate(s⃗ₗ):
        sᵗ = transform(s)
        for iᵖ, p in enumerate(plotters):
            kw = kwargs.copy()
            kw.update()
            fig.add_subplot(gs[divmod(iᶜ*nᵖ+iᵖ, nʳ)])
            p(sᵗ, *scatters[iᶜ], hold=True, **kw)

    if suptitle is not None:
        fig.suptitle(suptitle, y=1.1, fontsize=FONT_SIZE*1.5)

    return fig


# pylint: disable=too-many-arguments
def add_scatterplot(
        ax,
        right_ascensions,
        declinations,
        marker=DEFAULT_SCATTER_MARKER,
        color=NEUTRINO_COLOR,
        pt_labels=None,
        zorder=1000,
        **kwargs
):
    """Add a scatterplot to the current HEALpy axis. Useful for plotting
    neutrinos etc.; remaining ``kwargs`` are passed to
    ``healpy.projscatter``. Returns the current figure."""
    import numpy as np
    from matplotlib.colors import ColorConverter
    from matplotlib.transforms import ScaledTranslation
    from matplotlib import pyplot as plt

    # make the fonts bigger than the default 10pt
    plt.rcParams['font.size'] = FONT_SIZE
    events = np.array([right_ascensions, declinations])
    pt_labels = pt_labels or [None]*len(right_ascensions)
    # canonicalize input colors as RGBA
    color = ColorConverter.to_rgba(color)

    ainfo = ax.proj.arrayinfo
    if 'xsize' in ainfo and 'ysize' in ainfo:
        i, j = ax.proj.xy2ij(ax.proj.ang2xy(right_ascensions,
                                            declinations, lonlat=True))
        include = np.array((i>=0)&(i<ainfo['ysize'])&
                           (j>=0)&(j<ainfo['xsize'])).reshape((-1,))
        pt_labels = [p for p, i in zip(pt_labels, include) if i]
        events = events[:, include]


    # add an 'x' at each event location
    ax.projscatter(
        events,
        lonlat=True,
        # https://stackoverflow.com/questions/50706901
        edgecolor='white',
        linewidth=OUTLINE_STROKE,
        facecolor=color,
        rasterized=False,
        marker=marker,
        s=(2*plt.rcParams['lines.markersize']) ** 2,  # double default size
        zorder=zorder,
        **kwargs
    )
    # Make a matplotlib translation to offset text labels by a bit so that
    # they don't cover up the scatter plot markers they are labeling. See:
    # https://matplotlib.org/users/transforms_tutorial.html
    transdata = ax.transData
    ntrans = transdata + ScaledTranslation(N_X_OFFSET, N_Y_OFFSET,
                                           ax.figure.dpi_scale_trans)

    # label events
    for i, event in enumerate(events.transpose()):
        ax.projtext(
            event[0],
            event[1],
            pt_labels[i] or str(i+1),
            lonlat=True,
            color=color,
            transform=ntrans,
            zorder=zorder,
            path_effects=[outline_effect()],
        )
    return ax.figure


def visufunc_defaults(s⃗, default_kwargs, *scatter, sigmas=1,
                      graticule=graticule, **kwargs):
    """
    Get default arguments for ``visufunc`` wrapped functions.

    Parameters
    ----------
    s⃗
        Skymap to plot.
    default_kwargs
        Default keyword arguments from ``@visufunc`` wrapper.
    sigmas
        Error regions of point sources to plot.
    *scatter
        Scatterplot points.
    **kwargs
        Keyword arguments.

    Returns
    -------
    s⃗, scatter, kwargs
        Input arguments updated to defaults. All keyword arguments are folded
        into ``kwargs``.
    """
    import numpy as np
    from healpy import visufunc as UNSEEN
    from astropy.units import Quantity as Qty

    default_kwargs = default_kwargs.copy()

    if not np.iterable(sigmas):
        sigmas = [sigmas]
    kwargs['sigmas'] = sigmas
    kwargs['graticule'] = graticule

    if 'fig' in kwargs:
        raise ValueError("`fig` argument not yet supported.")

    # handle partial skymaps
    if isinstance(s⃗, AbstractPartialUniqSkymap):
        scatter = [*s⃗.point_sources, *scatter]

    if 'min' not in kwargs or 'max' not in kwargs:
        if isinstance(s⃗, AbstractPartialUniqSkymap):
            val = s⃗.value.s
        elif isinstance(s⃗, Qty):
            val = s⃗.value
        else:
            val = s⃗
        val = val[val!=UNSEEN]
        try:
            kwargs['min'] = kwargs.get('min', val.min())
            kwargs['max'] = kwargs.get('max', val.max())
        except ValueError as err:
            if not str(err).startswith('zero-size array to reduction'):
                raise
            kwargs['min'] = kwargs['max'] = 0
        del val

    # set some LLAMA default keyword arguments
    kwargs['nest'] = kwargs.get('nest', True)
    kwargs['cmap'] = kwargs.get('cmap') or default_cmap()
    kwargs['unit'] = kwargs.get('unit') or getattr(s⃗, 'unit', None)
    one_pt = len(scatter) == 1 and len(scatter[0][0]) == 1
    if kwargs.get('rot') is None:
        if one_pt:
            kwargs['rot'] = (*scatter[0][0][0][:2], 0)
            default_xsize = default_kwargs.pop('xsize', None)
            if default_xsize:
                sig = scatter[0][0][0][2]
                if sig:
                    size = kwargs.get('xsize', default_xsize)
                    size = min(kwargs.get('ysize', size), size)
                    kwargs['reso'] = 60*4*max(sigmas)*sig/size
        else:
            kwargs['rot'] = DEFAULT_ROT

    # if default figure settings are given, pass them through
    kwargs['dpi'] = kwargs.get('dpi', DPI)
    kwargs['figsize'] = kwargs.get('figsize', (WIDTH, HEIGHT))

    # apply any default kwargs for the wrapped function
    for key, value in default_kwargs.items():
        kwargs[key] = kwargs.get(key, value)

    kwargs['return_projected_map'] = False

    return s⃗, scatter, kwargs
