# pylint: disable=line-too-long,invalid-name,bad-continuation
# flake8: noqa
# (c) Stefan Countryman, 2019

"""
A partial HEALPix skymap class supporting multi-resolution HEALPix skymaps in
NUNIQ ordering.
"""

import re
import sys
import functools
import operator
import base64
from io import BytesIO
from textwrap import wrap, dedent
from collections import OrderedDict
from typing import List, Iterator, Iterable, Callable, Optional, Union, IO
from .utils import (
    uniq2nest,
    uniq2dangle,
    uniq_diadic,
    uniq_intersection,
    fill,
    render,
    reraster,
    check_valid_nuniq,
    uniq2nest_and_nside,
    uniq2nside,
    uniq2order,
    nest2ang,
    nside2pixarea,
    nside_quantile_indices,
    nside_slices,
    read_partial_skymap,
)
from .abstract import AbstractPartialUniqSkymap
from . import plotters
from .plotters import (
    DEFAULT_ROT,
    MAX_NSIDE,
    multiplot,
)
from .points import PT_META_REGEX, PT_META_KW_REGEX, PT_META_COLOR_REGEX, _vecs_for_repr_, PointsTuple

DIADIC_EXCEPTIONS = {'and': operator.and_, 'or': operator.or_,
                     'divmod': divmod}


def plot_fill(s⃗):
    """
    ``PartialUniqSkymap.fill`` a skymap either to its native max resolution or
    to the max resolution suggested by ``plotters.MAX_NSIDE``. Does nothing if
    not passed a ``PartialUniqSkymap``.
    """
    if not isinstance(s⃗, PartialUniqSkymap):
        return s⃗
    return s⃗.fill(nˢ=min(MAX_NSIDE, s⃗.n⃗ˢ().max()))


def partial_visufunc(meth):
    """
    Plot a ``PartialUniqSkymap`` using the ``healpy.visufunc`` function of the
    same name as the wrapped method ``meth``, then call ``meth`` passing it the
    plot as ``fig``. This allows ``meth`` to focus on any special handling of
    the figure after it's been called.
    """
    func = vars(plotters)[meth.__name__]
    addendum = '\n'.join(wrap("""
        This method is a wrapper around ``{plotters.__name__}.{func.__name__}``
        that uses this partial skymap's pixels.
    """))
    doc = re.sub(r'\n\nParameters', f"\n\n{addendum}\n\nParameters",
                 re.sub(r'\n(s⃗ : .*)(\n\*scatter)', lambda m: m.group(2),
                        func.__doc__, 0, re.DOTALL))

    #@functools.wraps(meth)
    #def wrapper(self, *args, nest=True, **kwargs):
    #    return func(plot_fill(self), *self.point_sources, *args, nest=True,
    #                **kwargs)
    #wrapper.__doc__ = (wrapper.__doc__ or '') + doc

    wrapped = functools.wraps(meth)(func)
    wrapped.__doc__ = (wrapped.__doc__ or '') + doc
    return wrapped


def _get_op(name):
    if name in DIADIC_EXCEPTIONS:
        return DIADIC_EXCEPTIONS[name]
    return getattr(operator, name)


def diadic_dunder(pad=None, coarse=False, post=None):
    """
    Implement diadic dunder methods like ``__add__``, ``__radd__``, etc. for
    scalar, array, and ``PartialUniqSkymap`` arguments using ``uniq_diadic``.
    ``pad`` and ``coarse`` are passed to ``uniq_diadic`` when
    performing operations between ``PartialUniqSkymap`` instances.
    If provided, run ``post`` on the result array before creating a new
    ``PartialUniqSkymap`` instance out of it (for example, to cast booleans to
    integers).
    """

    def decorator(meth):
        "The actual decorator."
        name = meth.__name__[2:-2]
        srt = lambda s, o: (s, o)
        if name.startswith('i'):
            Ω = Ωᵢ = _get_op(name)
            try:
                Ω = _get_op(name[1:])
            except AttributeError:
                pass
        elif name.startswith('r'):
            try:
                Ω = Ωᵢ = _get_op(name)
            except AttributeError:
                Ω = Ωᵢ = _get_op(name[1:])
                srt = lambda s, o: (o, s)
        else:
            Ω = Ωᵢ = _get_op(name)

        @functools.wraps(meth)
        def wrapper(s, o, pad=pad, coarse=coarse, post=post):
            import numpy as np
            from astropy.units import Quantity as Qty
            # from IPython.core.debugger import Tracer; Tracer()()

            if isinstance(o, PartialUniqSkymap):
                u⃗, s⃗ = uniq_diadic(Ω, srt(s.u⃗, o.u⃗), srt(s.s⃗, o.s⃗),
                                   pad=pad, coarse=coarse)
                pts = o.point_sources
                oname = o.name or 'PIXELS'
            elif (not np.iterable(o)) or isinstance(o, (np.ndarray, Qty)):
                s⃗ = Ωᵢ(*srt(s.s⃗, o))
                u⃗ = s.u⃗
                pts = []
                oname = 'array'
            else:
                return NotImplemented
            pts = [*s.point_sources, *pts]
            m = s.meta.copy()
            m['HISTORY'] = m.get('HISTORY', []) + [
                f'DIAD: {meth.__name__}({s.name or "PIXELS"}, {oname})']
            return PartialUniqSkymap(s⃗ if post is None else post(s⃗), u⃗,
                                     point_sources=pts, copy=False, meta=m)

        wrapper.__doc__ = dedent(f"""
            ``__{name}__`` for scalars, arrays, and `PartialUniqSkymap`
            instances. Arrays must match ``s⃗`` pixel-for-pixel. Provide keyword
            arguments ``pad`` to provide a pad value for missing pixels and/or
            ``coarse`` to specify whether the resulting skymap should take the
            higher or lower resolution in overlapping areas (default coarse
            value: {coarse})
        """)

        return wrapper

    return decorator


def bool_to_uint8(s):
    "Convert a boolean value to ``np.uint8``."
    import numpy as np

    return np.array(s, dtype=np.uint8)


class PartialUniqSkymap(AbstractPartialUniqSkymap):
    """
    A HEALPix skymap object residing in memory with NUNIQ ordering. Only
    a subset of the full sky. You can index into a ``PartialUniqSkymap`` with
    NUNIQ indices to get a skymap with the same shape (optionally padding
    missing values with a second index argument). You can also use index
    notation to set pixel values at the specified NUNIQ index locations.
    """

    def __init__(self, s⃗, u⃗, copy=False, name=None, point_sources=None,
                 meta=None, empty=None, compress=False):
        """
        Initialize a skymap with the pixel values and NUNIQ indices used.

        Parameters
        ----------
        s⃗ : array-like
            Pixel values. Must be numeric.
        u⃗ : array-like
            NUNIQ indices corresponding to pixels in s⃗.
        copy : bool, optional
            Whether to make copies of the input arrays.
        name : str, optional
            The name of the skymap column. Used as the pixel column name when
            saving ``to_table``.
        point_sources : List[PointsTuple], optional
            If this skymap is associated with a list of point sources, you can
            provide it as an argument. These point sources will be included in
            data products for this skymap as well as plots.
        meta : OrderedDict, optional
            Metadata for this skymap. Used when saving to file. If this skymap
            was loaded from a file, this field will contain the metadata from
            that file. ``point_sources`` are removed from metadata before
            storing it in a ``PartialUniqSkymap``. ``PIXTYPE``, ``ORDERING``,
            and ``PARTIAL`` are set automatically.
        empty : scalar, optional
            Pixels with this value are interpreted as being empty and are
            discarded from the skymap on initialization to save storage space.
            This requires reindexing into the input arguments and therefore
            implies ``copy=True``. For example, set this to ``healpy.UNSEEN``
            to automatically discard pixels not included in a standard full-sky
            ``healpy`` skymap.

        Raises
        ------
        ValueError
            If ``s⃗`` is not a numeric data type.
        """
        import numpy as np
        from astropy.units import Quantity as Qty
        from astropy.table.column import Column as Col

        if len(s⃗) != len(u⃗):
            raise ValueError(f"Must have same lengths: s⃗={s⃗}, u⃗={u⃗}")
        self.name = name
        if empty is None:
            self.s⃗ = np.array(s⃗, copy=copy)
            self.u⃗ = np.array(u⃗, copy=copy)
        else:
            s⃗̇ = s⃗ == empty
            self.s⃗ = np.array(s⃗, copy=False)[s⃗̇]
            self.u⃗ = np.array(u⃗, copy=False)[s⃗̇]
        check_valid_nuniq(self.u⃗)
        if not np.issubdtype(self.s⃗.dtype, np.number):
            raise ValueError(f"`s⃗` must be numeric. got: {s⃗}")

        # provide point sources and deduplicate
        self.point_sources = PointsTuple.dedup(*(point_sources or []))

        meta = meta or {}
        newmeta = OrderedDict()
        for k, v in meta.items():
            if not (PT_META_REGEX.match(k) or
                    PT_META_KW_REGEX.match(k) or
                    PT_META_COLOR_REGEX.match(k)):
                newmeta[k] = v
        newmeta['PIXTYPE'] = 'HEALPIX'
        newmeta['ORDERING'] = 'NUNIQ'
        newmeta['PARTIAL'] = True
        self.meta = newmeta

        if isinstance(s⃗, (Qty, Col)):              # preserve astropy unit
            self.s⃗ = Qty(self.s⃗, s⃗.unit, copy=False)

    def n⃗ˢ(self, as_skymap=False, copy=False, **kwargs):
        """
        Pixel NSIDE values. If ``as_skymap=True``, return as a
        ``PartialUniqSkymap`` instance (with ``**kwargs`` passed to init).
        """
        n = uniq2nside(self.u⃗)
        if as_skymap:
            u⃗ = np.array(self.u⃗, copy=True) if copy else self.u⃗
            m = self.meta.copy()
            m['HISTORY'] = m.get('HISTORY', []) + ['Take HEALPix NSIDE.']
            return PartialUniqSkymap(n, u⃗, copy=False, name='NSIDE', meta=m,
                                     point_sources=self.point_sources,
                                     **kwargs)
        return n

    def o⃗(self, as_skymap=False, copy=False, **kwargs):
        """
        HEALPix order values. If ``as_skymap=True``, return as a
        ``PartialUniqSkymap`` instance (with ``**kwargs`` passed to init).
        """
        o = uniq2order(self.u⃗)
        if as_skymap:
            u⃗ = np.array(self.u⃗, copy=True) if copy else self.u⃗
            m = self.meta.copy()
            m['HISTORY'] = m.get('HISTORY', []) + ['Take HEALPix order.']
            return PartialUniqSkymap(o, u⃗, copy=False, name='ORDER', meta=m,
                                     point_sources=self.point_sources,
                                     **kwargs)
        return o

    def astype(self, dtype, copy=True, **kwargs):
        """
        Return a new ``PartialUniqSkymap`` with the data-type of ``s⃗`` set to
        ``dtype``. If ``copy=True``, always make sure both ``u⃗`` and ``s⃗`` are
        copies of the original data in the new array. Otherwise, re-use ``u⃗``
        and (if possible given the provided ``dtype`` and ``**kwargs``) ``s⃗``.
        ``copy`` and ``**kwargs`` are passed on to ``s⃗.astype`` to make the
        conversion.
        """
        return PartialUniqSkymap(self.s⃗.astype(dtype, copy=copy, **kwargs),
                                 self.u⃗, name=self.name, meta=self.meta,
                                 point_sources=self.point_sources)

    def to(self, *args, **kwargs):
        """
        Convert the units of this skymap's pixels (if they are stored as
        an ``astropy.units.Quantity`` instance).

        Parameters
        ----------
        *args, **kwargs
            Passed on to ``astropy.units.Quantity.to``.

        Raises
        ------
        TypeError
            If ``self.s⃗`` is not a ``Quantity``.
        """
        from astropy.units import Quantity

        if not isinstance(self.s⃗, Quantity):
            raise TypeError("Can only convert dimensions of a ``Quantity``")
        return PartialUniqSkymap(self.s⃗.to(*args, **kwargs), self.u⃗,
                                 copy=False, name=self.name,
                                 meta=self.meta,
                                 point_sources=self.point_sources)

    def compress(self, ϵ, cmp=lambda _m, δ, ϵ: δ < ϵ, stype=None, utype=None):
        """
        Eliminate redundant pixels with ``utils.uniq_minimize`` and store
        indices ``u⃗`` in the smallest integer size that represents all values.

        Parameters
        ----------
        ϵ : float or astropy.units.Quantity
            The maximum difference between adjacent pixels under which they are
            considered equal and combined into a single pixel. You can modify
            its meaning with ``cmp``.
        cmp : Callable[[array, array, Union[float, Quantity]], bool], optional
            A function taking the minimum value of a group of pixels as well as
            ``max-min`` as well as ``ϵ`` and returning a boolean indicating
            whether the pixels are the same. By default, simply checks whether
            ``max-min`` is less than ``ϵ``. Must be broadcastable to an array
            of ``min`` and ``max-min`` values, returning a boolean ``array``
            result.
        stype : type, optional
            If provided, store ``s⃗`` as this type. Defaults to ``s⃗.dtype``.
        utype : type, optional
            If provided, store ``u⃗`` as this type. Defaults to the smallest
            ``np.int`` type required to store all values of ``u⃗``.

        Returns
        -------
        compressed : PartialUniqSkymap
            A compressed version of this skymap.
        """
        raise NotImplementedError("Not yet implemented.")

    def sort(self, copy=True):
        """
        Sort this skymap by UNIQ indices ``u⃗`` (sorting ``s⃗`` as well, of
        course). If ``copy=True``, copy ``u⃗`` and ``s⃗`` and return a new
        ``PartialUniqSkymap``; otherwise, sort them in-place and return this
        ``PartialUniqSkymap``.
        """
        u⃗̇ = self.u⃗.argsort()
        if not copy:
            self.u⃗[:] = self.u⃗[u⃗̇]
            self.s⃗[:] = self.u⃗[u⃗̇]
            return self
        return PartialUniqSkymap(self.s⃗[u⃗̇], self.u⃗[u⃗̇], name=self.name,
                                 meta=self.meta, copy=True,
                                 point_sources=self.point_sources)

    @property
    def value(self):
        """
        Get a dimensionless view of this skymap (no effect if ``s⃗`` is not an
        ``astropy.units.Quantity``).
        """
        from astropy.units import Quantity

        s⃗ = self.s⃗.value if isinstance(self.s⃗, Quantity) else self.s⃗
        return PartialUniqSkymap(s⃗, self.u⃗, copy=False, name=self.name,
                                 meta=self.meta,
                                 point_sources=self.point_sources)

    def to_table(self, name=None, uname='UNIQ'):
        """
        Return a new ``astropy.table.Table`` whose ``UNIQ`` column is the NUNIQ
        indices ``n⃗ˢ`` and ``PIXELS`` (or ``self.name``, if set) column is the
        skymap pixel values ``s⃗``. Optionally override the pixel value column
        name and/or the NUNIQ column name with the ``name`` and ``uname``
        arguments respectively.
        """
        from astropy.table import Table

        name = name or self.name or 'PIXELS'
        t = Table([self.u⃗, self.s⃗], names=[uname, name], meta=self.meta)
        for pt in self.point_sources:
            t.meta.update(PointsTuple(*pt).meta_dict())
        return t

    def write(
            self,
            file: Union[IO, str],
            *args,
            strategy: str = 'basic',
            **kwargs
    ):
        """
        Write a ``PartialUniqSkymap`` instance to file using the specified
        ``IoStrategy``. See ``IoRegistry`` attributes for details.

        Parameters
        ----------
        file : file or str
            The file object or filename to write to.
        *args
            Arguments to pass on to ``IoRegistry``.
        strategy : str, optional
            Name of the ``IoRegistry`` strategy to use for reads/writes.
        **kwargs
            Keyword arguments to pass on to a specific ``IoStrategy``. See
            ``IoRegistry`` properties for details.

        See Also
        --------
        astropy.table.Table.write
        astropy.table.Table.write.help
        astropy.io.fits.open
        IoRegistry
        IoRegistry.basic
        """
        return getattr(IoRegistry, strategy).write(self, file, *args, **kwargs)

    def read(
            *args,
            strategy: str = 'basic',
            **kwargs
    ):
        """
        Read a ``PartialUniqSkymap`` from file using
        ``astropy.table.Table.read``. See ``IoRegistry`` attributes for
        details. When called as a bound method or passed a
        ``PartialUniqSkymap`` as the first argument, uses that skymap as a
        mask for the bound skymap and only loads those pixels.

        Parameters
        ----------
        mask : PartialUniqSkymap, optional
            Only read pixels overlapping with this mask.
        file : file or str
            The file object or filename to read from.
        *args
            Arguments to pass on to ``IoRegistry``.
        strategy : str, optional
            Name of the ``IoRegistry`` strategy to use for reads/writes.
        **kwargs
            Keyword arguments to pass on to a specific ``IoStrategy``. See
            ``IoRegistry`` properties for details.

        Returns
        -------
        m : PartialUniqSkymap
            A new ``PartialUniqSkymap`` instance with the specified data.

        Examples
        --------
        For multi-skymap fits files, you can load the full set of skymaps with
        ``hdul = astropy.io.fits.open(fitsfile)`` and then load the skymap of
        interest with ``PartialUniqSkymap.read(hdul[i])``. With fits files, you
        can memory-map the resulting skymap to save memory (at the expense of
        speed) by adding ``memmap=True`` as a keyword argument.

        See Also
        --------
        astropy.table.Table.read
        astropy.table.Table.read.help
        astropy.io.fits.open
        IoRegistry
        IoRegistry.basic
        """
        if not isinstance(args[0], PartialUniqSkymap):  # unbound method; add
            args = (None,) + args                       #   placeholder to args
        return getattr(IoRegistry, strategy).read(*args, **kwargs)

    def fill(self, nˢ=None, pad=None, as_skymap=False):
        """
        Return a full-sky *nested* HEALPix skymap at NSIDE resolution ``nˢ``.

        Parameters
        ----------
        nˢ : int
            HEALPix NSIDE value of the output map. If not provided, use the
            highest NSIDE value in this skymap's ``n⃗ˢ`` values to preserve
            detail.
        pad : float, optional
            Fill in missing values with ``pad`` (if not provided, use
            ``healpy.UNSEEN``). Preserves ``astropy.units.Unit`` of this
            skymap's pixel values (if ``s⃗`` is an ``astropy.units.Quantity``).
        as_skymap : bool, optional
            If ``True``, return a ``PartialUniqSkymap`` instance with the new
            pixelization (instead of a bare array with implicit indexing).

        Returns
        -------
        s⃗ : array or PartialUniqSkymap
            The filled-in skymap, either as an array if ``as_skymap == False``
            or as a new ``PartialUniqSkymap`` instance.

        See Also
        --------
        PartialUniqSkymap.fixed
        """
        import numpy as np

        nˢ = nˢ or uniq2nside(self.u⃗.max())
        s⃗ᵒ = fill(self.u⃗, self.s⃗, nˢ, pad=pad)
        if not as_skymap:
            return s⃗ᵒ
        m = self.meta.copy()
        m['HISTORY'] = m.get('HISTORY', []) + [f'Filled to NEST, NSIDE={nˢ}.']
        return PartialUniqSkymap(s⃗ᵒ, np.arange(4*nˢ**2, 16*nˢ**2), copy=False,
                                 meta=m, point_sources=self.point_sources)

    def quantiles(
            self,
            quantiles: Iterable[float]
    ) -> Iterator:
        """
        Get an iterator of downselected skymaps partitioned by ``quantiles``.
        For example, get the smallest sky area containing 90% of the
        probability (or whatever other intensive quantity this skymap
        represents) with ``quantiles=[0.1, 1]``.

        Parameters
        ----------
        quantiles : Iterable[float]
            Quantiles from which to select pixels. Must be in ascending order
            with values in the interval ``[0, 1]``. These will form endpoints
            for partitions of the ``skymap``. For example, ``[0.1, 0.9]`` will
            omit the lowest and highest value pixels, giving the intermediate
            pixels accounting for 80% of the integrated skymap.  Note that
            quantiles returned by this function are non-intersecting and
            half-open on the right (as with python indices), with the exception
            of ``1`` for the last pixel; for example, ``[0, 1]`` will include
            all pixels, ``[0.5, 1]`` will include the highest density pixels
            accounting for 50% of the integrated skymap value, ``[0, 0.5, 1]``
            will partition the skymap into non-intersecting sets of pixels
            accounting for the high- and low-density partitions of the skymap
            by integrated value, etc.

        Returns
        -------
        partitions : Iterator[PartialUniqSkymap]
            A generator containing non-overlapping skymaps falling into the
            specified ``quantiles``. Will have length one less than the number
            of quantiles, which form the boundaries of the partitions. Always
            an iterable, even if only two quantiles are provided; you can
            unpack a single value with, e.g., ``x, = m.quantiles(...)``.

        Raises
        ------
        ValueError
            If ``quantiles`` has length less than 2; if its values are not in
            order and contained in the interval ``[0, 1]``; if ``nside`` and
            ``skymap`` cannot be broadcast together; if any values in
            ``skymap`` are negative; or if the total integrated skymap equals
            zero, in which case quantiles are undefined.

        See Also
        --------
        hpmoc.utils.nside_quantile_indices
        """
        import numpy as np

        quantiles = np.array(quantiles, dtype=float)
        indices, norm = nside_quantile_indices(self.n⃗ˢ(), self.s⃗, quantiles)
        for i, l, u in zip(indices, quantiles[:-1], quantiles[1:]):
            m = self.meta.copy()
            m['HISTORY'] = m.get('HISTORY', []) + wrap(
                f'Downselected to [{l:.2g}, {u:.2g}] quantile '
                f'({(u-l)*100:.2g}%) of {norm:.2g} ({norm*(u-l):.2g} total)',
                70
            )
            yield PartialUniqSkymap(self.s⃗[i], self.u⃗[i], copy=False, meta=m,
                                    point_sources=self.point_sources)

    def fixed(self, nˢ=None):
        """
        Re-raster to a fixed NSIDE. Like ``fill`` but for partial skymaps.

        Parameters
        ----------
        nˢ : int
            HEALPix NSIDE value of the output map. If not provided, use the
            highest NSIDE value in this skymap's ``n⃗ˢ`` values to preserve
            detail.
        """
        nˢ = nˢ or uniq2nside(self.u⃗.max())
        u⃗ᵒ = uniq2nest(self.u⃗, nˢ, nest=True)
        s⃗ = self.reraster(u⃗ᵒ, copy=False)
        s⃗.meta['HISTORY'][-1] += f' (fixed NSIDE={nˢ})'
        return s⃗

    def __getitem__(self, idx) -> '__class__':
        """
        Get a view into this skymap with the given index applied to ``u⃗`` and
        ``s⃗``. Uses their provided ``__getitem__`` semantics, so you'll get
        e.g. a view on the same data if using a slice index.

        Note that the return value will *always* be a ``PartialUniqSkymap``,
        even if you provide a scalar index; scalar return values are made into
        length-1 lists, as if you'd asked for a slice of length 1.

        **NB: the provided indices are treated as simple array indices, NOT as
        UNIQ indices; order matters!**

        If you want to guarantee a copy on a view that you know is not copied,
        make a copy with the returned array.
        """
        import numpy as np

        m = self.meta.copy()
        repidx = repr(idx)
        msg = repidx if len(repidx) < 60 else repidx[:58]+'...'
        m['HISTORY'] = m.get('HISTORY', []) + [f'Got view: {msg}']
        args = [a if np.iterable(a) else np.array(a).reshape((1,))
                for a in (self.s⃗[idx], self.u⃗[idx])]
        return PartialUniqSkymap(*args, copy=False, meta=m,
                                 point_sources=self.point_sources)

    def __setitem__(self, idx, value):
        """
        Like ``__getitem__``, will set values using the semantics of ``s⃗``
        datatype.
        """
        self.s⃗[idx] = value

    def _iparser(self, item):
        """
        Parse the ``item`` argument for ``__getitem__``, ``__setitem__``, and
        ``__delitem__``.
        """

    def intersection(self, u⃗):
        """
        See ``utils.uniq_intersection``.
        """
        if isinstance(u⃗, AbstractPartialUniqSkymap):
            u⃗ = u⃗.u⃗
        return uniq_intersection(self.u⃗, u⃗)

    def render(self, u⃗ᵒ, pad=None):
        """
        Like ``reraster``, but ``u⃗ᵒ`` does not need to be unique. Use this to
        e.g. render a skymap to a plot. Unlike ``reraster``, will not return a
        ``PartialUniqSkymap``; instead, simply returns the pixel values
        corresponding to ``u⃗ᵒ``.

        ``u⃗ᵒ`` can also be an ``astropy.wcs.WCS`` world coordinate system, in
        which case the returned array will contain the pixel values of this
        skymap in that coordinate system (with regions outside of the
        projection set to ``np.nan``).

        Parameters
        ----------
        u⃗ᵒ: array or astropy.wcs.WCS
            The pixels to fill. If an array, treated as UNIQ indices
            (duplicates allowed); if WCS, treated as a set of pixels to render
            to.
        pad: float, optional
            A pad value to use for pixels not contained in the maps. Defaults
            to ``None``, which will raise an error if suitable values cannot
            be found for every valid pixel in ``u⃗ᵒ`` (this does not apply to
            values outside a ``WCS`` projection, which will take on ``np.nan``
            values).

        Returns
        -------
        pixels: array
            Pixel values at locations specified in ``u⃗ᵒ``. If ``u⃗ᵒ`` is a
            ``WCS`` instance, then values outside of the projection will be
            set to ``np.nan``.
        """
        import numpy as np

        return render(self.u⃗, self.s⃗, u⃗ᵒ, pad)

    def reraster(self, u⃗ᵒ, pad=None, copy=True):
        """
        Return a new ``PartialUniqSkymap`` instance with the same pixel values
        rerasterized to match the output NUNIQ indices ``u⃗ᵒ``. Fill in missing
        values in the output skymap with ``pad``. If ``pad`` is not provided
        and this skymap does not cover the full region defined in ``u⃗ᵒ``,
        raises a ``ValueError``. Preserves ``astropy.units.Unit`` of this
        skymap's pixel values (if ``s⃗`` is an ``astropy.units.Quantity``). If
        ``copy`` is ``False``, use ``u⃗ᵒ`` as the indices of the new skymap;
        otherwise, use a copy.
        """
        import numpy as np

        s⃗ᵒ = reraster(self.u⃗, self.s⃗, u⃗ᵒ, pad)
        m = self.meta.copy()
        m['HISTORY'] = m.get('HISTORY', []) + ['Rerasterized.']
        return PartialUniqSkymap(s⃗ᵒ, np.array(u⃗ᵒ, copy=copy), copy=False,
                                 meta=m, point_sources=self.point_sources)

    def Ω⃗(self):
        """
        Get the sky coordinates (right-ascension and declination) corresponding
        to each pixel in the skymap.

        Returns
        -------
        ra_dec : astropy.units.Quantity
            2D array whose first row is the right-ascension and second row is
            the declination (in degrees) of each pixel. You can get each of
            these individually with ``ra, dec = self.Ω⃗()``.
            ``self.ang()[:, i]`` corresponds to RA, Dec for ``self.s⃗[i]``.
        """
        return nest2ang(*uniq2nest_and_nside(self.u⃗))

    def A⃗(self):
        "Area per-pixel for pixels in this skymap in ``astropy.unit.sr``."
        from astropy.units import sr  # pylint: disable=no-name-in-module
        return nside2pixarea(self.n⃗ˢ(), degrees=False)*sr     # steradian

    def Δθ⃗(self, ra, dec, degrees=True):
        """
        Get distances from each pixel in this skymap to the point at
        right-ascension ``ra`` and declination ``dec``.

        Parameters
        ----------
        ra : array-like or astropy.units.Quantity
            Right-ascension of the point
        dec : array-like or astropy.units.Quantity
            Declination of the point
        degrees : bool, optional
            If ``ra`` and ``dec`` are ``astropy.units.Quantity`` instances,
            they will be automatically converted. If they are unitless scalars,
            they will be interpreted as degrees if ``degrees=True``, radians
            otherwise.

        Returns
        -------
        Δθ⃗ : astropy.units.Quantity
            The distances of each pixel in this skymap to the point at ``ra``,
            ``dec`` in degrees.

        Examples
        --------
        We should find that the distance from any pixel to the North pole is
        equal to 90 minus the declination (within some small error):

        >>> import numpy as np
        >>> from astropy.units import deg
        >>> skymap = PartialUniqSkymap(*([4+np.arange(12)]*2))
        >>> _, dec = skymap.Ω⃗()
        >>> dec
        <Quantity [ 41.8103149,  41.8103149,  41.8103149,  41.8103149,   0.       ,
                     0.       ,   0.       ,   0.       , -41.8103149, -41.8103149,
                   -41.8103149, -41.8103149] deg>
        >>> Δθ⃗ = skymap.Δθ⃗(32, 90)
        >>> Δθ⃗
        <Quantity [0.84106867, 0.84106867, 0.84106867, 0.84106867, 1.57079633,
                   1.57079633, 1.57079633, 1.57079633, 2.30052398, 2.30052398,
                   2.30052398, 2.30052398] rad>
        >>> np.all(abs(Δθ⃗+dec-90*deg).to('deg').value<1e-13)
        True

        Likewise, the distance from any pixel to the South pole should be
        equal to 90 plus the declination:

        >>> not np.around(skymap.Δθ⃗(359, -90)-dec-90*deg, 15).value.any()
        True
        """
        return uniq2dangle(self.u⃗, ra, dec, degrees=degrees)

    def unzip_orders(self):
        """
        Return a list of ``PartialUniqSkymap`` instances corresponding to the
        parts of this sky imaged and each HEALPix order. Length equals the
        maximum order of this skymap. Empty terms indicate that this skymap
        does not have pixels of the corresponding HEALPix order.
        """
        srt = self.sort()
        [[s⃗], o⃗] = nside_slices(srt.u⃗)[1:3]
        return [srt[s] for s in [slice(0, 0)]*o⃗[0]+s⃗]

    def unzip_atlas(self):
        "Return 12 sub-skymaps corresponding to the HEALPix base pixels."
        raise NotImplementedError()

    def min(self):
        "Minimum skymap value == ``self.s⃗.min()``."
        return self.s⃗.min()

    def max(self):
        "Maximum skymap value == ``self.s⃗.max()``."
        return self.s⃗.max()

    @property
    def unit(self):
        "``self.s⃗.unit``, if defined; otherwise ``None``."
        return getattr(self.s⃗, 'unit', None)

    # pylint: disable=unused-argument,missing-docstring
    @partial_visufunc
    def azeqview(self, *scatter, rot=DEFAULT_ROT, **kwargs):
        pass

    # pylint: disable=unused-argument,missing-docstring
    @partial_visufunc
    def cartview(self, *scatter, rot=DEFAULT_ROT, **kwargs):
        pass

    # pylint: disable=unused-argument,missing-docstring
    @partial_visufunc
    def gnomview(self, *scatter, rot=DEFAULT_ROT, **kwargs):
        pass

    # pylint: disable=unused-argument,missing-docstring
    @partial_visufunc
    def orthview(self, *scatter, rot=DEFAULT_ROT, **kwargs):
        pass

    # pylint: disable=unused-argument,missing-docstring
    @partial_visufunc
    def mollview(self, *scatter, rot=DEFAULT_ROT, **kwargs):
        pass

    def multiplot(*s⃗ₗ: List['__class__'], nest: bool = True, **kwargs):
        """
        Call ``plotters.multiplot`` with the default ``transform``
        suitable for a ``PartialUniqSkymap``.

        Parameters
        ----------
        *s⃗ₗ : List[Union[PartialUniqSkymap, array]]
            Skymaps to plot. Can be ``PartialUniqSkymap`` instances or
            full-sky single-resolution skymaps.
        **kwargs
            Keyword arguments passed to ``plotters.multiplot``.

        Returns
        -------
        fig : matplotlib.figure.Figure
            A new ``matplotlib`` figure containing the specified subplots.

        See Also
        --------
        plotters.multiplot
        plotters.mollview
        plotters.orthview
        plotters.gnomview
        plotters.cartview
        plotters.azeqview
        """
        # s⃗ₗ = [*s⃗ₗ]
        # s = kwargs.get('scatters', [[]]*len(s⃗ₗ))
        # kwargs['scatters'] = [s⃗.point_sources+s for s⃗, s in zip(s⃗ₗ, s)]
        return multiplot(*s⃗ₗ, **kwargs)

    def _vecs_for_repr_(self, maxlen, *vecs):
        if not vecs:
            vecs = self.u⃗, self.s⃗
        return _vecs_for_repr_(maxlen, *vecs)

    def __str__(self):
        return self.to_table().__str__()

    def __repr__(self):
        return self.to_table().__repr__()

    def _repr_html_(self):
        [u⃗, s⃗], [_, unit] = self._vecs_for_repr_(20)
        pts = self.point_sources
        unit = f'<thead><tr><th></th><th>{unit}</th></tr></thead>'
        rows = "\n".join(f'<tr><td>{u}</td><td>{s}</td></tr>'
                         for u, s in zip(u⃗, s⃗))
        meta_chunks = [
            (
                k,
                type(v).__name__,
                (
                    f"<td>{v}</td>" if not isinstance(v, (list, tuple)) else
                    "<td><table>{}</table></td>".format(
                        "\n".join(f"<tr><td>{vv}</td></tr>" for vv in v))
                )
            ) for k, v in  self.meta.items()
        ]
        meta = "\n".join(f'<tr><th>{k}</th><td><em>{t}</em></td>{c}</tr>'
                         for k, t, c in meta_chunks)
        name = self.name or 'PIXELS'
        pt_srcs = "\n".join(f'<div><h5>Point Sources {i}</h5>'
                            f'{PointsTuple(*p)._repr_html_()}</div>'
                            for i, p in enumerate(self.point_sources))

        if 'matplotlib' in sys.modules:
            import matplotlib.pyplot as plt

            img = BytesIO()
            plotters = ['cartview']
            widths = [2]
            if len(pts) == 1 and len(pts[0][0]) == 1:
                plotters.append('gnomview')
                widths.append(1)
            if 'IPython' in sys.modules:
                from IPython.utils import io

                with io.capture_output():
                    fig = self.multiplot(plotters=plotters, dpi=50,
                                         widths=widths, ncols=1, title=None)
            else:
                fig = self.multiplot(plotters=plotters, dpi=50, widths=widths,
                                     ncols=1, title=None)
            fig.savefig(img, format='png')
            plt.close(fig)
            img.seek(0)
            #b64 = "".join(base64.encodebytes(img.read()).decode().split('\n'))
            b64 = base64.b64encode(img.read()).decode()
            pd=f'''
                <div style="vertical-align: top;">
                    <h5>Plot</h5>
                    <img src="data:image/png;base64,{b64}" alt="Preview plot"
                         style="min-width: 200px; max-width: 400px;"/>
                </div>
            '''
        else:
            pd = ''

        tab = f'''
        <style>
            .partialuniq_flexbox {{
                display: flex;
                flex-wrap: wrap;
                pad: -0.5em;
            }}

            .partialuniq_flexbox > div {{
                margin: 0.5em;
            }}

            .partialuniq_flexbox_vert {{
                flex-direction: column;
            }}
        </style>
        <div class="partialuniq_flexbox">
                <div style="vertical-align: top;">
                    <h5>Skymap ({len(self.s⃗)} pixels)</h5>
                    <table>
                        <thead><tr><th>UNIQ</th><th>{name}</th></tr></thead>
                        {unit}
                        <thead><tr><th>int64</th><th>float64</th></tr></thead>
                        {rows}
                    </table>
                </div>
                <div style="vertical-align: top;">
                    <h5>Metadata</h5>
                    <table>
                        {meta}
                    </table>
                </div>
                <div class="partialuniq_flexbox partialuniq_flexbox_vert">
                    {pd}
                    {pt_srcs}
                </div>
        </div>
        '''
        return tab

    # BEGIN COMPARATOR METHODS

    @diadic_dunder(post=bool_to_uint8)
    def __eq__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder(post=bool_to_uint8)
    def __le__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder(post=bool_to_uint8)
    def __lt__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder(post=bool_to_uint8)
    def __ge__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder(post=bool_to_uint8)
    def __gt__(self, other): pass  # pylint: disable=C0321

    # BEGIN NUMERIC METHODS

    @diadic_dunder()
    def __add__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __sub__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __mul__(self, other): pass  # pylint: disable=C0321

    # @diadic_dunder()
    # def __matmul__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __truediv__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __floordiv__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __mod__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __divmod__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __pow__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __lshift__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __rshift__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __and__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __xor__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __or__(self, other): pass  # pylint: disable=C0321

    # REVERSE NUMERIC METHODS

    @diadic_dunder()
    def __radd__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __rsub__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __rmul__(self, other): pass  # pylint: disable=C0321

    # @diadic_dunder()
    # def __rmatmul__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __rtruediv__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __rfloordiv__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __rmod__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __rdivmod__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __rpow__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __rlshift__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __rrshift__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __rand__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __rxor__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __ror__(self, other): pass  # pylint: disable=C0321

    # IN-PLACE NUMERIC METHODS

    @diadic_dunder()
    def __iadd__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __isub__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __imul__(self, other): pass  # pylint: disable=C0321

    # @diadic_dunder()
    # def __imatmul__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __itruediv__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __ifloordiv__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __imod__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __ipow__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __ilshift__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __irshift__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __iand__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __ixor__(self, other): pass  # pylint: disable=C0321

    @diadic_dunder()
    def __ior__(self, other): pass  # pylint: disable=C0321


class IoStrategy:
    """
    Methods for reading and writing ``PartialUniqSkymap`` instances from/to
    file.
    """
    read: Callable
    write: Callable


class BasicIo(IoStrategy):
    """
    Read/write files saved in the default format used by ``PartialUniqSkymap``.
    """

    #FIXME add mask
    @staticmethod
    def read(
            _skymap: PartialUniqSkymap,
            file: Union[IO, str],
            *args,
            name: Optional[str] = None,
            uname: str = 'UNIQ',
            empty = None,
            **kwargs
    ) -> PartialUniqSkymap:
        """
        Read a file saved in the default format used by ``PartialUniqSkymap``.

        Parameters
        ----------
        mask : PartialUniqSkymap
            Only read in pixels overlapping with ``mask``.
        file : file or str
            The file object or filename to read from.
        name : str, optional
            The column-name of the pixel data. If not specified and if reading
            from a file with only one non-index column, that column will be
            chosen automatically.
        uname : str, optional
            The column-name of the HEALPix NUNIQ pixel data, if different from
            the default value.
        empty : scalar, optional
            ``empty`` argument to pass to ``PartialUniqSkymap`` initializer.
            **Not used when writing.**
        *args, **kwargs
            Arguments to pass on to ``astropy.table.Table.read``.

        Returns
        -------
        m : PartialUniqSkymap
            A new ``PartialUniqSkymap`` instance with the specified data.
        """
        from astropy.table import Table

        t = Table.read(file, **kwargs)
        if not name:
            c = [t for t in t.colnames if t != uname]
            if len(c) != 1:
                raise ValueError(f"Ambiguous colname; pick from {c}")
            name = c[0]
        #from IPython.core.debugger import set_trace; set_trace()
        return PartialUniqSkymap(t[name], t[uname], name=name, empty=empty,
                                 meta=t.meta,
                                 point_sources=PointsTuple.meta_read(t.meta))

    @staticmethod
    def write(
            skymap: PartialUniqSkymap,
            file: Union[IO, str],
            name: Optional[str] = None,
            uname: Optional[str] = 'UNIQ',
            *args,
            **kwargs
    ):
        """
        Read a file saved in the default format used by ``PartialUniqSkymap``.

        Parameters
        ----------
        skymap : PartialUniqSkymap
            The skymap to save.
        file : file or str
            The file object or filename to write to.
        name : str, optional
            The column-name of the pixel data in the saved file, if different
            from that specified by the skymap.
        uname : str, optional
            The column-name of the HEALPix NUNIQ pixel data in the saved file,
            if different from the default value.
        *args, **kwargs
            Arguments to pass on to ``astropy.table.Table.write``.
        """
        skymap.to_table(name=name, uname=uname).write(file, *args, **kwargs)


class LigoIo(IoStrategy):
    """
    Read/write files in the format used by LIGO/Virgo for their skymaps.
    """

    @staticmethod
    def read(
            mask: Optional[PartialUniqSkymap],
            file: Union[IO, str],
            *args,
            name: str = 'PROBDENSITY',
            memmap: bool = True,
            coarsen: int = 0,
            **kwargs
    ):
        """
        Read a file saved in the format used by LIGO/Virgo for their skymaps.

        Parameters
        ----------
        mask : PartialUniqSkymap
            Only read in pixels overlapping with ``mask``.
        file : file or str
            The file object or filename to read from.
        name : str, optional
            The column-name of the pixel data.
        memmap : bool, optional
            Whether to memory-map the input file during read. Useful when
            reading small sky areas from large files to conserve memory.
            The returned skymap will be stored as a copy in memory.
        coarsen : int, optional
            If provided, coarsen the ``mask`` by up to this many HEALPix
            orders (up to order 0) to speed up read times. This will select
            a superset of the sky region defined in ``mask``.
        *args, **kwargs
            Arguments to pass on to ``astropy.table.Table.read``.
        """
        import numpy as np

        if mask is None:
            pt = []
            m = np.arange(12)+4  # 12 base pixels = whole sky
        else:
            pt = mask.point_sources
            m = mask.u⃗
        m = np.unique(m >> (2*min(uniq2order(m.min()), coarsen)))
        p = read_partial_skymap(file, m, memmap=memmap)
        return PartialUniqSkymap(p[name], p['UNIQ'],
                                 name=name, meta=p.meta,
                                 point_sources=pt)

    def write(
            skymap: PartialUniqSkymap,
            file: Union[IO, str],
            name: Optional[str] = None,
            *args,
            **kwargs
    ):
        """
        Write a skymap to file in the format used by LIGO/Virgo for their
        skymaps. A thin wrapper around ``BasicIo.write``.
        """
        BasicIo.write(skymap, file, name=name, *args, **kwargs)


class IoRegistry:
    """
    Handle IO for ``PartialUniqSkymap`` instances.
    """
    basic = BasicIo
    ligo = LigoIo
