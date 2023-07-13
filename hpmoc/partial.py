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
A partial HEALPix skymap class supporting multi-resolution HEALPix skymaps in
NUNIQ ordering.
"""

import sys
import functools
import operator
import base64
from io import BytesIO
from textwrap import wrap, dedent
from collections import OrderedDict
from typing import (
    List,
    Iterator,
    Mapping,
    Callable,
    Literal,
    Sequence,
    Optional,
    Union,
    IO,
    Tuple,
    Any,
    Type,
    TypeVar,
    TYPE_CHECKING,
    cast,
    overload
)
from .utils import (
    max_uint_type,
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
    nest2uniq,
    uniq_minimize,
    interp_wcs,
)
from .abstract import AbstractPartialUniqSkymap
from .plot import plot, gridplot, _one_pt
from .plotters import (
    multiplot,
)
from .healpy import healpy as hp
from .points import PT_META_REGEX, PT_META_KW_REGEX, PT_META_COLOR_REGEX, _vecs_for_repr_, PointsTuple

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from astropy.visualization import wcsaxes
    from astropy.wcs.wcs import WCS
    import matplotlib.gridspec

# _depr_visufunc = deprecated(
#     reason=dedent("""
#         healpy visufunc plotting methods have been replaced with more powerful,
#         cross-platform WCSAxes-based plotters.  Use ``hpmoc.plot.plot`` and
#         ``PartialUniqSkymap.plot`` instead, for which this method is now a thin
#         wrapper. This method is retained for convenience, but its interface
#         has changed somewhat, and it may be removed or further modified without
#         warning in the future.
#     """.rstrip().strip('\n')).replace('\n', ' '),
#     version = "0.3.0"
# )

_depr_visufunc = lambda x: x

_XType = TypeVar('_XType', bound='np.generic')
_YType = TypeVar('_YType', bound='np.generic')
_OType = TypeVar('_OType', bound='np.generic')

@overload
def apply_diadic(
    op: Callable[[NDArray[_XType], NDArray[_YType]], NDArray[_OType]],
    x: 'PartialUniqSkymap[_XType]',
    y: 'PartialUniqSkymap[_YType]',
    /,
    **kwargs
) -> 'PartialUniqSkymap[_OType]': ...

@overload
def apply_diadic(
    op: Callable[[NDArray[_XType], Union[_YType, NDArray[_YType]]], NDArray[_OType]],
    x: 'PartialUniqSkymap[_XType]',
    y: Union[_XType, NDArray[_XType]],
    /,
    **kwargs
) -> 'PartialUniqSkymap[_OType]': ...

@overload
def apply_diadic(
    op: Callable[[Union[_XType, NDArray[_XType]], NDArray[_YType]], NDArray[_OType]],
    x: Union[_XType, NDArray[_XType]],
    y: 'PartialUniqSkymap[_YType]',
    /,
    **kwargs
) -> 'PartialUniqSkymap[_OType]': ...

def apply_diadic(f, x, y, /, iop=None, post=None, **kwargs):
    """
    Apply the diadic function ``f(x, y) -> o``. If both ``x`` and `y``, are 
    ``PartialUniqSkymap``s, the operation is performed efficiently on a MOC
    healpix grid either upsampling (if ``coarse=False``, default) to the
    highest input resolution in each area of the sky or downsampling to the
    lowest input resolution (if ``coarse=True``). Additional kwargs are passed
    to ``hpmoc.utils.uniq_diadic``. If one of ``x`` or ``y`` is not a skymap,
    the operation is the value array with standard numpy array/scalar arithmetic.

    The resulting ``PartialUniqSkymap`` will have the point sources from both
    ``x`` and ``y``. The metadata and history of ``x`` will be preserved,
    unless ``x`` is not a ``PartialUniqSkymap``.

    ``PartialUniqSkymap`` support all Python operators like *, +, -, and /
    already. This function should be used for calculations that require more
    than one basic arithmetic operation.

    Parameters
    ----------
    f : function (x, y) -> o
        The operation to apply
    x : PartialUniqSkymap or array-like
        The left side operand. At least one of ``x`` or ``y`` must be a
        ``PartialUniqSkymap``
    y : PartialUniqSkymap or array-like
        The right side operand. At least one of ``x`` or ``y`` must be a
        ``PartialUniqSkymap``
    iop : function (x, y) -> o, optional
        NUNIQ indices of the output skymap.
    **kwargs
        Additional kwargs are passed to ``hpmoc.utils.uniq_diadic``
    Returns
    -------
    o : PartialUniqSkymap
        The result of the operation ``f(x, y)``. If ``x`` and ``y`` are both
        ``PartialUniqSkymap``s, ``o`` may have different UNIQ indices from
        either due to resolution conformation.

    See Also
    --------
    hpmoc.utils.uniq_diadic

    """
    if iop is None:
        iop = f
    px = isinstance(x, PartialUniqSkymap)
    py = isinstance(y, PartialUniqSkymap)
    if px and py:
        uu, ss = uniq_diadic(f, (x.u, y.u), (x.s, y.s), **kwargs)
        pts = [*x.point_sources, *y.point_sources]
        xname = x.name or 'PIXELS'
        yname = y.name or 'PIXELS'
        s = x
    elif px:
        ss = f(x.s, y)
        uu = x.u
        pts = x.point_sources
        xname = x.name or 'PIXELS'
        yname = 'array'
        s = x
    elif py:
        ss = f(x, y.s)
        uu = y.u
        pts = y.point_sources
        xname = 'array'
        yname = y.name or 'PIXELS'
        s = y
    else:
        raise TypeError("At least one of `x` or `y` must be a `PartialUniqSkymap`.")

    meta = s.meta.copy()
    meta['HISTORY'] = meta.get('HISTORY', []) + [
        f'DIAD: {f.__name__}({xname}, {yname})']
    return PartialUniqSkymap(ss if post is None else post(ss), uu,
                                point_sources=pts, copy=False, meta=meta)
    

DIADIC_EXCEPTIONS = {'and': operator.and_, 'or': operator.or_,
                     'divmod': divmod}

def _get_op(name):
    if name in DIADIC_EXCEPTIONS:
        return DIADIC_EXCEPTIONS[name]
    return getattr(operator, name)


def diadic_dunder(pad=None, coarse: bool = False, post=None):
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
        def wrapper(s, o, pad=pad, coarse=coarse, post=post, **kwargs):
            try:
                return apply_diadic(Ω, *srt(s, o), iop=Ωᵢ, pad=pad, coarse=coarse, post=post, **kwargs)
            except TypeError:
                return NotImplemented

        wrapper.__doc__ = dedent(f"""
            ``__{name}__`` for scalars, arrays, and `PartialUniqSkymap`
            instances. Arrays must match ``s`` pixel-for-pixel. Provide keyword
            arguments ``pad`` to provide a pad value for missing pixels and/or
            ``coarse`` to specify whether the resulting skymap should take the
            higher or lower resolution in overlapping areas (default coarse
            value: {coarse}). For additional kwargs, see documentation of
            ``hpmoc.utils.uniq_diadic``.
        """)

        return wrapper

    return decorator


def bool_to_uint8(s):
    "Convert a boolean value to ``np.uint8``."
    import numpy as np

    return np.array(s, dtype=np.uint8)


_DType = TypeVar('_DType', covariant=True, bound='np.generic')
_Other_DType = TypeVar('_Other_DType', bound='np.generic')
class PartialUniqSkymap(AbstractPartialUniqSkymap[_DType]):
    """
    A HEALPix skymap object residing in memory with NUNIQ ordering. Only
    a subset of the full sky. You can index into a ``PartialUniqSkymap`` with
    NUNIQ indices to get a skymap with the same shape (optionally padding
    missing values with a second index argument). You can also use index
    notation to set pixel values at the specified NUNIQ index locations.
    """
    point_sources: List[PointsTuple]
    meta: Mapping[str, Any]

    def __init__(
        self,
        s: 'NDArray[_DType]',
        u: Union['NDArray[np.integer[Any]]', 'WCS'],
        copy: bool = False, name=None, point_sources=None,
        meta=None, empty=None, compress=False, interp="nearest"):
        """
        Initialize a skymap with the pixel values and NUNIQ indices used.

        Parameters
        ----------
        s : array-like
            Pixel values. Must be numeric.
        u : array-like, WCS, or None
            NUNIQ indices corresponding to pixels in s. If ``None``, assume
            ``s`` is given as a single-resolution, all-sky NEST HEALPix skymap.
            Pass ``u='RING'`` to indicate a HEALPix RING-ordered all-sky
            skymap, or ``u='NEST'`` to explicitly indicate NEST ordering
            If a ``WCS`` instance, assume ``s`` is the image described by that
            ``WCS`` and convert the image into an MOC HEALPix image using
            bilinear interpolation (pixel sizes will locally be approximately
            the same as those in the original ``WCS``).
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
        interp : int, optional
            The interpolation strategy to use. Pass ``"nearest"`` for
            nearest-neighbor or ``"bilinear"`` for bilinear. See
            ``hpmoc.utils.interp_wcs`` for a full list of valid interpolation
            strategies. *Ignored if* ``u`` *is not a* ``WCS`` *instance.*

        Raises
        ------
        ValueError
            If ``s`` is not a numeric data type; if ``u`` is not provided and
            ``s`` cannot be interpreted as an all-sky fixed-resolution NEST
            HEALPix skymap; or if ``interp`` is not a valid value.
        """
        import numpy as np
        from astropy.wcs import WCS
        from astropy.units import Quantity as Qty
        from astropy.table.column import Column as Col

        if u is None or isinstance(u, str):
            idx = np.arange(len(s))
            ns = hp.npix2nside(len(s))
            if u is not None and u.upper() == 'RING':
                idx = hp.ring2nest(ns, idx)
            u = nest2uniq(idx, ns, in_place=True)
            del idx, ns
        elif isinstance(u, WCS):
            u, s = interp_wcs(u, s, interp=interp)
        if len(s) != len(u):
            raise ValueError(f"Must have same lengths: s={s}, u={u}")
        self.name = name
        if empty is None:
            self.s = np.array(s, copy=copy)
            self.u = np.array(u, copy=copy)
        else:
            si = s == empty
            self.s = np.array(s, copy=False)[si]
            self.u = np.array(u, copy=False)[si]
        check_valid_nuniq(self.u)
        if not np.issubdtype(self.s.dtype, np.number):
            raise ValueError(f"`s` must be numeric. got: {s}")

        # provide point sources and deduplicate
        self.point_sources = PointsTuple.dedup(*(point_sources or []))

        meta = meta or {}
        newmeta = {}
        for k, v in meta.items():
            if not (PT_META_REGEX.match(k) or
                    PT_META_KW_REGEX.match(k) or
                    PT_META_COLOR_REGEX.match(k)):
                newmeta[k] = v
        newmeta['PIXTYPE'] = 'HEALPIX'
        newmeta['ORDERING'] = 'NUNIQ'
        newmeta['PARTIAL'] = True
        self.meta = newmeta

        if isinstance(s, (Qty, Col)):              # preserve astropy unit
            self.s = Qty(self.s, cast(Union[Qty, Col], s).unit, copy=False)

    def _meta_copy(self):
        """Returns a shallow copy of self.meta. Needed since .copy() isn't in
        the ABC for Mapping"""
        return {k: v for k, v in self.meta.items()}

    @overload
    def nside(self, as_skymap: Literal[True], copy: bool = False, **kwargs) -> 'PartialUniqSkymap[np.integer[Any]]': ...

    @overload
    def nside(self, as_skymap: Literal[False] = False, copy: bool = False, **kwargs) -> 'NDArray[np.integer[Any]]': ...

    def nside(self, as_skymap=False, copy=False, **kwargs): # type: ignore
        """
        Pixel NSIDE values. If ``as_skymap=True``, return as a
        ``PartialUniqSkymap`` instance (with ``**kwargs`` passed to init).
        """
        import numpy as np

        n = uniq2nside(self.u)
        if as_skymap:
            u = np.array(self.u, copy=True) if copy else self.u
            m = self._meta_copy()
            m['HISTORY'] = m.get('HISTORY', []) + ['Take HEALPix NSIDE.']
            return type(self)(n, u, copy=False, name='NSIDE', meta=m,
                              point_sources=self.point_sources,
                              **kwargs)
        return n

    @overload
    def orders(self, as_skymap: Literal[True] = True, copy: bool = False, **kwargs) -> 'PartialUniqSkymap[np.integer[Any]]': ...

    @overload
    def orders(self, as_skymap: Literal[False], copy: bool = False, **kwargs) -> 'NDArray[np.integer]': ...

    def orders(self, as_skymap=False, copy=True, **kwargs): # type: ignore
        """
        HEALPix order values. If ``as_skymap=True``, return as a
        ``PartialUniqSkymap`` instance (with ``**kwargs`` passed to init).
        """
        import numpy as np

        o = uniq2order(self.u)
        if as_skymap:
            u = np.array(self.u, copy=True) if copy else self.u
            m = self._meta_copy()
            m['HISTORY'] = m.get('HISTORY', []) + ['Take HEALPix order.']
            return type(self)(o, u, copy=False, name='ORDER', meta=m,
                              point_sources=self.point_sources,
                              **kwargs)
        return o

    def copy(self: 'PartialUniqSkymap[_DType]') -> 'PartialUniqSkymap[_DType]':
        """
        Return a copy of this instance with separate copies of its data. The
        copy can be edited without affecting the original.
        """
        return type(self)(self.s, self.u, copy=True, name=self.name,
                          point_sources=self.point_sources, meta=self.meta)

    def astype(self, dtype: Type['_Other_DType'], copy=True, **kwargs) -> 'PartialUniqSkymap[_Other_DType]':
        """
        Return a new ``PartialUniqSkymap`` with the data-type of ``s`` set to
        ``dtype``. If ``copy=True``, always make sure both ``u`` and ``s`` are
        copies of the original data in the new array. Otherwise, re-use ``u``
        and (if possible given the provided ``dtype`` and ``**kwargs``) ``s``.
        ``copy`` and ``**kwargs`` are passed on to ``s.astype`` to make the
        conversion.
        """

        # https://github.com/microsoft/pyright/issues/5412
        return type(self)(self.s.astype(dtype, copy=copy, **kwargs), # type: ignore
                          self.u, name=self.name, meta=self.meta,
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
            If ``self.s`` is not a ``Quantity``.
        """
        from astropy.units import Quantity

        if not isinstance(self.s, Quantity):
            raise TypeError("Can only convert dimensions of a ``Quantity``")
        _s = cast(Quantity, self.s) # make pyright happy
        return type(self)(_s.to(*args, **kwargs), self.u,
                          copy=False, name=self.name,
                          meta=self._meta_copy(),
                          point_sources=self.point_sources)

    def compress(self, stype: Optional['np.dtype'] = None, utype: Optional['np.dtype'] = None, **kwargs) -> 'PartialUniqSkymap':
        """
        Eliminate redundant pixels with ``utils.uniq_minimize`` and store
        indices ``u`` in the smallest integer size that represents all values.

        Parameters
        ----------
        stype : type, optional
            If provided, store ``s`` as this type. Defaults to ``s.dtype``.
        utype : type, optional
            If provided, store ``u`` as this type. Defaults to the smallest
            ``int`` type required to store all values of ``u``.
        kwargs
            Keyword arguments to pass to ``hpmoc.utils.uniq_minimize``.

        Returns
        -------
        compressed : PartialUniqSkymap
            A compressed version of this skymap.

        See Also
        --------
        hpmoc.utils.uniq_minimize
        """
        u, s = uniq_minimize(self.u, self.s)
        if utype is None:
            utype = max_uint_type(self.u.max())
        s = s if stype is None else s.astype(stype)
        return type(self)(s, u.astype(utype), copy=False,
                          name=self.name, meta=self._meta_copy(),
                          point_sources=self.point_sources)

    def sort(self: 'PartialUniqSkymap[_DType]', copy=True) -> 'PartialUniqSkymap[_DType]':
        """
        Sort this skymap by UNIQ indices ``u`` (sorting ``s`` as well, of
        course). If ``copy=True``, copy ``u`` and ``s`` and return a new
        ``PartialUniqSkymap``; otherwise, sort them in-place and return this
        ``PartialUniqSkymap``.
        """
        ui = self.u.argsort()
        if not copy:
            self.u[:] = self.u[ui]
            self.s[:] = self.s[ui]
            return self
        return type(self)(self.s[ui], self.u[ui], name=self.name,
                          meta=self.meta, copy=True,
                          point_sources=self.point_sources)

    @property
    def value(self):
        """
        Get a dimensionless view of this skymap (no effect if ``s`` is not an
        ``astropy.units.Quantity``).
        """
        from astropy.units import Quantity

        if isinstance(self.s, Quantity):
            s = cast(Quantity, self.s).value # make pyright happy
            return type(self)(s, self.u, copy=False, name=self.name,
                            meta=self.meta,
                            point_sources=self.point_sources)
        return self

    def apply(
            self,
            func: Callable,
            copy: bool = True,
            quantity: bool = True
    ) -> 'PartialUniqSkymap':
        """
        Map a function to this skymap's pixels and return a new skymap whose
        values are the returned values of ``func``. Note that ``func(self.s)``
        must therefore return values corresponding to the same pixels as the
        input skymap. Use this to, for example, get the logarithm of this
        skymap as ``self.apply(np.log)``. If ``copy == True``, make a copy of
        the NUNIQ indices ``self.u``; otherwise, share them with the new
        skymap. To operate directly on the skymap value (even if it is stored
        as an ``astropy.units.Quantity``), use ``quantity = True``. Note that
        this will strip units from the result.
        """
        from astropy.units import Quantity

        m = self._meta_copy()
        mod = getattr(func, '__module__',
                      getattr(type(func), '__module__',
                              sys._getframe(-1).f_locals.get('__name__')))
        mod = '' if mod is None else mod + '.'
        m['HISTORY'] = (m.get('HISTORY', []) +
                        [f'Applied {mod}{func.__name__}'])
        n = f"{func.__name__}({self.name})"
        if not quantity and isinstance(self.s, Quantity):
            s = cast(Quantity, self.s).value
        else:
            s = self.s
        return type(self)(func(s), self.u.copy() if copy else self.u,
                          copy=False, name=n, meta=m,
                          point_sources=self.point_sources)

    def to_table(self, name=None, uname='UNIQ', meta_compat=False):
        """
        Return a new ``astropy.table.Table`` whose ``UNIQ`` column is the NUNIQ
        indices ``self.u`` and ``PIXELS`` (or ``self.name``, if set) column is
        the skymap pixel values ``s``. Optionally override the pixel value
        column name and/or the NUNIQ column name with the ``name`` and
        ``uname`` arguments respectively. If ``meta_compat`` is ``True``, drop
        metadata keys that are added by ``ligo.skymap.io.fits.write_sky_map``
        so they don't get duplicated.
        """
        from astropy.table import Table

        name = name or self.name or 'PIXELS'
        filtered_keys = {
            "PIXTYPE",
            "ORDERING",
            "COORDSYS",
            "MOCORDER",
            "INDXSCHM"
        } if self.meta else set()
        
        meta = {k: v for k, v in self.meta.items()
                if k not in filtered_keys}

        for pt in self.point_sources:
            meta.update(PointsTuple(*pt).meta_dict())

        t = Table([self.u, self.s], names=[uname, name], meta=meta)
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
        from .io import IoRegistry

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
        from .io import IoRegistry

        if not isinstance(args[0], PartialUniqSkymap):  # unbound method; add
            args = (None,) + args                       #   placeholder to args
        return getattr(IoRegistry, strategy).read(*args, **kwargs)

    def fill(self, nside=None, pad=None, as_skymap=False):
        """
        Return a full-sky *nested* HEALPix skymap at NSIDE resolution
        ``nside``.

        Parameters
        ----------
        nside : int
            HEALPix NSIDE value of the output map. If not provided, use the
            highest NSIDE value in this skymap's ``nside`` values to preserve
            detail.
        pad : float, optional
            Fill in missing values with ``pad`` (if not provided, use
            ``healpy.UNSEEN``). Preserves ``astropy.units.Unit`` of this
            skymap's pixel values (if ``s`` is an ``astropy.units.Quantity``).
        as_skymap : bool, optional
            If ``True``, return a ``PartialUniqSkymap`` instance with the new
            pixelization (instead of a bare array with implicit indexing).

        Returns
        -------
        s : array or PartialUniqSkymap
            The filled-in skymap, either as an array if ``as_skymap == False``
            or as a new ``PartialUniqSkymap`` instance.

        See Also
        --------
        PartialUniqSkymap.fixed
        """
        import numpy as np

        nside = nside or uniq2nside(self.u.max())
        sᵒ = fill(self.u, self.s, nside, pad=pad)
        if not as_skymap:
            return sᵒ
        m = self._meta_copy()
        m['HISTORY'] = (m.get('HISTORY', []) +
                        [f'Filled to NEST, NSIDE={nside}.'])
        return PartialUniqSkymap(sᵒ, np.arange(4*nside**2, 16*nside**2),
                                 copy=False, meta=m,
                                 point_sources=self.point_sources)

    def quantiles(
            self: 'PartialUniqSkymap[_DType]',
            quantiles: Sequence[float],
    ) -> Tuple[Iterator, 'NDArray[_DType]', '_DType']:
        """
        Get an iterator of downselected skymaps partitioned by ``quantiles``.
        For example, get the smallest sky area containing 90% of the
        probability (or whatever other intensive quantity this skymap
        represents) with ``quantiles=[0.1, 1]``.

        Parameters
        ----------
        quantiles : Sequence[float]
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
        partitions: Iterator[PartialUniqSkymap]
            A generator containing non-overlapping skymaps falling into the
            specified ``quantiles``. Will have length one less than the number
            of quantiles, which form the boundaries of the partitions. Always
            an iterable, even if only two quantiles are provided; you can
            unpack a single value with, e.g., ``x, = m.quantiles(...)``.
        levels: astropy.units.Quantity or array
            Values of the skymap at each quantile. Useful for e.g. contour
            plots (though ``PartialUniqSkymap.plot`` will handle this
            automatically).
        norm: astropy.units.Quantity or float
            The integrated value of the skymap. Useful for calculating the
            integral of each quantile.

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

        quantiles_arr = np.asarray(quantiles, dtype=np.float_)
        indices, levels, norm = nside_quantile_indices(self.nside(), self.s,
                                                       quantiles_arr)

        def skymaps():
            for i, l, u in zip(indices, quantiles_arr[:-1], quantiles_arr[1:]):
                m = self._meta_copy()
                m['HISTORY'] = m.get('HISTORY', []) + wrap(
                    f'Downselected to [{l:.2g}, {u:.2g}] quantile '
                    f'({(u-l)*100:.2g}%) of {norm:.2g} ({norm*(u-l):.2g} ',
                    70
                )
                yield PartialUniqSkymap(self.s[i], self.u[i], copy=False,
                                        meta=m,
                                        point_sources=self.point_sources)

        return skymaps(), levels, norm

    def fixed(self, nside=None):
        """
        Re-raster to a fixed NSIDE. Like ``fill`` but for partial skymaps.

        Parameters
        ----------
        nside : int
            HEALPix NSIDE value of the output map. If not provided, use the
            highest NSIDE value in this skymap's ``nside`` values to preserve
            detail.
        """
        nside = nside or uniq2nside(self.u.max())
        u⃗ᵒ = uniq2nest(self.u, nside, nest=False)
        s = self.reraster(u⃗ᵒ, copy=False)
        s.meta['HISTORY'][-1] += f' (fixed NSIDE={nside})'
        return s

    def __getitem__(self, idx):
        """
        Get a view into this skymap with the given index applied to ``u`` and
        ``s``. Uses their provided ``__getitem__`` semantics, so you'll get
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

        m = self._meta_copy()
        repidx = repr(idx)
        msg = repidx if len(repidx) < 60 else repidx[:58]+'...'
        m['HISTORY'] = m.get('HISTORY', []) + [f'Got view: {msg}']
        s = np.atleast_1d(self.s[idx])
        u = np.atleast_1d(self.u[idx])
        return type(self)(s, u, copy=False, meta=m,
                          point_sources=self.point_sources)

    def __setitem__(self, idx, value):
        """
        Like ``__getitem__``, will set values using the semantics of ``s``
        datatype.
        """
        raise NotImplementedError("Revisit this.")
        self[idx].s = value

    def _iparser(self, item):
        """
        Parse the ``item`` argument for ``__getitem__``, ``__setitem__``, and
        ``__delitem__``.
        """

    def intersection(self, u):
        """
        See ``utils.uniq_intersection``.
        """
        if isinstance(u, AbstractPartialUniqSkymap):
            u = u.u
        return uniq_intersection(self.u, u)

    def render(self, u⃗ᵒ, pad=None, valid=None, mask_missing=False):
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
        valid: array, optional
            If provided, results will be scattered into an array of the same
            shape as ``valid``, filling the indices where ``valid==True``. The
            number of ``True`` values in ``valid`` must therefore equal the
            length of ``u⃗ᵒ``. This argument only makes sense if ``u⃗ᵒ`` is an
            array of NUNIQ indices; if it is a ``WCS`` instance and ``valid``
            is provided, an error is raised. Use ``valid`` to produce plots or
            to reuse indices produced by ``wcs2mask_and_uniq`` in several
            ``render`` invocations.  See note on how ``mask_missing`` affects
            the result.
        mask_missing : bool
            If ``mask_missing=True``, return a ``np.ma.core.MaskedArray``.
            Missing values are tolerated and are marked as ``True`` in the
            ``mask_missing``. They will be set to ``pad or 0`` in the ``data``
            field. If ``valid`` is also provided, then the output will still be
            a ``np.ma.core.MaskedArray``, but will be set to ``True`` wherever
            ``valid == False`` in addition to wherever pixels are missing (and
            will still take on masked values of ``np.nan`` in the invalid
            regions).

        Returns
        -------
        pixels: array
            Pixel values at locations specified in ``u⃗ᵒ``. If ``u⃗ᵒ`` is a
            ``WCS`` instance, then values outside of the projection will be
            set to ``np.nan``.
        """
        return render(self.u, self.s, u⃗ᵒ, pad=pad, valid=valid,
                      mask_missing=mask_missing)

    def reraster(self, u_out, pad=None, mask_missing=False, copy=True, **kwargs):
        """
        Return a new ``PartialUniqSkymap`` instance with the same pixel values
        rerasterized to match the output NUNIQ indices ``u_out``. Fill in missing
        values in the output skymap with ``pad``. If ``pad`` is not provided
        and this skymap does not cover the full region defined in ``u_out``,
        raises a ``ValueError``. Preserves ``astropy.units.Unit`` of this
        skymap's pixel values (if ``s`` is an ``astropy.units.Quantity``). If
        ``copy`` is ``False``, use ``u_out`` as the indices of the new skymap;
        otherwise, use a copy.
        """
        import numpy as np

        s_out = reraster(self.u, self.s, u_out, pad=pad, mask_missing=mask_missing, **kwargs)
        m = self._meta_copy()
        m['HISTORY'] = m.get('HISTORY', []) + ['Rerasterized.']
        return PartialUniqSkymap(s_out, np.array(u_out, copy=copy), copy=False,
                                 meta=m, point_sources=self.point_sources)

    def coords(self):
        """
        Get the sky coordinates (right-ascension and declination, ICRS)
        corresponding to each pixel in the skymap.

        Returns
        -------
        ra_dec : astropy.units.Quantity
            2D array whose first row is the right-ascension and second row is
            the declination (in degrees) of each pixel. You can get each of
            these individually with ``ra, dec = self.coords()``.
            ``self.coords()[:, i]`` corresponds to RA, Dec for ``self.s[i]``.
        """
        return nest2ang(*uniq2nest_and_nside(self.u))

    def area(self):
        "Area per-pixel for pixels in this skymap in ``astropy.unit.sr``."
        from astropy.units import sr  # pylint: disable=no-name-in-module
        return nside2pixarea(self.nside(), degrees=False)*sr     # steradian

    def ang_dist(self, ra, dec, degrees=True):
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
        ang_dist : astropy.units.Quantity
            The distances of each pixel in this skymap to the point at ``ra``,
            ``dec`` in degrees.

        Examples
        --------
        We should find that the distance from any pixel to the North pole is
        equal to 90 minus the declination (within some small error):

        >>> import numpy as np
        >>> from astropy.units import deg
        >>> skymap = PartialUniqSkymap(*([4+np.arange(12)]*2))
        >>> _, dec = skymap.coords()
        >>> dec
        <Quantity [ 41.8103149,  41.8103149,  41.8103149,  41.8103149,   0.       ,
                     0.       ,   0.       ,   0.       , -41.8103149, -41.8103149,
                   -41.8103149, -41.8103149] deg>
        >>> Δθ⃗ = skymap.ang_dist(32, 90)
        >>> Δθ⃗
        <Quantity [0.84106867, 0.84106867, 0.84106867, 0.84106867, 1.57079633,
                   1.57079633, 1.57079633, 1.57079633, 2.30052398, 2.30052398,
                   2.30052398, 2.30052398] rad>
        >>> np.all(abs(Δθ⃗+dec-90*deg).to('deg').value<1e-13)
        True

        Likewise, the distance from any pixel to the South pole should be
        equal to 90 plus the declination:

        >>> not np.around(skymap.ang_dist(359, -90)-dec-90*deg,
        ...               14).value.any()
        True
        """
        return uniq2dangle(self.u, ra, dec, degrees=degrees)

    def unzip_orders(self):
        """
        Return a list of ``PartialUniqSkymap`` instances corresponding to the
        parts of this sky imaged and each HEALPix order. Length equals the
        maximum order of this skymap. Empty terms indicate that this skymap
        does not have pixels of the corresponding HEALPix order.
        """
        # TODO test
        srt = self.sort()
        [[s], o⃗] = nside_slices(srt.u)[1:3]
        return [srt[s] for s in [slice(0, 0)]*o⃗[0]+s]

    def unzip_atlas(self):
        "Return 12 sub-skymaps corresponding to the HEALPix base pixels."
        raise NotImplementedError()

    def min(self):
        "Minimum skymap value = ``self.s.min()``."
        return self.s.min()

    def max(self):
        "Maximum skymap value = ``self.s.max()``."
        return self.s.max()

    @property
    def unit(self):
        "``self.s.unit``, if defined; otherwise ``None``."
        return getattr(self.s, 'unit', None)

    @_depr_visufunc
    def azeqview(self, *scatter, **kwargs):
        return self.plot(*scatter, projection='ARC', **kwargs)

    @_depr_visufunc
    def cartview(self, *scatter, **kwargs):
        return self.plot(*scatter, projection='CAR', **kwargs)

    @_depr_visufunc
    def gnomview(self, *scatter, **kwargs):
        return self.plot(*scatter, projection='TAN', **kwargs)

    @_depr_visufunc
    def orthview(self, *scatter, **kwargs):
        return self.plot(*scatter, projection='SIN', **kwargs)

    @_depr_visufunc
    def mollview(self, *scatter, **kwargs):
        return self.plot(*scatter, projection='MOL', **kwargs)

    def plot(self, *scatter: PointsTuple, **kwargs) -> Union[
            'wcsaxes.WCSAxes',
            'wcsaxes.WCSAxesSubplot'
    ]:
        """
        Plot this skymap. A thin wrapper around ``hpmoc.plot.plot`` that
        automatically includes scatter plots for this instance's
        ``point_sources``.

        Parameters
        ----------
        scatter: PointsTuple
            Additional point sources to plot (in addition to those stored
            in ``self.point_sources``).
        kwargs
            Keywork arguments to be passed to ``hpmoc.plot.plot``.

        Returns
        -------
        ax: WCSAxes or WCSAxesSubplot
            The axes that were just plotted to.

        See Also
        --------
        hpmoc.plot.plot
        """
        return plot(self, *(*self.point_sources, *scatter), **kwargs)

    def gridplot(
            self,
            *skymaps: Union[
                'PartialUniqSkymap',
                'NDArray[Any]',
                Tuple[
                    'NDArray[Any]',
                    Optional[
                        Union[
                            'NDArray[Any]',
                            'WCS',
                            str,
                        ]
                    ],
                ],
            ],
            **kwargs
    ) -> Tuple[
            'matplotlib.gridspec.GridSpec',
            Sequence[Sequence['wcsaxes.WCSAxes']]
    ]:
        """
        Plot this skymap and any others in ``skymaps``. A thin wrapper around
        ``hpmoc.plot.gridplot``.

        Parameters
        ----------
        *skymaps : PartialUniqSkymap or map-like
            Skymaps to pass to ``gridplot``. Can be anything accepted by
            ``hpmoc.plot.plot``, which ``hpmoc.plot.gridplot`` will use to
            display them.
        **kwargs
            Keyword arguments for ``hpmoc.plot.gridplot``.

        See Also
        --------
        hpmoc.plot.gridplot
        """
        return gridplot(self, *skymaps, **kwargs)

    @_depr_visufunc
    def multiplot(self, *skymaps: Sequence['PartialUniqSkymap'], nest: bool = True, **kwargs):
        """
        Call ``plotters.multiplot`` with the default ``transform``
        suitable for a ``PartialUniqSkymap``.

        Parameters
        ----------
        *skymapsₗ : List[Union[PartialUniqSkymap, array]]
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
        plot.multiplot
        PartialUniqSkymap.plot
        plot.plot
        """
        return multiplot(self, *skymaps, **kwargs)

    def _vecs_for_repr_(self, maxlen, *vecs):
        if not vecs:
            vecs = self.u, self.s
        return _vecs_for_repr_(maxlen, *vecs)

    def __str__(self):
        return self.to_table().__str__()

    def __repr__(self):
        return self.to_table().__repr__()

    def _repr_html_(self):
        [u, s], [_, unit] = self._vecs_for_repr_(20)
        pts = self.point_sources
        unit = f'<thead><tr><th></th><th>{unit}</th></tr></thead>'
        rows = "\n".join(f'<tr><td>{u}</td><td>{s}</td></tr>'
                         for u, s in zip(u, s))
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
            projs = ['CEA']
            widths = [2]
            one_pt = _one_pt(pts, None)
            if one_pt:
                projs.append('ARC')
                widths.append(1)
            kwargs = {'projections': projs, 'bottom': 0.1, 'left': 0.04,
                      'missing_color': 'gray'}
            if 'IPython' in sys.modules:
                from IPython.utils.capture import capture_output

                with capture_output():
                    gs, [[ax], *_] = self.gridplot(**kwargs)
            else:
                gs, [[ax], *_] = self.gridplot(**kwargs)
            if one_pt:
                for co in ax.coords:
                    co.set_ticklabel_visible(False)
            fig = gs.figure
            assert fig is not None
            fig.savefig(img, format='png')
            plt.close(fig)
            img.seek(0)
            #b64 = "".join(base64.encodebytes(img.read()).decode().split('\n'))
            b64 = base64.b64encode(img.read()).decode()
            pd=f'''
                <div style="vertical-align: top;">
                    <h5>Plot</h5>
                    <img src="data:image/png;base64,{b64}" alt="Preview plot"
                         style="min-width: 200px; max-width: 800px;"/>
                </div>
            '''
        else:
            pd = '''
                <div>
                    <em>Import matplotlib to display plot preview.</em>
                </div>
            '''

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
                <div class="partialuniq_flexbox partialuniq_flexbox_vert">
                    {pd}
                    {pt_srcs}
                </div>
                <div style="vertical-align: top;">
                    <h5>Skymap ({len(self.s)} pixels)</h5>
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
        </div>
        '''
        return tab

    # BEGIN COMPARATOR METHODS
    # None of these methods should ever be reached, since they will be replaced
    # by a constructed method that calls ``uniq_diadic`` with the appropriate
    # operator. See ``diadic_dunder`` for more information.  

    @diadic_dunder(post=bool_to_uint8)
    def __eq__(self, other): raise NotImplementedError("__eq__")

    @diadic_dunder(post=bool_to_uint8)
    def __ne__(self, other): raise NotImplementedError("__ne__")

    @diadic_dunder(post=bool_to_uint8)
    def __le__(self, other): raise NotImplementedError("__le__")

    @diadic_dunder(post=bool_to_uint8)
    def __lt__(self, other): raise NotImplementedError("__lt__")

    @diadic_dunder(post=bool_to_uint8)
    def __ge__(self, other): raise NotImplementedError("__ge__")

    @diadic_dunder(post=bool_to_uint8)
    def __gt__(self, other): raise NotImplementedError("__gt__")

    # BEGIN NUMERIC METHODS

    @diadic_dunder()
    def __add__(self, other): raise NotImplementedError("__add__")

    @diadic_dunder()
    def __sub__(self, other): raise NotImplementedError("__sub__")

    @diadic_dunder()
    def __mul__(self, other): raise NotImplementedError("__mul__")

    # @diadic_dunder()
    # def __matmul__(self, other): raise NotImplementedError("__matmul__")

    @diadic_dunder()
    def __truediv__(self, other): raise NotImplementedError("__truediv__")

    @diadic_dunder()
    def __floordiv__(self, other): raise NotImplementedError("__floordiv__")

    @diadic_dunder()
    def __mod__(self, other): raise NotImplementedError("__mod__")

    @diadic_dunder()
    def __divmod__(self, other): raise NotImplementedError("__divmod__")

    @diadic_dunder()
    def __pow__(self, other): raise NotImplementedError("__pow__")

    @diadic_dunder()
    def __lshift__(self, other): raise NotImplementedError("__lshift__")

    @diadic_dunder()
    def __rshift__(self, other): raise NotImplementedError("__rshift__")

    @diadic_dunder()
    def __and__(self, other): raise NotImplementedError("__and__")

    @diadic_dunder()
    def __xor__(self, other): raise NotImplementedError("__xor__")

    @diadic_dunder()
    def __or__(self, other): raise NotImplementedError("__or__")

    # REVERSE NUMERIC METHODS

    @diadic_dunder()
    def __radd__(self, other): raise NotImplementedError("__radd__")

    @diadic_dunder()
    def __rsub__(self, other): raise NotImplementedError("__rsub__")

    @diadic_dunder()
    def __rmul__(self, other): raise NotImplementedError("__rmul__")

    # @diadic_dunder()
    # def __rmatmul__(self, other): raise NotImplementedError("__rmatmul__")

    @diadic_dunder()
    def __rtruediv__(self, other): raise NotImplementedError("__rtruediv__")

    @diadic_dunder()
    def __rfloordiv__(self, other): raise NotImplementedError("__rfloordiv__")

    @diadic_dunder()
    def __rmod__(self, other): raise NotImplementedError("__rmod__")

    @diadic_dunder()
    def __rdivmod__(self, other): raise NotImplementedError("__rdivmod__")

    @diadic_dunder()
    def __rpow__(self, other): raise NotImplementedError("__rpow__")

    @diadic_dunder()
    def __rlshift__(self, other): raise NotImplementedError("__rlshift__")

    @diadic_dunder()
    def __rrshift__(self, other): raise NotImplementedError("__rrshift__")

    @diadic_dunder()
    def __rand__(self, other): raise NotImplementedError("__rand__")

    @diadic_dunder()
    def __rxor__(self, other): raise NotImplementedError("__rxor__")

    @diadic_dunder()
    def __ror__(self, other): raise NotImplementedError("__ror__")

    # IN-PLACE NUMERIC METHODS

    @diadic_dunder()
    def __iadd__(self, other): raise NotImplementedError("__iadd__")

    @diadic_dunder()
    def __isub__(self, other): raise NotImplementedError("__isub__")

    @diadic_dunder()
    def __imul__(self, other): raise NotImplementedError("__imul__")

    # @diadic_dunder()
    # def __imatmul__(self, other): raise NotImplementedError("__imatmul__")

    @diadic_dunder()
    def __itruediv__(self, other): raise NotImplementedError("__itruediv__")

    @diadic_dunder()
    def __ifloordiv__(self, other): raise NotImplementedError("__ifloordiv__")

    @diadic_dunder()
    def __imod__(self, other): raise NotImplementedError("__imod__")

    @diadic_dunder()
    def __ipow__(self, other): raise NotImplementedError("__ipow__")

    @diadic_dunder()
    def __ilshift__(self, other): raise NotImplementedError("__ilshift__")

    @diadic_dunder()
    def __irshift__(self, other): raise NotImplementedError("__irshift__")

    @diadic_dunder()
    def __iand__(self, other): raise NotImplementedError("__iand__")

    @diadic_dunder()
    def __ixor__(self, other): raise NotImplementedError("__ixor__")

    @diadic_dunder()
    def __ior__(self, other): raise NotImplementedError("__ior__")
