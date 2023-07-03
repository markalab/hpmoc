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


import re
from collections import OrderedDict
from typing import NamedTuple, Union, List, Tuple, Optional
from .utils import wcs2ang, uniq2nest_and_nside, monochrome_opacity_colormap

PT_META_REGEX = re.compile('^PT([0-9A-Fa-f])_([0-9A-Fa-f]{2})(RA|DC|SG|ST)$')
PT_META_KW_REGEX = re.compile('^PT([0-9A-Fa-f])_(MRK|LABEL)$')
PT_META_COLOR_REGEX = re.compile('^PT([0-9A-Fa-f])_([RGBA])$')
PT_META_LUT = {
    'RA': 0,
    'DC': 1,
    'SG': 2,
    'ST': 3,
    'MRK': 'marker',
    'LABEL': 'label',
    'R': 0,
    'G': 1,
    'B': 2,
    'A': 3
}


class Rgba(NamedTuple):
    """
    An RGBA color tuple. If ``alpha`` is omitted, set to ``1``.
    """
    red: Union[float, int]
    green: Union[float, int]
    blue: Union[float, int]
    alpha: Union[float, int] = 1

    @classmethod
    def from_hex(cls, hexcolor):
        """
        Convert a color of the forms ``rgb``, ``rgba``, ``rrggbb``,
        ``rrggbbaa``, ``#rgb``, ``#rrggbb``, ``#rgba``, or ``#rrggbbaa`` to a
        new ``Rgba`` instance. Alpha can be omitted, in which case it is set to
        1.
        """
        h = hexcolor.strip('#')
        l = len(h)
        if l not in (3, 4, 6, 8):
            raise ValueError(f"Unrecognized hex color format: {hexcolor}")
        c = 1 if l in (3, 4) else 2
        return cls(*(int(h[i:i+c], 16)/(16**c-1) for i in range(0, l, c)))

    def to_hex(self, include_alpha=True):
        """
        Get a hex string of the form ``#rrggbbaa`` for this ``Rgba`` tuple.
        """
        cs = self if include_alpha else self[:-1]
        return "#"+("{:02x}"*len(cs)).format(*(int(255*c) for c in cs))


def _vecs_for_repr_(maxlen, *vecs):
    l = set(len(v) for v in vecs)
    if len(l) > 1:
        raise ValueError("Vecs must have the same length.")
    l = l.pop()
    vˡ, uˡ = zip(*((v.value, v.unit) if hasattr(v, 'value') else (v, None)
                    for v in vecs))
    if l <= maxlen:
        return vˡ, uˡ
    e = maxlen//2
    s = maxlen-e-1
    return [[*d[:s], '...', *d[-e:]] for d in vˡ], uˡ


class NoDisks(ValueError):
    """
    Indicates that there are no point sources with nonzero support (sigma).
    Catch this to avoid wasting an extra artist plotting non-existent disks.
    """


class PointsTuple(NamedTuple):
    """
    A collection of points for scatterplots.
    """
    points: List[Tuple[float, float, Optional[float], Optional[str]]]
    rgba: Rgba = Rgba.from_hex('#0F7F12')
    marker: str = 'x'
    label: str = None

    def _repr_html_(self):
        pts = _vecs_for_repr_(20, *zip(*self.points))[0]
        rowa = []
        for i, [r, d, *σs] in enumerate(zip(*pts)):
            σ = '--' if len(σs) == 0 else σs[0]
            s = '--' if len(σs) < 2 else σs[1]
            rowa.append(f'<tr><td>{s or i}</td><td>{r}</td><td>{d}</td>'
                        f'<td>{σ}</td></tr>')
        rows = "\n".join(rowa)
        bgcolor = Rgba(*self.rgba).to_hex()
        return f'''
            <table>
                <thead>
                    <tr style="background-color: {bgcolor};">
                        <th>
                            {self.label or "<em>no label</em>"}: {self.marker}
                        </th>
                        <th></th><th></th><th></th>
                    </tr>
                </thead>
                <thead>
                    <tr>
                        <th>Source</th>
                        <th>RA [deg]</th>
                        <th>Dec [deg]</th>
                        <th>σ [deg]</th>
                    </tr>
                </thead>
                {rows}
            </table>
        '''

    def scale_sigma(self, factor, **kwargs):
        """
        Return a new ``PointsTuple`` with sigma scaled for each point by
        ``factor`` (if sigma is defined). Optionally also change ``rgba``,
        ``marker``, or ``label`` (will be unchanged by default). A convenience
        method for plotting multiple sigmas around a point source.
        """
        kw = dict(zip(self._fields, self))
        kw.update(kwargs)
        kw['points'] = [(r, d) + ((2*s[0],) if s else ())
                        for r, d, *s in self.points]
        return type(self)(**kw)

    @classmethod
    def meta_read(cls, meta: dict) -> List['__class__']:
        f"""
        Read points following the regular expression {PT_META_REGEX}
        from a dictionary ``meta`` into a ``PointsTuple``. Specify ``PTRGBAi``,
        as color, ``PTMRKi`` as the marker, and ``PTLABELi`` as the legend
        label. Use for e.g. reading points from a fits file header. Returns a
        list of ``PointsTuple`` instances.

        See Also
        --------
        PointsTuple.meta_dict
        """
        d = cls.__new__.__defaults__
        pts = [*zip(*sum([[((int(a, 16), int(b, 16)), PT_META_LUT[c], meta[k])
                           for a, b, c in PT_META_REGEX.findall(k)]
                          for k in meta], []))]
        if not pts:
            return
        uniq_pts = set(pts[0])
        kws = {lst: {'points': {}, 'rgba': [*d[0]], 'marker': d[1],
                     'label': d[2]}
               for lst, _ in uniq_pts}
        for lst, pt in uniq_pts:
            kws[lst]['points'][pt] = [None, None, None, None]
        for [lst, pt], pos, val in zip(*pts):
            kws[lst]['points'][pt][pos] = val
        for k in meta:
            m = [(PT_META_LUT[b], int(a, 16))
                 for a, b in PT_META_KW_REGEX.findall(k)]
            if m:
                kws[m[0][1]][m[0][0]] = meta[k]
            m = [(PT_META_LUT[b], int(a, 16))
                 for a, b in PT_META_COLOR_REGEX.findall(k)]
            if m:
                kws[m[0][1]]['rgba'][m[0][0]] = meta[k]
        kws = [*kws.values()]
        for lst in kws:
            lst['points'] = [*lst['points'].values()]
            lst['rgba'] = Rgba(*lst['rgba'])
        return [cls(**kw) for kw in kws]

    def meta_dict(*pts, start=0) -> dict:
        """
        Create a flattened meta dictionary, e.g. for a fits file, out of the
        provided ``PointsTuple`` instances.

        See Also
        --------
        PointsTuple.meta_read
        """
        res = OrderedDict()
        if len(pts) > 16:
            raise ValueError("Can only save up to 16 point lists to meta. "
                             "Use tables for large numbers of point sources.")
        for i, [pt, rgba, m, l] in enumerate(pts):
            if len(pt) > 256:
                raise ValueError("Can only save up to 256 points per list to "
                                 "meta. Use tables for large numbers of point "
                                 "sources.")
            pre = f"PT{i+start:X}_"
            if l is not None:
                res[pre+'LABEL'] = l
            if m is not None:
                res[pre+'MRK'] = m
            for k, c in zip('RGBA', rgba):
                if c is not None:
                    res[pre+k] = c
            for j, p in enumerate(pt):
                for k, v in zip(('RA', 'DC', 'SG', 'ST'), p):
                    if v is not None:
                        res[f"{pre}{j:02X}{k}"] = v
        return res

    @classmethod
    def dedup(cls, *pts):
        """
        Deduplicate a collection of ``PointsTuple`` instances, converting
        color tuples to ``Rgba`` and converting all input tuples to
        ``PointsTuple`` instances. Preserves ordering.
        """
        unique = []
        for p, r, m, l in pts:
            pt = cls([*p], Rgba(*r), m, l)
            if pt not in unique:
                unique.append(pt)
        return unique

    def render(self, u⃗ᵒ, extent=1.):
        """
        Similar to ``hpmoc.PartialUniqSkymap.render``, but for support disks
        specified by the input points' sigma parameters. Will raise
        ``NoDisks`` if all point source sigmas are undefined and/or zero.
        Use this with the ``cmap`` method and your ``WCSAxes`` instance's
        ``imshow`` method to display these disks. You will need to call
        ``WCSAxes.scatter`` separately to display the point locations
        themselves using ``self.marker``. You will also need to label the
        point names separately using ``self.label_points``.

        All of this is done automatically if this instance is passed as a
        ``scatter`` parameter to ``PartialUniqSkymap.plot``.

        Parameters
        ----------
        u⃗ᵒ: array or astropy.wcs.WCS
            The pixels to fill. If an array, treated as UNIQ indices
            (duplicates allowed); if WCS, treated as a set of pixels to render
            to.
        extent: float, optional
            How many multiples of ``sigma`` to extend the disk out to.

        Returns
        -------
        pixels: array
            Pixel values at locations specified in ``u⃗ᵒ``. If a ``WCS``
            instance, then the pixels will be suitable for immediate display
            in a ``WCSAxes`` instance using ``imshow``; otherwise, you might
            wish to use the nonzero values as a mask to select pixels in a
            ``PartialUniqSkymap`` near the point sources in this instance.

        Raises
        ------
        NoDisks
            If all sigmas are zero or undefined. Use this to avoid an
            extraneous call to ``imshow``, or to catch missing sigmas.

        See Also
        --------
        hpmoc.partial.PartialUniqSkymap.plot
        hpmoc.partial.PartialUniqSkymap.render
        WCSAxes.scatter
        """

        import numpy as np
        from astropy.wcs import WCS
        from astropy.units import rad
        from .healpy import healpy as hp

        # get the disks to be plotted, raising ``NoDisks`` if none match
        p, t, s = np.radians([(r, d, s[0]) for (r, d, *s) in self.points
                              if len(s) > 0 and s[0] > 0]).T
        if p.size == 0:
            raise NoDisks()
        a = self.rgba.alpha
        d = np.float32 if len(self.points) < 16777216 else np.float64

        # get angles of disks as well as the max dot-product with s (sigma)
        # and normalize angles p, t to phi, theta conventions.
        np.multiply(s, extent, out=s)
        np.cos(s, out=s)
        np.subtract(np.radians(90), t, out=t)
        points = hp.ang2vec(t, p, lonlat=False).T
        del t, p

        # get the coordinates to plot to and convert them to vectors
        if isinstance(u⃗ᵒ, WCS):
            valid, tp, pp  = wcs2ang(u⃗ᵒ, lonlat=False)
            tp = tp.to(rad).value
            pp = pp.to(rad).value
        else:
            valid = None
            tp, pp = hp.pix2ang(*uniq2nest_and_nside(u⃗ᵒ)[::-1], nest=True,
                                 lonlat=False)
        plot = hp.ang2vec(tp, pp, lonlat=False)
        del tp, pp

        # take dot products and sum overlaps of each pixel before applying
        # alpha
        m = (plot@points>s).sum(axis=1, dtype=d)
        del plot, points
        np.power(1-a, m, out=m)
        np.subtract(1, m, out=m)

        # fill the output with rendered values if necessary and return
        if valid is None:
            return m
        o = np.full(valid.shape, np.nan)
        o[valid] = m
        return o


    def cmap(self):
        """
        Returns
        -------
        cmap: matplotlib.colors.LinearSegmentedColormap
            A color map to be used as the ``cmap`` argument to ``imshow``
            alongside the return value of ``render``.
        """
        return monochrome_opacity_colormap(self.rgba.to_hex()[:-2], self.rgba[:-1])
