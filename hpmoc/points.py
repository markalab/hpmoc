import re
from collections import OrderedDict
from typing import NamedTuple, Union, List, Tuple, Optional

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

    def to_hex(self):
        """
        Get a hex string of the form ``#rrggbbaa`` for this ``Rgba`` tuple.
        """
        return "#"+("{:02x}"*4).format(*(int(255*c) for c in self))


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


class PointsTuple(NamedTuple):
    """
    A collection of points for scatterplots.
    """
    points: List[Tuple[float, float, Optional[float], Optional[str]]]
    rgba: Rgba = Rgba(0, 1, 0, 0.2)
    marker: str = 'x'
    label: str = None

    def _repr_html_(self):
        pts = _vecs_for_repr_(20, *zip(*self.points))[0]
        rows = "\n".join(f'<tr><td>{s or i}</td><td>{r}</td><td>{d}</td>'
                         f'<td>{σ}</td></tr>'
                         for i, [r, d, σ, s] in enumerate(zip(*pts)))
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
            pt = cls(p, Rgba(*r), m, l)
            if pt not in unique:
                unique.append(pt)
        return unique