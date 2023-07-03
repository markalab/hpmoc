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

"HEALPix helper utilities needed for both ``utils`` and ``healpy``."

import numpy as np

COMPRESS_MASKS = np.array([
    0x5555555555555555,
    0x3333333333333333,
    0x0f0f0f0f0f0f0f0f,
    0x00ff00ff00ff00ff,
    0x0000ffff0000ffff,
    0x00000000ffffffff,
], dtype=np.uint64)


def alt_compress(x, in_place=False):
    """
    Start in 0x55... state.

    https://help.dyalog.com/18.0/Content/Language/Primitive%20Functions/Replicate.htm

    Examples
    --------
    >>> alt_compress(0b011101)
    7
    >>> alt_compress(0b110010)
    4
    >>> alt_compress(100)
    10
    >>> f'{alt_compress(0b10011100):04b}'
    '0110'

    See Also
    --------
    alt_expand
    """

    if isinstance(x, int):
        x = np.uint64(x)
    elif not (in_place and np.issubdtype(x.dtype, np.uint64)):
        x = x.astype(np.uint64)
    x &= COMPRESS_MASKS[0]
    for i, m in enumerate(COMPRESS_MASKS[1:]):
        hold = m&x
        x &= ~m
        x >>= np.uint64(1<<i)
        x |= hold
    return x


def alt_expand(x, in_place=False):
    """
    Start in 0x00000000ffffffff state.

    https://help.dyalog.com/18.0/Content/Language/Primitive%20Functions/Expand.htm

    Examples
    --------
    >>> f'{alt_expand(0b100101):012b}'
    '010000010001'

    See Also
    --------
    alt_compress
    """

    o = len(COMPRESS_MASKS)
    if isinstance(x, int):
        x = np.uint64(x)
    elif not (in_place and np.issubdtype(x.dtype, np.uint64)):
        x = x.astype(np.uint64)
    x &= COMPRESS_MASKS[-1]
    for i, m in enumerate(COMPRESS_MASKS[-2::-1]):
        hold = m&x
        x &= ~m
        x <<= np.uint64(1<<(o-i-2))
        x |= hold
    return x
