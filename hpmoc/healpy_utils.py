"HEALPix helper utilities needed for both ``utils`` and ``healpy``."


def compress_masks():
    """
    Must use ``numpy.uint64`` due to signedness issues when bit-flipping large
    python floats as bitmasks.
    """
    import numpy as np

    return np.array([
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
    import numpy as np

    if isinstance(x, int):
        x = np.uint64(x)
    elif not (in_place and np.issubdtype(x.dtype, np.uint64)):
        x = x.astype(np.uint64)
    masks = compress_masks()
    x &= masks[0]
    for i, m in enumerate(masks[1:]):
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
    import numpy as np

    masks = compress_masks()
    o = len(masks)
    if isinstance(x, int):
        x = np.uint64(x)
    elif not (in_place and np.issubdtype(x.dtype, np.uint64)):
        x = x.astype(np.uint64)
    x &= masks[-1]
    for i, m in enumerate(masks[-2::-1]):
        hold = m&x
        x &= ~m
        x <<= np.uint64(1<<(o-i-2))
        x |= hold
    return x
