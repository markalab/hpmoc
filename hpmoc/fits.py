# (c) Stefan Countryman 2021

"""
Load skymaps from fits files.
"""

from __future__ import annotations

import re
import gzip
import logging
from pathlib import Path
from math import ceil
from collections import OrderedDict
from typing import (
    Union,
    IO,
    List,
    Tuple,
    Callable,
    Iterator,
    Any,
    Optional,
    Iterable,
    cast,
    TYPE_CHECKING
)
from .healpy import healpy as hp
from .utils import (
    set_partial_skymap_metadata,
    EmptyStream,
    nest2uniq,
    uniq2nside,
    nside2pixarea,
    uniq_minimize,
    uniq_intersection,
    is_gz,
)

if TYPE_CHECKING:
    from nptyping import NDArray, Int
    from astropy.io.fits import (
        Header,
        BinTableHDU,
        ColDefs
    )
    from astropy.io.fits.hdu.base import ExtensionHDU
    from astropy.units.quantity import Quantity

LOGGER = logging.getLogger(__name__)
FITS_BLOCKSIZE = 2880
FITS_CARD_WIDTH = 80
FITS_CARD_NAME_WIDTH = 8
FITS_HEADER_LENGTH = FITS_BLOCKSIZE // FITS_CARD_WIDTH
BINTABLE_TO_NUMPY_TYPES = {
    'L': '?',    # boolean
    # 'X': bit, not supported for now.
    'B': 'B',    # unsigned byte
    'I': '>i2',  # 16-bit big-endian int
    'J': '>i4',  # 32-bit big-endian int
    'K': '>i8',  # 64-bit big-endian int
    'A': 'b',    # char/signed byte
    'E': '>f4',  # 32-bit big-endian floating point
    'D': '>f8',  # 64-bit big-endian floating point
    'C': '>c8',  # 64-bit big-endian complex floating point
    'M': '>c16', # 128-bit big-endian complex floating point
    # 'P': 32-bit array descriptor, not supported for now
    # 'Q': 64-bit array descriptor, not supported for now
}
HEADER_NON_META = re.compile(
    "|".join(
        (
            'XTENSION',
            'BITPIX',
            'NAXIS[0-9]*',
            'PCOUNT',
            'GCOUNT',
            'TFIELDS',
            'TTYPE[0-9]*',
            'TFORM[0-9]*',
            'TUNIT[0-9]*',
        )
    )
)
TFORM = re.compile(r'([0-9]*)([LXBIJKAEDCMPQ])(.*)')
BUFFER_ROWS = 4**8


def calculate_max_rows_read(hdu, dtype, buf_rows):
    "Meant to handle BAYESTAR MOC and NEST + CWB skymaps."
    for name, coltype, *repeat in dtype:
        if name in ('PROBDENSITY', 'PROB', 'PROBABILITY'):
            if repeat:
                return buf_rows // repeat[0]
            return buf_rows
    raise ValueError(f"Could not find a PROBABILITY column in {hdu.columns}, "
                     "do not know how to proceed.")


def extract_probdensity(hdu, chunk, offset):
    "Meant to handle BAYESTAR MOC and NEST + CWB skymaps."
    import numpy as np
    from astropy.units import Quantity

    # define up here for typing purposes
    nside = -1

    ordering = hdu.header['ORDERING']
    if ordering == 'NUNIQ':
        u = chunk['UNIQ']
    else:
        for probname in ('PROBDENSITY', 'PROBABILITY', 'PROB'):
            if probname in hdu.columns.names:
                fmt = cast(str, hdu.columns[probname].format)
                m = TFORM.match(fmt)
                if m is None:
                    raise ValueError(f"Could not parse column format {fmt}")
                repeat = int(m[1]) if m[1] else 1
                break
        else:
            raise ValueError("Could not find a probability column in "
                             f"HDU columns: {hdu.columns}")
        nside = hdu.header['NSIDE']
        inds = np.arange(repeat*offset, repeat*(offset+len(chunk)))
        if ordering == 'RING':
            inds = hp.ring2nest(nside, inds)
        elif ordering != 'NESTED':
            raise ValueError(f"Unrecognized ordering: {ordering}")
        u = nest2uniq(inds, nside, in_place=True)
    if 'PROBDENSITY' in hdu.columns.names:
        return u, [Quantity(chunk['PROBDENSITY'], copy=False,
                            unit=hdu.columns['PROBDENSITY'].unit)]
    LOGGER.debug(f"PROBDENSITY not found in {hdu.columns}, trying to "
                 f"calculate it from PROB column and NSIDE")
    if 'PROBABILITY' in hdu.columns.names:
        LOGGER.debug("Column named PROBABILITY found. Using for probability "
                     "(Fermi GBM convention?)")
        probname = 'PROBABILITY'
    elif 'PROB' in hdu.columns.names:
        probname = 'PROB'
    else:
        raise ValueError(f"PROB not found in HDU columns: {hdu.columns}")
    probunit = hdu.columns[probname].unit
    if not probunit == 'pix-1':
        raise ValueError(f"Unexpected unit for {probname} column: {probunit}")
    prob = chunk[probname].ravel()  # handle CWB skymaps with repeated cols
    if ordering == 'NUNIQ':
        nside = uniq2nside(u)
    return u, [Quantity(prob/nside2pixarea(nside, degrees=False),
                        copy=False, unit='sr-1')]


def read_bintable_chunks(
        stream: IO,
        tables: int = -1,
        buf_rows: int = BUFFER_ROWS,
        extractor: Callable[
            ['BinTableHDU', NDArray[Any, Any], int],
            Tuple[
                NDArray[Any, Int],
                List['Quantity'],
            ]
        ] = extract_probdensity,
        row_calculator: Callable[
            [
                'BinTableHDU',
                List[Union[Tuple[str, str], Tuple[str, str, Tuple[int]]]],
                int,
            ],
            int
        ] = calculate_max_rows_read,
) -> Iterator[
        Tuple[
            'BinTableHDU',
            Iterator[
                Tuple[
                    NDArray[Any, Any],
                    List['Quantity'],
                ]
            ],
        ]
]:
    import numpy as np
    from astropy.io import fits

    # assume first HDU has been skipped already
    while tables != 0:
        try:
            hdu = next_hdu(stream)
        except EmptyStream:
            break

        # make sure the HDUs are bintables
        assert isinstance(hdu, fits.BinTableHDU)

        dtype = bintable_dtype(hdu)
        max_rows = row_calculator(hdu, dtype, buf_rows)
        total_rows = cast(int, hdu.header['NAXIS2'])
        width = cast(int, hdu.header['NAXIS1'])
        assert total_rows * width == hdu.size

        # create a generator for each HDU
        def chunk_iter():
            offset = 0
            while offset < total_rows:
                rows = min(max_rows, total_rows - offset)
                chunk = np.frombuffer(stream.read(width*rows), dtype=dtype)
                yield extractor(cast('BinTableHDU', hdu), chunk, offset)
                offset += rows
            assert offset == total_rows
            position_in_block = hdu.size % FITS_BLOCKSIZE
            if position_in_block != 0:
                stream.read(FITS_BLOCKSIZE - position_in_block)

        yield hdu, chunk_iter()
        tables -= 1


def load_ligo(
        infile: Union[IO, str, Path],
        mask: Optional[NDArray[Any, Int]] = None,
        maps: Optional[Union[int, Iterable[int]]] = None,
        extractor: Callable[
            ['BinTableHDU', NDArray[Any, Any], int],
            Tuple[
                NDArray[Any, Int],
                List['Quantity'],
            ]
        ] = extract_probdensity,
        row_calculator: Callable[
            [
                'BinTableHDU',
                List[Union[Tuple[str, str], Tuple[str, str, Tuple[int]]]],
                int,
            ],
            int
        ] = calculate_max_rows_read,
        chunk_processor: Optional[Callable[
            [NDArray[Any, Int], NDArray[Any, Any]],
            Tuple[NDArray[Any, Int], NDArray[Any, Any]]
        ]] = uniq_minimize,
        post_processor: Optional[Callable[
            [NDArray[Any, Int], NDArray[Any, Any]],
            Tuple[NDArray[Any, Int], NDArray[Any, Any]]
        ]] = uniq_minimize,
        buf_rows: int = BUFFER_ROWS,
) -> Iterator[Tuple[NDArray[Any, Int], 'Quantity', OrderedDict]]:
    import numpy as np

    if isinstance(infile, (str, Path)):
        with (gzip.open if is_gz(infile) else open)(infile, 'rb') as stream:
            for tab in load_ligo(stream, mask, maps, extractor, row_calculator,
                                 chunk_processor, post_processor, buf_rows):
                yield tab
        return
    if mask is not None:
        mask, = uniq_minimize(mask)
    if maps is None:
        tables = -1
    elif isinstance(maps, int):
        tables = maps
        maps = None
    else:
        maps = np.array(list(maps))
        umaps = np.unique(maps)
        if len(umaps) != len(maps) or (umaps != maps).any():
            raise ValueError("Must provide a unique sorted list of `maps`.")
        del umaps
        tables = max(maps)
    # read the first HDU, which will be empty for a BINTABLE extension
    # fits file
    next_hdu(infile)
    for i, [hdu, table] in enumerate(
            read_bintable_chunks(
                infile,
                tables=tables,
                buf_rows=buf_rows,
                extractor=extractor,
                row_calculator=row_calculator
            )
    ):
        us = []
        ss = []
        for u, [s] in table:
            if maps is None or i in maps:
                if chunk_processor is not None:
                    u, s = chunk_processor(u, s)
                if mask is not None:
                    ui = np.unique(uniq_intersection(u, mask)[0])
                    us.append(u[ui])
                    ss.append(s[ui])
                else:
                    us.append(u)
                    ss.append(s)
        u = np.concatenate(us)
        s = np.concatenate(ss)
        del us, ss
        meta = OrderedDict()
        for k, v, *_ in hdu.header.cards:
            if HEADER_NON_META.match(k) is not None:
                continue
            if k == 'HISTORY':
                if k in meta:
                    meta[k].append(v)
                else:
                    meta[k] = [v]
            elif k not in meta:
                meta[k] = v
        set_partial_skymap_metadata(meta, mask, load_ligo.__name__)
        if post_processor is not None:
            u, s = post_processor(u, s)
        yield u, s, meta


def bintable_dtype(
        hdu: 'BinTableHDU',
) -> List[Union[Tuple[str, str], Tuple[str, str, Tuple[int]]]]:
    """
    Get list that can be passed to ``numpy.dtype`` to define a structured
    datatype corresponding to the data in the table ``hdu``.
    """
    types = []
    for col in cast('ColDefs', hdu.columns):
        m = TFORM.match(cast(str, col.format))
        assert m is not None
        g = m.groups()
        dt = BINTABLE_TO_NUMPY_TYPES[g[2]]
        tup = (col.name or '', dt)
        if g[1]:
            tup += (int(g[1]),)
        types.append(tup)
    return types


def _next_header_blocks(stream: IO):
    block = stream.read(FITS_BLOCKSIZE)
    if not block:
        return block
    for i in range(FITS_HEADER_LENGTH):
        start = i*FITS_CARD_WIDTH
        if block[start:start+FITS_CARD_NAME_WIDTH] == b'END     ':
            return block
    return block + _next_header_blocks(stream)


def next_header(stream: IO) -> 'Header':
    "Read the next ``astropy.fits.Header`` from a fits file stream (bytes)."
    from astropy.io import fits

    header = _next_header_blocks(stream)
    if not header:
        raise EmptyStream("Stream ended.")
    return fits.Header.fromstring(header)


def next_hdu(stream: IO) -> 'ExtensionHDU':
    "Read the next FITS HDU from a fits file stream (bytes)."
    from astropy.io import fits

    header = _next_header_blocks(stream)
    if not header:
        raise EmptyStream("Stream ended.")
    return cast('ExtensionHDU', fits.HDUList.fromstring(header)[0])
