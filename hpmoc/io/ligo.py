"""
Strategies for loading LIGO/Virgo/KAGRA skymaps into ``PartialUniqSkymap``
instances.
"""

from typing import Optional, Union, IO
from nptyping import NDArray
from .abstract import IoStrategy
from ..fits import load_ligo
from ..partial import PartialUniqSkymap
from ..utils import uniq_coarsen, uniq_minimize


class LigoIo(IoStrategy):
    """
    Read/write files in the format used by LIGO/Virgo for their skymaps.
    """

    @staticmethod
    def read(
            mask: Optional[Union[PartialUniqSkymap, NDArray]],
            file: Union[IO, str],
            *args,
            name: str = 'PROBDENSITY',
            coarsen: Optional[int] = None,
            **kwargs
    ):
        """
        Read a file saved in the format used by LIGO/Virgo for their skymaps.

        Parameters
        ----------
        mask : PartialUniqSkymap or array, optional
            Only read in pixels overlapping with ``mask``.
        file : file or str
            The file object or filename to read from. Can be a stream as no
            seeking will be performed.
        name : str, optional
            The column-name of the pixel data.
        coarsen : int, optional
            If provided, coarsen the ``mask`` by up to this many HEALPix
            orders (up to order 0) to speed up read times. This will select
            a superset of the sky region defined in ``mask``.
        *args, **kwargs
            Arguments to pass on to ``hpmoc.fits.load_ligo``.
        """
        pt = mask.point_sources if isinstance(mask, PartialUniqSkymap) else []
        if mask is not None:
            mask = mask.u if isinstance(mask, PartialUniqSkymap) else mask
            mask = uniq_coarsen(mask, coarsen) if coarsen is not None else mask
            mask, = uniq_minimize(mask)
        [[u, s, meta]] = load_ligo(file, mask=mask, **kwargs)
        return PartialUniqSkymap(s, u, name=name, meta=meta, point_sources=pt)

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
