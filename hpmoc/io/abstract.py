"""
Abstract definitions of ``IoStrategy``.
"""

from typing import Optional, Union, IO
from ..partial import PartialUniqSkymap
from numpy.typing import ArrayLike

class IoStrategy:
    """
    Methods for reading and writing ``PartialUniqSkymap`` instances from/to
    file.
    """
    @classmethod
    def read(
        cls,
        skymap: Optional[Union[PartialUniqSkymap, ArrayLike]],
        file: Union[IO, str],
        *args,
        name: Optional[str] = None,
        uname: str = 'UNIQ',
        empty = None,
        **kwargs
    ) -> PartialUniqSkymap:
        raise NotImplementedError("read")

    @classmethod
    def write(
        cls,
        skymap: PartialUniqSkymap,
        file: Union[IO, str],
        name: Optional[str] = None,
        uname: Optional[str] = 'UNIQ',
        *args,
        **kwargs
    ):
        raise NotImplementedError("write")

class StubIo(IoStrategy):
    """
    A placeholder for an IO strategy that is included with ``HPMOC`` but
    requires dependencies that are not currently installed.
    """
    qualname: str
    requirements: str

    @classmethod
    def read(cls, *args, **kwargs):
        raise ImportError(f"This is a stub for {cls.qualname}. You need to "
                          f"install {cls.requirements} to be able to use it.")

    write = read


class ReadonlyIo(IoStrategy):
    @classmethod
    def write(cls, *args, **kwargs):
        raise NotImplementedError("This is a read-only IO method.")
