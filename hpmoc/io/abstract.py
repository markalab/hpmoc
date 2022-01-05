"""
Abstract definitions of ``IoStrategy``.
"""

from typing import Callable
from ..partial import PartialUniqSkymap


class IoStrategy:
    """
    Methods for reading and writing ``PartialUniqSkymap`` instances from/to
    file.
    """
    read: Callable
    write: Callable


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

    @staticmethod
    def write(*args, **kwargs):
        raise NotImplementedError("This is a read-only IO method.")
