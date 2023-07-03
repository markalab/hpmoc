# Copyright (C) 2022-2023 Columbia Experimental Gravity Group (GECo)

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

"""
Abstract definitions of ``IoStrategy``.
"""

from typing import Optional, Union, IO, ClassVar
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
    qualname: ClassVar[str]
    requirements: ClassVar[str]

    @classmethod
    def read(cls, *args, **kwargs):
        raise ImportError(f"This is a stub for {cls.qualname}. You need to "
                          f"install {cls.requirements} to be able to use it.")

    write = read


class ReadonlyIo(IoStrategy):
    @classmethod
    def write(cls, *args, **kwargs):
        raise NotImplementedError("This is a read-only IO method.")
