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

"""
Define an ``AbstractPartialUniqSkymap`` interface.
"""

from typing import Any, Generic, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

_DType = TypeVar('_DType', covariant=True, bound='np.generic')
class AbstractPartialUniqSkymap(Generic[_DType]):
    s: 'npt.NDArray[_DType]'
    u: 'npt.NDArray[np.integer[Any]]'
