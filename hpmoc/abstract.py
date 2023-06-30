"""
Define an ``AbstractPartialUniqSkymap`` interface.
"""

from typing import Union, Any, Generic, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

_DType = TypeVar('_DType', covariant=True, bound='np.generic')
class AbstractPartialUniqSkymap(Generic[_DType]):
    s: 'npt.NDArray[_DType]'
    u: 'npt.NDArray[np.integer[Any]]'
