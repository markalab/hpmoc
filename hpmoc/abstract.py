"""
Define an ``AbstractPartialUniqSkymap`` interface.
"""

from typing import Union, Any, TYPE_CHECKING
from nptyping import NDArray, Int

if TYPE_CHECKING:
    from astropy.units.quantity import Quantity

class AbstractPartialUniqSkymap:
    s: Union[NDArray[Any, Any], 'Quantity']
    u: NDArray[Any, Int]