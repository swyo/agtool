import random
import string
from typing import Tuple, List, Optional


def randomString(length: int, samples: Optional[List[str]] = None) -> str:
    """A random string is generated.

    Args:
        length: the size of string.
        samples: cumstomized letters for sampling. e.g., ['a', 'b'].
    """
    letters = string.ascii_lowercase
    if samples:
        letters = samples
    return ''.join(random.choice(letters) for i in range(length))


def random2Dcharacters(
    shape: Tuple[int, int], samples: Optional[List[str]] = None
) -> List[List[str]]:
    """A random 2D characters array is generated.

    Args:
        shape: (m, n) shape of the characters array.
        samples: cumstomized letters for sampling. e.g., ['a', 'b'].
    """
    m, n = shape
    letters = string.ascii_lowercase
    if samples:
        letters = samples
    return [[random.choice(letters) for i in range(n)] for _ in range(m)]
