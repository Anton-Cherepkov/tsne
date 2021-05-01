from typing import Callable, Optional
from numpy import inf
import math
import sys


def binary_search(
    func: Callable[[float], float],
    target: float,
    x0: float = 1.0,
    tol: float = 1e-5,
    n_steps: int = 100
) -> Optional[float]:
    """
    func is some decreasing function
    """

    left = -inf
    right = inf
    mid = x0
    
    for _ in range(n_steps):
        diff = func(mid) - target

        if math.fabs(diff) < tol:
            return mid
        
        if diff < 0:
            left = mid
            if right == inf:
                mid *= 2
            else:
                mid = (mid + right) / 2
        else:
            right = mid
            if left == -inf:
                mid /= 2
            else:
                mid = (mid + left) / 2
    
    print(f'Binary search has not converged, current diff={diff}', file=sys.stderr)
    return mid
