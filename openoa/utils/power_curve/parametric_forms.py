"""
Power Curves

These curve functions are written in the style of Scipy.optimize:

fun : callable
The objective function to be minimized.
fun(x, *args) -> float
where x is an 1-D array with shape (n,) and args is a tuple of the fixed parameters needed to completely specify the
function.

ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize

"""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import pandas as pd
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.float64]
ArrayLike = TypeVar("ArrayLike", NDArrayFloat, "pd.Series[float]")


def _power_curve(
    x: NDArrayFloat | pd.Series[float], a: float, b: float, c: float, d: float, g: float
) -> NDArrayFloat:
    """The actual power curve implementation.

    Args:
        x(:obj:`numpy.ndarray` | `pandas.Series`): input data
        a(:obj:`float`): the minimum value asymptote
        b(:obj:`float`): steepness of the curve
        c(:obj:`float`): the point of inflection
        d(:obj:`float`): the maximum value asymptote
        g(:obj:`float`): asymmetry of the curve

    Returns:
        (:obj:`numpy.ndarray`): The converted power data.
    """
    values = _to_array(x)
    result: NDArrayFloat = d + (a - d) / (1 + (values / c) ** b) ** g
    return result


def _to_array(x: NDArrayFloat | pd.Series[float]) -> NDArrayFloat:
    if isinstance(x, pd.Series):
        return np.asarray(x.to_numpy(), dtype=np.float64)
    return x


def logistic5param(
    x: NDArrayFloat | pd.Series[float], a: float, b: float, c: float, d: float, g: float
) -> NDArrayFloat:
    """Create and return a 5 parameter logistic function

    Args:
        x(:obj:`numpy.ndarray` | `pandas.Series`): Input data.
        a(:obj:`float`): The minimum value asymptote.
        b(:obj:`float`): Steepness of the curve.
        c(:obj:`float`): The point of inflection.
        d(:obj:`float`): The maximum value asymptote.
        g(:obj:`float`): Asymmetry of the curve.

    Returns:
        Function[numpy.ndarray[real]] -> numpy.ndarray[real]

    """

    values = _to_array(x)
    res: NDArrayFloat = np.ones_like(values, dtype=np.float64)
    # In the case where b<0, x==0, there is a divide by zero error. The answer should be "d" when x==0 and b<0.
    dom: npt.NDArray[np.bool_] | slice
    if b < 0:
        res *= d  # Initialize result, default value is d
        dom = values != 0.0  # Only nonzero elements in domain
    else:
        dom = slice(None)  # All elements in domain

    # Apply power curve definition to point within domain
    res[dom] = _power_curve(values[dom], a, b, c, d, g)

    return res


def logistic5param_capped(
    x: NDArrayFloat | pd.Series[float],
    a: float,
    b: float,
    c: float,
    d: float,
    g: float,
    lower: float,
    upper: float,
) -> NDArrayFloat:
    """Create and return a capped 5 parameter logistic function whose output is capped by lower and upper bounds.

    Args:
        x(:obj:`numpy.ndarray` | `pandas.Series`): Input data.
        a(:obj:`float`): The minimum value asymptote.
        b(:obj:`float`): Steepness of the curve.
        c(:obj:`float`): The point of inflection.
        d(:obj:`float`): The maximum value asymptote.
        g(:obj:`float`): Asymmetry of the curve.
        lower(:obj:`float`): Input values below this number are set to this number.
        upper(:obj:`float`): Input values above this number are set to this number.

    Returns:
        Function[numpy.ndarray[real]] -> numpy.ndarray[real]

    """
    return _cap(logistic5param(x, a, b, c, d, g), lower, upper)


"""
Helpers for Power Curve
"""


def _cap(y: ArrayLike, lower: float, upper: float) -> ArrayLike:
    if isinstance(y, np.ndarray):
        y = np.where(y < lower, lower, y)
        y = np.where(y > upper, upper, y)
    else:
        y = y.where(y > lower, lower)
        y = y.where(y < upper, upper)
    return y
