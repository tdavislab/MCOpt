"""
Utilities for generating datasets
"""

from typing import Any, Tuple, Union, cast

from numpy.typing import NDArray
import numpy as np

__all__ = ["Sinusoidal", "Distance", "Normal"]


def Sinusoidal(shape: Tuple[int, int], npeaks: int = 3) -> NDArray[np.float_]:
    def height(x: int, y: int):
        return 0.5 * (
            np.sin(np.pi * (2 * x * npeaks / shape[0] - 0.5))
            + np.sin(np.pi * (2 * y * npeaks / shape[1] - 0.5))
        )

    return np.fromfunction(height, shape, dtype=np.float_)


def Distance(shape: Tuple[int, int]) -> NDArray[np.float_]:
    def height(x: int, y: int):
        return np.sqrt((x / shape[0] - 0.5) ** 2 + (y / shape[1] - 0.5) ** 2)

    return np.fromfunction(height, shape=shape, dtype=np.float_)


def Normal(
    shape: Tuple[int, int],
    center: Tuple[int, int],
    sigma: Union[float, Tuple[float, float]] = 1,
) -> NDArray[np.float_]:
    if not hasattr(sigma, "__len__"):
        sigma_x = cast(float, sigma)
        sigma_y = cast(float, sigma)
    else:
        sigma_x, sigma_y = sigma  # type: ignore

    def gauss(x: int, y: int):
        return (
            1
            / (2 * np.pi * sigma_x * sigma_y)
            * np.exp(
                -((x - center[0]) ** 2) / (2 * sigma_x**2)
                - (y - center[1]) ** 2 / (2 * sigma_y**2)
            )
        )

    return np.fromfunction(gauss, shape=shape, dtype=np.float_)


def GaussianNoise(
    shape: Tuple[int, int], random_state: Any = None
) -> NDArray[np.float_]:
    rng = np.random.default_rng(random_state)

    return rng.normal(size=shape)
