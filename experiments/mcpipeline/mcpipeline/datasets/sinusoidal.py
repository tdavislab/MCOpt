"""
TODO
"""

from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from mcpipeline.vtk import WarpFilter
from mcpipeline.datasets.dataset import GenDataset
import mcpipeline.gen as gen


class Sinusoidal(GenDataset):
    """TODO"""

    def __init__(self):
        super().__init__([WarpFilter(scale_factor=50)])

    @property
    def name(self) -> str:
        return "sinusoidal"

    def generate_steps(self) -> Iterable[NDArray[np.float_]]:
        shape = (100, 100)

        initial = gen.Sinusoidal(shape=shape, npeaks=3) * 0.5
        initial += gen.Distance(shape=shape) * -0.5

        yield initial + gen.Noise(shape=shape, scale=0.1, random_state=42)

        yield initial + gen.Noise(shape=shape, scale=0.25, random_state=42)