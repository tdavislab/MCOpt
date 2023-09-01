"""
TODO
"""

from collections import UserDict
from pathlib import Path
import os

from mcpipeline.vtk import MorseComplex

class Complexes(UserDict[int, MorseComplex]):
    def __init__(self, steps):
        super().__init__(steps)

    def save(self, output_dir: Path):
        os.makedirs(output_dir, exist_ok=True)

        for t, mc in self.items():
            mc.critical_points.save(output_dir / f'critical_points{t}', append_ext=True)
            mc.separatrices.save(output_dir / f'separatrices{t}', append_ext=True)
            mc.segmentation.save(output_dir / f'segmentation{t}', append_ext=True)

