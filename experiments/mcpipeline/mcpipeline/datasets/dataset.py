"""
Definitions for dataset classes
"""

from __future__ import annotations

from typing import Optional, List, Dict, Iterable
from abc import ABC, abstractmethod
import textwrap

from numpy.typing import NDArray
import numpy as np
from experiments.mcpipeline.mcpipeline.vtk import VTKData, VTKFilter


from mcpipeline.vtk import VTKFilter, VTKData, VTKPolyData


class Dataset(ABC):
    """An abstract class defining the general behavior of datasets.

    Attributes
    ----------
    steps: Dict[int, VTKData]
        A mapping from time steps to the VTKData associated with that time.
    """

    def __init__(
        self, steps: Dict[int, VTKData], filters: Optional[List[VTKFilter]] = None
    ):
        """Constructs a new dataset, applying the given filters to the given steps.
        Steps are a mapping from a time step to the VTKData for that step.

        Parameters
        ----------
        steps : Dict[int, VTKData]
            The steps.
        filters : Optional[List[VTKFilter]], optional
            A list of filters to apply to the time steps, by default None
        """

        super().__init__()

        if filters is not None:
            for t, dat in steps.items():
                for f in filters:
                    dat = f(dat)

                steps[t] = dat

        self.steps = steps

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the dataset."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def _desc(self) -> str:
        """A description for the dataset in Markdown."""
        raise NotImplementedError()

    @property
    def description(self) -> str:
        """A formatted description for the dataset."""
        return textwrap.dedent(self._desc)


class GenDataset(Dataset, ABC):
    """A dataset which derives it's timesteps from the return value of `generate_steps`."""

    def __init__(self, filters: Optional[List[VTKFilter]] = None) -> None:
        steps: Dict[int, VTKData] = {
            t: VTKPolyData.make_plane(arr)
            for t, arr in enumerate(self.generate_steps())
        }

        super().__init__(steps, filters)

    @abstractmethod
    def generate_steps(self) -> Iterable[NDArray[np.float_]]:
        """Generates the steps for the dataset, the timesteps will be assigned as
        0, 1, 2, ...

        Returns
        -------
        Iterable[NDArray[np.float_]]
            The steps for the dataset.
            _description_
        """

        raise NotImplementedError()
