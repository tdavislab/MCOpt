"""
Definitions for dataset classes
"""

from __future__ import annotations

from typing import Optional, List, Dict, Iterable, Any
from abc import ABC, abstractmethod
from collections import UserDict
import inspect
import os
from pathlib import Path
import textwrap
import re
from functools import cached_property

from numpy.typing import NDArray
import numpy as np

from mcpipeline.vtk import VTKFilter, VTKData, VTKPolyData, MorseComplex
from mcpipeline.topo import p_curve, make_morse_complex
from mcpipeline.curves import Curve
from mcpipeline.complexes import Complexes

__all__ = ["Dataset", "GenDataset"]


class Dataset(ABC, UserDict[int, VTKData]):
    """An abstract class defining the general behavior of datasets.

    Attributes
    ----------
    steps: Dict[int, VTKData]
        A mapping from time steps to the VTKData associated with that time.
    """

    def __init__(
        self,
        steps: Dict[int, VTKData],
        filters: Optional[List[VTKFilter]] = None,
        scalar_field_name: Optional[str] = None,
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

        if filters is not None:
            for t, dat in steps.items():
                for f in filters:
                    dat = f(dat)

                steps[t] = dat

        super().__init__(steps)

        self.scalar_field_name = scalar_field_name

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the dataset."""
        raise NotImplementedError()

    @property
    def description(self) -> str:
        """A formatted description for the dataset."""

        doc = inspect.getdoc(self)

        if doc is not None:
            doc = textwrap.dedent(doc)

        return f"# {self.name.title()}\n{doc}"

    @cached_property
    def sanitized_name(self) -> str:
        return re.sub(r"\s+", "_", re.sub(r"[^\w\s]", "", self.name))

    def save(self, output_dir: Path):
        os.makedirs(output_dir, exist_ok=True)

        for t, dat in self.items():
            dat.save(output_dir / f"{self.sanitized_name}{t}", append_ext=True)

    def p_curves(self) -> Dict[int, Curve]:
        return {t: p_curve(dat, table=0) for t, dat in self.items()}

    def morse_complexes(
        self, persistence_threshold: float, ascending: bool = True
    ) -> Complexes:
        return Complexes({
            t: make_morse_complex(
                dat,
                persistence_threshold,
                ascending=ascending,
                scalar_field=self.scalar_field_name,
            )
            for t, dat in self.items()
        })


class GenDataset(Dataset, ABC):
    """A dataset which derives it's timesteps from the return value of `generate_steps`."""

    def __init__(
        self,
        filters: Optional[List[VTKFilter]] = None,
        gen_args: Optional[List[Any]] = None,
        gen_kwargs: Optional[dict] = None,
    ) -> None:
        if gen_args is None:
            gen_args = []
        if gen_kwargs is None:
            gen_kwargs = {}

        steps: Dict[int, VTKData] = {
            t: VTKPolyData.make_plane(arr)
            for t, arr in enumerate(self.generate_steps(*gen_args, **gen_kwargs))
        }

        super().__init__(steps, filters)

    @abstractmethod
    def generate_steps(self, *args, **kwargs) -> Iterable[NDArray[np.float_]]:
        """Generates the steps for the dataset, the timesteps will be assigned as
        0, 1, 2, ...

        Returns
        -------
        Iterable[NDArray[np.float_]]
            The steps for the dataset.
            _description_
        """

        raise NotImplementedError()
