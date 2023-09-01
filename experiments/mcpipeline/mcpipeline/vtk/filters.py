"""
A collection of VTK Filter wrappers
"""

from abc import ABC, abstractmethod
from mcpipeline.vtk.data import VTKData

from vtkmodules.vtkFiltersGeneral import (
    vtkWarpScalar,
    vtkBoxClipDataSet,
    vtkTransformFilter,
)
from vtkmodules.vtkImagingCore import vtkImageClip
from vtkmodules.vtkCommonTransforms import vtkTransform
import vtk

from mcpipeline.vtk.data import VTKData

__all__ = [
    "VTKFilter",
    "WarpFilter",
    "BoxClipDataSetFilter",
    "ImageClipFilter",
    "TranslateFilter",
]


class VTKFilter(ABC):
    """An abstract class representing VTK Filters."""

    @abstractmethod
    def __call__(self, input: VTKData) -> VTKData:
        raise NotImplementedError()


class WarpFilter(VTKFilter):
    def __init__(self, scale_factor: float):
        super().__init__()

        self.scale_factor = scale_factor

    def __call__(self, input: VTKData) -> VTKData:
        warp = vtkWarpScalar()
        warp.SetInputConnection(input.conn())
        warp.SetScaleFactor(self.scale_factor)

        return VTKData(warp)


class BoxClipDataSetFilter(VTKFilter):
    def __init__(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        zmin: float,
        zmax: float,
    ) -> None:
        super().__init__()

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax

    def __call__(self, input: VTKData) -> VTKData:
        clip = vtkBoxClipDataSet()
        clip.SetInputConnection(input.conn())
        clip.SetBoxClip(
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
            self.zmin,
            self.zmax,
        )

        return VTKData(clip)


class ImageClipFilter(VTKFilter):
    def __init__(
        self,
        xmin: int,
        xmax: int,
        ymin: int,
        ymax: int,
        zmin: int,
        zmax: int,
    ) -> None:
        super().__init__()

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax

    def __call__(self, input: VTKData) -> VTKData:
        clip = vtkImageClip()
        clip.SetInputConnection(input.conn())
        clip.SetOutputWholeExtent(
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
            self.zmin,
            self.zmax,
        )
        clip.ClipDataOn()

        return VTKData(clip)


class TranslateFilter(VTKFilter):
    def __init__(self, dx: float = 0, dy: float = 0, dz: float = 0) -> None:
        super().__init__()

        self.dx = dx
        self.dy = dy
        self.dz = dz

    def __call__(self, input: VTKData) -> VTKData:
        filter = vtkTransformFilter()
        filter.SetInputConnection(input.conn())

        trans = vtkTransform()
        trans.Translate(self.dx, self.dy, self.dz)

        filter.SetTransform(trans)

        return VTKData(filter)
