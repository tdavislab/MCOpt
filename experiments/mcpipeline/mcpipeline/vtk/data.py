"""
Wrappers for VTK data that add some utilities and allow for caching
"""

from __future__ import annotations

from typing import NamedTuple, Dict
from pathlib import Path

import pandas as pd
from numpy.typing import NDArray
import numpy as np

from vtkmodules.vtkCommonCore import vtkIdList, VTK_DOUBLE
from vtkmodules.vtkCommonDataModel import (
    vtkDataObject,
    vtkImageData,
    vtkPolyData,
    vtkRectilinearGrid,
    vtkStructuredGrid,
    vtkTable,
    vtkUnstructuredGrid,
)
from vtkmodules.vtkCommonExecutionModel import vtkAlgorithm, vtkAlgorithmOutput
from vtkmodules.vtkFiltersSources import vtkPlaneSource
from vtkmodules.vtkIOXML import *
from vtkmodules.numpy_interface import dataset_adapter
from vtkmodules.util.numpy_support import numpy_to_vtk
import vtk

__all__ = ["VTKData", "VTKTable", "VTKPolyData"]

# IO interfaces for supported data types
IFace = NamedTuple("IFace", writer=vtkXMLWriter, reader=vtkXMLReader, ext=str)

VTK_XML_IO: Dict[type, IFace] = {
    vtkImageData: IFace(
        ext="vti",
        writer=vtkXMLImageDataWriter,
        reader=vtkXMLImageDataReader,
    ),
    vtkPolyData: IFace(
        ext="vtp",
        writer=vtkXMLPolyDataWriter,
        reader=vtkXMLPolyDataReader,
    ),
    vtkRectilinearGrid: IFace(
        ext="vtr",
        writer=vtkXMLRectilinearGridWriter,
        reader=vtkXMLRectilinearGridReader,
    ),
    vtkStructuredGrid: IFace(
        ext="vts",
        writer=vtkXMLStructuredGridWriter,
        reader=vtkXMLStructuredGridReader,
    ),
    vtkTable: IFace(
        ext="vtt",
        writer=vtkXMLTableWriter,
        reader=vtkXMLTableReader,
    ),
    vtkUnstructuredGrid: IFace(
        ext="vtu",
        writer=vtkXMLUnstructuredGridWriter,
        reader=vtkXMLUnstructuredGridReader,
    ),
}

VTK_EXT_TO_READER = {iface.ext: iface.reader for iface in VTK_XML_IO.values()}


def get_io_iface(ty: type) -> IFace:
    if ty not in VTK_XML_IO:
        raise ValueError(f"No supported VTK IO interface for type {ty}")

    return VTK_XML_IO[ty]


class VTKData:
    """A wrapper for VTK data objects that enables caching."""

    @staticmethod
    def load(path: Path) -> VTKData:
        """Loads VTK data from disk using the given path.

        Parameters
        ----------
        path : Path
            The path to load the date from.

        Returns
        -------
        VTKData
            The loaded data.

        Raises
        ------
        ValueError
            Raised if the filetype is not recognized.
        """

        ext = path.suffix

        if f".{ext}" not in VTK_EXT_TO_READER:
            raise ValueError(f"No supported VTK reader for file type {ext}")

        reader = VTK_EXT_TO_READER[ext]()
        reader.SetFileName(path)
        reader.Update()

        return VTKData(reader)

    def __init__(self, src: vtkAlgorithm, port: int = 0) -> None:
        """Wraps the output of a vtkAlgorithm. We wrap the algorithm instead of
        the data object, because it prevents some memory issues with the vtk python
        wrapper and improves the performance of chaining algorithms.

        Parameters
        ----------
        src : vtkAlgorithm
                        The algorithm that produces the data to wrap.
        port : int, optional
                        The index of the port from `src` that produces the desired data, by default 0.
        """

        self._src = src
        self._index = port

    def get(self) -> vtkDataObject:
        """Get the VTK wrapped data object.

        Returns
        -------
        vtkDataObject
                        The data object.
        """
        self._src.Update()

        return self._src.GetOutputDataObject(self._index)

    def conn(self) -> vtkAlgorithmOutput:
        """Get a port that produces the wrapped data.

        Returns
        -------
        vtkAlgorithmOutput
                        The port.
        """
        return self._src.GetOutputPort(self._index)

    def save(self, path: Path, append_ext: bool = True):
        """Saves the data to disk at the given path.

        Parameters
        ----------
        path : Path
            The path to save the file to.
        append_ext : bool, optional
            If `True` and the path does not have an extension, then an appropriate
            extension is appended, by default True.
        """

        ty = type(self.get())

        iface = get_io_iface(ty)

        if path.suffix == "" and append_ext:
            path.with_suffix(f".{iface.ext}")

        writer = iface.writer()
        writer.SetInputConnection(self.conn())
        writer.SetFileName(path)
        writer.Write()

    # __getstate__ and __setstate__ override the behavior of the pickler to use
    # the return value. This allows for caching using joblib or similar.

    def __getstate__(self) -> dict:
        ty = type(self.get())

        iface = get_io_iface(ty)

        writer = iface.writer()
        writer.SetInputConnection(self.conn())
        writer.SetWriteOutputToString(True)
        writer.Write()

        return {"data": writer.GetOutputString(), "ty": ty}

    def __setstate__(self, state: dict):
        ty = state["ty"]
        data_str = state["data"]

        iface = get_io_iface(ty)

        reader = iface.reader()
        reader.ReadFromInputStringOn()
        reader.SetInputString(data_str)
        reader.Update()  # ensure there are no errors with encoding

        self._src = reader
        self._index = 0


class VTKTable(VTKData):
    """A wrapper for VTK tables."""

    def __init__(self, src: vtkAlgorithm, port: int = 0) -> None:
        super().__init__(src, port)

        assert isinstance(self.get(), vtkTable)

    def get(self) -> vtkTable:
        return super().get()  # type: ignore


class VTKPolyData(VTKData):
    """A wrapper for VTK polygon data."""

    @staticmethod
    def make_plane(scalars: NDArray[np.float_]):
        assert scalars.ndim == 2

        plane = vtkPlaneSource()
        plane.SetResolution(scalars[0] - 1, scalars[1] - 1)
        plane.SetOrigin([0, 0, 0])
        plane.SetPoint1(scalars.shape[0], 0, 0)
        plane.SetPoint2(0, scalars.shape[1], 0)

        data = numpy_to_vtk(scalars.ravel(), deep=True, array_type=VTK_DOUBLE)
        data.SetName("data")

        plane.GetOutput().GetPointData().SetScalars(data)

        return VTKPolyData(plane)

    def __init__(self, src: vtkAlgorithm, port: int = 0) -> None:
        super().__init__(src, port)

        assert isinstance(self.get(), vtkPolyData)

    def get(self) -> vtkPolyData:
        return super().get()  # type: ignore

    def point_data_df(self) -> pd.DataFrame:
        """Creates a DataFrame containing the point data that mimics the
        conventions of Paraview's exported spreadsheets.

        Returns
        -------
        pd.DataFrame
            The point data DataFrame.
        """

        adapter = dataset_adapter.WrapDataObject(self.get())

        point_data = pd.DataFrame(dict(adapter.PointData))  # type: ignore

        point_data.index.names = ["Point ID"]

        points = pd.DataFrame(adapter.Points, columns=["Points_0", "Points_1"])  # type: ignore

        return pd.concat([point_data, points], axis=1)

    def cell_data_df(self) -> pd.DataFrame:
        """Creates a DataFrame containing the cell data that mimics the
        conventions of Paraview's exported spreadsheets.

        Returns
        -------
        pd.DataFrame
            The cell data DataFrame.
        """

        poly = self.get()
        adapter = dataset_adapter.WrapDataObject(poly)

        cells = pd.DataFrame(dict(adapter.CellData))  # type: ignore

        cells.index.names = ["Cell Id"]

        # XXX: This is kinda slow
        id_list = vtkIdList()
        for cell_id in range(cells.shape[0]):
            poly.GetCellPoints(cell_id, id_list)

            for i in range(id_list.GetNumberOfIds()):
                k = f"Point Index {i}"

                if k not in cells:
                    cells[k] = pd.Series(dtype="Int64")

                cells.at[cell_id, k] = id_list

        return cells
