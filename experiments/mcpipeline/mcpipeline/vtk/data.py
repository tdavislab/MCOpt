"""
Wrappers for VTK data that add some utilities and allow for caching
"""

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
from vtkmodules.vtkIOXML import *

# import vtk

__all__ = ["VTKData", "VTKTable"]

# IO interfaces for supported data types
VTK_XML_IO = {
    vtkImageData: {
        "ext": "vti",
        "writer": vtkXMLImageDataWriter,
        "reader": vtkXMLImageDataReader,
    },
    vtkPolyData: {
        "ext": "vtp",
        "writer": vtkXMLPolyDataWriter,
        "reader": vtkXMLPolyDataReader,
    },
    vtkRectilinearGrid: {
        "ext": "vtr",
        "writer": vtkXMLRectilinearGridWriter,
        "reader": vtkXMLRectilinearGridReader,
    },
    vtkStructuredGrid: {
        "ext": "vts",
        "writer": vtkXMLStructuredGridWriter,
        "reader": vtkXMLStructuredGridReader,
    },
    vtkTable: {
        "ext": "vtt",
        "writer": vtkXMLTableWriter,
        "reader": vtkXMLTableReader,
    },
    vtkUnstructuredGrid: {
        "ext": "vtu",
        "writer": vtkXMLUnstructuredGridWriter,
        "reader": vtkXMLUnstructuredGridReader,
    },
}


def get_io_iface(ty: type) -> vtkXMLWriter:
    if ty not in VTK_XML_IO:
        raise ValueError(f"No supported VTK IO interface for type {ty}")

    return VTK_XML_IO[ty]


class VTKData:
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

    # __getstate__ and __setstate__ override the behavior of the pickler to use
    # the return value. This allows for caching using joblib or similar.

    def __getstate__(self) -> dict:
        ty = type(self.get())

        iface = get_io_iface(ty)

        writer = iface["writer"]()
        writer.SetInputConnection(self.conn())
        writer.SetWriteOutputToString(True)
        writer.Write()

        return {"data": writer.GetOutputString(), "ty": ty}

    def __setstate__(self, state: dict):
        ty = state["ty"]
        data_str = state["data"]

        iface = get_io_iface(ty)

        reader = iface["reader"]()
        reader.ReadFromInputStringOn()
        reader.SetInputString(data_str)
        reader.Update()  # ensure there are no errors with encoding

        self._src = reader
        self._index = 0


class VTKTable(VTKData):
    def __init__(self, src: vtkAlgorithm, port: int = 0) -> None:
        super().__init__(src, port)

        assert isinstance(self.get(), vtkTable)
