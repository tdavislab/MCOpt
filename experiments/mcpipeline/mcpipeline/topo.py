"""
TODO
"""

from typing import Optional, Dict
from collections import UserDict

from vtkmodules.vtkFiltersCore import vtkThreshold
from vtkmodules.vtkFiltersGeneral import vtkDataSetTriangleFilter
from vtkmodules.vtkCommonDataModel import vtkDataSetAttributes, vtkDataObject


from mcpipeline.cache import cache
from mcpipeline.vtk import VTKData, VTKPolyData, VTKTable, MorseComplex
from mcpipeline.curves import Curve

__all__ = ["p_curve", "make_morse_complex"]


@cache.cache
def p_curve(data: VTKData, table: int = 0) -> Curve:
    from topologytoolkit import ttkPersistenceCurve  # type: ignore

    tetra = vtkDataSetTriangleFilter()
    tetra.SetInputConnection(data.conn())

    curve = ttkPersistenceCurve()  # type: ignore
    curve.SetInputConnection(tetra.GetOutputPort())
    curve.SetInputArrayToProcess(
        0, 0, 0, 0, vtkDataSetAttributes.SCALARS  # type: ignore
    )

    tbl = VTKTable(curve, index=table)

    return (tbl.column_numpy(0), tbl.column_numpy(1))


@cache.cache
def make_morse_complex(
    data: VTKData,
    persistence_threshold: float,
    ascending: bool = True,
    scalar_field: Optional[str] = None,
) -> MorseComplex:
    from topologytoolkit import ttkPersistenceDiagram, ttkTopologicalSimplification, ttkMorseSmaleComplex  # type: ignore

    def set_input_array(alg):
        if scalar_field is None:
            alg.SetInputArrayToProcess(0, 0, 0, 0, vtkDataSetAttributes.SCALARS)
        else:
            alg.SetInputArrayToProcess(0, 0, 0, 0, scalar_field)

    tetra = vtkDataSetTriangleFilter()
    tetra.SetInputConnection(data.conn())

    diagram = ttkPersistenceDiagram()
    diagram.SetInputConnection(tetra.GetOutputPort())
    set_input_array(diagram)

    crit_pairs = vtkThreshold()
    crit_pairs.SetInputConnection(diagram.GetOutputPort())
    crit_pairs.SetInputArrayToProcess(
        0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_CELLS, "PairIdentifier"
    )
    crit_pairs.SetThresholdFunction(vtkThreshold.THRESHOLD_BETWEEN)
    crit_pairs.SetLowerThreshold(-0.1)

    per_pairs = vtkThreshold()
    per_pairs.SetInputConnection(crit_pairs.GetOutputPort())
    per_pairs.SetInputArrayToProcess(
        0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_CELLS, "Persistence"
    )
    per_pairs.SetThresholdFunction(vtkThreshold.THRESHOLD_BETWEEN)
    per_pairs.SetLowerThreshold(persistence_threshold)

    simp = ttkTopologicalSimplification()
    simp.SetInputConnection(0, tetra.GetOutputPort())
    set_input_array(simp)
    simp.SetInputConnection(1, per_pairs.GetOutputPort())

    mc = ttkMorseSmaleComplex()
    mc.SetInputConnection(simp.GetOutputPort())
    set_input_array(mc)

    mc.SetComputeCriticalPoints(True)
    mc.SetComputeAscendingSeparatrices1(False)
    mc.SetComputeAscendingSeparatrices2(False)
    mc.SetComputeAscendingSegmentation(ascending)
    mc.SetComputeDescendingSegmentation(not ascending)
    mc.SetComputeDescendingSeparatrices1(ascending)
    mc.SetComputeDescendingSeparatrices2(not ascending)

    return MorseComplex(
        critical_points=VTKPolyData(mc, 0),
        separatrices=VTKPolyData(mc, 1),
        segmentation=VTKData(mc, 3),
    )
