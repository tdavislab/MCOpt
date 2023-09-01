"""
TODO
"""

from typing import Optional

from mcopt import MorseGraph

from mcpipeline.vtk import MorseComplex
from mcpipeline.cache import cache


@cache.cache
def make_morse_graph(mc: MorseComplex, sample_rate: Optional[int] = None) -> MorseGraph:
    graph = MorseGraph.from_paraview_df(
        critical_points=mc.critical_points.point_data_df(),
        separatrices_points=mc.separatrices.point_data_df(),
        separatrices_cells=mc.separatrices.cell_data_df(),
    )

    if sample_rate is not None:
        return graph.sample(sample_rate)

    return graph
