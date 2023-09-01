"""
TODO
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast, Any, Dict

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.axis import Axis
import matplotlib.pyplot as plt

from mcpipeline.curves import Curve


__all__ = ["Plotter"]


class Plotter:
    def __init__(self, save_fig_kwargs: Optional[dict] = None) -> None:
        self.save_fig_kwargs = {} if save_fig_kwargs is None else save_fig_kwargs

    def plot_curve(
        self,
        curves: List[Curve],
        ax: Optional[Axes] = None,
        rules: Optional[List] = None,
    ) -> Tuple[Figure, Axes]:
        if ax is None:
            fig = plt.gcf()

            ax = fig.gca()
        else:
            fig = ax.get_figure()

        for x, y in curves:
            ax.plot(x, y)

        if rules is not None:
            extra_ticks = []

            y_min, y_max = ax.get_ylim()

            for rule in rules:
                if type(rule) is tuple:
                    rule, kwargs = rule
                else:
                    kwargs = dict()

                ax.vlines(
                    cast(float, rule), y_min, y_max, linestyles="dashed", **kwargs
                )

                extra_ticks.append(rule)

            lim = ax.get_xlim()
            ax.set_xticks(list(ax.get_xticks()) + extra_ticks)
            ax.set_xlim(lim)

            ax.legend(loc="best")

        return (fig, ax)

    def save_fig(self, fig: Figure, fname, **kwargs):
        """TODO

        Parameters
        ----------
        fig : Figure
            _description_
        fname : _type_
            _description_
        """
        fname = Path(fname)

        os.makedirs(fname.parent, exist_ok=True)

        merged_kwargs = {**self.save_fig_kwargs, **kwargs}

        fig.savefig(str(fname), **merged_kwargs)
