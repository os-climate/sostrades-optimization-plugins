"""
Copyright 2024 Capgemini

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    TwoAxesInstanciatedChart as BaseTwoAxesInstanciatedChart,
)
from sostrades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import (
    InstantiatedPlotlyNativeChart as BaseInstantiatedPlotlyNativeChart,
)

from sostrades_optimization_plugins.tools.plot_tools.color_map import ColorMap
from sostrades_optimization_plugins.tools.plot_tools.colormaps import (
    available_colormaps,
)
from sostrades_optimization_plugins.tools.plot_tools.palettes import available_palettes

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from sostrades_optimization_plugins.tools.plot_tools.color_palette import (
        ColorPalette,
    )


DEFAULT_PALETTE: ColorPalette | None = None
DEFAULT_COLORMAP: ColorMap | None = None


def set_default_palette(
    palette: str | ColorPalette | None = None,
) -> ColorPalette | None:
    """Set default palette to new value.

    Example:
        >>> from sostrades_optimization_plugins.tools.plot_tools.color_palette import ColorPalette
        >>> palette = set_default_palette(ColorPalette(name="witness"))
        >>> palette.name
        'witness'
    """
    global DEFAULT_PALETTE

    if isinstance(palette, str):
        if palette in available_palettes:
            DEFAULT_PALETTE = available_palettes[palette]
            return DEFAULT_PALETTE

    DEFAULT_PALETTE = palette
    return DEFAULT_PALETTE


def set_default_colormap(colormap: str | ColorMap | None = None) -> ColorMap | None:
    """Set default colormap to new value.

    Example:
        >>> colormap = set_default_colormap(ColorMap(name="sectors", color_map={}))
        >>> print(colormap.name)
        sectors
    """
    global DEFAULT_COLORMAP

    if isinstance(colormap, str):
        if colormap in available_colormaps:
            DEFAULT_COLORMAP = available_colormaps[colormap]
        return DEFAULT_COLORMAP

    DEFAULT_COLORMAP = colormap
    return DEFAULT_COLORMAP


T = TypeVar("T", bound="ExtendedMixin")


# Mixin class with common additional methods
class ExtendedMixin(Generic[T]):
    color_palette: ColorPalette | None = None
    color_map: ColorMap | None = None
    group_name: str | None = None
    layout_custom_updates: dict | None = None
    xaxes_custom_updates: dict | None = None
    yaxes_custom_updates: dict | None = None

    # Vars that do not have setters yet.
    subtitle: str | None = None
    use_scattergl: bool = False
    flag_remove_empty_traces: bool = False

    def __init__(self, *args, **kwargs):
        if "color_palette" in kwargs:
            self.set_color_palette(kwargs.pop("color_palette"))
        else:
            self.set_color_palette(DEFAULT_PALETTE)

        if "group_name" in kwargs:
            self.set_group(kwargs.pop("group_name"))

        if "color_map" in kwargs:
            self.set_color_map(kwargs.pop("color_map"))
        else:
            self.set_color_map(DEFAULT_COLORMAP)

        if "use_scattergl" in kwargs:
            self.use_scattergl = kwargs.pop("use_scattergl")

        super().__init__(*args, **kwargs)

    def set_group(self, group_name: str) -> T:
        """Set the color palette group."""
        if self.color_palette is None:
            msg = "No palette as been set yet. Please set it specifying a group."
            raise ValueError(msg)

        if group_name not in self.color_palette.predefined_groups:
            msg = f"{group_name} is not defined in the palette ({self.color_palette.name})."
            raise ValueError(msg)

        self.group_name = group_name

        return self

    def set_color_palette(
        self,
        color_palette: ColorPalette | str | None = None,
        group_name: str | None = None,
    ) -> T:
        """Set color palette."""
        if isinstance(color_palette, str):
            color_palette = color_palette.lower()
            if color_palette not in available_palettes and color_palette:
                possible_palettes = list(available_palettes)
                msg = (
                    f"{color_palette} not available in predefined color palettes. "
                    f"Possible values are: {possible_palettes}"
                )
                raise ValueError(msg)
            self.color_palette = available_palettes[color_palette]
        else:
            self.color_palette = color_palette

        self.group_name = group_name

        return self

    def set_color_map(
        self,
        color_map: dict | ColorMap | str | None = None,
        fill_nonexistent: bool = False,
    ) -> T:
        """Set color map."""

        if isinstance(color_map, str):
            color_map = color_map.lower()
            if color_map not in available_colormaps:
                msg = f"No colormap named {color_map} is available. Possible colormap names are {available_colormaps.keys()}"
                raise ValueError(msg)
            color_map = available_colormaps[color_map]

        elif isinstance(color_map, dict):
            color_map = ColorMap(color_map=color_map, fill_nonexistent=fill_nonexistent)

        self.color_map = color_map

        return self

    def set_layout_custom_updates(self, layout_updates: dict) -> T:
        """Set layout custom updates."""
        self.layout_custom_updates = layout_updates
        return self

    def add_layout_custom_updates(self, layout_updates: dict) -> T:
        """Set layout custom updates."""
        if self.layout_custom_updates is None:
            self.layout_custom_updates = {}
        self.layout_custom_updates.update(layout_updates)
        return self

    def set_xaxes_custom_updates(self, xaxes_updates: dict) -> T:
        """Set layout custom updates."""
        self.xaxes_custom_updates = xaxes_updates
        return self

    def add_xaxes_custom_updates(self, xaxes_updates: dict) -> T:
        """Update layout custom updates."""
        if self.xaxes_custom_updates is None:
            self.xaxes_custom_updates = {}
        self.xaxes_custom_updates.update(xaxes_updates)
        return self

    def set_yaxes_custom_updates(self, yaxes_updates: dict) -> T:
        """Set layout custom updates."""
        self.yaxes_custom_updates = yaxes_updates
        return self

    def add_yaxes_custom_updates(self, yaxes_updates: dict) -> T:
        """Update layout custom updates."""
        if self.yaxes_custom_updates is None:
            self.yaxes_custom_updates = {}
        self.yaxes_custom_updates.update(yaxes_updates)
        return self

    def add_rangeslider(self, options: dict | None = None) -> T:
        """Add a x-axis rangeslider to the plot."""

        if options is None:
            options = {
                "rangeslider_visible": True,
                "rangeselector": {
                    "buttons": [
                        {
                            "count": 1,
                            "label": "1m",
                            "step": "month",
                            "stepmode": "backward",
                        },
                        {
                            "count": 6,
                            "label": "6m",
                            "step": "month",
                            "stepmode": "backward",
                        },
                        # {
                        #     "count": 1,
                        #     "label": "YTD",
                        #     "step": "year",
                        #     "stepmode": "todate",
                        # },
                        {
                            "count": 1,
                            "label": "1y",
                            "step": "year",
                            "stepmode": "backward",
                        },
                        {"step": "all"},
                    ]
                },
                "type": "date",
            }

        self.add_xaxes_custom_updates(options)
        return self

    def to_plotly(self, logger=None) -> go.Figure:
        """Convert to plotly figure."""
        fig: go.Figure = super().to_plotly(logger=logger)

        # Set the colorway in the layout of the figure
        if self.color_palette is not None:
            # Check if color palette has enough colors to plot all the traces
            if len(self.color_palette.main_colors) < len(fig.data):
                msg = f"Palette ({self.color_palette.name}) does not have enough colors for plotting all the data."
                raise ValueError(msg)

            fig.update_layout(colorway=self.color_palette.main_colors)

        # Force the first N colors to follow the group name, if given
        if self.group_name is not None:
            group_colors = self.color_palette.get_group_by_name(self.group_name)
            len_group = len(group_colors)
            for i, trace in enumerate(fig.data):
                if i >= len_group:
                    break
                trace.update(line={"color": group_colors[i]})

        # Loop through each trace and update the color based on its name
        if self.color_map is not None:
            for trace in fig.data:
                series_name = trace.name
                if series_name in self.color_map:
                    color = self.color_map[series_name]
                elif self.color_map.fill_nonexistent:
                    color = self.color_map.get_color(series_name)
                else:
                    continue

                # Update marker color if trace has marker properties
                if hasattr(trace, "marker"):
                    trace.marker.color = color

                # Update line color if trace has line properties
                if hasattr(trace, "line"):
                    trace.line.color = color

        # Remove vertical lines in x axis
        fig.update_layout(xaxis={"showgrid": False})
        fig.update_yaxes(rangemode="tozero")

        # Make ticks larger
        fig.update_layout(
            {
                f"{axis}": {"tickfont": {"size": 12}}
                for axis in ["xaxis", "yaxis", "yaxis2"]
            },
        )

        # Update layout with custom layout updates
        if self.layout_custom_updates:
            fig.update_layout(**self.layout_custom_updates)

        if self.xaxes_custom_updates:
            fig.update_xaxes(**self.xaxes_custom_updates)

        if self.yaxes_custom_updates:
            fig.update_yaxes(**self.yaxes_custom_updates)

        if self.flag_remove_empty_traces:
            return self.remove_empty_traces(fig)

        return fig

    def get_default_title_layout(self, title_name="", pos_x=0.1, pos_y=0.9):
        """
        Generate a plotly layout dictionary for title configuration.

        Args:
            title_name (str): Title of the chart.
            pos_x (float): Position of title on x axis.
            pos_y (float): Position of title on y axis.

        Returns:
            dict: Dictionary containing plotly layout configuration for the title.

        """
        # Make titles look nicer
        subtitle_text = (
            f"<br><span style='font-size: 12px'>{self.subtitle}</span>"
            if self.subtitle is not None
            else ""
        )
        title_dict = {
            "text": f"<b>{title_name}</b>{subtitle_text}",
            "x": pos_x,  # 0 means left alignment (0 to 1 scale)
            "y": pos_y,
            "xanchor": "left",
            "yanchor": "top",
            "font": {
                "size": 16,  # Main title size
            },
        }

        return title_dict

    def get_default_font_layout(self):
        """Generate plotly layout dict for font

        :return: font_dict : dict that contains plotly layout for the font
        :type: dict
        """
        font_dict = {
            "family": 'Roboto, "Open Sans", "Helvetica Neue", Arial, sans-serif',
            "size": 10,
            "color": "#333333",
        }
        return font_dict

    @staticmethod
    def remove_empty_traces(fig: go.Figure) -> go.Figure:
        """
        Remove traces that contain only zero values from a Plotly figure.

        Returns the modified figure.
        """
        # Check if figure has no traces
        if len(fig.data) == 0:
            return fig

        # Create a list to store indices of traces to remove
        traces_to_remove = []

        # Check each trace
        for i, trace in enumerate(fig.data):
            # Get y data (we only need to check y values for zero series)
            y_data = np.array(trace.y) if trace.y is not None else np.array([])

            # If y contains only zeros (or is empty), mark for removal
            if len(y_data) == 0 or np.all(np.isclose(y_data, 0, atol=1e-10)):
                traces_to_remove.append(i)

        # Remove traces in reverse order to avoid index shifting
        for index in sorted(traces_to_remove, reverse=True):
            fig.data = tuple(trace for i, trace in enumerate(fig.data) if i != index)

        return fig

    @staticmethod
    def is_empty_or_zero_figure(fig: go.Figure) -> bool:
        """
        Check if a Plotly figure has no traces or if all traces contain only zero values.
        """
        # Check if figure has no traces
        if len(fig.data) == 0:
            return True

        # Check each trace
        for trace in fig.data:
            # Get x and y data
            x_data = np.array(trace.x) if trace.x is not None else np.array([])
            y_data = np.array(trace.y) if trace.y is not None else np.array([])

            # If either x or y contains non-zero values, return False
            if not (
                np.all(np.isclose(x_data, 0, atol=1e-10))
                and np.all(np.isclose(y_data, 0, atol=1e-10))
            ):
                return False

        # If we get here, all traces contain only zeros
        return True


class WITNESSTwoAxesInstanciatedChart(
    ExtendedMixin["WITNESSTwoAxesInstanciatedChart"], BaseTwoAxesInstanciatedChart
):
    pass


class WITNESSInstantiatedPlotlyNativeChart(
    ExtendedMixin["WITNESSInstantiatedPlotlyNativeChart"],
    BaseInstantiatedPlotlyNativeChart,
):
    pass


class TwoAxesInstanciatedChart(
    ExtendedMixin["TwoAxesInstanciatedChart"], BaseTwoAxesInstanciatedChart
):
    pass


class InstantiatedPlotlyNativeChart(
    ExtendedMixin["InstantiatedPlotlyNativeChart"], BaseInstantiatedPlotlyNativeChart
):
    pass
