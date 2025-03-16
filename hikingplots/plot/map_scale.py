import numpy as np
from matplotlib import pyplot as plt

from .map import MapSection
from .plot_tools import MapPlottable, PlotDefinition, plt_to_numpy


class MapScale(MapPlottable):
    def __init__(self):
        super().__init__()
        self._line_width_scale = 1
        self._color = "black"

    def get_plot_id(self):
        return None

    def iterate_nice_distances(self):
        v = 1
        while True:
            yield v
            yield v * 2
            yield v * 5
            v *= 10

    def format_distance(self, distance_m: int):
        if distance_m < 1000:
            return f"{distance_m} m"
        else:
            return f"{distance_m//1000} km"

    def plot(
        self, map_section: MapSection, plot_definition: PlotDefinition
    ) -> np.ndarray:
        fig_scale = 100.0
        fig = plt.figure(
            figsize=(
                (plot_definition.width + 0.1) / fig_scale,
                (plot_definition.height + 0.1) / fig_scale,
            ),
            dpi=100,
        )

        ONE_PIXEL = (
            72
            / fig_scale  # linewidth is in points, there are 72 points per inch, 1 pixel per inch
        )
        linewidth = 2 * ONE_PIXEL * self._line_width_scale

        ax = fig.add_axes([0, 0, 1, 1])

        margin_left_right = 20 * ONE_PIXEL
        pos_left = margin_left_right

        for width_m in self.iterate_nice_distances():
            width_rel = width_m / map_section.width_meters
            if width_rel > 0.2:
                break

        pos_right = pos_left + width_rel * plot_definition.width

        pos_y = 10 * ONE_PIXEL
        winglet = 5 * ONE_PIXEL

        plt.plot(
            [pos_left, pos_right],
            [pos_y, pos_y],
            linewidth=linewidth,
            color=self._color,
            antialiased=plot_definition.antialiased,
        )
        # winglets
        plt.plot(
            [pos_left, pos_left],
            [pos_y - winglet, pos_y + winglet],
            linewidth=linewidth,
            color=self._color,
            antialiased=plot_definition.antialiased,
        )
        plt.plot(
            [pos_right, pos_right],
            [pos_y - winglet, pos_y + winglet],
            linewidth=linewidth,
            color=self._color,
            antialiased=plot_definition.antialiased,
        )

        # text
        t = plt.text(
            (pos_left + pos_right) / 2,
            pos_y + 4 * ONE_PIXEL,
            self.format_distance(width_m),
            fontsize=18,
            antialiased=plot_definition.antialiased,
            horizontalalignment="center",
            zorder=-1,
        )
        t.set_bbox(dict(facecolor="white", edgecolor="white"))

        # white bakground
        plt.fill_between(
            [pos_left - margin_left_right, pos_right + margin_left_right],
            0,
            pos_y + 20 * ONE_PIXEL,
            color="white",
            zorder=-2,
        )

        ax.set_xlim([0, plot_definition.width])
        ax.set_ylim([0, plot_definition.height])
        ax.set_axis_off()

        result = plt_to_numpy(fig, dpi=100)
        plt.close(fig)

        return result
