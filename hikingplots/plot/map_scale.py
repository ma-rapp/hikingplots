from matplotlib import pyplot as plt

from .map import MapSection
from .plot_tools import MapPlottableUsingMatplotlib, PlotDefinition


class MapScale(MapPlottableUsingMatplotlib):
    def __init__(self):
        super().__init__()
        self._line_width_scale = 1
        self._color = "black"

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
            return f"{distance_m // 1000} km"

    def plot_on_fig(
        self,
        map_section: MapSection,
        plot_definition: PlotDefinition,
        ax: plt.Axes,
        one_pixel: float,
    ) -> None:
        ax.set_xlim([0, plot_definition.width])
        ax.set_ylim([0, plot_definition.height])

        linewidth = 2 * one_pixel * self._line_width_scale

        margin_left_right = 20 * one_pixel
        pos_left = margin_left_right

        for width_m in self.iterate_nice_distances():
            width_rel = width_m / map_section.width_meters
            if width_rel > 0.2:
                break

        pos_right = pos_left + width_rel * plot_definition.width

        pos_y = 10 * one_pixel
        winglet = 5 * one_pixel

        ax.plot(
            [pos_left, pos_right],
            [pos_y, pos_y],
            linewidth=linewidth,
            color=self._color,
            antialiased=plot_definition.antialiased,
        )
        # winglets
        ax.plot(
            [pos_left, pos_left],
            [pos_y - winglet, pos_y + winglet],
            linewidth=linewidth,
            color=self._color,
            antialiased=plot_definition.antialiased,
        )
        ax.plot(
            [pos_right, pos_right],
            [pos_y - winglet, pos_y + winglet],
            linewidth=linewidth,
            color=self._color,
            antialiased=plot_definition.antialiased,
        )

        # text
        t = ax.text(
            (pos_left + pos_right) / 2,
            pos_y + 4 * one_pixel,
            self.format_distance(width_m),
            fontsize=18,
            antialiased=plot_definition.antialiased,
            horizontalalignment="center",
            zorder=-1,
        )
        t.set_bbox(dict(facecolor="white", edgecolor="white"))

        # white bakground
        ax.fill_between(
            [pos_left - margin_left_right, pos_right + margin_left_right],
            0,
            pos_y + 20 * one_pixel,
            color="white",
            zorder=-2,
        )
