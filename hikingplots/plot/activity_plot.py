from typing import List, Union

import numpy as np
from PIL import Image

from .map import MapSection
from .plot_tools import MapPlottable, PlotDefinition, convert_to_duotone, overlay_alpha


class ActivityPlot:
    def __init__(
        self,
        area_of_interest: MapSection,
        pixels_per_degree_latitude: int,
        background_color: List[int],
        antialiased: bool = True,
    ):
        self.area_of_interest = area_of_interest
        self.pixels_per_degree_latitude = pixels_per_degree_latitude
        self.background_color = background_color
        self.antialiased = antialiased

        self.plot_area = self.setup_plot_area()

    def setup_plot_area(self) -> np.ndarray:
        color = np.array(self.background_color, dtype=np.float32)
        height = int(
            self.area_of_interest.height * self.pixels_per_degree_latitude + 0.5
        )
        width = int(
            self.area_of_interest.width
            * self.pixels_per_degree_latitude
            * ActivityPlot.get_longitude_scale(self.area_of_interest)
            + 0.5
        )
        empty = np.zeros(
            (
                height,
                width,
                4,
            ),
            dtype=np.float32,
        )  # 0-1 range
        empty[:, :] = color
        return empty

    @staticmethod
    def get_longitude_scale(area_of_interest: MapSection):
        return np.cos(area_of_interest.center_latitude / 180 * np.pi)

    @staticmethod
    def compute_pixels_per_degree_latitude_to_fit(
        area_of_interest: MapSection, max_width: int, max_height: int
    ) -> int:
        pixels_per_degree_latitude = min(
            int(max_height / area_of_interest.height),
            int(
                max_width
                / area_of_interest.width
                / ActivityPlot.get_longitude_scale(area_of_interest)
            ),
        )
        return pixels_per_degree_latitude

    @staticmethod
    def _cached_plot(
        *, area_of_interest, plot_definition, ids, plottables: List[MapPlottable]
    ):
        if len(plottables) == 1:
            return plottables[0].plot(area_of_interest, plot_definition)
        else:
            plotted = ActivityPlot._cached_plot(
                area_of_interest=area_of_interest,
                plot_definition=plot_definition,
                ids=ids[:-1],
                plottables=plottables[:-1],
            )
            return overlay_alpha(
                plotted, plottables[-1].plot(area_of_interest, plot_definition)
            )

    def plot(self, plottable: Union[MapPlottable, List[MapPlottable]]) -> None:
        if isinstance(plottable, MapPlottable):
            plottables = [plottable]
        else:
            plottables = plottable
        ids = [p.get_plot_id() for p in plottables]
        plot_definition = PlotDefinition(
            width=self.plot_area_width,
            height=self.plot_area_height,
            antialiased=self.antialiased,
        )
        plotted = ActivityPlot._cached_plot(
            area_of_interest=self.area_of_interest,
            plot_definition=plot_definition,
            ids=ids,
            plottables=plottables,
        )
        self.plot_area = overlay_alpha(self.plot_area, plotted)

    def convert_to_duotone(self, mode=None):
        self.plot_area = convert_to_duotone(self.plot_area, mode=mode)

    @property
    def plot_area_width(self):
        return self.plot_area.shape[1]

    @property
    def plot_area_height(self):
        return self.plot_area.shape[0]

    def get_image(self) -> Image:
        pixel_values = (self.plot_area * 255).astype(np.uint8)
        return Image.fromarray(pixel_values)
