import pathlib
from typing import Generator, Union

import gpxpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .geolocation import GeoLocator
from .map import MapSection
from .plot_tools import MapPlottable, PlotDefinition, plt_to_numpy


class Track(MapPlottable):
    def __init__(
        self,
        gpx,
        color="black",
        plot_width_scale=1,
        plot_solid=False,
        plot_start: bool = False,
        plot_end: bool = False,
        draw_partial_track: float = 1,
    ):
        super().__init__()
        self._gpx = gpx
        self._color = color
        self._plot_width_scale = plot_width_scale
        self._plot_solid = plot_solid
        self._plot_start = plot_start
        self._plot_end = plot_end
        self._draw_partial_track = draw_partial_track
        self._waypoints: pd.DataFrame | None = None

    @property
    def tags(self):
        return self.metadata.get("tags")

    @property
    def waypoints(self) -> pd.DataFrame:
        if self._waypoints is None:
            rows = [
                (
                    point.time,
                    point.latitude,
                    point.longitude,
                )
                for point in self._gpx.walk(only_points=True)
            ]
            columns = ["time", "latitude", "longitude"]
            self._waypoints = pd.DataFrame(rows, columns=columns)
        return self._waypoints

    @property
    def waypoints_for_plotting(self) -> pd.DataFrame:
        if self._draw_partial_track >= 1:
            return self.waypoints
        else:
            start, end = self.waypoints["time"].iloc[[0, -1]]
            cutoff = start + self._draw_partial_track * (end - start)

            return self.waypoints[self.waypoints["time"] <= cutoff]

    @property
    def first_waypoint(self):
        return self.waypoints.iloc[0]

    @property
    def year(self):
        return self.first_waypoint["time"].year

    @property
    def month(self):
        return self.first_waypoint["time"].month

    @property
    def start_address(self):
        coord = (self.first_waypoint["latitude"], self.first_waypoint["longitude"])
        address = GeoLocator.lookup(*coord).raw["address"]
        return address

    @property
    def country(self):
        return self.start_address["country"]

    @property
    def state(self):
        return self.start_address["state"]

    @property
    def city(self):
        start_address = self.start_address
        for key in ["village", "town", "city", "suburb", "municipality"]:
            if key in start_address:
                return start_address[key]
        raise ValueError(f"do not know how to parse city from {start_address}")

    @property
    def landmarks(self):
        peaks_in_area = GeoLocator.get_named_mountain_peaks(self.bounding_box)
        saddles_in_area = GeoLocator.get_named_mountain_saddles(self.bounding_box)
        alpine_huts_in_area = GeoLocator.get_alpine_huts(self.bounding_box)
        water_bodies_in_area = GeoLocator.get_named_water_bodies(self.bounding_box)
        cave_entrances_in_area = GeoLocator.get_named_cave_entrances(self.bounding_box)

        landmarks = (
            peaks_in_area
            + saddles_in_area
            + alpine_huts_in_area
            + water_bodies_in_area
            + cave_entrances_in_area
        )

        min_distance = 40 / 40_000_000 * 360  # roughly 40 meters to degrees

        close_landmarks = []
        for landmark in landmarks:
            if len(landmark["nodes"]) == 0:
                continue
            landmark_nodes = np.array(
                [
                    (float(node["latitude"]), float(node["longitude"]))
                    for node in landmark["nodes"]
                ]
            )

            # find all waypoints that are close to any of the landmark nodes
            # close means within min_distance in both latitude and longitude
            waypoints = self.waypoints[["latitude", "longitude"]].to_numpy()
            # Process waypoints in batches to avoid memory issues
            batch_size = 1000
            close_waypoint_indices = []
            for waypoint_start_idx in range(0, waypoints.shape[0], batch_size):
                waypoint_end_idx = min(
                    waypoint_start_idx + batch_size, waypoints.shape[0]
                )
                waypoint_batch = waypoints[waypoint_start_idx:waypoint_end_idx]

                for landmark_batch_idx in range(0, landmark_nodes.shape[0], batch_size):
                    landmark_batch_end_idx = min(
                        landmark_batch_idx + batch_size, landmark_nodes.shape[0]
                    )
                    landmark_batch = landmark_nodes[
                        landmark_batch_idx:landmark_batch_end_idx
                    ]

                    distance = (
                        waypoint_batch[:, np.newaxis, :]
                        - landmark_batch[np.newaxis, :, :]
                    )
                    abs_distance = np.abs(distance)
                    close = (abs_distance[:, :, 0] < min_distance) & (
                        abs_distance[:, :, 1] < min_distance
                    )  # shape (batch_size, n_nodes_in_batch)
                    batch_indices = np.where(close.any(axis=1))[0]
                    # Adjust indices to refer to the original waypoints array
                    close_waypoint_indices.extend(batch_indices + waypoint_start_idx)

            # add the landmark if we found any close waypoints
            if close_waypoint_indices:
                close_landmarks.append(
                    {
                        "name": landmark["name"],
                        "first_waypoint_index": min(close_waypoint_indices),
                    }
                )

        close_landmarks.sort(key=lambda landmark: landmark["first_waypoint_index"])
        landmark_names = [landmark["name"] for landmark in close_landmarks]

        # remove duplicates while preserving order
        seen = set()
        unique_landmark_names = []
        for name in landmark_names:
            if name not in seen:
                unique_landmark_names.append(name)
                seen.add(name)
        return unique_landmark_names

    @property
    def bounding_box(self) -> MapSection:
        return MapSection(
            north_latitude=self.waypoints["latitude"].max(),
            south_latitude=self.waypoints["latitude"].min(),
            east_longitude=self.waypoints["longitude"].max(),
            west_longitude=self.waypoints["longitude"].min(),
        )

    def get_plot_id(self):
        return (
            self._gpx.tracks[0].name,
            self._plot_solid,
            self._plot_width_scale,
            self._color,
            self._plot_start,
            self._plot_end,
            self._draw_partial_track,
        )

    def plot(
        self, map_section: MapSection, plot_definition: PlotDefinition
    ) -> np.ndarray:
        fig = plt.figure(figsize=(plot_definition.width, plot_definition.height), dpi=1)

        ax = fig.add_axes([0, 0, 1, 1])

        ONE_PIXEL = (
            72  # linewidth is in points, there are 72 points per inch, 1 pixel per inch
        )

        sigma = 1
        max_width = 4 * sigma if not self._plot_solid else 1
        for rel_linewidth in range(1, max_width + 1):
            if self._plot_solid:
                alpha = 1
            else:
                alpha = norm_gaussian(rel_linewidth - 1, sigma=sigma) - norm_gaussian(
                    rel_linewidth, sigma=sigma
                )
            linewidth = rel_linewidth * 2 * ONE_PIXEL * self._plot_width_scale
            ax.plot(
                self.waypoints_for_plotting["longitude"],
                self.waypoints_for_plotting["latitude"],
                color=self._color,
                linewidth=linewidth,
                alpha=alpha,
                antialiased=plot_definition.antialiased,
            )

        if self._plot_start or self._plot_end:
            idx = []
            if self._plot_start:
                idx.append(0)
            if self._plot_end:
                idx.append(-1)
            ax.scatter(
                self.waypoints["longitude"].iloc[idx],
                self.waypoints["latitude"].iloc[idx],
                s=150 * ONE_PIXEL * ONE_PIXEL * self._plot_width_scale,
                c=self._color,
                marker="o",
                alpha=1,
                antialiased=plot_definition.antialiased,
            )

        ax.set_xlim([map_section.west_longitude, map_section.east_longitude])
        ax.set_ylim([map_section.south_latitude, map_section.north_latitude])
        ax.set_axis_off()

        result = plt_to_numpy(fig)
        plt.close(fig)
        return result

    @classmethod
    def load_many(cls, path, **kwargs) -> Generator["Track", None, None]:
        path = pathlib.Path(path)
        for folder in path.iterdir():
            track = cls.from_folder(folder, **kwargs)
            if track is not None:
                yield track

    @classmethod
    def from_folder(
        cls,
        path: Union[str, pathlib.Path],
        limit_tag=None,
        colormap=None,
        **kwargs,
    ):
        path = pathlib.Path(path)
        if colormap is None:
            colormap = {"hiking": "crimson", "cycling": "darkorange"}

        gpx_filenames = list(path.glob("*.gpx"))
        assert len(gpx_filenames) <= 1
        if len(gpx_filenames) == 1:
            gpx_filename = gpx_filenames[0]
            metadata_filename = path / "metadata.yaml"
            if metadata_filename.exists():
                with open(metadata_filename, "r") as f:
                    metadata = yaml.load(f, Loader=yaml.SafeLoader)
            else:
                metadata = {}

            if limit_tag is not None:
                if limit_tag not in metadata.get("tags", []):
                    return None

            track_type = metadata.get("type", "hiking")
            color = colormap[track_type]

            with open(gpx_filename, "r") as f:
                return cls(gpxpy.parse(f), color=color, **kwargs)
        else:
            return None


def norm_gaussian(x, mu=0, sigma=1):
    """
    gaussian but scaled such that its maximum is at y=1
    """
    return np.exp(-np.power((x - mu) / sigma, 2.0) / 2)
