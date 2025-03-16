import math
from dataclasses import dataclass
from typing import List

import numpy as np
import shapely.geometry


@dataclass
class MapSection:
    north_latitude: float
    south_latitude: float
    east_longitude: float
    west_longitude: float

    def enlarge(self, factor: float) -> "MapSection":
        smaller_size = min(self.width, self.height)
        return MapSection(
            north_latitude=self.north_latitude + smaller_size * factor,
            south_latitude=self.south_latitude - smaller_size * factor,
            east_longitude=self.east_longitude + smaller_size * factor,
            west_longitude=self.west_longitude - smaller_size * factor,
        )

    def round_to(self, resolution: float) -> "MapSection":
        # round towards outside
        return MapSection(
            north_latitude=math.ceil(self.north_latitude / resolution) * resolution,
            south_latitude=math.floor(self.south_latitude / resolution) * resolution,
            east_longitude=math.ceil(self.east_longitude / resolution) * resolution,
            west_longitude=math.floor(self.west_longitude / resolution) * resolution,
        )

    @property
    def width(self) -> float:
        return self.east_longitude - self.west_longitude

    @property
    def width_meters(self) -> float:
        meters_per_degree_at_equator = 40_075_000 / 360
        meters_per_degree_at_this_center = meters_per_degree_at_equator * np.cos(
            self.center_latitude / 180 * np.pi
        )
        return self.width * meters_per_degree_at_this_center

    @property
    def height(self) -> float:
        return self.north_latitude - self.south_latitude

    @property
    def height_meters(self) -> float:
        meters_per_degree = 40_075_000 / 360
        return self.height * meters_per_degree

    @property
    def center_latitude(self) -> float:
        return (self.north_latitude + self.south_latitude) / 2

    @property
    def center_longitude(self) -> float:
        return (self.east_longitude + self.west_longitude) / 2

    def to_polygon(self) -> shapely.Polygon:
        return shapely.geometry.box(
            self.west_longitude,
            self.south_latitude,
            self.east_longitude,
            self.north_latitude,
        )

    def contains(self, other: "MapSection") -> bool:
        return (
            self.north_latitude >= other.north_latitude
            and self.south_latitude <= other.south_latitude
            and self.east_longitude >= other.east_longitude
            and self.west_longitude <= other.west_longitude
        )

    @staticmethod
    def create_envelope(map_sections: List["MapSection"]) -> "MapSection":
        return MapSection(
            north_latitude=max([section.north_latitude for section in map_sections]),
            south_latitude=min([section.south_latitude for section in map_sections]),
            east_longitude=max([section.east_longitude for section in map_sections]),
            west_longitude=min([section.west_longitude for section in map_sections]),
        )
