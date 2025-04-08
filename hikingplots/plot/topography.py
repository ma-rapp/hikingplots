import os
import pathlib
import re
from dataclasses import dataclass

import diskcache
import earthpy.spatial as es
import geopandas as gpd
import lxml.etree
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import scipy
import scipy.ndimage
from matplotlib import colors
from rasterio.windows import Window

from .map import MapSection
from .plot_tools import (
    MapPlottable,
    PlotDefinition,
    convert_to_duotone,
    overlay_alpha,
    plt_to_numpy,
)


@dataclass
class TopographyData:
    data: np.ndarray
    area: MapSection


class LandTopography(MapPlottable):
    def __init__(self, topography_data: TopographyData):
        super().__init__()
        assert topography_data.data.min() != topography_data.data.max(), (
            topography_data.data.min(),
            topography_data.data.max(),
        )
        self.topography_data = topography_data

    def get_plot_id(self):
        return ("land", str(self.topography_data.area))

    def plot(
        self, map_section: MapSection, plot_definition: PlotDefinition
    ) -> np.ndarray:
        height_map = self.get_scaled_height_map(map_section, plot_definition)

        terrain_base_color = self.render_terrain_base_color(height_map)
        terrain_hillshade = self.render_terrain_hillshade(height_map)
        terrain_hillshade[:, :, 3] = 0.6
        terrain = overlay_alpha(terrain_base_color, terrain_hillshade)

        return terrain

    def render_terrain_base_color(self, height_map: np.ndarray) -> np.ndarray:
        cmap = matplotlib.cm.get_cmap("terrain")
        norm = matplotlib.colors.Normalize(vmin=-1000, vmax=2500)
        return cmap(norm(height_map)).astype(np.float32)

    def render_terrain_hillshade(
        self,
        height_map: np.ndarray,
        azimuth: float = 30.0,
        angle_altitude: float = 30.0,
    ) -> np.ndarray:
        x, y = np.gradient(height_map)
        slope = np.pi / 2.0 - np.arctan(np.sqrt(x * x + y * y))
        aspect = np.arctan2(-x, y)
        azimuthrad = azimuth * np.pi / 180.0
        altituderad = angle_altitude * np.pi / 180.0
        shaded = np.sin(altituderad) * np.sin(slope) + np.cos(altituderad) * np.cos(
            slope
        ) * np.cos(azimuthrad - aspect)

        cmap = matplotlib.cm.get_cmap("Greys")
        norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
        shade_colors = cmap(norm(shaded))
        return shade_colors.astype(np.float32)

    def get_scaled_height_map_with_padding(
        self, area: MapSection, plot_definition: PlotDefinition
    ) -> np.ndarray:
        if not self.topography_data.area.contains(area):
            raise ValueError(
                f"topology dataset too small. {area} not fully contained in {self.topography_data.area}"
            )
        padding_left = int(
            plot_definition.width
            * (area.west_longitude - self.topography_data.area.west_longitude)
            / (self.topography_data.area.width)
            + 0.5
        )
        padding_right = int(
            plot_definition.width
            * (self.topography_data.area.east_longitude - area.east_longitude)
            / (self.topography_data.area.width)
            + 0.5
        )
        padding_top = int(
            plot_definition.height
            * (self.topography_data.area.north_latitude - area.north_latitude)
            / (self.topography_data.area.height)
            + 0.5
        )
        padding_bottom = int(
            plot_definition.height
            * (area.south_latitude - self.topography_data.area.south_latitude)
            / (self.topography_data.area.height)
            + 0.5
        )
        padded_size = (
            plot_definition.height + padding_top + padding_bottom,
            plot_definition.width + padding_left + padding_right,
        )

        zoom = (
            padded_size[0] / self.topography_data.data.shape[0],
            padded_size[1] / self.topography_data.data.shape[1],
        )
        return scipy.ndimage.zoom(self.topography_data.data, zoom, order=2), (
            (padding_top, padding_bottom),
            (padding_left, padding_right),
        )

    def get_scaled_height_map(
        self, area: MapSection, plot_definition: PlotDefinition
    ) -> np.ndarray:
        height_map, ((padding_top, padding_bottom), (padding_left, padding_right)) = (
            self.get_scaled_height_map_with_padding(area, plot_definition)
        )
        if padding_top > 0:
            height_map = height_map[padding_top:, :]
        if padding_bottom > 0:
            height_map = height_map[:-padding_bottom, :]
        if padding_left > 0:
            height_map = height_map[:, padding_left:]
        if padding_right > 0:
            height_map = height_map[:, :-padding_right]
        return height_map

    @classmethod
    def load_area(
        cls, topo_land_path: pathlib.Path, area: MapSection, **kwargs
    ) -> "LandTopography":
        topography = rasterio.open(topo_land_path / "merged.tif")

        fully_contained = (
            topography.bounds.left <= area.west_longitude
            and topography.bounds.bottom <= area.south_latitude
            and topography.bounds.right >= area.east_longitude
            and topography.bounds.top >= area.north_latitude
        )
        if not fully_contained:
            raise ValueError(
                f"topology dataset too small. {area} not fully contained in {topography.bounds}"
            )

        transformer = rasterio.transform.AffineTransformer(topography.transform)
        p_row_top, p_col_left = transformer.rowcol(
            area.west_longitude, area.north_latitude, op=float
        )
        p_row_bottom, p_col_right = transformer.rowcol(
            area.east_longitude, area.south_latitude, op=float
        )

        margin = 0
        p_row_top = np.floor(p_row_top) - margin
        p_row_bottom = np.ceil(p_row_bottom) + margin
        p_col_left = np.floor(p_col_left) - margin
        p_col_right = np.ceil(p_col_right) + margin

        window = Window.from_slices(
            (p_row_top, p_row_bottom), (p_col_left, p_col_right)
        )
        topography_data = TopographyData(
            data=topography.read(1, window=window),
            area=MapSection(
                north_latitude=transformer.xy(p_row_top, 0, offset="ul")[1],
                south_latitude=transformer.xy(p_row_bottom, 0, offset="ul")[1],
                east_longitude=transformer.xy(0, p_col_right, offset="ul")[0],
                west_longitude=transformer.xy(0, p_col_left, offset="ul")[0],
            ),
        )

        return cls(topography_data=topography_data, **kwargs)


class ContourLandTopography(LandTopography):
    def __init__(
        self,
        major_level_step_size: int = 100,
        minor_level_step_size: int = 20,
        draw_zero_level: bool = False,
        draw_major_level_labels: bool = True,
        draw_hillshade: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._major_level_step_size = major_level_step_size
        self._minor_level_step_size = minor_level_step_size
        self._draw_zero_level = draw_zero_level
        self._draw_major_level_labels = draw_major_level_labels
        self._draw_hillshade = draw_hillshade

    def get_plot_id(self):
        return super().get_plot_id() + (
            self._major_level_step_size,
            self._minor_level_step_size,
            self._draw_major_level_labels,
        )

    def _get_levels(self, vmin: float, vmax: float, step_size: int):
        min_level = (vmin // step_size) * step_size
        levels = [min_level]
        while vmax > max(levels):
            levels.append(levels[-1] + step_size)
        return levels

    def plot(
        self, map_section: MapSection, plot_definition: PlotDefinition
    ) -> np.ndarray:
        height_map = self.get_scaled_height_map(map_section, plot_definition)

        fig = plt.figure(
            figsize=(
                (plot_definition.width + 0.1) / 100.0,
                (plot_definition.height + 0.1) / 100.0,
            ),
            dpi=100,
        )

        ax = fig.add_axes([0, 0, 1, 1])

        major_levels = self._get_levels(
            height_map.min(), height_map.max(), self._major_level_step_size
        )
        if height_map.max() - height_map.min() < self._minor_level_step_size * 10:
            minor_levels = self._get_levels(
                height_map.min(), height_map.max(), self._minor_level_step_size
            )
            minor_levels = [
                level for level in minor_levels if level not in major_levels
            ]
        else:
            minor_levels = []

        if not self._draw_zero_level:
            major_levels = [level for level in major_levels if level != 0]
            minor_levels = [level for level in minor_levels if level != 0]

        ax.contour(
            height_map[::-1],
            minor_levels,
            colors="black",
            linewidths=0.5,
            linestyles=[(0, (1, 8))],
            antialiased=plot_definition.antialiased,
        )
        major_contours = ax.contour(
            height_map[::-1],
            major_levels,
            colors="black",
            linewidths=0.7,
            antialiased=plot_definition.antialiased,
        )

        if self._draw_major_level_labels:
            labels = ax.clabel(major_contours, inline=True, fontsize=18)
            for label in labels:
                label.set_antialiased(plot_definition.antialiased)

        ax.set_xlim([0, height_map.shape[1]])
        ax.set_ylim([0, height_map.shape[0]])
        ax.set_axis_off()

        result = plt_to_numpy(fig, dpi=100)
        plt.close(fig)

        if self._draw_hillshade:
            hillshade = es.hillshade(height_map)
            hillshade = hillshade[:, :, None].repeat(4, axis=2).astype(np.float32)
            hillshade = hillshade / 255
            hillshade = (
                0.4 * hillshade + 0.7
            )  # turn brightness up -> fewer black pixels
            hillshade[hillshade < 0] = 0
            hillshade[hillshade > 1] = 1
            hillshade_duotone = convert_to_duotone(hillshade, mode="noise")

            return overlay_alpha(hillshade_duotone, result)
        else:
            return result


class WaterTopography(MapPlottable):
    _cache = diskcache.Cache("cache/water_topography")

    def __init__(
        self,
        water_bodies: list[gpd.GeoDataFrame],
        land_topography: LandTopography,
        area: MapSection,
        color: str,
        plot_width_scale: float = 2,
    ):
        super().__init__()
        self.water_bodies = water_bodies
        self.land_topography = land_topography
        self.area = area
        self.color = color
        self.plot_width_scale = plot_width_scale

    def get_plot_id(self):
        return ("water", str(self.area), self.color, self.plot_width_scale)

    def plot(
        self, map_section: MapSection, plot_definition: PlotDefinition
    ) -> np.ndarray:
        fig = plt.figure(
            figsize=(
                (plot_definition.width + 0.1) / 100.0,
                (plot_definition.height + 0.1) / 100.0,
            ),
            dpi=100,
        )

        ax = fig.add_axes([0, 0, 1, 1])

        ONE_PIXEL = (
            72
            / 100.0  # linewidth is in points, there are 72 points per inch, 1 pixel per inch
        )

        for water_body in self.water_bodies:
            water_body.plot(
                ax=ax,
                color=self.color,
                linewidth=self.plot_width_scale * ONE_PIXEL,
                aspect=None,
                antialiased=plot_definition.antialiased,
            )

        ax.set_xlim([map_section.west_longitude, map_section.east_longitude])
        ax.set_ylim([map_section.south_latitude, map_section.north_latitude])
        ax.set_axis_off()

        result = plt_to_numpy(fig, dpi=100)
        plt.close(fig)

        ocean = self.render_ocean(map_section, plot_definition)
        return overlay_alpha(result, ocean)

    def render_ocean(
        self,
        area: MapSection,
        plot_definition: PlotDefinition,
        threshold: float = 0.001,
    ) -> np.ndarray:
        height_map = self.land_topography.get_scaled_height_map(area, plot_definition)
        x, y = np.gradient(height_map)
        slope = np.sqrt(x * x + y * y)
        flat = slope < threshold
        sea_level = height_map < 100
        sea = np.logical_and(flat, sea_level)

        # remove isolated pixels
        sea = scipy.ndimage.binary_erosion(sea, iterations=50, border_value=1).astype(
            sea.dtype
        )
        sea = scipy.ndimage.binary_dilation(sea, iterations=50, border_value=0).astype(
            sea.dtype
        )

        sea_color = np.zeros(
            (plot_definition.height, plot_definition.width, 4), dtype=np.float32
        )
        sea_color[sea, :] = np.asarray(colors.to_rgba(self.color), dtype=np.float32)
        return sea_color

    @classmethod
    def load_area(
        cls,
        topo_water_path: pathlib.Path,
        area: MapSection,
        land_topography: LandTopography,
        strahler_limit: int = 3,
        **kwargs,
    ) -> "WaterTopography":
        river_names = []
        for folder in os.listdir(topo_water_path / "EU_hydro_gpkg_eu"):
            if m := re.match(r"euhydro_(?P<rivername>[a-z]+)_v013_GPKG", folder):
                river_names.append(m.group("rivername"))

        water_bodies = []
        for river in river_names:
            filename = (
                topo_water_path
                / f"EU_hydro_gpkg_eu/euhydro_{river}_v013_GPKG/euhydro_{river}_v013.gpkg"
            )
            metadata_filename = (
                topo_water_path
                / f"EU_hydro_gpkg_eu/euhydro_{river}_v013_GPKG/Metadata/{river}_metadata.xml"
            )
            water_bodies.extend(
                WaterTopography._load_water_bodies_from_file(
                    filename, metadata_filename, area, strahler_limit
                )
            )
        return cls(water_bodies, land_topography, area, **kwargs)

    @staticmethod
    @_cache.memoize()
    def _load_water_bodies_from_file(
        filename: str, metadata_filename: str, area: MapSection, strahler_limit: int
    ):
        metadata = lxml.etree.parse(metadata_filename)
        file_area = MapSection(
            west_longitude=float(
                metadata.xpath(
                    "//ns0:westBoundLongitude/ns2:Decimal",
                    namespaces=metadata.getroot().nsmap,
                )[0].text
            ),
            east_longitude=float(
                metadata.xpath(
                    "//ns0:eastBoundLongitude/ns2:Decimal",
                    namespaces=metadata.getroot().nsmap,
                )[0].text
            ),
            north_latitude=float(
                metadata.xpath(
                    "//ns0:northBoundLatitude/ns2:Decimal",
                    namespaces=metadata.getroot().nsmap,
                )[0].text
            ),
            south_latitude=float(
                metadata.xpath(
                    "//ns0:southBoundLatitude/ns2:Decimal",
                    namespaces=metadata.getroot().nsmap,
                )[0].text
            ),
        )
        file_area_polygon = file_area.to_polygon()
        area_of_interest_polygon = area.to_polygon()
        if not file_area_polygon.intersects(area_of_interest_polygon):
            return []

        large_rivers = gpd.read_file(filename, layer="River_Net_p")
        small_rivers = gpd.read_file(filename, layer="River_Net_l")
        small_rivers = small_rivers[small_rivers.STRAHLER >= strahler_limit]
        large_canals = gpd.read_file(filename, layer="Canals_p")
        small_canals = gpd.read_file(filename, layer="Canals_l")
        small_canals = small_canals[small_canals.STRAHLER >= strahler_limit]
        inland_water = gpd.read_file(filename, layer="InlandWater")
        coastal = gpd.read_file(filename, layer="Coastal_p")
        transit = gpd.read_file(filename, layer="Transit_p")
        water_bodies = [
            large_rivers,
            small_rivers,
            large_canals,
            small_canals,
            inland_water,
            coastal,
            transit,
        ]

        water_bodies = [w.to_crs("WGS84") for w in water_bodies]  # WGS84 is lat/lon

        water_bodies = [
            w[w.geometry.intersects(area_of_interest_polygon)] for w in water_bodies
        ]
        water_bodies = [w for w in water_bodies if len(w) > 0]
        return water_bodies
