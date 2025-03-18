from collections import defaultdict
import pathlib

from .activity_plot import ActivityPlot
from .console import NoStepIndicator, StepIndicator
from .map import MapSection
from .map_scale import MapScale
from .topography import ContourLandTopography, LandTopography, WaterTopography
from .track import Track


def plot_area(
    tracks_path: pathlib.Path | str,
    tag: str | None,
    topo_land_path: pathlib.Path | str,
    topo_water_path: pathlib.Path | str,
    no_terrain: bool,
    no_water: bool,
    no_tracks: bool,
    track_plot_width_scale: int,
    track_plot_solid: bool,
    show_steps:bool =True,
):
    step_indicator_cls = StepIndicator if show_steps else NoStepIndicator

    with step_indicator_cls("Loading tracks") as ind:
        tracks = list(
            Track.load_many(
                pathlib.Path(tracks_path),
                limit_tag=tag,
                plot_width_scale=track_plot_width_scale,
                plot_solid=track_plot_solid,
            )
        )
        ind.print_info(f"found {len(tracks)} tracks")
        tracks = sorted(tracks, key=lambda track: track.get_plot_id())

    with step_indicator_cls("Getting area of interest"):
        area_of_interest = MapSection.create_envelope(
            [track.bounding_box for track in tracks]
        )
        area_of_interest = area_of_interest.enlarge(0.2)
        area_of_interest = area_of_interest.round_to(
            7.5 / 60 / 60
        )  # round to resolution of topograhpy data

    with step_indicator_cls("Loading land topography"):
        land = LandTopography.load_area(pathlib.Path(topo_land_path), area_of_interest)

    with step_indicator_cls("Loading water topography"):
        water = WaterTopography.load_area(pathlib.Path(topo_water_path), area_of_interest, land, color="royalblue")

    with step_indicator_cls("Setting up plot") as ind:
        plot = ActivityPlot(
            area_of_interest,
            pixels_per_degree_latitude=480 * 4,
            background_color=[182 / 255, 203 / 252, 99 / 252, 1],
        )
        ind.print_info(f"{plot.plot_area_width}x{plot.plot_area_height} px")

    if not no_terrain:
        with step_indicator_cls("Plotting land topography"):
            plot.plot(land)

    if not no_water:
        with step_indicator_cls("Plotting water topography"):
            plot.plot(water)

    if not no_tracks:
        with step_indicator_cls("Plotting tracks"):
            plot.plot(tracks)

    return plot.get_image()


def plot_track_duotone(
    track_path: pathlib.Path | str,
    topo_land_path: pathlib.Path | str,
    topo_water_path: pathlib.Path | str,
    draw_partial_track: float = 1.0,
    track_halign: str="center",
    draw_topo: bool = True,
    draw_major_level_labels: bool = True,
    add_scale: bool = False,
    show_steps: bool = True,
):
    """Plot a track in duotone style.

    Args:
        track_path: Path to the track folder.
        topo_land_path: Path to the land topography data (see `topo-land` in README.md).
        topo_water_path: Path to the water topography data (see `topo-water` in README.md).
        draw_partial_track: Fraction of the track to draw (if 0 <= `draw_partial_track` < 1). Draw full track if `draw_partial_track` >= 1.
        track_halign: Horizontal alignment of the track. One of "left", "center", "right".
        draw_topo: Whether to draw the topography.
        draw_major_level_labels: Whether to draw the major level labels.
        show_steps: Whether to log the steps on the console.
        add_scale: Whether to add a scale to the plot.
    """
    target_height = 480
    target_width = 800

    step_indicator_cls = StepIndicator if show_steps else NoStepIndicator

    with step_indicator_cls("Loading track"):
        track = Track.from_folder(
            pathlib.Path(track_path),
            plot_width_scale=2,
            plot_solid=True,
            colormap=defaultdict(lambda: "black"),
            plot_start=True,
            plot_end=draw_partial_track >= 1,
            draw_partial_track=draw_partial_track,
        )

    with step_indicator_cls("Getting area of interest"):
        area_of_interest = track.bounding_box
        area_of_interest = area_of_interest.enlarge(0.1)

        longitude_scale = ActivityPlot.get_longitude_scale(area_of_interest)
        target_aspect = target_width / target_height
        actual_aspect = (
            longitude_scale * area_of_interest.width
        ) / area_of_interest.height

        while (
            area_of_interest.width_meters < 1_800
            and area_of_interest.height_meters < 1_800 / target_aspect
        ):
            area_of_interest = area_of_interest.enlarge(0.1)

        if actual_aspect > target_aspect:
            # too wide -> make higher
            area_of_interest.south_latitude = (
                area_of_interest.north_latitude
                - area_of_interest.width * longitude_scale / target_aspect
            )
        else:
            # too high -> make wider
            target_width_longitude = area_of_interest.height / longitude_scale * target_aspect
            if track_halign == "center":
                current_center = area_of_interest.center_longitude
                area_of_interest.east_longitude = current_center + 0.5 * target_width_longitude
                area_of_interest.west_longitude = current_center - 0.5 * target_width_longitude
            elif track_halign == "left":
                area_of_interest.east_longitude = area_of_interest.west_longitude + target_width_longitude
            elif track_halign == "right":
                area_of_interest.west_longitude = area_of_interest.east_longitude - target_width_longitude
            else:
                raise ValueError(f"Invalid track_halign: {track_halign}")

    if draw_topo:
        with step_indicator_cls("Loading land topography"):
            land = ContourLandTopography.load_area(
                pathlib.Path(topo_land_path),
                area_of_interest,
                major_level_step_size=200,
                minor_level_step_size=50,
                draw_major_level_labels=draw_major_level_labels,
                draw_hillshade=False,
            )

        with step_indicator_cls("Loading water topography"):
            water = WaterTopography.load_area(
                pathlib.Path(topo_water_path),
                area_of_interest,
                land,
                color=[0.5, 0.5, 0.5, 1],
                plot_width_scale=3,
                strahler_limit=2,
            )

    with step_indicator_cls("Setting up plot") as ind:
        pixels_per_degree_latitude = (
            ActivityPlot.compute_pixels_per_degree_latitude_to_fit(
                area_of_interest=area_of_interest,
                max_width=target_width,
                max_height=target_height,
            )
        )
        plot = ActivityPlot(
            area_of_interest,
            pixels_per_degree_latitude=pixels_per_degree_latitude,
            background_color=[1, 1, 1, 1],
            antialiased=False,
        )
        ind.print_info(f"{plot.plot_area_width}x{plot.plot_area_height} px")

    if draw_topo:
        with step_indicator_cls("Plotting land topography"):
            plot.plot(land)

        with step_indicator_cls("Plotting water topography"):
            plot.plot(water)

    if add_scale:
        plot.plot(MapScale())

    with step_indicator_cls("Plotting track"):
        plot.plot(track)

    with step_indicator_cls("Converting to duotone"):
        plot.convert_to_duotone(mode="checkerboard")

    return plot.get_image()
