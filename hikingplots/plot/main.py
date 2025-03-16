import click

from . import plot


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--tracks-path",
    default="tracks",
    help="Path to the folder containing the tracks",
    type=click.Path(exists=True),
)
@click.option(
    "--topo-land-path",
    default="topo-land",
    help="Path to the folder containing the land topology data",
    type=click.Path(exists=True),
)
@click.option(
    "--topo-water-path",
    default="topo-water",
    help="Path to the folder containing the water topology data",
    type=click.Path(exists=True),
)
@click.option("--tag", help="Limit the plot to a specific tag", type=str)
@click.option(
    "--output-path",
    help="Path to the output file",
    type=click.Path(dir_okay=False),
    required=True,
)
@click.option("--no-terrain", is_flag=True)
@click.option("--no-water", is_flag=True)
@click.option("--no-tracks", is_flag=True)
@click.option("--track-plot-width-scale", type=int, default=1)
@click.option("--track-plot-solid", is_flag=True)
def plot_area(*args, output_path, **kwargs):
    image = plot.plot_area(*args, **kwargs)
    image.save(output_path)


@cli.command()
@click.argument(
    "track-path",
    type=click.Path(exists=True),
)
@click.option(
    "--topo-land-path",
    default="topo-land",
    help="Path to the folder containing the land topology data",
    type=click.Path(exists=True),
)
@click.option(
    "--topo-water-path",
    default="topo-water",
    help="Path to the folder containing the water topology data",
    type=click.Path(exists=True),
)
@click.option(
    "--output-path",
    help="Path to the output file",
    type=click.Path(dir_okay=False),
    required=True,
)
@click.option("--draw-topo/--no-draw-topo", is_flag=True, default=True)
@click.option(
    "--draw-major-level-labels/--no-draw-major-level-labels", is_flag=True, default=True
)
@click.option("--add-scale/--no-add-scale", is_flag=True, default=True)
def plot_track_duotone(*args, output_path, **kwargs):
    image = plot.plot_track_duotone(*args, **kwargs)
    image.save(output_path)


if __name__ == "__main__":
    cli()
