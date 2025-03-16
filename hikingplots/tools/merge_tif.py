import click
import rasterio.merge


@click.command()
@click.argument("filenames", nargs=-1)
@click.argument("output_filename")
def main(filenames, output_filename):
    datasets = [rasterio.open(f) for f in filenames]
    rasterio.merge.merge(datasets, dst_path=output_filename)


if __name__ == "__main__":
    main()
