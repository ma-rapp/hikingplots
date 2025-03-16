import io
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .map import MapSection


@dataclass
class PlotDefinition:
    width: int
    height: int
    antialiased: bool


class MapPlottable(ABC):
    """
    An object that is plottable onto a map.
    """

    @abstractmethod
    def get_plot_id(self):
        pass

    @abstractmethod
    def plot(
        self, map_section: MapSection, plot_definition: PlotDefinition
    ) -> np.ndarray:
        pass


def overlay_alpha(first, second):
    """
    Overlay two images with alpha channel.
    The first image is the background, the second image is the overlay.
    """
    assert first.dtype == np.float32, first.dtype
    assert second.dtype == np.float32, second.dtype

    if first.shape != second.shape:
        raise ValueError(f"shapes do not match: {first.shape} != {second.shape}")

    second_color = second[:, :, :3]
    second_alpha = second[:, :, 3]

    first_color = first[:, :, :3]
    first_alpha = first[:, :, 3]

    result = np.zeros_like(first)
    result[:, :, 3] = second_alpha + first_alpha * (1 - second_alpha)
    new_alpha = result[:, :, 3]
    result[:, :, :3] = (
        second_color * second_alpha[:, :, None]
        + first_color * (first_alpha * (1 - second_alpha))[:, :, None]
    ) / (new_alpha[:, :, None] + 1e-6)
    if np.any(np.isnan(result)):
        raise ValueError("NaN in result")
    return result


def plt_to_numpy(fig, dpi=1):
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format="raw", dpi=dpi, transparent=True)

    io_buf.seek(0)
    img_arr = np.reshape(
        np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
    )
    io_buf.close()

    result = img_arr.astype(np.float32) / 255.0
    return result


def convert_to_duotone(image, mode=None):
    assert 0 <= image.min() <= image.max() <= 1
    if mode is None or mode == "noise":
        level_patterns = {0: np.zeros_like(image), 1: np.ones_like(image)}
    elif mode == "checkerboard":
        checkerboard = np.zeros_like(image)
        checkerboard[::2, ::2] = 1
        checkerboard[1::2, 1::2] = 1
        level_patterns = {
            0: np.zeros_like(image),
            0.49: checkerboard,
            0.51: checkerboard,
            1: np.ones_like(image),
        }
    else:
        raise ValueError(f"unknown mode: {mode}")

    duotone = np.zeros_like(image)

    levels = sorted(level_patterns.keys())
    for low, high in zip(levels[:-1], levels[1:]):
        rand = np.random.RandomState(1)
        x = rand.uniform(low, high, image.shape)
        higher = np.logical_and(x <= image, image <= high)
        lower = np.logical_and(low <= image, image < x)

        duotone[higher] = level_patterns[high][higher]
        duotone[lower] = level_patterns[low][lower]

    duotone[:, :, 1] = duotone[:, :, 0]
    duotone[:, :, 2] = duotone[:, :, 0]
    duotone[:, :, 3] = 1
    return duotone
