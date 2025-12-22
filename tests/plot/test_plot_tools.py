from matplotlib import pyplot as plt

from hikingplots.plot.plot_tools import plt_to_numpy


def test_plt_to_numpy():
    fig = plt.figure(figsize=(100, 100), dpi=1)
    fig_array = plt_to_numpy(fig, dpi=1)
    plt.close(fig)

    assert fig_array.shape == (100, 100, 4)
