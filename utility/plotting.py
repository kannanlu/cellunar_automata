"""some utility functions
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_binary_grid_2d(data: np.ndarray) -> None:
    """plot 2D binary grid data 
    """
    list_cmap = ListedColormap(['w', 'k'], N=2)
    fig, ax = plt.subplots(1, 1, figsize=(3.375, 3))
    ax.imshow(data, cmap=list_cmap, vmin=0, vmax=1)
    ax.set_title('Zeros and Ones')
    plt.show()
    return


#########################


def test_plot():
    data = np.zeros((100, 100))
    data[:, 50:] = 1
    plot_binary_grid_2d(data)
    return


if __name__ == '__main__':
    test_plot()
