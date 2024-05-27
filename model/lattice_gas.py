"""lattice gas automata model
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import gzip
from utility.plotting import plot_binary_grid_2d


class LatticeGasAutomata():

    def __init__(self, n: int = 100) -> None:
        self.n = n
        self.state = np.zeros((n, n))
        self.position = (0, 0)
        self.neighbor = (0, 0)

    def init_state(self, plot: bool = False) -> None:
        """generate nxn 2D grid state with 0 at each grid
        """
        self.state = np.zeros((self.n, self.n))
        if plot:
            plot_binary_grid_2d(self.state)
        return

    def prepare_state(self, plot: bool = False) -> None:
        """update the state with half filled
        """
        m = self.n // 2
        self.state[:, m:] = 1
        if plot:
            plot_binary_grid_2d(self.state)
        return

    def random_select_site(self) -> None:
        """ randomly select a sit position 
        """
        id_x = np.random.randint(low=0, high=self.n, size=1)
        id_y = np.random.randint(low=0, high=self.n, size=1)
        self.position = (id_x, id_y)
        return

    def random_select_neighbor(self) -> None:
        """ randomly select a neighbor
        """
        curr_x, curr_y = self.position
        x_or_y = np.random.randint(low=0, high=1, size=1)
        step = np.random.randint(low=0, high=1, size=1)
        step = -1 if step == 0 else 1
        if x_or_y == 0:
            self.neighbor = (curr_x + step, curr_y)
        else:
            self.neighbor = (curr_x, curr_y + step)
        return

    def swap(self) -> None:
        """ swap the site with the neighbor
        """
        curr_x, curr_y = self.position
        next_x, next_y = self.neighbor
        if (next_x >= self.n) or (next_x < 0) or (next_y >= self.n) or (next_y
                                                                        < 0):
            tmp2 = 0
        else:
            tmp2 = self.state[next_x, next_y]
        tmp = self.state[curr_x, curr_y]
        self.state[next_x, next_y] = tmp
        self.state[curr_x, curr_y] = tmp2
        return

    def update(self, plot: bool = False) -> None:
        self.random_select_site()
        self.random_select_neighbor()
        self.swap()
        if plot:
            plot_binary_grid_2d(self.state)
        return

    def sweep(self, n_step: int = 10000) -> np.ndarray:
        num_bits = np.array([])
        self.init_state()
        self.prepare_state()
        for ii in range(n_step):
            self.update()
            nb = gzip.compress(self.state.tobytes())
            num_bits = np.append(num_bits, nb)
        return num_bits
