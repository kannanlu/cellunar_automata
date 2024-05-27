"""lattice gas automata model
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import gzip
from utility.plotting import plot_binary_grid_2d
from tqdm import tqdm


class LatticeGasAutomata():
    """Binary state on a square lattice with n x n,
    n: number of sites in x and y direction,
    state: the n x n 2D array values,
    position: the current curser position,
    neighbor: the current neighbor position.
    This automata selects the curser position and swap with 
    the neighbor position under a certain boundary condition.
    """

    def __init__(self, n: int = 100) -> None:
        self.n = n
        # raise error if n <=1
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

    def halffill_state(self, plot: bool = False) -> None:
        """update the state with half filled on the right
        """
        m = self.n // 2
        self.state[:, m:] = 1
        if plot:
            plot_binary_grid_2d(self.state)
        return

    def bc_reflective(self, site: tuple = (0, 0)) -> tuple:
        """ A boundary condition that maps the site 
        that is out of the region reflectively to the interior
        """
        curr_x, curr_y = site
        if (curr_x >= self.n):
            curr_x = self.n - 2
        elif (curr_x < 0):
            curr_x = 1
        elif (curr_y >= self.n):
            curr_y = self.n - 2
        elif (curr_y < 0):
            curr_y = 1
        else:
            pass
        return (curr_x, curr_y)

    def random_select_site(self, check: bool = False) -> None:
        """ randomly select a curser position 
        """
        id_x = np.random.randint(low=0, high=self.n, size=1).item()
        id_y = np.random.randint(low=0, high=self.n, size=1).item()
        self.position = (id_x, id_y)
        if check:
            print("Selected site: ", self.position, "site value: ",
                  self.state[self.position])
        return

    def random_select_neighbor(self, check: bool = False) -> None:
        """ randomly select a neighbor, the boundary treatment is performed
        """
        curr_x, curr_y = self.position
        x_or_y = np.random.randint(low=0, high=2, size=1).item()
        step = np.random.randint(low=0, high=2, size=1).item()
        step = -1 if step == 0 else 1
        if x_or_y == 0:
            self.neighbor = (curr_x + step, curr_y)
        else:
            self.neighbor = (curr_x, curr_y + step)
        # perform the boundary treatment
        self.neighbor = self.bc_reflective(self.neighbor)
        if check:
            print("Selected neighbor: ", self.neighbor, "neighbor value: ",
                  self.state[self.neighbor])
        return

    def swap(self, check: bool = False) -> None:
        """ swap the site with the neighbor
        """
        curr_x, curr_y = self.position
        next_x, next_y = self.neighbor
        tmp2 = self.state[next_x, next_y]
        tmp = self.state[curr_x, curr_y]
        self.state[next_x, next_y] = tmp
        self.state[curr_x, curr_y] = tmp2
        if check:
            print("Updated site: ", self.position, "updated site value: ",
                  self.state[self.position])
            print("Updated neighbor: ", self.neighbor,
                  "updated neighbor value: ", self.state[self.neighbor])
        return

    def update(self, check: bool = False, plot: bool = False) -> None:
        """The update rule is to select a curser position 
        and swap with one of its neighbors randomly.
        """
        self.random_select_site(check)
        self.random_select_neighbor(check)
        self.swap(check)
        if plot:
            plot_binary_grid_2d(self.state)
        return

    def entropy_sweep(self, n_step: int = 10000) -> np.ndarray:
        """Compress the state at each interation 
        and obtain the entropy (roughly) of each iteration. 
        """
        num_bits = np.array([])
        self.init_state()
        self.halffill_state()
        for ii in tqdm(range(n_step)):
            self.update()
            # nb = gzip.compress(self.state.tobytes())
            nb = len(gzip.compress(self.state.tobytes()))
            num_bits = np.append(num_bits, nb)
        return num_bits
