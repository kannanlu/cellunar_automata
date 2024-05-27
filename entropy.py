"""iteratively calculate the entropy
"""
from model.lattice_gas import LatticeGasAutomata
from utility.plotting import plot_binary_grid_2d
import numpy as np
import matplotlib.pyplot as plt
import gzip

# initialize the model and evolve
size = 100
n_step = 10000
lga = LatticeGasAutomata(n=size)
res = lga.entropy_sweep(n_step=n_step)

# plot the entropy
x = np.linspace(0, n_step, n_step)
fig, ax = plt.subplots(1, 1, figsize=(3.375, 3))
ax.plot(x, res, 'o', markersize=3)
ax.set_xlabel('step')
ax.set_ylabel('Number of compressed bits')
plt.show()
