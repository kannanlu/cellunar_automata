from model.lattice_gas import LatticeGasAutomata
from utility.plotting import plot_binary_grid_2d

lga = LatticeGasAutomata(10)
lga.prepare_state(plot=True)
lga.update(plot=True)
ii = 0
while ii <= 500:
    lga.update()
    ii += 1
lga.update(plot=True)

# n_step = 10000

# x = np.linspace(0, 10000, 10001)
# res = lga.sweep(n_step)
