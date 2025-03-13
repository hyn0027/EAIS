import numpy as np
import hj_reachability as hj

grid_min = np.array([-2.0, -2.0, 0.0])  # in meters
grid_max = np.array([2.0, 2.0, 2 * np.pi])  # in meters
num_cells = (51, 51, 51)  # in cells
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
    hj.sets.Box(grid_min, grid_max), (51, 51, 51), periodic_dims=2
)
grid_np = np.array(
    [
        [i, j, k]
        for i in range(num_cells[0])
        for j in range(num_cells[1])
        for k in range(num_cells[2])
    ],
    dtype=np.float32,
) / np.array(num_cells)
