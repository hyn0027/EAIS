import jax.numpy as jnp
from config import grid, num_cells, grid_np
from typing import Tuple, List
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import time


def ground_truth_lx():
    x_c, y_c, radius = -1.0, -1.0, 0.5
    obstacle1 = (
        jnp.linalg.norm(np.array([0.0, 1.0]) - grid.states[..., :2], axis=-1) - radius
    )
    obstacle2 = (
        jnp.linalg.norm(np.array([x_c, y_c]) - grid.states[..., :2], axis=-1) - radius
    )
    failure_lx = jnp.minimum(obstacle1, obstacle2)
    return failure_lx


def GP(points: List[Tuple[int, int, int, float]], noise_level=1e-2):
    X = np.array([list(point[:3]) for point in points], dtype=np.float32)
    X = X / np.array(num_cells)
    y = np.array([point[3] for point in points], dtype=np.float32)
    time_0 = time.time()
    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(
        noise_level=noise_level, noise_level_bounds=(1e-8, 1e6)
    )
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=4)
    gp.fit(X, y)
    time_1 = time.time()
    print(f"Training time: {time_1 - time_0}")

    # Predict in one go
    preds = gp.predict(grid_np, return_std=False)

    # Reshape to 3D grid
    lx = preds.reshape(num_cells)
    print("time to predict: ", time.time() - time_1)
    return lx


def random_point_location() -> Tuple[int, int, int]:
    return (
        np.random.randint(0, num_cells[0]),
        np.random.randint(0, num_cells[1]),
        np.random.randint(0, num_cells[2]),
    )


def get_random_points(
    num_points, golden_lx, noise=0.0
) -> List[Tuple[int, int, int, float]]:
    known_points = []
    for _ in range(num_points):
        x, y, z = random_point_location()
        known_points.append(
            (x, y, z, golden_lx[x, y, z] + np.random.normal(0, noise * noise))
        )
    return known_points
