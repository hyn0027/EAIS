import jax.numpy as jnp
from config import grid, num_cells, grid_np
from typing import Tuple, List
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import time


def ground_truth_lx():
    class Sinoides:
        def __init__(self) -> None:
            self.parameters = [np.random.normal(0, 2) for i in range(14)]

        def f(self, x, y, z):
            return (
                np.sin(2 * x * self.parameters[0] + self.parameters[1])
                + np.sin(2 * y * self.parameters[2] + self.parameters[3])
                + np.sin(2 * z * self.parameters[4] + self.parameters[5])
                + np.sin(4.0 * x * y * self.parameters[6] + self.parameters[7])
                + np.sin(4.0 * y * z * self.parameters[8] + self.parameters[9])
                + np.sin(4.0 * z * x * self.parameters[10] + self.parameters[11])
                + np.sin(16.0 * x * y * z * self.parameters[12] + self.parameters[13])
                + 1
            )

    func = Sinoides()

    lx = np.zeros(num_cells)
    for i in range(num_cells[0]):
        for j in range(num_cells[1]):
            for k in range(num_cells[2]):
                lx[i, j, k] = func.f(
                    (i - num_cells[0] / 2) / num_cells[0],
                    (j - num_cells[1] / 2) / num_cells[1],
                    (k - num_cells[2] / 2) / num_cells[2],
                )
    return lx


def GP(points: List[Tuple[int, int, int, float]], noise_level=1e-2):
    X = np.array([list(point[:3]) for point in points], dtype=np.float32)
    X = X / np.array(num_cells)
    y = np.array([point[3] for point in points], dtype=np.float32)
    time_0 = time.time()
    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5)) + WhiteKernel(
        noise_level=noise_level * noise_level, noise_level_bounds="fixed"
    )

    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
    gp.fit(X, y)
    time_1 = time.time()
    # print(f"Training time: {time_1 - time_0}")

    # Predict in one go
    preds = gp.predict(grid_np, return_std=False)

    # Reshape to 3D grid
    lx = preds.reshape(num_cells)
    # print("time to predict: ", time.time() - time_1)
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
        known_points.append((x, y, z, golden_lx[x, y, z] + np.random.normal(0, noise)))
    return known_points
