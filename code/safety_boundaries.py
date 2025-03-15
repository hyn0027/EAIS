import numpy as np
from typing import Tuple, List
from config import num_cells, grid_np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import time
import scipy.optimize
import warnings


class Sinoides:
    def __init__(self) -> None:
        # Initialize parameters using a single call to np.random.normal
        self.parameters = np.random.normal(0, 2, 12)

    def f(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        p = self.parameters  # For convenience
        return (
            np.sin(2 * x * p[0] + p[1])
            + np.sin(2 * y * p[2] + p[3])
            + np.sin(2 * z * p[4] + p[5])
            + np.sin(4.0 * x * y * p[6] + p[7])
            + np.sin(4.0 * y * z * p[8] + p[9])
            + np.sin(4.0 * z * x * p[10] + p[11])
        )


def ground_truth_lx() -> np.ndarray:
    """Generate ground truth using Sinoides function over the grid."""
    func = Sinoides()

    # Normalized coordinate arrays
    x = (np.arange(num_cells[0]) - num_cells[0] / 2) / num_cells[0]
    y = (np.arange(num_cells[1]) - num_cells[1] / 2) / num_cells[1]
    z = (np.arange(num_cells[2]) - num_cells[2] / 2) / num_cells[2]

    # Create meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Vectorized evaluation of the function
    lx = func.f(X, Y, Z)
    return lx


def GP(
    points: List[Tuple[int, int, int, float]], noise_level: float = 1e-2, restarts: int = 8
) -> np.ndarray:
    """Gaussian Process regression on sparse points and prediction on grid."""
    X = (
        np.array([point[:3] for point in points], dtype=np.float32)
        - np.array(num_cells, dtype=np.float32) / 2
    ) / np.array(num_cells) * np.sqrt(12)
    y = np.array([point[3] for point in points], dtype=np.float32)

    # Define kernel with RBF and fixed noise
    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-8, 1e5)) + WhiteKernel(
        noise_level=noise_level**2, noise_level_bounds="fixed"
    )

    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=restarts, normalize_y=True
    )

    start_time = time.time()
    gp.fit(X, y)
    training_time = time.time() - start_time
    # print(f"GP training completed in {training_time:.2f} seconds.")

    # Prediction on grid
    preds = gp.predict(grid_np, return_std=False)
    return preds.reshape(num_cells)


def random_point_location() -> Tuple[int, int, int]:
    """Generate a random point in the grid."""
    return tuple(np.random.randint(0, n) for n in num_cells)


def get_random_points(
    num_points: int, golden_lx: np.ndarray, noise: float = 0.0
) -> List[Tuple[int, int, int, float]]:
    """Sample random points from ground truth with optional noise."""
    points = []
    for _ in range(num_points):
        x, y, z = random_point_location()
        value = golden_lx[x, y, z] + np.random.normal(0, noise)
        points.append((x, y, z, value))
    return points
