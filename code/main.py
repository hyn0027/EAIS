import os
import numpy as np
import matplotlib.pyplot as plt
from config import num_cells
from safety_boundaries import ground_truth_lx, get_random_points, GP


def ensure_dir(path: str) -> None:
    """Ensure that a directory exists."""
    os.makedirs(path, exist_ok=True)


def plot_slice(
    data: np.ndarray, title: str, filename: str, slice_idx: int, threshold: bool = False
) -> None:
    """Helper to plot and save a slice of data."""
    plt.clf()
    plt.title(title)
    slice_data = data[:, :, slice_idx].T > 0 if threshold else data[:, :, slice_idx].T
    plt.imshow(slice_data, origin="lower")
    plt.colorbar()
    plt.savefig(filename)


def solve(failure_lx: np.ndarray, path: str) -> None:
    """Visualize slices of the failure level set."""
    # solver_settings = hj.SolverSettings.with_accuracy(
    #     "very_high", hamiltonian_postprocessor=hj.solver.backwards_reachable_tube
    # )

    # # Time
    # time = 0.0
    # target_time = -2.8

    # # Run the solver
    # target_values = hj.step(
    #     solver_settings, dyn_sys, grid, time, failure_lx, target_time
    # )

    ensure_dir(path)
    for slice_idx in [0, 25]:
        plot_slice(
            failure_lx,
            f"Failure Set ({slice_idx}th slice)",
            os.path.join(path, f"failure_set{slice_idx}.png"),
            slice_idx,
            True,
        )
        plot_slice(
            failure_lx,
            f"Failure lx value ({slice_idx}th slice)",
            os.path.join(path, f"failure_lx{slice_idx}.png"),
            slice_idx,
        )


def eval_difference(
    golden_lx: np.ndarray, sim_lx: np.ndarray, path: str = "difference"
) -> np.ndarray:
    """Evaluate and visualize difference between golden and simulated lx."""
    ensure_dir(path)
    diff = golden_lx - sim_lx
    print(
        f"[Difference] Max: {diff.max()}, Min: {diff.min()}, Mean: {diff.mean()}, Std: {diff.std()}"
    )
    for slice_idx in [0, 25]:
        plot_slice(
            diff,
            f"Difference ({slice_idx}th slice)",
            os.path.join(path, f"difference{slice_idx}.png"),
            slice_idx,
        )
    return diff


def vanilla_sample(
    golden_lx: np.ndarray,
    noise: float = 0.0,
    gp_noise_level: float = 1e-2,
    path: str = "vanilla_sample",
    total_points: int = 200,
) -> np.ndarray:
    """Perform vanilla (random) sampling and evaluate."""
    sim_lx = GP(get_random_points(total_points, golden_lx, noise), gp_noise_level)
    solve(sim_lx, path)
    total_possible_points = np.prod(num_cells)
    print(f"[Vanilla Sample] Noise: {noise}, GP Noise Level: {gp_noise_level}")
    return eval_difference(golden_lx, sim_lx, path)


def active_sampling(
    golden_lx: np.ndarray,
    noise: float = 0.0,
    gp_noise_level: float = 1e-2,
    path: str = "active_sampling",
    random_prob: float = 0.5,
    total_points: int = 200,
) -> np.ndarray:
    """Perform active sampling based on discrepancy."""

    def find_most_discrepant_point(sim_lx, known_points):
        diffs = [abs(sim_lx[x, y, z] - v) for x, y, z, v in known_points]
        return known_points[np.argmax(diffs)]

    def add_new_points(sim_lx, known_points):
        if np.random.rand() < random_prob:
            return get_random_points(12, golden_lx, noise)
        center = find_most_discrepant_point(sim_lx, known_points)
        x, y, z = center[:3]
        deltas = [-10, -5, 5, 10]
        res = []
        for dx, dy, dz in (
            [(d, 0, 0) for d in deltas]
            + [(0, d, 0) for d in deltas]
            + [(0, 0, d) for d in deltas]
        ):
            nx, ny, nz = x + dx, y + dy, z + dz
            if (
                0 <= nx < num_cells[0]
                and 0 <= ny < num_cells[1]
                and 0 <= nz < num_cells[2]
            ):
                res.append(
                    (nx, ny, nz, golden_lx[nx, ny, nz] + np.random.normal(0, noise**2))
                )
        return res

    known_points = get_random_points(80, golden_lx, noise)
    sim_lx = GP(known_points, gp_noise_level)
    while len(known_points) < total_points:
        known_points.extend(add_new_points(sim_lx, known_points))
        sim_lx = GP(known_points, gp_noise_level)

    known_points = known_points[:total_points]
    sim_lx = GP(known_points, gp_noise_level)

    print(f"[Active Sampling] Noise: {noise}, GP Noise Level: {gp_noise_level}")
    solve(sim_lx, path)

    # 3D scatter plot of sampled points
    x, y, z = zip(*[p[:3] for p in known_points])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig(os.path.join(path, "point_distribution.png"))

    return eval_difference(golden_lx, sim_lx, path)


def simulate() -> None:
    """Run all simulations with various parameters."""
    golden_lx = ground_truth_lx()
    solve(golden_lx, "golden")

    params = [
        (1e-7, 1e-7),
        (1e-6, 1e-6),
        (1e-5, 1e-5),
        (1e-4, 1e-4),
        (1e-3, 1e-3),
        (1e-2, 1e-2),
        (1e-1, 1e-1),
        (5e-1, 5e-1),
        (1, 1),
        (0, 5e-2),
        (5e-2, 1e-7),
    ]

    diff_dict = {}

    for noise, gp_noise in params:
        diff_active = active_sampling(
            golden_lx, noise, gp_noise, f"active_sampling_{noise}_{gp_noise}", 0.5, 300
        )
        diff_vanilla = vanilla_sample(
            golden_lx, noise, gp_noise, f"vanilla_sample_{noise}_{gp_noise}", 300
        )
        diff_dict[f"active_{noise}_{gp_noise}"] = diff_active
        diff_dict[f"vanilla_{noise}_{gp_noise}"] = diff_vanilla

    return diff_dict


def main():
    """Main function to run simulations."""
    diff = None
    for _ in range(20):
        print(f"Simulation Pass{_}")
        simulate()
        if diff is None:
            diff = simulate()
        else:
            diff_tmp = simulate()
            for key in diff.keys():
                diff[key] = np.concatenate((diff[key], diff_tmp[key]), axis=0)

    for key in diff.keys():
        print(
            f"[{key}] Max: {diff[key].max()}, Min: {diff[key].min()}, Mean: {diff[key].mean()}, Std: {diff[key].std()}"
        )


if __name__ == "__main__":
    main()
