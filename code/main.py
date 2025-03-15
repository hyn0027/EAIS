import os
import warnings
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

    extended_point_idx = []

    def find_k_most_discrepant_point(sim_lx, known_points, k=1):
        # diffs = [abs(sim_lx[x, y, z] - v) for x, y, z, v in known_points]
        # idx = np.argmax(diffs)
        # return known_points[idx], idx
        diffs = [abs(sim_lx[x, y, z] - v) for x, y, z, v in known_points]
        idx = np.argsort(diffs)[-k:]
        return [known_points[i] for i in idx], idx

    def add_new_points(sim_lx, known_points):
        if np.random.rand() < random_prob:
            return get_random_points(120, golden_lx, noise)
        centers, idxs = find_k_most_discrepant_point(sim_lx, known_points, k=10)
        res = []
        for center, idx in zip(centers, idxs):
            if idx in extended_point_idx:
                res.extend(get_random_points(12, golden_lx, noise))
            extended_point_idx.append(idx)
            x, y, z = center[:3]
            deltas = [-10, -5, 5, 10]
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
                        (
                            nx,
                            ny,
                            nz,
                            golden_lx[nx, ny, nz] + np.random.normal(0, noise),
                        )
                    )
        return res

    known_points = get_random_points(400, golden_lx, noise)
    # SUPPRESE WARNING
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim_lx = GP(known_points, gp_noise_level, restarts=1)

    while len(known_points) < total_points:
        known_points.extend(add_new_points(sim_lx, known_points))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim_lx = GP(known_points, gp_noise_level, restarts=1)

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


def simulate(pass_n) -> None:
    """Run all simulations with various parameters."""
    if not os.path.exists(f"res/{pass_n}"):
        os.makedirs(f"res/{pass_n}")
    golden_lx = ground_truth_lx()
    solve(golden_lx, f"res/{pass_n}/golden")
    # save golden_lx
    np.save(os.path.join(f"res/{pass_n}/golden", "lx.npy"), golden_lx)

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
            golden_lx,
            noise,
            gp_noise,
            f"res/{pass_n}/active_sampling_{noise}_{gp_noise}",
            0.5,
            800,
        )
        diff_vanilla = vanilla_sample(
            golden_lx,
            noise,
            gp_noise,
            f"res/{pass_n}/vanilla_sample_{noise}_{gp_noise}",
            800,
        )
        diff_dict[f"active_{noise}_{gp_noise}"] = diff_active
        diff_dict[f"vanilla_{noise}_{gp_noise}"] = diff_vanilla
        # save diff to npy
        np.save(
            os.path.join(
                f"res/{pass_n}/active_sampling_{noise}_{gp_noise}", "diff.npy"
            ),
            diff_active,
        )
        np.save(
            os.path.join(f"res/{pass_n}/vanilla_sample_{noise}_{gp_noise}", "diff.npy"),
            diff_vanilla,
        )

    return diff_dict


def main():
    """Main function to run simulations."""
    diff = None
    for i in range(4):
        print(f"Simulation Pass{i}")
        if diff is None:
            diff = simulate(i)
        else:
            diff_tmp = simulate(i)
            for key in diff.keys():
                diff[key] = np.concatenate((diff[key], diff_tmp[key]), axis=0)

    for key in diff.keys():
        print(
            f"[{key}] Max: {diff[key].max()}, Min: {diff[key].min()}, Mean: {diff[key].mean()}, Std: {diff[key].std()}"
        )


if __name__ == "__main__":
    main()
