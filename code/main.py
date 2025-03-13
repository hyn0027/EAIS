import os
import matplotlib.pyplot as plt
import hj_reachability as hj
from dubins import dyn_sys
from config import grid
from config import num_cells
from safety_boundaries import ground_truth_lx, get_random_points, GP
import numpy as np


def solve(failure_lx, path: str):
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

    if not os.path.exists(path):
        os.makedirs(path)

    plt.clf()
    plt.title("Failure Set (0th slice)")
    plt.imshow(failure_lx[:, :, 0].T > 0, origin="lower")
    plt.colorbar()
    plt.savefig(os.path.join(path, "failure_set0.png"))

    plt.clf()
    plt.title("falure lx value (0th slice)")
    plt.imshow(failure_lx[:, :, 0].T, origin="lower")
    plt.colorbar()
    plt.savefig(os.path.join(path, "failure_lx0.png"))

    plt.clf()
    plt.title("Failure Set (25th slice)")
    plt.imshow(failure_lx[:, :, 25].T > 0, origin="lower")
    plt.colorbar()
    plt.savefig(os.path.join(path, "failure_set25.png"))

    plt.clf()
    plt.title("falure lx value (25th slice)")
    plt.imshow(failure_lx[:, :, 25].T, origin="lower")
    plt.colorbar()
    plt.savefig(os.path.join(path, "failure_lx25.png"))


def eval_difference(golden_lx, sim_lx, path="difference"):
    if not os.path.exists(path):
        os.makedirs(path)
    diff = golden_lx - sim_lx
    print(f"Max difference: {diff.max()}")
    print(f"Min difference: {diff.min()}")
    print(f"Mean difference: {diff.mean()}")
    print(f"Std difference: {diff.std()}")
    plt.clf()
    plt.title("Difference (25th slice)")
    plt.imshow(diff[:, :, 25].T, origin="lower")
    plt.colorbar()
    plt.savefig(os.path.join(path, "difference25.png"))
    plt.clf()
    plt.title("Difference (0th slice)")
    plt.imshow(diff[:, :, 0].T, origin="lower")
    plt.colorbar()
    plt.savefig(os.path.join(path, "difference0.png"))


def vanilla_sample(
    golden_lx, noise=0.0, gp_noise_level=1e-2, path="vanilla_sample", total_points=200
):
    sim_lx = GP(get_random_points(total_points, golden_lx, noise), gp_noise_level)
    solve(sim_lx, path)
    total_possible_points = num_cells[0] * num_cells[1] * num_cells[2]
    print(f"vanilla sample with noise: {noise}, gp_noise_level: {gp_noise_level}")
    print(f"Total possible points: {total_possible_points}")
    print(f"Percetage of points used: {total_points/total_possible_points * 100}")
    eval_difference(golden_lx, sim_lx, path)
    print("-" * 50)


def active_sampling(
    golden_lx,
    noise=0.0,
    gp_noise_level=1e-2,
    path="active_sampling",
    random_prob=0.5,
    total_points=200,
):
    def find_most_discrepant_point(sim_lx, known_points):
        max_diff = 0
        max_point = None
        for point in known_points:
            x, y, z = point[:3]
            diff = abs(sim_lx[x, y, z] - point[3])
            if diff > max_diff:
                max_diff = diff
                max_point = point
        return max_point

    def add_new_points(sim_lx, known_points):
        if np.random.rand() < random_prob:
            return get_random_points(12, golden_lx, noise)
        else:
            center = find_most_discrepant_point(sim_lx, known_points)
            res = []
            for delta in [-10, -5, 5, 10]:
                x, y, z = center[:3]
                x += delta
                if x < 0 or x >= num_cells[0]:
                    continue
                res.append(
                    (
                        x,
                        y,
                        z,
                        golden_lx[x, y, z] + np.random.normal(0, noise * noise),
                    )
                )
            for delta in [-10, -5, 5, 10]:
                x, y, z = center[:3]
                y += delta
                if y < 0 or y >= num_cells[1]:
                    continue
                res.append(
                    (
                        x,
                        y,
                        z,
                        golden_lx[x, y, z] + np.random.normal(0, noise * noise),
                    )
                )
            for delta in [-10, -5, 5, 10]:
                x, y, z = center[:3]
                z += delta
                if z < 0 or z >= num_cells[2]:
                    continue
                res.append(
                    (
                        x,
                        y,
                        z,
                        golden_lx[x, y, z] + np.random.normal(0, noise * noise),
                    )
                )
            return res

    known_points = get_random_points(80, golden_lx, noise)
    sim_lx = GP(known_points, gp_noise_level)
    while len(known_points) < total_points:
        known_points.extend(add_new_points(sim_lx, known_points))
        sim_lx = GP(known_points, gp_noise_level)
    print(f"active sampling with noise: {noise}, gp_noise_level: {gp_noise_level}")
    eval_difference(golden_lx, sim_lx, path)
    print(f"number of points used: {len(known_points)}")
    print(
        f"Percentage of points used: {len(known_points)/(num_cells[0] * num_cells[1] * num_cells[2]) * 100}"
    )
    solve(sim_lx, path)

    # plot the distribution of the points
    x = [point[0] for point in known_points]
    y = [point[1] for point in known_points]
    z = [point[2] for point in known_points]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig(os.path.join(path, "point_distribution.png"))
    print("-" * 50)


def main():
    golden_lx = ground_truth_lx()
    solve(golden_lx, "golden")
    active_sampling(
        golden_lx,
        noise=1e-2,
        gp_noise_level=1e-2,
        path="active_sampling_1e-2_1e-2",
        random_prob=0.5,
        total_points=300,
    )
    active_sampling(
        golden_lx,
        noise=1e-1,
        gp_noise_level=1e-1,
        path="active_sampling_1e-1_1e-1",
        random_prob=0.5,
        total_points=300,
    )
    active_sampling(
        golden_lx,
        noise=5e-1,
        gp_noise_level=5e-1,
        path="active_sampling_5e-1_5e-1",
        random_prob=0.5,
        total_points=300,
    )
    active_sampling(
        golden_lx,
        noise=1e-5,
        gp_noise_level=1e-5,
        path="active_sampling_1e-5_1e-5",
        random_prob=0.5,
        total_points=300,
    )
    active_sampling(
        golden_lx,
        noise=1e-7,
        gp_noise_level=1e-7,
        path="active_sampling_1e-7_1e-7",
        random_prob=0.5,
        total_points=300,
    )
    active_sampling(
        golden_lx,
        noise=1,
        gp_noise_level=1,
        path="active_sampling_1_1",
        random_prob=0.5,
        total_points=300,
    )
    vanilla_sample(
        golden_lx,
        noise=1e-7,
        gp_noise_level=1e-7,
        path="vanilla_sample_1e-7_1e-7",
        total_points=300,
    )
    vanilla_sample(
        golden_lx,
        noise=1e-4,
        gp_noise_level=1e-4,
        path="vanilla_sample_1e-4_1e-4",
        total_points=300,
    )
    vanilla_sample(
        golden_lx,
        noise=1e-2,
        gp_noise_level=1e-2,
        path="vanilla_sample_1e-2_1e-2",
        total_points=300,
    )
    vanilla_sample(
        golden_lx,
        noise=1e-1,
        gp_noise_level=1e-1,
        path="vanilla_sample_1e-1_1e-1",
        total_points=300,
    )
    vanilla_sample(
        golden_lx,
        noise=1,
        gp_noise_level=1,
        path="vanilla_sample_1_1",
        total_points=300,
    )

    vanilla_sample(
        golden_lx,
        noise=5e-1,
        gp_noise_level=5e-1,
        path="vanilla_sample_5e-1_5e-1",
        total_points=300,
    )
    vanilla_sample(
        golden_lx,
        noise=0,
        gp_noise_level=5e-2,
        path="vanilla_sample_0_5e-2",
        total_points=300,
    )
    vanilla_sample(
        golden_lx,
        noise=5e-2,
        gp_noise_level=1e-7,
        path="vanilla_sample_5e-2_1e-7",
        total_points=300,
    )


if __name__ == "__main__":
    main()
