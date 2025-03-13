import os
import matplotlib.pyplot as plt
import hj_reachability as hj
from dubins import dyn_sys
from config import grid
from config import num_cells
from safety_boundaries import ground_truth_lx, get_random_points, GP
import numpy as np


def solve(failure_lx, path: str):
    solver_settings = hj.SolverSettings.with_accuracy(
        "very_high", hamiltonian_postprocessor=hj.solver.backwards_reachable_tube
    )
    # Time
    time = 0.0
    target_time = -2.8

    # Run the solver
    target_values = hj.step(
        solver_settings, dyn_sys, grid, time, failure_lx, target_time
    )

    if not os.path.exists(path):
        os.makedirs(path)

    plt.clf()
    plt.title("Failure Set (25th slice)")
    plt.imshow(failure_lx[:, :, 25].T > 0, origin="lower")
    plt.colorbar()
    plt.savefig(os.path.join(path, "failure_set.png"))

    plt.clf()
    plt.title("Target Set (25th slice)")
    plt.imshow(target_values[:, :, 25].T > 0, origin="lower")
    plt.colorbar()
    plt.savefig(os.path.join(path, "target_set.png"))

    plt.clf()
    plt.title("falure lx value (25th slice)")
    plt.imshow(failure_lx[:, :, 25].T, origin="lower")
    plt.colorbar()
    plt.savefig(os.path.join(path, "failure_lx.png"))


def eval_difference(golden_lx, sim_lx):
    diff = golden_lx - sim_lx
    print(f"Max difference: {diff.max()}")
    print(f"Min difference: {diff.min()}")
    print(f"Mean difference: {diff.mean()}")
    print(f"Std difference: {diff.std()}")
    plt.clf()
    plt.title("Difference (25th slice)")
    plt.imshow(diff[:, :, 25].T, origin="lower")
    plt.colorbar()
    plt.savefig("difference.png")


def vanilla_sample(golden_lx, noise=0.0, gp_noise_level=1e-2, path="vanilla_sample"):
    sim_lx = GP(get_random_points(200, golden_lx, noise), gp_noise_level)
    solve(sim_lx, path)
    total_possible_points = num_cells[0] * num_cells[1] * num_cells[2]
    print(f"Total possible points: {total_possible_points}")
    print(f"Percetage of points used: {200/total_possible_points * 100}")
    eval_difference(golden_lx, sim_lx)


def active_sampling(
    golden_lx, noise=0.0, gp_noise_level=1e-2, path="active_sampling", random_prob=0.5
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
            for delta in [-8, -4, 4, 8]:
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
            for delta in [-8, -4, 4, 8]:
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
            for delta in [-8, -4, 4, 8]:
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

    known_points = get_random_points(30, golden_lx, noise)
    sim_lx = GP(known_points, gp_noise_level)
    eval_difference(golden_lx, sim_lx)
    while len(known_points) < 200:
        known_points.extend(add_new_points(sim_lx, known_points))
        sim_lx = GP(known_points, gp_noise_level)
        eval_difference(golden_lx, sim_lx)
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
    plt.show()


def main():
    golden_lx = ground_truth_lx()
    solve(golden_lx, "golden")
    active_sampling(
        golden_lx,
        noise=1e-2,
        gp_noise_level=1e-2,
        path="active_sampling_1e-2_1e-2",
        random_prob=0.5,
    )
    # vanilla_sample(
    #     golden_lx, noise=1e-7, gp_noise_level=1e-7, path="vanilla_sample_1e-7_1e-7"
    # )
    # vanilla_sample(
    #     golden_lx, noise=1e-4, gp_noise_level=1e-4, path="vanilla_sample_1e-4_1e-4"
    # )
    # vanilla_sample(
    #     golden_lx, noise=1e-2, gp_noise_level=1e-2, path="vanilla_sample_1e-2_1e-2"
    # )
    # vanilla_sample(
    #     golden_lx, noise=1e-1, gp_noise_level=1e-1, path="vanilla_sample_1e-1_1e-1"
    # )
    # vanilla_sample(golden_lx, noise=1, gp_noise_level=1, path="vanilla_sample_1_1")

    # vanilla_sample(
    #     golden_lx, noise=0, gp_noise_level=5e-2, path="vanilla_sample_1_5e-2"
    # )
    # vanilla_sample(
    #     golden_lx, noise=5e-2, gp_noise_level=1e-7, path="vanilla_sample_5e-2_1e-7"
    # )


if __name__ == "__main__":
    main()
