import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from diffusers import DDPMScheduler
from tqdm import tqdm
import imageio
from typing import List, Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from collections import deque
import time

num_training_timesteps = 100


class TrajectoryDataset(Dataset):
    def __init__(self, num_samples=1000, steps_length=0.02, steps=32):
        self.step_length = steps_length
        self.obstacles = [
            ((0.4, 0.4), 0.1),
            ((0.7, 0.1), 0.05),
            ((0.8, 0.6), 0.08),
            ((0.2, 0.7), 0.07),
        ]
        self.steps = steps
        self.data = self._generate_data(num_samples)

    def _is_colliding(self, point):
        for (cx, cy), r in self.obstacles:
            if np.linalg.norm(point - np.array([cx, cy])) <= r:
                return True
        return False

    def continuous_score(self, point):
        # how continuouse (smooth) the trajectory is
        diff = np.linalg.norm(point[1:] - point[:-1], axis=1)
        return np.mean(diff), np.std(diff)

    def get_safety_score(self, point):
        # calculate the safety score based on distance to obstacles
        min_distance = float("inf")
        for (cx, cy), r in self.obstacles:
            distance = np.linalg.norm(point - np.array([cx, cy])) - r
            if distance < min_distance:
                min_distance = distance
        return min_distance

    def check_traj_lowest_score(self, trajectory):
        # check the lowest safety score in the trajectory
        lowest_score = float("inf")
        for point in trajectory:
            score = self.get_safety_score(point)
            if score < lowest_score:
                lowest_score = score
        return lowest_score

    def get_noisy_safety_score(self, point, noise_level=0.01):
        safety_score = self.get_safety_score(point)
        noise = np.random.normal(0, noise_level)
        return safety_score + noise

    def _generate_random_point(self):
        # random points that does not collide with obstacles
        while True:
            point = np.random.rand(2)
            if not self._is_colliding(point):
                return point

    def _transform_traj(self, trajectory, steps):
        # transform trajectory to a fixed number of steps, using linear interpolation
        res = []
        traj_len = len(trajectory)
        for i in range(steps):
            t = i / (steps - 1) * (traj_len - 1)
            t0 = int(t)
            t1 = min(t0 + 1, traj_len - 1)
            alpha = t - t0
            point = trajectory[t0] * (1 - alpha) + trajectory[t1] * alpha
            res.append(point)
        return np.array(res)

    def _generate_trajectory(self, start, goal):
        path = [start]
        current_point = start
        prev_direction = (goal - start) / np.linalg.norm(goal - start)
        while np.linalg.norm(current_point - goal) > self.step_length:
            new_direction = (goal - current_point) / np.linalg.norm(
                goal - current_point
            )
            direction = (prev_direction + new_direction) / np.linalg.norm(
                prev_direction + new_direction
            )
            next_point = current_point + direction * self.step_length
            if self._is_colliding(next_point):
                # try different directions, from the closest to furthest
                next_point = None
                for base_angle in np.linspace(0, np.pi, 10):
                    for angle in [-base_angle, base_angle]:
                        rotated_direction = np.array(
                            [
                                direction[0] * np.cos(angle)
                                - direction[1] * np.sin(angle),
                                direction[0] * np.sin(angle)
                                + direction[1] * np.cos(angle),
                            ]
                        )
                        next_point_rotated = (
                            current_point + rotated_direction * self.step_length
                        )
                        if not self._is_colliding(next_point_rotated):
                            next_point = next_point_rotated
                            break
                    if next_point is not None:
                        break
                path.append(next_point)
                current_point = next_point
                prev_direction = rotated_direction
            else:
                path.append(next_point)
                current_point = next_point
                prev_direction = direction
        path.append(goal)
        path = self._transform_traj(path, self.steps)
        path = np.array(path)
        return path

    def _generate_data(self, n):
        data = []
        for _ in tqdm(range(n), desc="Generating dataset"):
            start = self._generate_random_point()
            goal = self._generate_random_point()
            path = self._generate_trajectory(start, goal)
            data.append((start, goal, path))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        start, goal, path = self.data[idx]
        cond = np.concatenate([start, goal])
        return torch.tensor(path, dtype=torch.float32), torch.tensor(
            cond, dtype=torch.float32
        )

    def visualize(self, idx):
        start, goal, path = self.data[idx]
        plt.figure(figsize=(5, 5))
        plt.plot(path[:, 0], path[:, 1], "-o", label="Path")
        for (cx, cy), r in self.obstacles:
            plt.gca().add_patch(plt.Circle((cx, cy), r, color="gray"))
        plt.scatter(*start, c="green", s=100, label="Start")
        plt.scatter(*goal, c="red", s=100, label="Goal")
        plt.legend()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.show()


class TrajectoryModel(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.d_model = d_model

        # Project (x, y) + condition vector to d_model
        self.input_proj = nn.Linear(2 + 4, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 32, d_model))  # 32 steps
        self.time_encoding = nn.Parameter(
            torch.randn(1, num_training_timesteps, d_model)
        )
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head: project back to 2D noise
        self.output_proj = nn.Linear(d_model, 2)

    def forward(self, sample, timestep, class_labels):
        """
        sample: [B, S, 2]
        class_labels: [B, 4] (start+goal)
        timestep: [B]
        """
        B, S, _ = sample.shape
        cond = class_labels.unsqueeze(1).expand(-1, S, -1)  # [B, S, 4]
        timestep = timestep.unsqueeze(1).expand(-1, S)  # [B, S]
        x = torch.cat([sample, cond], dim=-1)  # [B, S, 6]
        x = (
            self.input_proj(x)
            + self.pos_encoding[:, :S]
            + self.time_encoding[:, timestep].squeeze(dim=0)
        )  # [B, S, d_model]
        x = self.transformer(x)  # [B, S, d_model]
        out = self.output_proj(x)  # [B, S, 2]
        return out


class TrajectoryTrainer:
    def __init__(self, model, dataset, batch_size=128, device=None):
        self.model = model
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.scheduler = DDPMScheduler(num_train_timesteps=num_training_timesteps)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.device = device or torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def train(self, epochs=100):
        loss_tracker = []
        for epoch in range(epochs):
            losses = []
            for traj, cond in tqdm(self.dataloader, desc=f"Epoch {epoch}"):
                traj = traj.to(self.device)
                cond = cond.to(self.device)
                noise = torch.randn_like(traj)
                timesteps = torch.randint(
                    0,
                    self.scheduler.config.num_train_timesteps,
                    (traj.size(0),),
                    device=self.device,
                ).long()

                noisy = self.scheduler.add_noise(traj, noise, timesteps)
                pred = self.model(noisy, timesteps, cond)

                loss = nn.functional.mse_loss(pred, noise)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

            print(f"Epoch {epoch} - Loss: {np.mean(losses):.4f}")
            loss_tracker.append(np.mean(losses))
            # save model
            # if epoch % 40 == 0:
            save_model(self.model, f"trajectory_model_epoch_{epoch}.pth")
        # draw loss curve
        plt.plot(loss_tracker)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.show()
        return loss_tracker


class TrajectorySampler:
    def __init__(self, model, scheduler, steps=32, device=None):
        self.model = model
        self.scheduler = scheduler
        self.steps = steps
        self.device = device or torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )

    def sample(self, start, goal):
        self.model.eval()
        x_list = []
        x = torch.randn(1, self.steps, 2).to(self.device)
        x_list.append(x.squeeze(0).cpu().numpy())
        cond = (
            torch.tensor(np.concatenate([start, goal]), dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )

        for t in reversed(range(self.scheduler.config.num_train_timesteps)):
            timesteps = torch.tensor([t], device=self.device)
            with torch.no_grad():
                noise_pred = self.model(x, timesteps, cond)

            x = self.scheduler.step(noise_pred, timesteps, x).prev_sample
            x_list.append(x.squeeze(0).cpu().numpy())

        return x.squeeze(0).cpu().numpy(), x_list


class Visualizer:
    @staticmethod
    def plot(path, start, goal, obstacles):
        plt.figure(figsize=(5, 5))
        plt.plot(path[:, 0], path[:, 1], "-o", label="Path")
        plt.scatter(*start, c="green", label="Start", s=100)
        plt.scatter(*goal, c="red", label="Goal", s=100)
        for (cx, cy), r in obstacles:
            plt.gca().add_patch(plt.Circle((cx, cy), r, color="gray"))
        plt.legend()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_list(path_list, start, goal, obstacles):
        # generate a gif
        images = []
        for i, path in enumerate(path_list):
            plt.figure(figsize=(5, 5))
            plt.plot(path[:, 0], path[:, 1], "-o", label="Path")
            plt.scatter(*start, c="green", label="Start")
            plt.scatter(*goal, c="red", label="Goal")
            for (cx, cy), r in obstacles:
                plt.gca().add_patch(plt.Circle((cx, cy), r, color="gray"))
            plt.legend()
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.grid(True)
            plt.savefig(f"img/frame_{i}.png")
            images.append(imageio.imread(f"img/frame_{i}.png"))
        imageio.mimsave("trajectory.gif", images, fps=10)
        print("GIF saved as trajectory.gif")

    @staticmethod
    def plot_two_traj(nominal_path, safe_path, start, goal, obstacles):
        plt.figure(figsize=(5, 5))
        plt.plot(nominal_path[:, 0], nominal_path[:, 1], "-o", label="Nominal Path")
        plt.plot(safe_path[:, 0], safe_path[:, 1], "-o", label="Safe Path")
        plt.scatter(*start, c="green", label="Start", s=100)
        plt.scatter(*goal, c="red", label="Goal", s=100)
        for (cx, cy), r in obstacles:
            plt.gca().add_patch(plt.Circle((cx, cy), r, color="gray"))
        plt.legend()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.show()


class SafetyBoundary:
    def __init__(self, points: List[Tuple[float, float, float]], eps=0.01, offset=0.01):
        self.points = deque(maxlen=500)
        self.points.extend(points)
        self.gp = self.compute_GP()
        self.eps = eps
        self.offset = offset

    def add_point(self, point: Tuple[float, float, float]):
        self.points.append(point)

    def recompute_GP(self, restarts: int = 8, noise_level: float = 1e-2):
        self.gp = self.compute_GP(restarts, noise_level)

    def compute_GP(self, restarts: int = 8, noise_level: float = 1e-2):
        if len(self.points) == 0:
            print("No points to compute GP.")
            return None
        points = np.array([point[:2] for point in self.points], dtype=np.float32)
        values = (
            np.array([point[2] for point in self.points], dtype=np.float32)
            - self.offset
        )
        kernel = 1.0 * RBF(
            length_scale=1.0, length_scale_bounds=(1e-8, 1e5)
        ) + WhiteKernel(noise_level=noise_level**2, noise_level_bounds="fixed")

        gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=restarts, normalize_y=False
        )

        gp.fit(points, values)
        return gp

    def visualize_safety_boundary(self, threshold=0):
        if self.gp is None:
            print("GP not computed.")
            return
        # draw those under threshold
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x, y)
        Z = self.gp.predict(np.c_[X.ravel(), Y.ravel()])
        Z = Z.reshape(X.shape)
        plt.figure(figsize=(5, 5))
        # draw those z under threshold
        plt.imshow(
            Z < threshold,
            extent=(0, 1, 0, 1),
            origin="lower",
            cmap="gray",
            alpha=0.5,
        )
        plt.colorbar(label="Safety Score")
        plt.scatter(
            [point[0] for point in self.points],
            [point[1] for point in self.points],
            c=[point[2] for point in self.points],
            cmap="viridis",
            s=10,
        )
        plt.title("Safety Boundary")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.show()

    def get_gp_score(self, point: Tuple[float, float]):
        if self.gp is None:
            print("GP not computed.")
            return None
        point = np.array(point).reshape(1, -1)
        score = self.gp.predict(point)
        return score[0]

    def get_high_score_points_around(
        self, point: Tuple[float, float], scale: float = 0.8
    ):
        score = self.get_gp_score(point)
        highest_gradient = None
        highest_gradient_direction = None
        for angle in np.linspace(0, 2 * np.pi, 32):
            new_point = (
                point[0] + self.eps * np.cos(angle),
                point[1] + self.eps * np.sin(angle),
            )
            new_score = self.get_gp_score(new_point)
            gradient = (new_score - score) / self.eps
            if highest_gradient is None or gradient > highest_gradient:
                highest_gradient = gradient
                highest_gradient_direction = angle
        if highest_gradient_direction is not None:
            new_point = (
                point[0]
                + np.cos(highest_gradient_direction)
                * scale
                / max(abs(highest_gradient), 0.5)
                * min(abs(score), 0.1),
                point[1]
                + np.sin(highest_gradient_direction)
                * scale
                / max(abs(highest_gradient), 0.5)
                * min(abs(score), 0.1),
            )
            return new_point


class SafeTrajectorySampler(TrajectorySampler):
    def __init__(
        self, model, scheduler, steps=32, device=None, dataset: TrajectoryDataset = None
    ):
        super().__init__(model, scheduler, steps, device)
        self.safety_boundary = SafetyBoundary([])
        self.dataset = dataset

    def sample(self, start, goal):
        self.model.eval()
        x_list = []
        x = torch.randn(1, self.steps, 2).to(self.device)
        x_list.append(x.squeeze(0).cpu().numpy())
        cond = (
            torch.tensor(np.concatenate([start, goal]), dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )

        for t in reversed(range(self.scheduler.config.num_train_timesteps)):
            timesteps = torch.tensor([t], device=self.device)
            with torch.no_grad():
                noise_pred = self.model(x, timesteps, cond)

            x = self.scheduler.step(noise_pred, timesteps, x).prev_sample
            for i in range(x.shape[1]):
                if np.random.rand() < 0.8:
                    continue
                point = x[0, i].cpu().numpy() + np.random.normal(0, 0.01, 2)
                noisy_score = self.dataset.get_noisy_safety_score(
                    point, noise_level=0.01
                )
                self.safety_boundary.add_point((point[0], point[1], noisy_score))
            if t % 50 == 0:
                self.safety_boundary.recompute_GP(noise_level=0.01)
            if self.safety_boundary.gp is not None:
                for i in range(x.shape[1]):
                    if self.safety_boundary.get_gp_score(x[0, i].cpu().numpy()) < 0:
                        new_point = self.safety_boundary.get_high_score_points_around(
                            x[0, i].cpu().numpy()
                        )
                        if new_point is not None:
                            x[0, i] = torch.tensor(new_point, dtype=torch.float32).to(
                                self.device
                            )

            x_list.append(x.squeeze(0).cpu().numpy())
        # self.safety_boundary.visualize_safety_boundary()

        return x.squeeze(0).cpu().numpy(), x_list


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)
    print(f"Model loaded from {path}, device: {device}")
    return model


def train():
    print("Generating dataset...")
    dataset = TrajectoryDataset(num_samples=102400)
    print(f"Dataset size: {len(dataset)}")
    dataset.visualize(0)
    model = TrajectoryModel()
    trainer = TrajectoryTrainer(model, dataset, batch_size=256)
    trainer.train(epochs=100)

    sampler = TrajectorySampler(model, trainer.scheduler)
    start = np.array([0.05, 0.1])
    goal = np.array([0.95, 0.9])
    sampled_path, sample_path_list = sampler.sample(start, goal)

    Visualizer.plot(sampled_path, start, goal, dataset.obstacles)

    save_model(model, "trajectory_model.pth")


def sample():
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = TrajectoryDataset(num_samples=1)
    dataset.visualize(0)
    model = TrajectoryModel()
    model = load_model(model, "trajectory_model_epoch_100.pth")
    scheduler = DDPMScheduler(num_train_timesteps=num_training_timesteps)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    sampler = TrajectorySampler(model, scheduler)
    safe_sampler = SafeTrajectorySampler(model, scheduler, dataset=dataset, steps=32)

    nominal_safe_score = []
    safety_safe_score = []
    nominal_compute_time = []
    safety_compute_time = []

    nominal_continuous_mean_scores = []
    safety_continuous_mean_scores = []
    nominal_continuous_std_scores = []
    safety_continuous_std_scores = []

    for i in tqdm(range(100), desc="Sampling"):
        start = dataset._generate_random_point()
        goal = dataset._generate_random_point()
        time_start = time.time()
        nominal_path, nominal_path_list = sampler.sample(start, goal)
        time_end = time.time()
        nominal_compute_time.append(time_end - time_start)
        time_start = time.time()
        safe_path, safe_path_list = safe_sampler.sample(start, goal)
        time_end = time.time()
        safety_compute_time.append(time_end - time_start)
        nominal_safe_score.append(dataset.check_traj_lowest_score(nominal_path))
        safety_safe_score.append(dataset.check_traj_lowest_score(safe_path))
        nominal_continuous_mean, nominal_continuous_std = dataset.continuous_score(
            nominal_path
        )
        safety_continuous_mean, safety_continuous_std = dataset.continuous_score(
            safe_path
        )
        nominal_continuous_mean_scores.append(nominal_continuous_mean)
        nominal_continuous_std_scores.append(nominal_continuous_std)
        safety_continuous_mean_scores.append(safety_continuous_mean)
        safety_continuous_std_scores.append(safety_continuous_std)
        Visualizer.plot_two_traj(
            nominal_path, safe_path, start, goal, dataset.obstacles
        )
    print(
        f"Nominal Path Safety Score: {np.mean(nominal_safe_score):.4f} ± {np.std(nominal_safe_score):.4f}"
    )
    print(
        f"Safe Path Safety Score: {np.mean(safety_safe_score):.4f} ± {np.std(safety_safe_score):.4f}"
    )
    print(
        f"Nominal rate for violating safety: {np.sum(np.array(nominal_safe_score) < 0) / len(nominal_safe_score):.4f}"
    )
    print(
        f"Safe rate for violating safety: {np.sum(np.array(safety_safe_score) < 0) / len(nominal_safe_score):.4f}"
    )
    print(
        f"Nominal Path Compute Time: {np.mean(nominal_compute_time):.4f} ± {np.std(nominal_compute_time):.4f}"
    )
    print(
        f"Safe Path Compute Time: {np.mean(safety_compute_time):.4f} ± {np.std(safety_compute_time):.4f}"
    )
    print(
        f"Nominal Path Continuous Score: {np.mean(nominal_continuous_mean_scores):.4f} ± {np.std(nominal_continuous_mean_scores):.4f}"
    )
    print(
        f"Safe Path Continuous Score: {np.mean(safety_continuous_mean_scores):.4f} ± {np.std(safety_continuous_mean_scores):.4f}"
    )
    print(
        f"Nominal Path Continuous Score Std: {np.mean(nominal_continuous_std_scores):.4f} ± {np.std(nominal_continuous_std_scores):.4f}"
    )
    print(
        f"Safe Path Continuous Score Std: {np.mean(safety_continuous_std_scores):.4f} ± {np.std(safety_continuous_std_scores):.4f}"
    )


if __name__ == "__main__":
    # train()
    sample()
