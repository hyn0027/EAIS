import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from diffusers import DDPMScheduler
from tqdm import tqdm
import imageio


class TrajectoryDataset(Dataset):
    def __init__(self, num_samples=1000, steps_length=0.02, steps=64):
        self.step_length = steps_length
        self.obstacles = [
            ((0.4, 0.4), 0.2, 0.2),
            ((0.7, 0.1), 0.1, 0.3),
            ((0.8, 0.6), 0.3, 0.1),
            ((0.2, 0.7), 0.1, 0.4),
        ]
        self.steps = steps
        self.data = self._generate_data(num_samples)

    def _is_colliding(self, point):
        for (ox, oy), w, h in self.obstacles:
            if ox <= point[0] <= ox + w and oy <= point[1] <= oy + h:
                return True
        return False

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
        for _ in range(n):
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
        for (ox, oy), w, h in self.obstacles:
            plt.gca().add_patch(plt.Rectangle((ox, oy), w, h, color="gray"))
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
        self.pos_encoding = nn.Parameter(torch.randn(1, 64, d_model))  # 64 steps

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
        """
        B, S, _ = sample.shape
        cond = class_labels.unsqueeze(1).expand(-1, S, -1)  # [B, S, 4]
        x = torch.cat([sample, cond], dim=-1)  # [B, S, 6]
        x = self.input_proj(x) + self.pos_encoding[:, :S]  # [B, S, d_model]
        x = self.transformer(x)  # [B, S, d_model]
        out = self.output_proj(x)  # [B, S, 2]
        return out


class TrajectoryTrainer:
    def __init__(self, model, dataset, batch_size=128, device=None):
        self.model = model
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.scheduler = DDPMScheduler(num_train_timesteps=100)
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
            if epoch % 10 == 0:
                save_model(self.model, f"trajectory_model_epoch_{epoch}.pth")
        # draw loss curve
        plt.plot(loss_tracker)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.show()
        return loss_tracker


class TrajectorySampler:
    def __init__(self, model, scheduler, steps=64, device=None):
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
        plt.scatter(*start, c="green", label="Start")
        plt.scatter(*goal, c="red", label="Goal")
        for (ox, oy), w, h in obstacles:
            plt.gca().add_patch(plt.Rectangle((ox, oy), w, h, color="gray"))
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
            for (ox, oy), w, h in obstacles:
                plt.gca().add_patch(plt.Rectangle((ox, oy), w, h, color="gray"))
            plt.legend()
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.grid(True)
            plt.savefig(f"img/frame_{i}.png")
            images.append(imageio.imread(f"img/frame_{i}.png"))
        imageio.mimsave("trajectory.gif", images, fps=10)
        print("GIF saved as trajectory.gif")


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
    model = TrajectoryModel()
    model = load_model(model, "trajectory_model.pth")
    scheduler = DDPMScheduler(num_train_timesteps=100)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    sampler = TrajectorySampler(model, scheduler)
    start = np.array([0.05, 0.1])
    goal = np.array([0.95, 0.9])
    sampled_path, sample_path_list = sampler.sample(start, goal)
    # Visualizer.plot(sampled_path, start, goal, dataset.obstacles)
    Visualizer.plot_list(sample_path_list, start, goal, dataset.obstacles)


if __name__ == "__main__":
    # train()
    sample()
