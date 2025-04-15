import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import UNet2DModel, DDPMScheduler
import numpy as np
import random
import logging

import robosuite as suite
from robosuite.wrappers import GymWrapper

# ---------------- Logging Setup ----------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {device}")


# Set seed for reproducibility
def set_seed(seed=42):
    logger.info(f"Setting random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --- robosuite-based Dataset ---
class RoboSuiteDataset(Dataset):
    def __init__(
        self,
        env_name="Lift",
        trajectory_len=1000,
        trajectory_dim=7,
        camera_name="agentview",
    ):
        logger.info("Initializing RoboSuiteDataset...")
        self.env = GymWrapper(
            suite.make(
                env_name=env_name,
                robots="Panda",
                has_renderer=False,
                use_camera_obs=True,
                camera_names=camera_name,
                control_freq=20,
            ),
            flatten_obs=False,
        )
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
        self.trajectory_dim = trajectory_dim
        self.camera_name = camera_name

        self.frames = []
        self.actions = []
        obs = self.env.reset()
        done = False
        steps = 0

        logger.info("Collecting environment data...")
        while steps < trajectory_len:
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            img = obs[f"{self.camera_name}_image"][:, :, ::-1]
            self.frames.append(self.transform(img))
            self.actions.append(
                torch.tensor(action[:trajectory_dim], dtype=torch.float32)
            )
            steps += 1
            if done:
                logger.info(f"Episode ended early at step {steps}, resetting env.")
                obs = self.env.reset()

        logger.info(f"Collected {len(self.frames)} frames.")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return {"image": self.frames[idx], "trajectory": self.actions[idx]}


# --- Model (unconditional UNet) ---
class TrajectoryDiffusionModel(nn.Module):
    def __init__(self, trajectory_dim=7):
        super().__init__()
        logger.info("Initializing TrajectoryDiffusionModel...")
        self.unet = UNet2DModel(
            sample_size=64,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(64, 128, 128),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        )
        self.output_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, trajectory_dim),
        )

    def forward(self, noisy_image, timestep):
        model_out = self.unet(noisy_image, timestep).sample
        traj = self.output_head(model_out)
        return traj


# --- Training ---
def train():
    logger.info("Starting training process...")
    set_seed()
    dataset = RoboSuiteDataset()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = TrajectoryDiffusionModel().to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    num_epochs = 3

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0.0

        for i, batch in enumerate(dataloader):
            image = batch["image"].to(device)
            traj = batch["trajectory"].to(device)

            noise = torch.randn_like(image).to(device)
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (image.size(0),),
                dtype=torch.long,
                device=device,
            )

            noisy_image = scheduler.add_noise(image, noise, timesteps)
            pred_traj = model(noisy_image, timesteps)

            loss = nn.MSELoss()(pred_traj, traj)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 10 == 0 or (i + 1) == len(dataloader):
                logger.info(
                    f"  Batch {i + 1}/{len(dataloader)} - Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1} completed - Avg Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    train()
