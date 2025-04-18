import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler
import logging
import random
import imageio
import numpy as np
from collections import OrderedDict

import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.environments.manipulation.lift import Lift
import robosuite.macros as macros
from torchvision import transforms

# ---------------- Logging Setup ----------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# device = "cpu"
logger.info(f"Using device: {device}")


class DiffusionPolicy(nn.Module):
    def __init__(self, obs_embed_dim=256, action_dim=7):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 15 * 15, obs_embed_dim),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(obs_embed_dim + action_dim + 1, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, noisy_action, obs_image, t):
        obs_embed = self.obs_encoder(obs_image)
        t = t.float().unsqueeze(1) / 10.0
        x = torch.cat([noisy_action, obs_embed, t], dim=1)
        return self.mlp(x)


class MyLift(Lift):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, **kwargs):
        res = super().reset(**kwargs)
        self.prev_dist = self._gripper_to_target(
            gripper=self.robots[0].gripper,
            target=self.cube.root_body,
            target_type="body",
            return_distance=True,
        )
        return res

    def reward(self, action=None):
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 10

        # reaching reward
        dist = self._gripper_to_target(
            gripper=self.robots[0].gripper,
            target=self.cube.root_body,
            target_type="body",
            return_distance=True,
        )
        dist_change = self.prev_dist - dist
        self.prev_dist = dist
        reward += dist_change * 50

        # grasping reward
        if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube):
            reward += 2

        return reward

    def _get_observations(self, force_update=False):
        """
        Grabs observations from the environment.
        Args:
            force_update (bool): If True, will force all the observables to update their internal values to the newest
                value. This is useful if, e.g., you want to grab observations when directly setting simulation states
                without actually stepping the simulation.
        Returns:
            OrderedDict: OrderedDict containing observations [(name_string, np.array), ...]
        """
        observations = OrderedDict()
        obs_by_modality = OrderedDict()

        # Force an update if requested
        if force_update:
            self._update_observables(force=True)

        # Loop through all observables and grab their current observation
        for obs_name, observable in self._observables.items():
            if observable.is_enabled() and observable.is_active():
                obs = observable.obs
                observations[obs_name] = obs
                modality = observable.modality + "-state"
                if modality not in obs_by_modality:
                    obs_by_modality[modality] = []
                # Make sure all observations are numpy arrays so we can concatenate them
                array_obs = [obs] if type(obs) in {int, float} or not obs.shape else obs
                obs_by_modality[modality].append(np.array(array_obs))

        # Add in modality observations
        for modality, obs in obs_by_modality.items():
            # To save memory, we only concatenate the image observations if explicitly requested
            if modality == "image-state" and not macros.CONCATENATE_IMAGES:
                continue
            observations[modality] = np.concatenate(obs, axis=-1)

        imageio.imwrite(
            f"/Users/yhong3/Documents/Notes/Notes/Courses/16-886 Embodied AI Safety/project/EAIS/code/img/debug/test.png",
            observations["agentview_image"],
        )
        return observations


def train_diffusion_policy():
    env = GymWrapper(
        MyLift(
            robots="Panda",
            has_renderer=False,
            use_camera_obs=True,
            camera_names="agentview",
            control_freq=20,
            reward_shaping=True,
        ),
        flatten_obs=False,
    )

    model = DiffusionPolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = DDPMScheduler(num_train_timesteps=10)
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    replay_buffer = []
    batch_size = 128
    max_buffer_size = 10000
    num_episodes = 100

    for ep in range(num_episodes):
        obs = env.reset()[0]
        total_reward = 0
        done = False
        step = 0
        losses = []
        frames = []

        while not done:
            step += 1
            obs_img = obs["agentview_image"][:, :, ::-1]
            img_tensor = transform(obs_img).unsqueeze(0).to(device)
            frames.append(obs["agentview_image"][:, :, ::-1])
            clipped_action = []

            with torch.no_grad():
                x_t = torch.randn((1, 7)).to(device)

                for t in scheduler.timesteps:
                    t_batch = torch.tensor([t], device=device, dtype=torch.long)
                    pred_noise = model(x_t, img_tensor, t_batch)
                    step_output = scheduler.step(pred_noise, t, x_t)
                    x_t = step_output.prev_sample

                action = x_t  # Final denoised action

                action_clipped = torch.tanh(action).squeeze().cpu().numpy()
                clipped_action.append(action_clipped)
                next_obs, reward, terminated, truncated, _ = env.step(action_clipped)
                done = terminated or truncated
                total_reward += reward

                # Store transition
                replay_buffer.append((img_tensor, action.detach(), reward))
                if len(replay_buffer) > max_buffer_size:
                    replay_buffer.pop(0)

            # Train from random batch
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                imgs, actions, rewards = zip(*batch)
                imgs = torch.cat(imgs)
                actions = torch.stack(actions).reshape(batch_size, -1).to(device)
                rewards = torch.tensor(
                    rewards, device=device, dtype=torch.float32
                ).unsqueeze(1)
                # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

                t = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=device,
                )
                noise = torch.randn_like(actions)
                noisy_actions = scheduler.add_noise(actions, noise, t)

                pred_noise = model(noisy_actions, imgs, t)
                loss = F.mse_loss(pred_noise, noise, reduction="none").mean(dim=1)
                loss = (loss * (torch.clamp(rewards, min=0.0) + 1)).mean()
                # loss = loss.mean()
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            obs = next_obs

        logger.info(
            f"[Episode {ep+1}] Total Reward: {total_reward:.2f}, Loss: {sum(losses)/len(losses):.4f}, Total Steps: {step}, Mean Action: {np.mean(clipped_action):.4f}, Std Action: {np.std(clipped_action):.4f}"
        )
        if ep % 10 == 0:
            imageio.mimsave(f"img/episode_{ep+1:03d}.gif", frames, fps=100)


if __name__ == "__main__":
    train_diffusion_policy()
