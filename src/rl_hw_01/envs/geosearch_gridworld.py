from typing import Optional
import numpy as np
import pygame
import os
import warnings
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
from gymnasium.envs.registration import register

os.environ["SDL_AUDIODRIVER"] = "dummy"


def ignore_pygame_warnings():
    warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
    return


ignore_pygame_warnings()


# ============================================================
# 2D Gaussian Density (Exactly as in Assignment)
# ============================================================
def gaussian_2d(x, y, mu_x, mu_y, sx, sy, rho):
    z_x = (x - mu_x) / sx
    z_y = (y - mu_y) / sy

    norm = 1.0 / (2 * np.pi * sx * sy * np.sqrt(1 - rho**2))
    expo = -1.0 / (2 * (1 - rho**2)) * (z_x**2 - 2 * rho * z_x * z_y + z_y**2)
    return norm * np.exp(expo)


# ============================================================
# GeoSearch Environment (25x25 with Rendering)
# ============================================================
class GeoSearchEnvCJ(gym.Env):
    """
    25x25 GeoSearch GridWorld
      - Two Gaussian resource layers f1 (water), f2 (gold)
      - Reward: R(x,y) = A*f1(x,y) + (1-A)*f2(x,y)
      - Deterministic motion
      - Start at (0,0)
    """

    metadata = {"render_modes": ["human", "rgb_array", "none"], "render_fps": 6}

    def __init__(
        self,
        render_mode=None,
        grid_size: int = 25,
        window_width: int = 700,
        window_height: int = 700,
        A: float = 0.75,
        step_penalty: float = 0.0,
        verbose: bool = False,
    ):
        self.width = grid_size
        self.height = grid_size
        self.A = A
        self.step_penalty = step_penalty
        self.window_width = window_width
        self.window_height = window_height
        self._verbose = verbose

        self.terminal_rewards = {}
        self.walls = set()

        # ----- Internal state -----
        self._agent_location = np.array([-1, -1], dtype=np.int32)

        # ----- Observation space -----
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    low=np.array([0, 0], dtype=np.int32),
                    high=np.array([self.width - 1, self.height - 1], dtype=np.int32),
                    shape=(2,),
                    dtype=np.int32,
                )
            }
        )

        # ----- Action space -----
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),  # RIGHT
            1: np.array([-1, 0]),  # LEFT
            2: np.array([0, -1]),  # UP
            3: np.array([0, 1]),  # DOWN
        }

        # ----- Gaussian parameters (Assignment Exact) -----
        self.f1_params = dict(mu_x=20, mu_y=20, sx=1, sy=1, rho=0.25)  # water
        self.f2_params = dict(mu_x=10, mu_y=10, sx=1, sy=1, rho=-0.25)  # gold

        # ----- Precompute resource layers -----
        self.f1 = np.zeros((self.width, self.height), dtype=np.float64)
        self.f2 = np.zeros((self.width, self.height), dtype=np.float64)
        self.R = np.zeros((self.width, self.height), dtype=np.float64)

        for x in range(self.width):
            for y in range(self.height):
                self.f1[x, y] = gaussian_2d(x, y, **self.f1_params)
                self.f2[x, y] = gaussian_2d(x, y, **self.f2_params)

        self.R = self.A * self.f1 + (1.0 - self.A) * self.f2
        self.R = self.R / np.max(self.R)  # normalize for stability

        self.start_state = (0, 0)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    # ----------------------------------------------------
    def _get_obs(self):
        return {"agent": self._agent_location.astype(np.int32, copy=False)}

    def _get_info(self):
        return {}

    # ----------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._agent_location = np.array(self.start_state, dtype=np.int32)
        return self._get_obs(), self._get_info()

    # ----------------------------------------------------
    def step(self, action):
        action = int(action)
        assert self.action_space.contains(action)

        direction = self._action_to_direction[action]
        proposed = self._agent_location + direction

        proposed = np.array(
            [
                np.clip(proposed[0], 0, self.width - 1),
                np.clip(proposed[1], 0, self.height - 1),
            ],
            dtype=np.int32,
        )

        old_state = tuple(self._agent_location)
        self._agent_location = proposed
        new_state = tuple(proposed)

        x, y = new_state
        reward = float(self.R[x, y] - self.step_penalty)

        terminated = False

        if self._verbose:
            print(
                f"Action={action} | Old={old_state} → "
                f"New={new_state} | Reward={reward:.6f}"
            )

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, False, self._get_info()

    # ----------------------------------------------------
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))

        cell_w = self.window_width / self.width
        cell_h = self.window_height / self.height

        # ----- draw reward heatmap -----
        for x in range(self.width):
            for y in range(self.height):
                v = self.R[x, y]
                color = int(255 * v)
                rect = pygame.Rect(x * cell_w, y * cell_h, cell_w, cell_h)
                pygame.draw.rect(canvas, (color, color, 255), rect)

        # ----- draw agent -----
        ax = (self._agent_location[0] + 0.5) * cell_w
        ay = (self._agent_location[1] + 0.5) * cell_h
        radius = 0.35 * min(cell_w, cell_h)
        pygame.draw.circle(canvas, (255, 0, 0), (ax, ay), radius)

        # ----- grid lines -----
        for x in range(self.width + 1):
            pygame.draw.line(
                canvas, 0, (x * cell_w, 0), (x * cell_w, self.window_height), width=1
            )
        for y in range(self.height + 1):
            pygame.draw.line(
                canvas, 0, (0, y * cell_h), (self.window_width, y * cell_h), width=1
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    # ----------------------------------------------------
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


# ============================================================
# TEST HARNESS
# ============================================================
def main(seed: int = 42, render_mode: str = "human"):
    register(
        id="gymnasium_env/GeoSearchEnvCJ-v0",
        entry_point=GeoSearchEnvCJ,
        max_episode_steps=500,
    )

    env = gym.make(
        "gymnasium_env/GeoSearchEnvCJ-v0",
        render_mode=render_mode,
    )

    try:
        check_env(env)
        print("✅ GeoSearch environment passes gym check!")
    except Exception as e:
        print(f"❌ Environment issues: {e}")

    obs, info = env.reset(seed=seed)
    rng = env.unwrapped.np_random

    for step_i in range(80):
        action = int(rng.integers(env.action_space.n))
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step_i:03d} | a={action} | obs={obs} | r={reward:.6f}")

    env.close()


if __name__ == "__main__":
    main()
