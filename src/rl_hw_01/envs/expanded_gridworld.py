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


class GridWorldExpandedEnvCJ(gym.Env):
    """
    6x6 GridWorld from assignment image:
      - Blue = wall (bounce)
      - White = -1
      - Red = -50 (terminal)
      - Green = +100 (terminal)
      - Start at (0,0)
      - Stochastic actions: success 0.75, random 0.25
    """

    metadata = {"render_modes": ["human", "rgb_array", "none"], "render_fps": 4}

    def __init__(
        self,
        render_mode=None,
        window_width: int = 600,
        window_height: int = 600,
        gamma: float = 0.25,
        verbose: bool = True,
    ):
        self.width = 6
        self.height = 6
        self.gamma = gamma
        self.window_width = window_width
        self.window_height = window_height
        self._verbose = verbose

        # internal state
        self._agent_location = np.array([-1, -1], dtype=np.int32)

        # observation space
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

        # 4-action grid
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),  # RIGHT
            1: np.array([-1, 0]),  # LEFT
            2: np.array([0, -1]),  # UP
            3: np.array([0, 1]),  # DOWN
        }

        # ----- WALLS (blue from image) -----
        self.walls = {(2, 0), (2, 1), (2, 3), (2, 4), (2, 5), (3, 3), (4, 3)}

        # ----- TERMINALS -----
        self.terminal_rewards = {
            (4, 1): -50,  # red
            (0, 5): -50,  # red
            (5, 5): +100,  # green
        }

        self.start_state = (0, 0)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    # -----------------------------
    def _get_obs(self):
        return {"agent": self._agent_location.astype(np.int32, copy=False)}

    def _get_info(self):
        return {"terminal_rewards": self.terminal_rewards}

    # -----------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._agent_location = np.array(self.start_state, dtype=np.int32)
        return self._get_obs(), self._get_info()

    # -----------------------------
    def step(self, action):
        action = int(action)
        assert self.action_space.contains(action)

        # stochastic malfunction
        if self.np_random.random() > (1 - self.gamma):
            action = int(self.np_random.integers(self.action_space.n))

        direction = self._action_to_direction[action]
        proposed = self._agent_location + direction

        proposed = np.array(
            [
                np.clip(proposed[0], 0, self.width - 1),
                np.clip(proposed[1], 0, self.height - 1),
            ],
            dtype=np.int32,
        )

        # ----- WALL BOUNCE -----
        if tuple(proposed.tolist()) in self.walls:
            new_state = self._agent_location.copy()
        else:
            new_state = proposed.astype(np.int32)

        old_state = tuple(self._agent_location)
        self._agent_location = new_state
        new_state_tuple = tuple(new_state)

        reward = -1
        terminated = False

        if new_state_tuple in self.terminal_rewards:
            reward = self.terminal_rewards[new_state_tuple]
            terminated = True

        if self._verbose:
            print(
                f"Action={action} | Old={old_state} → New={new_state_tuple} | Reward={reward}"
            )

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, False, self._get_info()

    # -----------------------------
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

        # ----- draw walls -----
        for x, y in self.walls:
            rect = pygame.Rect(x * cell_w, y * cell_h, cell_w, cell_h)
            pygame.draw.rect(canvas, (100, 100, 255), rect)

        # ----- draw terminals -----
        for (x, y), r in self.terminal_rewards.items():
            color = (0, 200, 0) if r > 0 else (200, 50, 50)
            rect = pygame.Rect(x * cell_w, y * cell_h, cell_w, cell_h)
            pygame.draw.rect(canvas, color, rect)

        # ----- draw agent -----
        ax = (self._agent_location[0] + 0.5) * cell_w
        ay = (self._agent_location[1] + 0.5) * cell_h
        radius = 0.35 * min(cell_w, cell_h)
        pygame.draw.circle(canvas, (0, 0, 255), (ax, ay), radius)

        # ----- grid lines -----
        for x in range(self.width + 1):
            pygame.draw.line(
                canvas, 0, (x * cell_w, 0), (x * cell_w, self.window_height), width=2
            )
        for y in range(self.height + 1):
            pygame.draw.line(
                canvas, 0, (0, y * cell_h), (self.window_width, y * cell_h), width=2
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

    # -----------------------------
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


# ---------------------------------------------------------
# TEST HARNESS (MATCHES YOUR BOAT MAIN)
# ---------------------------------------------------------
def main(seed: int = 42, render_mode: str = "human"):
    register(
        id="gymnasium_env/GridWorldImageEnvCJ-v0",
        entry_point=GridWorldExpandedEnvCJ,
        max_episode_steps=200,
    )

    env = gym.make(
        "gymnasium_env/GridWorldImageEnvCJ-v0",
        render_mode=render_mode,
    )

    try:
        check_env(env)
        print("✅ Environment passes gym check!")
    except Exception as e:
        print(f"❌ Environment issues: {e}")

    obs, info = env.reset(seed=seed)
    rng = env.unwrapped.np_random

    for step_i in range(40):
        action = int(rng.integers(env.action_space.n))
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step_i:02d} | a={action} | obs={obs} | r={reward}")

        if terminated or truncated:
            print("\n--- TERMINATED, RESETTING ---\n")
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
