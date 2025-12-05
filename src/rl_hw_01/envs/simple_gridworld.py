from typing import Optional

import numpy as np
import pygame

import os

os.environ["SDL_AUDIODRIVER"] = "dummy"

import warnings

import gymnasium as gym
from gymnasium import spaces

from gymnasium.utils.env_checker import check_env

from gymnasium.envs.registration import register


def ignore_pygame_warnings():
    warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
    return


ignore_pygame_warnings()


class GridWorldEnvCJ(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "none"], "render_fps": 4}

    def __init__(
        self,
        render_mode=None,
        width: int = 7,
        height: int = 4,
        window_width: int = 560,
        window_height: int = 400,
        verbose: bool = True,
    ):
        self.width = int(width)
        self.height = int(height)
        self.window_width = int(window_width)
        self.window_height = int(window_height)

        # Initialize positions - will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._wind_direction = np.array([-1, -1], dtype=np.int32)

        self._verbose = verbose

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        # observation space bounds differ per axis now
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

        # Define what actions are available (2 directions)
        self.action_space = gym.spaces.Discrete(2)

        # Define the wind directions that are available (2 directions)
        self.wind_space = spaces.Dict({"east_wind": spaces.Discrete(2)})

        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: np.array([1, 0]),  # Move right (positive x)
            1: np.array([-1, 0]),  # Move left (negative x)
        }

        self._wind_to_boat_direction = {
            0: np.array(
                [-1, 0]
            ),  # False or no east wind moves the boat to the left (negative x)
            1: np.array(
                [1, 0]
            ),  # Positive or yes east wind moves the boat to the right (positive x)
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        return {"agent": self._agent_location}

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            empty information currently, placeholder for potential information dumps
        """
        return {"information": "No information method setup"}

    def _sample_wind_direction(self):
        return 1 if self.np_random.random() < 0.7 else 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Randomly place the agent anywhere on the grid
        x = self.np_random.integers(0, self.width, dtype=np.int32)
        y = self.np_random.integers(0, self.height, dtype=np.int32)

        self._agent_location = np.array([x, y], dtype=np.int32)
        self._wind_direction = self._sample_wind_direction()

        # Randomly place target, ensuring it's different from agent position - unneeded with no target
        # self._target_location = self._agent_location
        # while np.array_equal(self._target_location, self._agent_location):
        #     self._target_location = self.np_random.integers(
        #         0, self.size, size=2, dtype=int
        #     )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0 or 1 for right and left directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-3) to a movement direction
        action = int(action)
        assert self.action_space.contains(action), f"Invalid action: {action}"

        wind = int(self._sample_wind_direction())
        self._wind_direction = wind
        assert self.wind_space["east_wind"].contains(wind), (
            f"Invalid wind direction: {wind}"
        )

        direction = self._action_to_direction[action]
        wind_direction_array = self._wind_to_boat_direction[wind]

        old_location = self._agent_location.copy()

        # Update agent position, ensuring it stays within grid bounds
        # np.clip prevents the agent from walking off the edge
        # Compute proposed position

        total_move = ((direction + wind_direction_array) // 2).astype(int)

        proposed = self._agent_location + total_move

        # Clamp to grid bounds
        self._agent_location = np.array(
            [
                np.clip(proposed[0], 0, self.width - 1),
                np.clip(proposed[1], 0, self.height - 1),
            ],
            dtype=np.int32,
        )

        # Setup up reward structure for various transition possibilities, check with assert that at least one reward is claimed

        reward = -1

        if (old_location[0] == self._agent_location[0]) and (
            total_move[0] == -1
        ):  # Start in left side and move left with wind blowing left
            reward = 0
        if (old_location[0] == self._agent_location[0]) and (
            total_move[0] == 1
        ):  # Start in right side and move right with wind blowing right
            reward = 4

        if (
            (total_move[0] == 0) and (self._agent_location[0] == 0)
        ):  # Start in left side - move left with right wind or move right with left wind
            reward = 1
        if (
            (total_move[0] == 0) and (self._agent_location[0] == 1)
        ):  # Start in right side - move right with left wind or move left with right wind
            reward = 3

        if (old_location[0] != self._agent_location[0]) and (
            total_move[0] == -1
        ):  # Start in right side and move left
            reward = 2
        if (old_location[0] != self._agent_location[0]) and (
            total_move[0] == 1
        ):  # Start in left side and move right
            reward = 2

        assert reward >= 0, (
            "Invalid reward transition structure and setup of"
            + f" Wind direction array {wind_direction_array} with action {direction} total move {total_move} old location {old_location} and new location {self._agent_location}"
        )

        if self._verbose:
            print(
                f"Wind direction array {wind_direction_array} with action {direction} total move {total_move} old location {old_location} and new location {self._agent_location}"
            )

        # Check if agent reached the target
        terminated = False

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            # pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))

        # per-axis cell size (rectangular cells are fine)
        cell_w = self.window_width / self.width
        cell_h = self.window_height / self.height

        # draw agent (centered in its cell)
        ax = (self._agent_location[0] + 0.5) * cell_w
        ay = (self._agent_location[1] + 0.5) * cell_h
        radius = 0.35 * min(cell_w, cell_h)
        pygame.draw.circle(canvas, (0, 0, 255), (ax, ay), radius)

        # grid lines: verticals
        for x in range(self.width + 1):
            X = x * cell_w
            pygame.draw.line(canvas, 0, (X, 0), (X, self.window_height), width=2)
        # horizontals
        for y in range(self.height + 1):
            Y = y * cell_h
            pygame.draw.line(canvas, 0, (0, Y), (self.window_width, Y), width=2)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


def main(seed: int = 42, render_mode: str = "human"):
    # You can pass any custom env settings via kwargs

    register(
        id="gymnasium_env/GridWorldEnvCJ-v2",
        entry_point=GridWorldEnvCJ,
        max_episode_steps=100,
    )

    env = gym.make(
        "gymnasium_env/GridWorldEnvCJ-v2", render_mode=render_mode, width=2, height=1
    )  # pass kwargs here
    obs, info = env.reset(seed=seed)

    try:
        check_env(env)
        print("Environment passes all checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")

    obs, info = env.reset(seed=seed)
    rng = env.unwrapped.np_random

    for _ in range(15):
        action = int(rng.integers(env.action_space.n))
        obs, reward, terminated, truncated, info = env.step(action=action)
        if getattr(env, "_verbose", False):
            print(
                f"Step {_}: Action Taken: {action}, Observation: {obs}, Reward: {reward}"
            )
        if terminated or truncated:
            print("\033[91m" + "Terminated!" + "\033[0m\n")
            obs, info = env.reset()
    env.close()


if __name__ == "__main__":
    # Example: set your custom arguments here
    main()
