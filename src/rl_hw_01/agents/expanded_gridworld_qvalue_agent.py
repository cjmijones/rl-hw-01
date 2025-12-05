import numpy as np
from dataclasses import dataclass
import gymnasium as gym


@dataclass
class QLConfig:
    gamma: float = 0.95
    alpha: float = 0.1
    epsilon: float = 0.1
    max_episode_steps: int = 200


class ExpandedGridWorldQLearningAgent:
    """
    Model-free tabular Q-Learning agent for GridWorldExpandedEnvCJ.
    """

    def __init__(self, env: gym.Env, cfg: QLConfig = QLConfig()):
        if hasattr(env, "unwrapped"):
            env = env.unwrapped
        self.env = env
        self.cfg = cfg

        self.width, self.height = env.width, env.height
        self.A = env.action_space.n

        # ----- State indexing (same as DP & MC) -----
        self.states = [(x, y) for x in range(self.width) for y in range(self.height)]
        self.S = len(self.states)

        self.to_idx = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state = {i: s for s, i in self.to_idx.items()}

        # ----- Q-table -----
        self.Q = np.zeros((self.width, self.height, self.A), dtype=np.float64)

        # Derived value + policy
        self.V = np.zeros(self.S, dtype=np.float64)
        self.pi = np.zeros(self.S, dtype=np.int64)

        # ----- Logs for convergence -----
        self.V_hist = []
        self.policy_change_hist = []
        self.episode_returns = []
        self.episode_lengths = []

        self._rng = np.random.default_rng()

    # ------------------------------------------------
    def _state_key(self, obs):
        return int(obs["agent"][0]), int(obs["agent"][1])

    def _epsilon_greedy(self, x, y):
        if self._rng.random() < self.cfg.epsilon:
            return int(self._rng.integers(self.A))
        return int(np.argmax(self.Q[x, y, :]))

    def _derive_V_pi_from_Q(self):
        V = np.zeros(self.S, dtype=np.float64)
        pi = np.zeros(self.S, dtype=np.int64)
        for i, (x, y) in enumerate(self.states):
            row = self.Q[x, y, :]
            V[i] = float(np.max(row))
            pi[i] = int(np.argmax(row))
        return V, pi

    # ------------------------------------------------
    def train(self, num_episodes=50_000, seed=0, log_every=1000):
        self._rng = np.random.default_rng(seed)

        for ep in range(1, num_episodes + 1):
            obs, _ = self.env.reset(seed=int(self._rng.integers(1_000_000)))
            done = False
            steps = 0
            ep_return = 0.0

            while not done and steps < self.cfg.max_episode_steps:
                x, y = self._state_key(obs)
                a = self._epsilon_greedy(x, y)

                obs2, r, terminated, truncated, _ = self.env.step(a)
                x2, y2 = self._state_key(obs2)

                # ----- Q-learning update -----
                td_target = r + self.cfg.gamma * np.max(self.Q[x2, y2, :])
                td_error = td_target - self.Q[x, y, a]

                self.Q[x, y, a] += self.cfg.alpha * td_error

                obs = obs2
                steps += 1
                ep_return += float(r)
                done = bool(terminated or truncated)

            # ----- Track convergence -----
            old_pi = self.pi.copy()
            self.V, self.pi = self._derive_V_pi_from_Q()

            policy_changes = int(np.sum(old_pi != self.pi))
            self.policy_change_hist.append(policy_changes)
            self.V_hist.append(self.V.copy())
            self.episode_returns.append(ep_return)
            self.episode_lengths.append(steps)

            if log_every and ep % log_every == 0:
                print(
                    f"[Q-Learning] Episode {ep:6d} | "
                    f"Return = {ep_return:8.2f} | "
                    f"Steps = {steps:3d} | "
                    f"Policy changes = {policy_changes}"
                )

        return self.V, self.pi

    # ------------------------------------------------
    def act(self, obs):
        x, y = self._state_key(obs)
        return int(np.argmax(self.Q[x, y, :]))

    def compute_Q_from_V(self):
        Q_flat = np.zeros((self.S, self.A), dtype=np.float64)
        for i, (x, y) in enumerate(self.states):
            Q_flat[i, :] = self.Q[x, y, :]
        return Q_flat
