import numpy as np
from dataclasses import dataclass
import gymnasium as gym


@dataclass
class MCConfig:
    gamma: float = 0.95
    epsilon: float = 0.1
    first_visit: bool = True
    max_episode_steps: int = 200


class SimpleGridWorldMCAgent:
    def __init__(self, env: gym.Env, cfg: MCConfig = MCConfig()):
        if hasattr(env, "unwrapped"):
            env = env.unwrapped
        self.env = env
        self.cfg = cfg

        self.width, self.height = env.width, env.height
        self.A = env.action_space.n

        self.states = [(x, y) for x in range(self.width) for y in range(self.height)]
        self.S = len(self.states)
        self.to_idx = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state = {i: s for s, i in self.to_idx.items()}

        self.Q = np.zeros((self.width, self.height, self.A), dtype=np.float64)
        self.N = np.zeros_like(self.Q, dtype=np.int64)

        self.V = np.zeros(self.S, dtype=np.float64)
        self.pi = np.zeros(self.S, dtype=np.int64)

        # ✅ Visualization compatibility
        self.Q_hist = []
        self.V_hist = []
        self.pi_hist = []
        self.delta_hist = []  # ✅ REQUIRED BY plot_convergence
        self.policy_change_hist = []  # ✅ REQUIRED BY plot_convergence
        self.episode_returns = []
        self.episode_lengths = []

        self.epsilon = float(cfg.epsilon)
        self._rng = np.random.default_rng()

    def _state_key(self, obs):
        return int(obs["agent"][0]), int(obs["agent"][1])

    def _epsilon_greedy(self, x, y):
        if self._rng.random() < self.epsilon:
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

    def _generate_episode(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        path = []
        steps = 0
        done = False
        seen = set()

        while not done and steps < self.cfg.max_episode_steps:
            x, y = self._state_key(obs)
            a = self._epsilon_greedy(x, y)

            key = (x, y, a)
            obs2, r, terminated, truncated, info = self.env.step(a)

            steps += 1
            obs = obs2

            if self.cfg.first_visit and key in seen:
                continue

            seen.add(key)
            path.append((x, y, a, float(r)))
            done = bool(terminated or truncated)

        return path

    def _calc_trajectory_return(self, path):
        G = 0.0
        for t in reversed(range(len(path))):
            x, y, a, r = path[t]
            G = self.cfg.gamma * G + r
        return G

    def train(self, num_episodes=10_000, seed=100, log_every=100):
        self._rng = np.random.default_rng(seed)

        for ep in range(1, num_episodes + 1):
            env_seed = int(self._rng.integers(0, 1_000_000))
            path = self._generate_episode(seed=env_seed)

            if len(path) == 0:
                continue

            episode_return = sum(r for *_, r in path)
            episode_length = len(path)

            self.episode_returns.append(episode_return)
            self.episode_lengths.append(episode_length)

            x, y, a, _ = path[0]

            self.N[x, y, a] += 1
            old_q = self.Q[x, y, a]

            G = self._calc_trajectory_return(path)
            self.Q[x, y, a] += (G - self.Q[x, y, a]) / self.N[x, y, a]

            # ✅ Proper delta tracking for plot_convergence
            self.delta_hist.append(abs(self.Q[x, y, a] - old_q))

            # ✅ Proper policy tracking
            old_pi = self.pi.copy()
            self.V, self.pi = self._derive_V_pi_from_Q()

            self.policy_change_hist.append(np.sum(old_pi != self.pi))

            self.Q_hist.append(self.Q.copy())
            self.V_hist.append(self.V.copy())
            self.pi_hist.append(self.pi.copy())

            if ep % log_every == 0:
                print(f"[MC] Episode {ep}")

        return self.V, self.pi

    def act(self, obs):
        x, y = self._state_key(obs)
        return int(np.argmax(self.Q[x, y, :]))

    def compute_Q_from_V(self):
        Q_flat = np.zeros((self.S, self.A), dtype=np.float64)
        for i, (x, y) in enumerate(self.states):
            Q_flat[i, :] = self.Q[x, y, :]
        return Q_flat
