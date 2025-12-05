import numpy as np
from dataclasses import dataclass
import gymnasium as gym


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
@dataclass
class MCConfig:
    gamma: float = 0.95
    epsilon: float = 0.1
    first_visit: bool = True
    max_episode_steps: int = 300


# ---------------------------------------------------------
# MONTE CARLO CONTROL AGENT
# ---------------------------------------------------------
class ExpandedGridWorldMCAgent:
    def __init__(self, env: gym.Env, cfg: MCConfig = MCConfig()):
        if hasattr(env, "unwrapped"):
            env = env.unwrapped
        self.env = env
        self.cfg = cfg

        self.width, self.height = env.width, env.height
        self.A = env.action_space.n

        # -----------------------------
        # State indexing (DP compatible)
        # -----------------------------
        self.states = [(x, y) for x in range(self.width) for y in range(self.height)]
        self.S = len(self.states)
        self.to_idx = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state = {i: s for s, i in self.to_idx.items()}

        # -----------------------------
        # Action-Value table Q(s,a)
        # -----------------------------
        self.Q = np.zeros((self.width, self.height, self.A), dtype=np.float64)
        self.N = np.zeros_like(self.Q, dtype=np.int64)

        # Derived state-value + policy
        self.V = np.zeros(self.S, dtype=np.float64)
        self.pi = np.zeros(self.S, dtype=np.int64)

        # Training logs
        self.episode_returns = []
        self.episode_lengths = []
        self.Q_hist = []
        self.V_hist = []
        self.pi_hist = []
        self.policy_change_hist = []

        self._rng = np.random.default_rng()

    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------
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
            V[i] = float(np.max(self.Q[x, y, :]))
            pi[i] = int(np.argmax(self.Q[x, y, :]))
        return V, pi

    # ---------------------------------------------------------
    # Generate One Monte Carlo Episode
    # ---------------------------------------------------------
    def _generate_episode(self, seed=None):
        obs, info = self.env.reset(seed=seed)

        path = []  # list of (x, y, a, r)
        steps = 0
        done = False

        while not done and steps < self.cfg.max_episode_steps:
            x, y = self._state_key(obs)

            a = self._epsilon_greedy(x, y)
            obs2, r, terminated, truncated, _ = self.env.step(a)

            path.append((x, y, a, float(r)))

            obs = obs2
            steps += 1
            done = bool(terminated or truncated)

        return path

    # ---------------------------------------------------------
    # Monte Carlo Return Computation
    # ---------------------------------------------------------
    def _compute_returns(self, path):
        G = 0.0
        returns = []

        for x, y, a, r in reversed(path):
            G = self.cfg.gamma * G + r
            returns.append((x, y, a, G))

        returns.reverse()
        return returns

    # ---------------------------------------------------------
    # TRAINING LOOP (MC CONTROL)
    # ---------------------------------------------------------
    def train(self, num_episodes=20_000, seed=0, log_every=1000):
        self._rng = np.random.default_rng(seed)

        for ep in range(1, num_episodes + 1):
            env_seed = int(self._rng.integers(0, 1_000_000))
            path = self._generate_episode(seed=env_seed)
            returns = self._compute_returns(path)

            seen = set()

            # -----------------------------
            # First-visit or Every-visit MC
            # -----------------------------
            for x, y, a, G in returns:
                key = (x, y, a)

                if self.cfg.first_visit and key in seen:
                    continue
                seen.add(key)

                self.N[x, y, a] += 1
                n = self.N[x, y, a]

                # Incremental mean update
                self.Q[x, y, a] += (G - self.Q[x, y, a]) / n

            # Episode stats
            ep_return = sum(r for *_, r in path)
            self.episode_returns.append(ep_return)
            self.episode_lengths.append(len(path))

            # Track derived policy
            old_pi = self.pi.copy()
            self.V, self.pi = self._derive_V_pi_from_Q()
            self.policy_change_hist.append(np.sum(old_pi != self.pi))

            if ep % log_every == 0:
                print(
                    f"[MC] Episode {ep:6d} | "
                    f"Return={ep_return:8.1f} | "
                    f"Length={len(path):4d} | "
                    f"Policy changes={self.policy_change_hist[-1]}"
                )

            self.Q_hist.append(self.Q.copy())
            self.V_hist.append(self.V.copy())
            self.pi_hist.append(self.pi.copy())

        return self.episode_returns, self.episode_lengths

    # ---------------------------------------------------------
    # Greedy evaluation policy after training
    # ---------------------------------------------------------
    def act(self, obs):
        x, y = self._state_key(obs)
        return int(np.argmax(self.Q[x, y, :]))

    # ---------------------------------------------------------
    # To match your DP plotting API
    # ---------------------------------------------------------
    def compute_Q_from_V(self):
        Q_flat = np.zeros((self.S, self.A), dtype=np.float64)
        for i, (x, y) in enumerate(self.states):
            Q_flat[i, :] = self.Q[x, y, :]
        return Q_flat
