from collections import defaultdict
import gymnasium as gym
import numpy as np


class SimpleGridWorldQValueAgent:
    """
    Visualization-compatible Q-Learning agent for Simple / Expanded GridWorld.

    Fully supports:
      - State value printing
      - Action value printing
      - Arrow policy visualization
      - Convergence plots
      - GIF rollout recording
      - DP/MC/QL shared interface
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        # ✅ Always unwrap for visualization compatibility
        if hasattr(env, "unwrapped"):
            env = env.unwrapped
        self.env = env

        # ---- Grid structure (REQUIRED by visualizers) ----
        self.width, self.height = env.width, env.height
        self.A = env.action_space.n

        self.states = [(x, y) for x in range(self.width) for y in range(self.height)]
        self.S = len(self.states)

        self.to_idx = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state = {i: s for s, i in self.to_idx.items()}

        # ---- Core learning hyperparameters ----
        self.lr = learning_rate
        self.discount_factor = discount_factor

        # ---- Exploration ----
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # ---- Dictionary-based Q-values (learning backend) ----
        self.q_values = defaultdict(lambda: np.zeros(self.A))

        # ---- Tensor-based Q-values (visualization backend) ----
        self.Q = np.zeros((self.width, self.height, self.A), dtype=np.float64)

        # ---- Derived V(s) and π(s) ----
        self.V = np.zeros(self.S, dtype=np.float64)
        self.pi = np.zeros(self.S, dtype=np.int64)

        # ---- Convergence Logs (REQUIRED BY VIS TOOLS) ----
        self.Q_hist = []
        self.V_hist = []
        self.pi_hist = []
        self.delta_hist = []
        self.policy_change_hist = []
        self.episode_returns = []
        self.episode_lengths = []

        # ---- Training error (extra diagnostic) ----
        self.training_error = []

    # ---------------------------------------------------------
    # State Key Utility
    # ---------------------------------------------------------
    def _action_key(self, obs):
        return int(obs["agent"][0]), int(obs["agent"][1])

    # ---------------------------------------------------------
    # Epsilon-Greedy Policy
    # ---------------------------------------------------------
    def get_action(self, obs):
        state = self._action_key(obs)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[state]))

    # ---------------------------------------------------------
    # Greedy Action (Used by rollout / visualization)
    # ---------------------------------------------------------
    def act(self, obs):
        x, y = self._action_key(obs)
        return int(np.argmax(self.Q[x, y, :]))

    # ---------------------------------------------------------
    # Bellman Q Update
    # ---------------------------------------------------------
    def update(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs,
    ):
        s = self._action_key(obs)
        s2 = self._action_key(next_obs)

        future_q_value = (not terminated) * np.max(self.q_values[s2])
        target = float(reward) + self.discount_factor * future_q_value

        td_error = target - self.q_values[s][action]
        self.q_values[s][action] += self.lr * td_error

        self.training_error.append(td_error)

        # ✅ Sync into the visualization tensor
        x, y = s
        self.Q[x, y, action] = self.q_values[s][action]

    # ---------------------------------------------------------
    # Derive V(s) and π(s)
    # ---------------------------------------------------------
    def _derive_V_pi_from_Q(self):
        for i, (x, y) in enumerate(self.states):
            row = self.Q[x, y, :]
            self.V[i] = float(np.max(row))
            self.pi[i] = int(np.argmax(row))

        return self.V, self.pi

    # ---------------------------------------------------------
    # Epsilon Decay
    # ---------------------------------------------------------
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    # ---------------------------------------------------------
    # Training Loop (Optional Helper)
    # ---------------------------------------------------------
    def train(self, num_episodes=50_000, max_steps=200, seed=0, log_every=1000):
        rng = np.random.default_rng(seed)

        for ep in range(1, num_episodes + 1):
            obs, _ = self.env.reset(seed=int(rng.integers(1_000_000)))
            done = False
            steps = 0
            ep_return = 0.0

            while not done and steps < max_steps:
                action = self.get_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)

                self.update(obs, action, reward, terminated, next_obs)

                obs = next_obs
                steps += 1
                ep_return += float(reward)
                done = bool(terminated or truncated)

            # ---- Track convergence ----
            old_V = self.V.copy()
            old_pi = self.pi.copy()

            # self._derive_V_pi_from_Q()
            self.V, self.pi = self._derive_V_pi_from_Q()

            self.delta_hist.append(np.max(np.abs(self.V - old_V)))
            self.policy_change_hist.append(np.sum(old_pi != self.pi))

            self.Q_hist.append(self.Q.copy())
            self.V_hist.append(self.V.copy())
            self.pi_hist.append(self.pi.copy())

            self.episode_returns.append(ep_return)
            self.episode_lengths.append(steps)

            self.decay_epsilon()

            if log_every and ep % log_every == 0:
                print(
                    f"[QL] Episode {ep:6d} | "
                    f"Return = {ep_return:8.2f} | "
                    f"Steps = {steps:3d} | "
                    f"Policy changes = {self.policy_change_hist[-1]}"
                )

        return self.V, self.pi

    # ---------------------------------------------------------
    # DP Compatibility Helper
    # ---------------------------------------------------------
    def compute_Q_from_V(self):
        Q_flat = np.zeros((self.S, self.A), dtype=np.float64)
        for i, (x, y) in enumerate(self.states):
            Q_flat[i, :] = self.Q[x, y, :]
        return Q_flat
