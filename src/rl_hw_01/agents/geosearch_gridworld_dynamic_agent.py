import numpy as np
from dataclasses import dataclass


# ============================================================
# Policy Iteration Config
# ============================================================
@dataclass
class PIConfig:
    gamma: float = 0.95
    theta: float = 1e-6
    max_eval_iters: int = 10_000


# ============================================================
# GeoSearch Dynamic Programming Agent (Analytic + Learned Model)
# ============================================================
class GeoSearchDynamicAgent:
    """
    Tabular DP Agent for GeoSearchEnvCJ (25x25 continuous reward field).

    Supports:
      ✅ Exact analytic DP from env transitions
      ✅ Learned model via rollouts + approximate DP

    Expected env API:
      - env.width, env.height
      - env.action_space.n = 4
      - env.R[x,y] reward surface
      - obs: {"agent": np.array([x,y])}
    """

    def __init__(self, env, config: PIConfig = PIConfig(), model_source: str = "known"):
        if hasattr(env, "unwrapped"):
            env = env.unwrapped
        self.env = env
        self.cfg = config

        # ----- State / Action space -----
        self.states = [(x, y) for x in range(env.width) for y in range(env.height)]
        self.S = len(self.states)
        self.A = env.action_space.n

        self.to_idx = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state = {i: s for s, i in self.to_idx.items()}

        # ----- Actions -----
        self._a_dir = {
            0: np.array([1, 0]),  # RIGHT
            1: np.array([-1, 0]),  # LEFT
            2: np.array([0, -1]),  # UP
            3: np.array([0, 1]),  # DOWN
        }

        # ----- Model tensors -----
        self.P = np.zeros((self.S, self.A, self.S), dtype=np.float64)
        self.R = np.zeros_like(self.P)

        # ----- Value + Policy -----
        self.V = np.zeros(self.S, dtype=np.float64)
        rng = np.random.default_rng(0)
        self.pi = rng.integers(self.A, size=self.S, dtype=np.int64)

        # ----- Logs -----
        self.V_hist = []
        self.delta_hist = []
        self.policy_change_hist = []
        self.pi_hist = []

        if model_source not in {"known", "learned"}:
            raise ValueError("model_source must be 'known' or 'learned'")
        self.model_source = model_source

        # ✅ Build analytic model only if requested
        if self.model_source == "known":
            self._build_known_model()

    # ============================================================
    # ✅ EXACT ANALYTIC GEOSEARCH TRANSITION MODEL
    # ============================================================
    def _build_known_model(self):
        width, height = self.env.width, self.env.height

        for si, s in enumerate(self.states):
            x, y = s
            for a in range(self.A):
                dx, dy = self._a_dir[a]
                px, py = x + dx, y + dy

                px = max(0, min(width - 1, px))
                py = max(0, min(height - 1, py))

                s_next = (px, py)
                sj = self.to_idx[s_next]

                self.P[si, a, sj] = 1.0
                self.R[si, a, sj] = float(self.env.R[px, py])

    # ============================================================
    # ✅ LEARNED MODEL VIA ROLLOUTS (APPROXIMATE DP)
    # ============================================================
    def estimate_model(
        self,
        episodes: int = 5_000,
        max_steps: int = 100,
        behavior: str = "random",
        epsilon: float = 0.2,
        policy_fn=None,
        seed: int = 0,
        verbose: bool = False,
    ):
        rng = np.random.default_rng(seed)

        N = np.zeros((self.S, self.A, self.S), dtype=np.int64)
        RS = np.zeros((self.S, self.A, self.S), dtype=np.float64)

        def _behavior_action(s):
            if behavior == "random":
                return int(rng.integers(self.A))
            elif behavior == "epsilon_greedy":
                if rng.random() < epsilon:
                    return int(rng.integers(self.A))
                return int(self.pi[self.to_idx[s]])
            elif behavior == "custom":
                return int(policy_fn(s))
            else:
                raise ValueError("Invalid behavior")

        for ep in range(episodes):
            obs, _ = self.env.reset(seed=int(rng.integers(1_000_000)))

            for _ in range(max_steps):
                s = (int(obs["agent"][0]), int(obs["agent"][1]))
                si = self.to_idx[s]
                a = _behavior_action(s)

                obs2, r, term, trunc, _ = self.env.step(a)
                s2 = (int(obs2["agent"][0]), int(obs2["agent"][1]))
                sj = self.to_idx[s2]

                N[si, a, sj] += 1
                RS[si, a, sj] += float(r)

                if term or trunc:
                    break
                obs = obs2

        with np.errstate(divide="ignore", invalid="ignore"):
            P_hat = N / np.maximum(N.sum(axis=2, keepdims=True), 1)
            R_hat = np.where(N > 0, RS / np.maximum(N, 1), 0.0)

        self.P = np.nan_to_num(P_hat)
        self.R = np.nan_to_num(R_hat)
        self.model_source = "learned"
        self.N_counts = N

        if verbose:
            print("✅ GeoSearch model learned from rollouts")

        return self.P, self.R, N

    # ============================================================
    # ✅ POLICY ITERATION CORE
    # ============================================================
    def fit(self):
        while True:
            self._policy_evaluation()
            if self._policy_improvement():
                break
        return self.V, self.pi

    def act(self, obs) -> int:
        x, y = int(obs["agent"][0]), int(obs["agent"][1])
        return int(self.pi[self.to_idx[(x, y)]])

    def compute_Q_from_V(self):
        gamma = self.cfg.gamma
        return np.sum(
            self.P * (self.R + gamma * self.V[np.newaxis, np.newaxis, :]), axis=2
        )

    # ============================================================
    # POLICY EVALUATION
    # ============================================================
    def _policy_evaluation(self):
        gamma, theta = self.cfg.gamma, self.cfg.theta

        while True:
            V_old = self.V.copy()

            for s in range(self.S):
                a = self.pi[s]
                self.V[s] = np.sum(self.P[s, a, :] * (self.R[s, a, :] + gamma * V_old))

            delta = float(np.max(np.abs(self.V - V_old)))
            self.V_hist.append(self.V.copy())
            self.delta_hist.append(delta)

            if delta < theta or len(self.V_hist) >= self.cfg.max_eval_iters:
                break

    # ============================================================
    # POLICY IMPROVEMENT
    # ============================================================
    def _policy_improvement(self) -> bool:
        gamma = self.cfg.gamma
        stable = True
        changes = 0

        for s in range(self.S):
            old_a = self.pi[s]
            q_sa = np.sum(self.P[s, :, :] * (self.R[s, :, :] + gamma * self.V), axis=1)
            best_a = int(np.argmax(q_sa))
            self.pi[s] = best_a

            if best_a != old_a:
                stable = False
                changes += 1

        self.policy_change_hist.append(changes)
        self.pi_hist.append(self.pi.copy())
        return stable
