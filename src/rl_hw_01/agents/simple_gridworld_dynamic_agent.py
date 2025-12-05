import numpy as np
from dataclasses import dataclass


@dataclass
class PIConfig:
    gamma: float = 0.95
    theta: float = 1e-6
    max_eval_iters: int = 10_000


class SimpleGridWorldDynamicAgent:
    def __init__(self, env, config: PIConfig = PIConfig(), model_source: str = "known"):
        if hasattr(env, "unwrapped"):
            env = env.unwrapped
        self.env = env
        self.cfg = config

        # State/action indexing
        self.states = [(x, y) for x in range(env.width) for y in range(env.height)]
        self.S = len(self.states)
        self.A = env.action_space.n
        self.to_idx = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state = {i: s for s, i in self.to_idx.items()}

        # Motion primitives (must match env)
        self._a_dir = {0: np.array([1, 0], dtype=int), 1: np.array([-1, 0], dtype=int)}
        self._w_dir = {0: np.array([-1, 0], dtype=int), 1: np.array([1, 0], dtype=int)}
        self.wind_probs = {
            0: 0.3,
            1: 0.7,
        }  # known wind model (used if model_source="known")

        # Model tensors (filled below)
        self.P = np.zeros((self.S, self.A, self.S), dtype=np.float64)
        self.R = np.zeros_like(self.P)

        # Value/policy + logs
        self.V = np.zeros(self.S, dtype=np.float64)
        self.pi = np.zeros(self.S, dtype=np.int64)
        self.V_hist, self.delta_hist, self.policy_change_hist = [], [], []
        self.pi_hist = []

        if model_source not in {"known", "learned"}:
            raise ValueError("model_source must be 'known' or 'learned'")
        self.model_source = model_source
        if self.model_source == "known":
            self._build_known_model()

    # --------- PUBLIC API EXTENSIONS ----------
    def estimate_model(
        self,
        episodes: int = 5000,
        max_steps: int = 50,
        *,
        behavior: str = "random",  # "random" | "epsilon_greedy" | "custom"
        epsilon: float = 0.2,  # used when behavior == "epsilon_greedy"
        policy_fn=None,  # used when behavior == "custom": fn(state_tuple)->action
        seed: int = 0,
        smooth_dirichlet: float = 0.0,  # >0 to add simple additive smoothing to counts
        verbose: bool = False,  # NEW ARG: Print training progress if True
        print_every: int = 500,  # NEW ARG: Print every this many episodes if verbose
    ):
        """Roll out to learn P̂,R̂ (on-policy/off-policy depending on behavior)."""
        rng = np.random.default_rng(seed)

        N = np.zeros((self.S, self.A, self.S), dtype=np.int64)
        RS = np.zeros((self.S, self.A, self.S), dtype=np.float64)

        def _eps_greedy_action(s_tuple):
            si = self.to_idx[s_tuple]
            if rng.random() < epsilon:
                return int(rng.integers(self.A))
            # greedy w.r.t current policy/value: compute Q from current (P,R,V)
            q = np.sum(
                self.P[si, :, :] * (self.R[si, :, :] + self.cfg.gamma * self.V), axis=1
            )
            return int(np.argmax(q))

        def _behavior_action(s_tuple):
            if behavior == "random":
                return int(rng.integers(self.A))
            elif behavior == "epsilon_greedy":
                return _eps_greedy_action(s_tuple)
            elif behavior == "custom":
                if policy_fn is None:
                    raise ValueError(
                        "Provide policy_fn(state)->action for behavior='custom'."
                    )
                return int(policy_fn(s_tuple))
            else:
                raise ValueError(
                    "behavior must be 'random' | 'epsilon_greedy' | 'custom'"
                )

        for ep in range(episodes):
            obs, _ = self.env.reset(seed=int(rng.integers(1_000_000)))
            for t in range(max_steps):
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

            if verbose and (ep + 1) % print_every == 0:
                total_transitions = N.sum()
                nonzero = N[N > 0]
                mean_counts = float(nonzero.mean()) if nonzero.size else 0.0
                max_counts = int(nonzero.max()) if nonzero.size else 0

                print(f"\n[estimate_model] Episode {ep + 1}/{episodes}")
                print(f"  Total transitions: {int(total_transitions):,}")
                print(
                    f"  Mean count (nonzero cells): {mean_counts:.2f} | Max count: {max_counts}"
                )
                print(
                    f"  Coverage: {np.count_nonzero(N)}/{N.size} "
                    f"(≈ {100 * np.count_nonzero(N) / N.size:.1f}%)"
                )

                # Current empirical model snapshot
                with np.errstate(divide="ignore", invalid="ignore"):
                    P_hat = N / np.maximum(N.sum(axis=2, keepdims=True), 1)
                    R_hat_tmp = np.where(N > 0, RS / np.maximum(N, 1), 0.0)

                P_hat = np.nan_to_num(P_hat, nan=0.0)
                R_hat_tmp = np.nan_to_num(R_hat_tmp, nan=0.0)

                print("  P̂(s'|s,a) (rounded):")
                print(np.round(P_hat, 2))

                # Preview greedy policy using the *current* empirical model
                Q_preview = np.sum(
                    P_hat
                    * (R_hat_tmp + self.cfg.gamma * self.V[np.newaxis, np.newaxis, :]),
                    axis=2,
                )
                pi_preview = np.argmax(Q_preview, axis=1)
                print("  Greedy π* preview (w.r.t. current P̂,R̂ and V):", pi_preview)

                # Optional: quick value preview (few sweeps) using P̂,R̂ (doesn't touch self.V)
                V_prev = self.V.copy()
                for _ in range(100):
                    V_prev_new = np.empty_like(V_prev)
                    for s in range(self.S):
                        # one-step greedy improvement for preview
                        q_sa = np.sum(
                            P_hat[s, :, :]
                            * (R_hat_tmp[s, :, :] + self.cfg.gamma * V_prev),
                            axis=1,
                        )
                        a = int(np.argmax(q_sa))
                        V_prev_new[s] = np.sum(
                            P_hat[s, a, :]
                            * (R_hat_tmp[s, a, :] + self.cfg.gamma * V_prev)
                        )
                    V_prev = V_prev_new

                V_table_preview = V_prev.reshape((self.env.height, self.env.width))
                print("  V(s) preview (5 sweeps):")
                print(np.round(V_table_preview, 2))

        if verbose:
            print("[estimate_model] Finished training. Building model...")

        if smooth_dirichlet > 0.0:
            N = N + smooth_dirichlet  # simple additive smoothing

        with np.errstate(divide="ignore", invalid="ignore"):
            P_hat = N / np.maximum(N.sum(axis=2, keepdims=True), 1)
            R_hat = np.where(N > 0, RS / np.maximum(N, 1), 0.0)

        P_hat = np.nan_to_num(P_hat, nan=0.0)
        R_hat = np.nan_to_num(R_hat, nan=0.0)
        self.P, self.R = P_hat, R_hat
        self.model_source = "learned"

        if verbose:
            print("[estimate_model] Model building complete. Returning learned model.")

        return P_hat, R_hat, N

    def use_model(self, P: np.ndarray, R: np.ndarray):
        """Inject a model (e.g., from estimate_model or elsewhere)."""
        assert P.shape == (self.S, self.A, self.S) and R.shape == P.shape
        self.P, self.R = P.copy(), R.copy()
        self.model_source = "learned"

    # --------- CORE PLANNING (unchanged) ----------
    def fit(self):
        while True:
            self._policy_evaluation()
            if self._policy_improvement():
                break
        return self.V, self.pi

    def act(self, obs) -> int:
        ax, ay = int(obs["agent"][0]), int(obs["agent"][1])
        s = self.to_idx[(ax, ay)]
        return int(self.pi[s])

    def compute_Q_from_V(self):
        gamma = self.cfg.gamma
        return np.sum(
            self.P * (self.R + gamma * self.V[np.newaxis, np.newaxis, :]), axis=2
        )

    # --------- INTERNALS ----------
    def _build_known_model(self):
        """Exact P,R by enumerating the known wind model and env reward rules."""
        width, height = self.env.width, self.env.height

        def _transition_and_reward(s, a, w):
            (x, y) = s

            a_vec = self._a_dir[a]
            w_vec = self._w_dir[w]

            total_move = ((a_vec + w_vec) // 2).astype(int)
            px, py = x + total_move[0], y + total_move[1]

            nx = max(0, min(width - 1, px))
            ny = max(0, min(height - 1, py))

            s_next = (nx, ny)
            old_x, new_x = x, nx
            r = -1

            if (old_x == new_x) and (total_move[0] == -1):
                r = 0
            if (old_x == new_x) and (total_move[0] == 1):
                r = 4
            if (total_move[0] == 0) and (new_x == 0):
                r = 1
            if (total_move[0] == 0) and (new_x == 1):
                r = 3
            if (old_x != new_x) and (total_move[0] == -1):
                r = 2
            if (old_x != new_x) and (total_move[0] == 1):
                r = 2
            assert r >= 0
            return s_next, float(r)

        P = np.zeros_like(self.P)
        R = np.zeros_like(self.R)

        for si, s in enumerate(self.states):
            for a in range(self.A):
                prob_acc, rew_acc, mass = {}, {}, 0.0
                for w, pw in self.wind_probs.items():
                    s2, r = _transition_and_reward(s, a, w)
                    sj = self.to_idx[s2]
                    prob_acc[sj] = prob_acc.get(sj, 0.0) + pw
                    rew_acc[sj] = rew_acc.get(sj, 0.0) + pw * r
                    mass += pw
                assert abs(mass - 1.0) < 1e-9
                for sj, p in prob_acc.items():
                    P[si, a, sj] = p
                    R[si, a, sj] = rew_acc[sj] / p
        self.P, self.R = P, R

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

    def _policy_improvement(self) -> bool:
        gamma = self.cfg.gamma
        stable, changes = True, 0
        for s in range(self.S):
            old_a = self.pi[s]
            q_sa = np.sum(self.P[s, :, :] * (self.R[s, :, :] + gamma * self.V), axis=1)
            best_a = int(np.argmax(q_sa))
            self.pi[s] = best_a
            if best_a != old_a:
                stable, changes = False, changes + 1
        self.policy_change_hist.append(changes)
        self.pi_hist.append(self.pi.copy())
        return stable
