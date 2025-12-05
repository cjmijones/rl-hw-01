import numpy as np
from dataclasses import dataclass


@dataclass
class PIConfig:
    """Policy-iteration configuration."""

    gamma: float = 0.95  # discount factor (for DP)
    theta: float = 1e-6  # convergence tolerance for policy evaluation
    max_eval_iters: int = 10_000  # max sweeps in policy evaluation


class ExpandedGridWorldDynamicAgent:
    """
    Tabular DP agent (policy iteration) for the GridWorld env.

    Expected env API (matched by GridWorldImageEnvCJ):

      - env.width, env.height : int
      - env.action_space.n    : int (should be 4: RIGHT, LEFT, UP, DOWN)
      - env.walls             : set[(x,y)] of wall cells (agent bounces off)
      - env.terminal_rewards  : dict[(x,y)] -> reward (e.g. -50, +100)
      - env.gamma             : malfunction probability in [0,1]
      - obs from env.{reset,step} : {"agent": np.array([x,y], dtype=int)}
    """

    def __init__(self, env, config: PIConfig = PIConfig(), model_source: str = "known"):
        # Strip wrappers so we see attributes like walls, terminal_rewards, gamma
        if hasattr(env, "unwrapped"):
            env = env.unwrapped
        self.env = env
        self.cfg = config

        # ----- State / action indexing -----
        # Enumerate all (x,y) pairs in the grid; walls are included as states,
        # but the environment's dynamics never move the agent into them.
        self.states = [(x, y) for x in range(env.width) for y in range(env.height)]
        self.S = len(self.states)
        self.A = env.action_space.n

        self.to_idx = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state = {i: s for s, i in self.to_idx.items()}

        # Motion primitives (must match env)
        # 0 = RIGHT, 1 = LEFT, 2 = UP, 3 = DOWN
        self._a_dir = {
            0: np.array([1, 0], dtype=int),  # RIGHT
            1: np.array([-1, 0], dtype=int),  # LEFT
            2: np.array([0, -1], dtype=int),  # UP
            3: np.array([0, 1], dtype=int),  # DOWN
        }

        # Pre-compute sets for convenience
        self.walls = getattr(env, "walls", set())
        self.terminal_rewards = getattr(env, "terminal_rewards", {})
        self.malfunction_prob = float(getattr(env, "gamma", 0.0))  # γ from assignment

        # ----- Model tensors -----
        # P[s,a,s'] = P(s' | s,a)
        # R[s,a,s'] = expected immediate reward when s->s' under (s,a)
        self.P = np.zeros((self.S, self.A, self.S), dtype=np.float64)
        self.R = np.zeros_like(self.P)

        # ----- Value / policy + logs -----
        self.V = np.zeros(self.S, dtype=np.float64)
        # start with a random policy (recommended in assignment)
        rng = np.random.default_rng(0)
        self.pi = rng.integers(self.A, size=self.S, dtype=np.int64)

        self.V_hist = []  # trajectory of V during evaluation
        self.delta_hist = []  # max |V_k - V_{k-1}|
        self.policy_change_hist = []  # #states changed at each improvement
        self.pi_hist = []  # full policy snapshots

        if model_source not in {"known", "learned"}:
            raise ValueError("model_source must be 'known' or 'learned'")
        self.model_source = model_source
        if self.model_source == "known":
            self._build_known_model()

    # ====================================================================
    # Public API: model estimation from rollouts
    # ====================================================================
    def estimate_model(
        self,
        episodes: int = 5000,
        max_steps: int = 50,
        *,
        behavior: str = "random",  # "random" | "epsilon_greedy" | "custom"
        epsilon: float = 0.2,  # used when behavior == "epsilon_greedy"
        policy_fn=None,  # used when behavior == "custom": fn(state_tuple)->action
        seed: int = 0,
        smooth_dirichlet: float = 0.0,  # >0 => additive smoothing
        verbose: bool = False,
        print_every: int = 500,
    ):
        """
        Roll out env trajectories to estimate P̂,R̂.

        This is the "approximate DP" part of the assignment: we approximate
        p(s', r | s,a) using empirical counts from any behavior policy.
        """
        rng = np.random.default_rng(seed)

        N = np.zeros((self.S, self.A, self.S), dtype=np.int64)
        RS = np.zeros((self.S, self.A, self.S), dtype=np.float64)

        def _eps_greedy_action(s_tuple):
            si = self.to_idx[s_tuple]
            if rng.random() < epsilon:
                return int(rng.integers(self.A))
            # greedy w.r.t *current* model + V
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

                with np.errstate(divide="ignore", invalid="ignore"):
                    P_hat = N / np.maximum(N.sum(axis=2, keepdims=True), 1)
                    R_hat_tmp = np.where(N > 0, RS / np.maximum(N, 1), 0.0)

                P_hat = np.nan_to_num(P_hat, nan=0.0)
                R_hat_tmp = np.nan_to_num(R_hat_tmp, nan=0.0)

                print("  P̂(s'|s,a) (rounded):")
                print(np.round(P_hat, 2))

                Q_preview = np.sum(
                    P_hat
                    * (R_hat_tmp + self.cfg.gamma * self.V[np.newaxis, np.newaxis, :]),
                    axis=2,
                )
                pi_preview = np.argmax(Q_preview, axis=1)
                print("  Greedy π* preview:", pi_preview)

                # Quick preview of V using the empirical model (few sweeps)
                V_prev = self.V.copy()
                for _ in range(100):
                    V_prev_new = np.empty_like(V_prev)
                    for s_idx in range(self.S):
                        q_sa = np.sum(
                            P_hat[s_idx, :, :]
                            * (R_hat_tmp[s_idx, :, :] + self.cfg.gamma * V_prev),
                            axis=1,
                        )
                        a_best = int(np.argmax(q_sa))
                        V_prev_new[s_idx] = np.sum(
                            P_hat[s_idx, a_best, :]
                            * (R_hat_tmp[s_idx, a_best, :] + self.cfg.gamma * V_prev)
                        )
                    V_prev = V_prev_new

                V_table_preview = V_prev.reshape((self.env.height, self.env.width))
                print("  V(s) preview (few sweeps):")
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

        self.N_counts = N

        if verbose:
            print("[estimate_model] Model building complete. Returning learned model.")

        return P_hat, R_hat, N

    def use_model(self, P: np.ndarray, R: np.ndarray):
        """Inject an externally computed model (e.g., from estimate_model)."""
        assert P.shape == (self.S, self.A, self.S) and R.shape == P.shape
        self.P, self.R = P.copy(), R.copy()
        self.model_source = "learned"

    # ====================================================================
    # Core planning (policy iteration)
    # ====================================================================
    def fit(self):
        """
        Run policy-iteration until policy is stable.

        Returns:
            V, pi: optimal value function and policy under the current model.
        """
        while True:
            self._policy_evaluation()
            if self._policy_improvement():
                break
        return self.V, self.pi

    def act(self, obs) -> int:
        """Greedy action under the current policy π."""
        ax, ay = int(obs["agent"][0]), int(obs["agent"][1])
        s = self.to_idx[(ax, ay)]
        return int(self.pi[s])

    def compute_Q_from_V(self):
        """Compute Q(s,a) from the current V via the Bellman optimality backup."""
        gamma = self.cfg.gamma
        return np.sum(
            self.P * (self.R + gamma * self.V[np.newaxis, np.newaxis, :]), axis=2
        )

    # ====================================================================
    # Internal: build exact model from known GridWorld dynamics
    # ====================================================================
    def _build_known_model(self):
        """
        Construct exact transition model P,R for the GridWorld:

          - Malfunction probability γ = env.gamma
          - With prob (1-γ): environment executes intended action a
          - With prob γ: executes a random action (uniform over 4)
          - Walls: bounce (stay in place)
          - Rewards: -1 for white, R_terminal for terminal states, 0 from terminal self-loops
        """
        width, height = self.env.width, self.env.height
        gamma_mal = self.malfunction_prob

        P = np.zeros_like(self.P)
        R = np.zeros_like(self.R)

        # helper: deterministic one-step transition for a *real* action
        def _deterministic_step(s, a_real):
            x, y = s

            # Terminal states are absorbing with zero reward from here on
            if s in self.terminal_rewards:
                return s, 0.0, True

            dx, dy = self._a_dir[a_real]
            px, py = x + dx, y + dy

            # Clip to grid
            px = max(0, min(width - 1, px))
            py = max(0, min(height - 1, py))

            # Bounce off walls
            if (px, py) in self.walls:
                nx, ny = x, y
            else:
                nx, ny = px, py

            s_next = (nx, ny)

            if s_next in self.terminal_rewards:
                r = float(self.terminal_rewards[s_next])
                terminated = True
            else:
                r = -1.0
                terminated = False
            return s_next, r, terminated

        for si, s in enumerate(self.states):
            for a in range(self.A):
                # If s is terminal: absorbing, zero reward
                if s in self.terminal_rewards:
                    P[si, a, si] = 1.0
                    R[si, a, si] = 0.0
                    continue

                prob_acc = {}
                rew_acc = {}

                # Actual action distribution given chosen action 'a':
                #   With prob 1-γ: execute 'a'
                #   With prob γ: execute random action (uniform over 4)
                for a_real in range(self.A):
                    if a_real == a:
                        p_a_real = (1.0 - gamma_mal) + gamma_mal / self.A
                    else:
                        p_a_real = gamma_mal / self.A

                    if p_a_real == 0.0:
                        continue

                    s2, r, _ = _deterministic_step(s, a_real)
                    sj = self.to_idx[s2]

                    prob_acc[sj] = prob_acc.get(sj, 0.0) + p_a_real
                    rew_acc[sj] = rew_acc.get(sj, 0.0) + p_a_real * r

                # Normalize / assign
                mass = sum(prob_acc.values())
                if abs(mass - 1.0) > 1e-8:
                    # Numerical guard – should be exactly 1
                    assert abs(mass - 1.0) < 1e-6, f"Transition mass != 1 (got {mass})"

                for sj, p in prob_acc.items():
                    P[si, a, sj] = p
                    R[si, a, sj] = rew_acc[sj] / p

        self.P, self.R = P, R

    # ====================================================================
    # Internal: policy evaluation / improvement
    # ====================================================================
    def _policy_evaluation(self):
        """Policy evaluation for the current π until convergence."""
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
        """Greedy policy improvement step; returns True if policy is stable."""
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
