import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm


def visualize_value_and_policy(agent, env_width, env_height, cmap_name="viridis"):
    import numpy as np
    import matplotlib.pyplot as plt

    # Build [row(y), col(x)] arrays directly (no transpose later)
    V_img = np.full((env_height, env_width), np.nan, dtype=float)
    A_img = np.full((env_height, env_width), -1, dtype=int)

    for (x, y), idx in agent.to_idx.items():
        V_img[y, x] = agent.V[idx]
        A_img[y, x] = agent.pi[idx]

    vmin, vmax = np.nanmin(V_img), np.nanmax(V_img)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)

    fig, ax = plt.subplots(figsize=(5 + env_width, 3 + env_height))
    im = ax.imshow(V_img, origin="lower", cmap=cmap, norm=norm)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Value (V*)", rotation=270, labelpad=15)
    ax.set_title("State Value Function V(s) and Greedy Policy π*")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Overlay policy arrows at the same coordinates used by imshow
    for y in range(env_height):
        for x in range(env_width):
            a = A_img[y, x]
            arrow = "→" if a == 0 else "←" if a == 1 else "·"
            ax.text(
                x,
                y,
                arrow,
                ha="center",
                va="center",
                fontsize=36,
                fontweight="bold",
                color="white",
                alpha=0.9,
            )

            v_val = V_img[y, x]
            rgb = cmap(norm(v_val))[:3]
            brightness = np.dot(rgb, [0.299, 0.587, 0.114])
            text_color = "black" if brightness > 0.5 else "white"
            ax.text(
                x,
                y - 0.32,
                f"{v_val:.2f}",
                ha="center",
                va="center",
                fontsize=14,
                color=text_color,
                fontweight="semibold",
            )

    ax.set_xticks(np.arange(env_width))
    ax.set_yticks(np.arange(env_height))
    ax.set_xticklabels(np.arange(env_width))
    ax.set_yticklabels(np.arange(env_height))
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.show()


def build_transition_table(agent, P_hat, R_hat, N, round_to=3):
    """Build readable transition table for reporting."""
    states = [(x, y) for x in range(agent.env.width) for y in range(agent.env.height)]
    to_idx = {s: i for i, s in enumerate(states)}
    rows = []
    for si, s in enumerate(states):
        for a in range(agent.A):
            for sj, s2 in enumerate(states):
                p = P_hat[si, a, sj]
                n = N[si, a, sj]
                r = R_hat[si, a, sj]
                if p > 0 or n > 0:
                    rows.append(
                        {
                            "s": s,
                            "a": "RIGHT" if a == 0 else "LEFT",
                            "s'": s2,
                            "P̂(s'|s,a)": round(p, round_to),
                            "R̂(s,a,s')": round(r, round_to),
                            "N": int(n),
                        }
                    )
    df = pd.DataFrame(rows)
    return df.sort_values(["s", "a", "s'"]).reset_index(drop=True)


def summarize_transition_table(df):
    """Aggregate summary per (s,a) pair."""
    summary = (
        df.groupby(["s", "a"])
        .agg(
            next_states=("s'", "count"),
            avg_prob=("P̂(s'|s,a)", "mean"),
            avg_reward=("R̂(s,a,s')", "mean"),
            total_visits=("N", "sum"),
        )
        .reset_index()
    )
    return summary


def compare_models(agent_known, agent_learned, P_hat):
    """Quick difference between analytic and learned transition models."""
    P_true = agent_known.P
    diffs = np.abs(P_hat - P_true)
    df_compare = pd.DataFrame(
        {
            "state": range(P_true.shape[0]),
            "mean |ΔP|": diffs.mean(axis=(1, 2)).round(3),
            "max |ΔP|": diffs.max(axis=(1, 2)).round(3),
        }
    )
    return df_compare


def show_V_table(agent, width, height, title="V*(s)"):
    V_grid = agent.V.reshape(height, width)
    print(f"\n=== {title} ===")
    print(np.round(V_grid, 3))


def show_Q_table(agent, title="Q*(s,a)"):
    Q = agent.compute_Q_from_V()
    print(f"\n=== {title} ===")
    for s in range(agent.S):
        q_row = ", ".join([f"a{a}:{Q[s, a]:.3f}" for a in range(agent.A)])
        print(f"s={s} -> {q_row}")


def _Q_from_V_snapshot(P, R, gamma, V_vec):
    return np.sum(P * (R + gamma * V_vec[np.newaxis, np.newaxis, :]), axis=2)


def plot_convergence(agent, title_suffix=""):
    """
    Plots:
      (A) V(s) convergence: max |V_k - V_{k-1}|
      (B) Q(s,a) convergence: max |Q_k - Q_{k-1}| inferred from V snapshots
      (C) Policy convergence: # of states changed at each improvement
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # --- (A) V convergence ---
    if len(agent.delta_hist) > 0:
        axes[0].plot(agent.delta_hist, marker="o")
        axes[0].set_yscale("log")  # values shrink quickly
        axes[0].set_title(f"V(s) max Δ per sweep {title_suffix}")
        axes[0].set_xlabel("Evaluation sweep")
        axes[0].set_ylabel("max |V_k − V_{k-1}|")
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "No V history", ha="center")

    # --- (B) Q convergence from V snapshots ---
    q_deltas = []
    if len(agent.V_hist) >= 2:
        Q_prev = _Q_from_V_snapshot(agent.P, agent.R, agent.cfg.gamma, agent.V_hist[0])
        for Vk in agent.V_hist[1:]:
            Q_now = _Q_from_V_snapshot(agent.P, agent.R, agent.cfg.gamma, Vk)
            q_deltas.append(float(np.max(np.abs(Q_now - Q_prev))))
            Q_prev = Q_now
        axes[1].plot(q_deltas, marker="o")
        axes[1].set_yscale("log")
        axes[1].set_title(f"Q(s,a) max Δ per sweep {title_suffix}")
        axes[1].set_xlabel("Evaluation sweep")
        axes[1].set_ylabel("max |Q_k − Q_{k-1}|")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "Need ≥2 V snapshots", ha="center")

    # --- (C) Policy convergence (#states changed) ---
    if len(agent.policy_change_hist) > 0:
        axes[2].plot(agent.policy_change_hist, marker="o")
        axes[2].set_title(f"Policy changes per improvement {title_suffix}")
        axes[2].set_xlabel("Improvement iteration")
        axes[2].set_ylabel("# states changed")
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "No policy-change history", ha="center")

    plt.tight_layout()
    plt.show()


def plot_Q_bars(agent, width, height, action_labels=None, title="Final Q*(s,a)"):
    if action_labels is None:
        action_labels = (
            {0: "RIGHT", 1: "LEFT"}
            if agent.A == 2
            else {a: f"a{a}" for a in range(agent.A)}
        )
    Q = agent.compute_Q_from_V()
    fig_rows, fig_cols = height, width
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(4 * fig_cols, 3.2 * fig_rows))
    axes = np.atleast_2d(axes)
    for y in range(height):
        for x in range(width):
            s = agent.to_idx[(x, y)]
            ax = axes[y, x]
            xs = list(range(agent.A))
            ax.bar(xs, Q[s, :])
            ax.set_xticks(xs)
            ax.set_xticklabels([action_labels[a] for a in xs])
            ax.set_title(f"State (x={x}, y={y})")
            ax.set_ylabel("Q*")
            ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()


###########################################################
######## EXPANDED GRIDWORLD VISUALIZATIONS #################
###########################################################


def visualize_expanded_gridworld_dp(
    agent, env, title="Optimal Policy and Value Function"
):
    """
    Fully synchronized visualization with GridWorldExpandedEnvCJ.
    Pulls walls, terminals, and start state DIRECTLY from env.
    Uses TOP-LEFT origin to match pygame rendering.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    width, height = env.width, env.height

    if hasattr(env, "walls"):
        walls = set(env.walls)

    if hasattr(env, "terminal_rewards"):
        terminals = dict(env.terminal_rewards)

    if hasattr(env, "start_state"):
        start_state = env.start_state

    # -------------------------
    # Build value + policy arrays
    # -------------------------
    V_img = np.full((height, width), np.nan)
    A_img = np.full((height, width), -1)

    for (x, y), si in agent.to_idx.items():
        if hasattr(env, "walls"):
            if (x, y) in walls:
                continue
        V_img[y, x] = agent.V[si]
        A_img[y, x] = agent.pi[si]

    # Overwrite terminal rewards
    if hasattr(env, "terminal_rewards"):
        for (x, y), r in terminals.items():
            V_img[y, x] = r
            A_img[y, x] = -1

    # Mask walls
    if hasattr(env, "walls"):
        for x, y in walls:
            V_img[y, x] = np.nan
            A_img[y, x] = -1

    # -------------------------
    # Plot heatmap (TOP-LEFT)
    # -------------------------
    fig, ax = plt.subplots(figsize=(10, 9))

    cmap = cm.get_cmap("viridis").copy()
    cmap.set_bad(color="lightgray")

    norm = Normalize(vmin=np.nanmin(V_img), vmax=np.nanmax(V_img))

    im = ax.imshow(V_img, origin="upper", cmap=cmap, norm=norm)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("State Value V*(s)")

    arrow_map = {
        0: "→",  # RIGHT
        1: "←",  # LEFT
        2: "↑",  # UP
        3: "↓",  # DOWN
    }

    # -------------------------
    # Draw overlays
    # -------------------------
    for y in range(height):
        for x in range(width):
            # WALL
            if hasattr(env, "walls"):
                if (x, y) in walls:
                    ax.text(
                        x, y, "■", ha="center", va="center", fontsize=22, color="black"
                    )
                    continue

            # TERMINAL
            if hasattr(env, "terminal_rewards"):
                if (x, y) in terminals:
                    r = terminals[(x, y)]
                    color = "green" if r > 0 else "red"
                    ax.text(
                        x,
                        y,
                        f"{int(r)}",
                        ha="center",
                        va="center",
                        fontsize=14,
                        fontweight="bold",
                        color=color,
                    )
                    continue

            # START
            if hasattr(env, "start_state"):
                if (x, y) == start_state:
                    ax.text(
                        x,
                        y + 0.1,
                        "S",
                        ha="center",
                        va="center",
                        fontsize=14,
                        fontweight="bold",
                        color="white",
                    )

            # POLICY
            a = A_img[y, x]
            if a >= 0:
                ax.text(
                    x,
                    y - 0.25,
                    arrow_map[a],
                    ha="center",
                    va="center",
                    fontsize=22,
                    fontweight="bold",
                    color="white",
                )

            # VALUE
            v = V_img[y, x]
            if not np.isnan(v):
                ax.text(
                    x,
                    y + 0.25,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="black",
                )

    # -------------------------
    # Axes formatting
    # -------------------------
    ax.set_xticks(np.arange(width))
    ax.set_yticks(np.arange(height))
    ax.set_xticklabels(np.arange(width))
    ax.set_yticklabels(np.arange(height))

    ax.set_xlabel("j")
    ax.set_ylabel("i")
    ax.set_title(title)

    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def print_transition_model_summary(agent, max_examples: int = 1, round_to: int = 3):
    """
    Pretty-print a *human* readable approximation of the learned transition model
    P̂(s'|s,a) and R̂(s,a,s') using only transitions that were actually observed
    (N > 0). Shows at most `max_examples` (s,a) pairs.
    """

    P_hat = agent.P
    R_hat = agent.R
    N = getattr(agent, "N_counts", None)

    if N is None:
        print(
            "⚠️ No N_counts stored on agent. "
            "Call estimate_model(...) after adding `self.N_counts = N`."
        )
        return

    print("\n" + "=" * 80)
    print("APPROXIMATED TRANSITION MODEL P̂(s'|s,a) AND R̂(s,a,s')")
    print(f"Shape P̂: {P_hat.shape}, Shape R̂: {R_hat.shape}")
    print("Only nonzero-count transitions (N > 0) are shown.")
    print("=" * 80)

    shown = 0
    action_names = ["RIGHT", "LEFT", "UP", "DOWN"]

    for si in range(agent.S):
        s_coord = agent.idx_to_state[si]  # (j, i)
        for a in range(agent.A):
            counts_row = N[si, a, :]
            if counts_row.sum() == 0:
                continue  # never saw this (s,a)

            print(
                f"\nState index s={si}, coord (j={s_coord[0]}, i={s_coord[1]}), "
                f"Action a={a} ({action_names[a]})"
            )
            print("-" * 80)
            s_prime_idx = "s'_idx"
            p_hat = "P̂(s'|s,a)"
            r_hat = "R̂(s,a,s')"
            print(
                f"{s_prime_idx:>6} | {'(j,i)':>8} | {p_hat:>10} | {r_hat:>11} | {'N':>5}"
            )
            print("-" * 80)

            # show only next states with non-zero count
            for sj in range(agent.S):
                n = counts_row[sj]
                if n == 0:
                    continue
                p = P_hat[si, a, sj]
                r = R_hat[si, a, sj]
                j2, i2 = agent.idx_to_state[sj]
                print(
                    f"{sj:6d} | ({j2:1d},{i2:1d}) | {p:10.{round_to}f} | {r:11.{round_to}f} | {int(n):5d}"
                )

            shown += 1
            if shown >= max_examples:
                print("\n[...] (truncated; increase max_examples to see more)")
                print("=" * 80)
                return

    print("\n(no (s,a) pairs with N > 0 were found — did estimate_model() run?)")
    print("=" * 80)


def print_state_value_function(agent, env, title="Optimal State-Value Function V*(s)"):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    width, height = env.width, env.height

    V_grid = [["" for _ in range(width)] for _ in range(height)]

    # Fill values
    for (x, y), si in agent.to_idx.items():
        V_grid[y][x] = f"{agent.V[si]:7.2f}"

    # Override terminals (if env defines them)
    if hasattr(env, "terminal_rewards"):
        for (x, y), r in env.terminal_rewards.items():
            V_grid[y][x] = f"{r:7.0f}"

    # Override walls
    if hasattr(env, "walls"):
        for x, y in env.walls:
            V_grid[y][x] = "   WALL "

    # Print with correct top-left origin
    for i in range(height):
        row = " | ".join(V_grid[i])
        print(f"i={i} | {row}")

    print("=" * 70)


def print_action_value_function(
    agent, env, title="Optimal Action-Value Function Q*(s,a)"
):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    Q = agent.compute_Q_from_V()

    action_names = ["RIGHT", "LEFT", "UP", "DOWN"]

    for i in range(env.height):
        for j in range(env.width):
            if hasattr(env, "walls"):
                if (j, i) in env.walls:
                    print(f"State (j={j}, i={i}) → WALL")
                    continue

            if hasattr(env, "terminal_rewards"):
                if (j, i) in env.terminal_rewards:
                    r = env.terminal_rewards[(j, i)]
                    print(f"State (j={j}, i={i}) → TERMINAL, Reward = {r}")
                    continue

            si = agent.to_idx[(j, i)]
            q_vals = Q[si]

            q_str = ", ".join(
                f"{action_names[a]}: {q_vals[a]:7.2f}" for a in range(len(q_vals))
            )

            print(f"State (j={j}, i={i}) → {q_str}")

        print("-" * 70)


def get_moving_avgs(arr, window=100, mode="valid"):
    """Compute moving average to smooth noisy data."""
    arr = np.asarray(arr, dtype=float)
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode=mode)


def plot_training_curves(
    episode_rewards,
    episode_lengths,
    agent,
    rolling_length=100,
):
    """
    Visualize learning curves for the GridWorld agent.

    Args:
        episode_rewards: list of total reward per episode
        episode_lengths: list of episode lengths
        agent: your SimpleGridWorldAgent (uses agent.training_error)
        rolling_length: smoothing window (episodes)
    """
    fig, axs = plt.subplots(ncols=3, figsize=(14, 4))

    # --- 1. Episode Rewards ---
    axs[0].set_title("Episode Rewards")
    rewards_smooth = get_moving_avgs(episode_rewards, rolling_length, "valid")
    axs[0].plot(rewards_smooth, color="tab:blue")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel(f"Reward (smoothed over {rolling_length})")

    # --- 2. Episode Lengths ---
    axs[1].set_title("Episode Lengths")
    lengths_smooth = get_moving_avgs(episode_lengths, rolling_length, "valid")
    axs[1].plot(lengths_smooth, color="tab:green")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel(f"Length (smoothed over {rolling_length})")

    # --- 3. Training Error (TD Error) ---
    axs[2].set_title("Temporal-Difference (TD) Error")
    td_error_smooth = get_moving_avgs(agent.training_error, rolling_length, "same")
    axs[2].plot(td_error_smooth, color="tab:orange")
    axs[2].set_xlabel("Step")
    axs[2].set_ylabel("TD Error (smoothed)")

    plt.tight_layout()
    plt.show()


def plot_mc_convergence(agent, title_suffix=""):
    """
    Monte Carlo convergence plots:
      (A) Episode return
      (B) Episode length
      (C) Policy changes per episode
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # -------------------------------------------------
    # (A) Episode Return
    # -------------------------------------------------
    if hasattr(agent, "episode_returns") and len(agent.episode_returns) > 0:
        axes[0].plot(agent.episode_returns, alpha=0.8)
        axes[0].set_title(f"Episode Return {title_suffix}")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Return")
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "No return history", ha="center")

    # -------------------------------------------------
    # (B) Episode Length
    # -------------------------------------------------
    if hasattr(agent, "episode_lengths") and len(agent.episode_lengths) > 0:
        axes[1].plot(agent.episode_lengths, alpha=0.8)
        axes[1].set_title(f"Episode Length {title_suffix}")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Steps")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No episode length history", ha="center")

    # -------------------------------------------------
    # (C) Policy Change Convergence
    # -------------------------------------------------
    if hasattr(agent, "policy_change_hist") and len(agent.policy_change_hist) > 0:
        axes[2].plot(agent.policy_change_hist, alpha=0.8)
        axes[2].set_title(f"Policy Changes {title_suffix}")
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("# States Changed")
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "No policy-change history", ha="center")

    plt.tight_layout()
    plt.show()
