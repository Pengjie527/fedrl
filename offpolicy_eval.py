import numpy as np
import pandas as pd
from typing import Tuple


def _build_behavior_policy(df: pd.DataFrame, n_states: int, n_actions: int) -> np.ndarray:
    """Empirical behaviour policy π_b(a|s) from dataframe."""
    counts = np.zeros((n_states, n_actions), dtype=np.float64)
    for s, a in zip(df['state'].astype(int), df['action'].astype(int)):
        if s < n_states and a < n_actions:
            counts[s, a] += 1
    # add small smooth to avoid zero
    probs = counts + 1e-6
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def wis_bootstrap(df: pd.DataFrame, policy: np.ndarray, n_states: int, n_actions: int,
                  gamma: float = 0.99, n_boot: int = 200, seed: int = 42) -> Tuple[float, float, float]:
    """Weighted importance sampling with bootstrap CI.
       Returns (mean, CI_low, CI_high) at 95% level.
    """
    rng = np.random.default_rng(seed)
    beh_pi = _build_behavior_policy(df, n_states, n_actions)

    # group trajectories by patient
    trajs = []
    for _, g in df.sort_values(['icustayid', 'bloc']).groupby('icustayid'):
        states  = g['state'].astype(int).to_numpy()
        actions = g['action'].astype(int).to_numpy()
        rewards = g['reward'].to_numpy()
        trajs.append((states, actions, rewards))

    def wis_estimate(sample_idxs, w_max=50.0, eps=0.01):
        """Per-decision self-normalised IS with weight clipping."""
        numer = 0.0
        denom = 0.0
        for idx in sample_idxs:
            s, a, r = trajs[idx]
            w = 1.0
            for t in range(len(r)):
                st = s[t]
                at = a[t]
                rho = (1.0 / beh_pi[st, at]) if (policy[st] == at) else eps
                w *= rho
                # weight clipping
                if w > w_max:
                    w = w_max
                if w == 0.0:
                    continue
                numer += w * (gamma ** t) * r[t]
                denom += w
        if denom == 0:
            return 0.0
        return numer / denom

    n_traj = len(trajs)
    estimates = []
    for _ in range(n_boot):
        sample_idx = rng.integers(0, n_traj, size=n_traj)
        estimates.append(wis_estimate(sample_idx))
    estimates = np.array(estimates)
    mean = estimates.mean()
    lb, ub = np.quantile(estimates, [0.05, 0.95])
    return mean, lb, ub 