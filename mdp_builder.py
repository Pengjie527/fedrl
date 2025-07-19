import numpy as np
import pandas as pd
from typing import Tuple

def build_mdp_from_df(df: pd.DataFrame, n_states: int, n_actions: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build transition probability tensor T[s,a,s'] and reward matrix R[s,a]
       from trajectory dataframe containing columns ['icustayid','bloc','state','action','reward'].
       Returns (T, R). Unseen (s,a) keep zeros; T rows are normalised.
    """
    # Initialise counts and reward sums
    T_counts = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    R_sums   = np.zeros((n_states, n_actions), dtype=np.float64)

    # Ensure correct ordering
    df_sorted = df.sort_values(['icustayid', 'bloc'])

    for _, group in df_sorted.groupby('icustayid'):
        states  = group['state'].to_numpy(dtype=int)
        actions = group['action'].to_numpy(dtype=int)
        rewards = group['reward'].to_numpy(dtype=float)
        for i in range(len(group)-1):
            s, a, s_next, r = states[i], actions[i], states[i+1], rewards[i]
            if s < n_states and a < n_actions and s_next < n_states:
                T_counts[s, a, s_next] += 1
                R_sums[s, a] += r
    # Compute probabilities and mean rewards
    sa_counts = T_counts.sum(axis=2)
    with np.errstate(divide='ignore', invalid='ignore'):
        T = np.divide(T_counts, sa_counts[:, :, None], where=sa_counts[:, :, None] > 0)
        R = np.divide(R_sums, sa_counts, where=sa_counts > 0)
    T[np.isnan(T)] = 0
    R[np.isnan(R)] = 0
    return T, R 