import numpy as np
from scipy.stats import entropy, spearmanr

def extract_features(bids, prev_bids=None):
    vals = np.array(list(bids.values()))

    spread = (vals.max() - vals.min()) / (vals.max() + 1e-9)
    cv = np.std(vals) / (np.mean(vals) + 1e-9)

    mean_val = np.mean(vals)
    cluster = np.mean(np.abs(vals - mean_val) < 0.05 * mean_val)

    floor = min(vals)
    floor_hug = np.mean(vals < floor * 1.1)

    hist, _ = np.histogram(vals, bins=5)
    ent = entropy(hist + 1e-9)

    rank_corr = 0
    if prev_bids is not None:
        prev_vals = np.array(list(prev_bids.values()))
        rank_corr = spearmanr(vals, prev_vals).correlation or 0

    return np.array([spread, cv, cluster, floor_hug, ent, rank_corr])