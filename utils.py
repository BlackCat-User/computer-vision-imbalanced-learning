import numpy as np

groups_c10 = {"many": [0, 1, 2], "medium": [3, 4, 5, 6], "few": [7, 8, 9]}

def make_lt_indices(targets, C=10, rho=100, n_max=5000, seed=0):
    rng = np.random.RandomState(seed)
    targets = np.array(targets)
    counts = [int(n_max * rho ** (-i / (C - 1))) for i in range(C)]
    idx = []
    for c, n in enumerate(counts):
        cls_idx = np.where(targets == c)[0]
        idx += rng.choice(cls_idx, n, replace=False).tolist()
    return sorted(idx), counts

def compute_metrics(preds, targets, groups):
    preds, targets = np.array(preds), np.array(targets)
    C = int(targets.max()) + 1
    per_class = [(preds[targets == c] == c).mean() for c in range(C)]
    out = {"overall": float((preds == targets).mean()),
           "per_class": [float(a) for a in per_class]}
    for name, cls in groups.items():
        out[name] = float(np.mean([per_class[c] for c in cls]))
    return out


def make_groups(class_counts, many_thr=100, few_thr=20):
    many   = [c for c, n in enumerate(class_counts) if n > many_thr]
    few    = [c for c, n in enumerate(class_counts) if n < few_thr]
    medium = [c for c, n in enumerate(class_counts) if few_thr <= n <= many_thr]
    return {"many": many, "medium": medium, "few": few}


counts_c100 = [int(500 * 100 ** (-i / 99)) for i in range(100)]
groups_c100 = make_groups(counts_c100)
