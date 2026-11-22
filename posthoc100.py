import os, sys
import numpy as np, torch

BASE = "/content/drive/MyDrive/Colab Notebooks/cv-longtail"
sys.path.insert(0, BASE)
from decoupled100 import load_model, get_data, extract, bal_acc, save_result
from losses import log_prior


def posthoc(seed):
    n_val, n_one = f"cifar100_posthoc_la_rho100_seed{seed}", f"cifar100_posthoc_la1_rho100_seed{seed}"
    if all(os.path.exists(f"{BASE}/results/{n}.json") for n in (n_val, n_one)):
        print("already done: seed", seed); return
    model = load_model(seed)
    _, vl, tl, counts, _ = get_data()
    fv, yv = extract(model, vl); ft, yt = extract(model, tl)
    W, b = model.fc.weight.detach().cpu(), model.fc.bias.detach().cpu()
    lp = log_prior(counts)
    logits = lambda f, tau: f @ W.T + b - tau * lp          # same form as losses.posthoc_adjust

    taus = [round(t, 2) for t in np.arange(0, 3.01, 0.1)]
    va = [bal_acc(logits(fv, t).argmax(1), yv) for t in taus]
    top = [t for t, a in zip(taus, va) if a >= max(va) - 1e-9]; best = top[len(top) // 2]
    print(f"seed {seed} val acc per tau:", {t: round(a, 3) for t, a in zip(taus, va)}, "-> tau", best)

    for name, tau, tag in [(n_val, best, "posthoc_la"), (n_one, 1.0, "posthoc_la1")]:
        cfg = {"dataset": "cifar100", "imbalance": 100, "seed": seed, "loss": tag, "tau": tau}
        save_result(name, cfg, counts, logits(ft, tau).argmax(1), yt,
                    {"tau": tau, "val_curve": {str(t): a for t, a in zip(taus, va)}})
