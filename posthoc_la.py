"""Post-hoc corrections of a trained ERM model (no training); tau is tuned on the balanced validation set.

  tau_norm_wide : Kang et al. tau-normalisation, w_k / ||w_k||^tau (no bias term)
  posthoc_la    : Menon et al. logit adjustment, argmax_k f_k(x) - tau * log(pi_k)

Tau grid follows Menon et al. (Fig. 3, same axis for both methods): 0..4 for CIFAR-10, 0..2 for CIFAR-100, step 0.1.
Test curves are stored for analysis only (Menon's "best possible tau" = oracle); tau is never selected on them.
"""
import glob, json, os, re
import numpy as np
import torch
import decoupled
from utils import compute_metrics, groups_c10


def grid_for(num_classes):
    top = 4.0 if num_classes <= 10 else 2.0
    return [round(t, 1) for t in np.arange(0, top + 1e-9, 0.1)]


def bal_acc(preds, y):
    preds, y = np.asarray(preds), np.asarray(y)
    return float(np.mean([(preds[y == c] == c).mean() for c in np.unique(y)]))


def pick_tau(taus, val_acc):
    """Best validation accuracy; on ties take the tied tau nearest the middle of the tied values."""
    val_acc = np.asarray(val_acc)
    ties = np.asarray(taus)[val_acc >= val_acc.max() - 1e-9]
    return float(ties[np.argmin(np.abs(ties - np.median(ties)))])


def _save(name, cfg, counts, preds, targets, extra):
    res = {"config": cfg, "counts": [int(c) for c in counts],
           **compute_metrics(preds, targets, groups_c10), **extra}
    path = f"{decoupled.BASE}/results/{name}.json"
    with open(path + ".tmp", "w") as f:
        json.dump(res, f)
    os.replace(path + ".tmp", path)
    print(name, "| overall:", round(res["overall"], 4), "many/med/few:",
          round(res["many"], 3), round(res["medium"], 3), round(res["few"], 3), "| tau:", extra.get("tau"))


def _load(seed, imbalance):
    model = decoupled.load_model(seed, imbalance)
    _, val_loader, test_loader, counts, _ = decoupled.get_data(imbalance)
    fv, yv = decoupled.extract(model, val_loader)
    ft, yt = decoupled.extract(model, test_loader)
    return model, fv, yv.numpy(), ft, yt.numpy(), counts


def _search(predict, zv, yv, zt, yt, taus):
    val = [bal_acc(predict(zv, t), yv) for t in taus]
    test = [bal_acc(predict(zt, t), yt) for t in taus]
    best = pick_tau(taus, val)
    if best == taus[-1]:
        print(f"  WARNING: selected tau {best} is at the edge of the grid")
    return best, {"tau": best, "val_curve": {str(t): a for t, a in zip(taus, val)},
                  "test_curve": {str(t): a for t, a in zip(taus, test)}}


def tau_norm_wide(seed, imbalance=100, taus=None):
    name = f"cifar10_taunorm_rho{imbalance}_seed{seed}"
    if os.path.exists(f"{decoupled.BASE}/results/{name}.json"):
        print("already done:", name)
        return
    decoupled.set_seed(seed)
    model, fv, yv, ft, yt, counts = _load(seed, imbalance)
    taus = list(taus) if taus is not None else grid_for(len(counts))
    W = model.fc.weight.detach().cpu()
    norms = W.norm(dim=1, keepdim=True)
    predict = lambda f, t: (f @ (W / norms ** t).T).argmax(1).numpy()
    best, extra = _search(predict, fv, yv, ft, yt, taus)
    extra["w_norms"] = norms.squeeze(1).tolist()
    cfg = {"dataset": "cifar10", "imbalance": imbalance, "seed": seed, "loss": "taunorm",
           "tau": best, "split": "val"}
    _save(name, cfg, counts, predict(ft, best), yt, extra)


def posthoc_la(seed, imbalance=100, taus=None):
    names = {"tuned": f"cifar10_posthoc_la_rho{imbalance}_seed{seed}",
             "tau1": f"cifar10_posthoc_la1_rho{imbalance}_seed{seed}"}
    if all(os.path.exists(f"{decoupled.BASE}/results/{n}.json") for n in names.values()):
        print("already done: posthoc LA seed", seed)
        return
    model, fv, yv, ft, yt, counts = _load(seed, imbalance)
    taus = list(taus) if taus is not None else grid_for(len(counts))
    W = model.fc.weight.detach().cpu()
    b = model.fc.bias.detach().cpu() if model.fc.bias is not None else torch.zeros(W.shape[0])
    zv, zt = (fv @ W.T + b).numpy(), (ft @ W.T + b).numpy()      # ERM logits, bias included
    log_prior = np.log(np.asarray(counts) / np.sum(counts))
    predict = lambda z, t: (z - t * log_prior).argmax(1)
    best, extra = _search(predict, zv, yv, zt, yt, taus)
    for key, tau in [("tuned", best), ("tau1", 1.0)]:
        cfg = {"dataset": "cifar10", "imbalance": imbalance, "seed": seed,
               "loss": "posthoc_la" if key == "tuned" else "posthoc_la1", "tau": tau, "split": "val"}
        _save(names[key], cfg, counts, predict(zt, tau), yt, {**extra, "tau": tau})


def _results(method, imbalance=100):
    out = {}
    for f in sorted(glob.glob(f"{decoupled.BASE}/results/cifar10_{method}_rho{imbalance}_seed*.json")):
        m = re.search(r"seed(\d+)\.json$", f)
        if m:
            out[int(m.group(1))] = json.load(open(f))
    return out


def _sd(v):
    v = np.asarray(v)
    return float(v.std(ddof=1)) if len(v) > 1 else 0.0


def summarize(imbalance=100):
    """Test accuracy per method: tau chosen on validation, tau=1, and the oracle (best tau on test, as in Menon's table)."""
    for method, label in [("taunorm", "tau-norm"), ("posthoc_la1", "post-hoc LA, tau=1"),
                          ("posthoc_la", "post-hoc LA, tau on val")]:
        rs = _results(method, imbalance)
        if not rs:
            continue
        acc = np.array([r["overall"] for r in rs.values()]) * 100
        line = f"{label:26s} val-selected tau {[r.get('tau') for r in rs.values()]} -> test {acc.mean():.2f} +- {_sd(acc):.2f}"
        if method != "posthoc_la1" and all("test_curve" in r for r in rs.values()):
            t1 = np.array([r["test_curve"]["1.0"] for r in rs.values()]) * 100
            orc = np.array([max(r["test_curve"].values()) for r in rs.values()]) * 100
            line += f" | tau=1: {t1.mean():.2f} | ORACLE (best tau on test): {orc.mean():.2f} +- {_sd(orc):.2f}"
        print(line)


def fig_tau_compare(imbalance=100):
    """Test accuracy vs tau for the two post-hoc methods (cf. Menon et al., Fig. 3)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for method, label in [("taunorm", "weight normalisation (tau-norm)"), ("posthoc_la", "logit adjustment")]:
        rs = _results(method, imbalance)
        if not rs or not all("test_curve" in r for r in rs.values()):
            continue
        keys = list(next(iter(rs.values()))["test_curve"])
        t = [float(k) for k in keys]
        a = np.array([[r["test_curve"][k] for k in keys] for r in rs.values()]) * 100
        line, = ax.plot(t, a.mean(0), label=f"{label} (n={len(rs)})")
        ax.fill_between(t, a.mean(0) - a.std(0), a.mean(0) + a.std(0), color=line.get_color(), alpha=0.2)
        for r in rs.values():
            ax.scatter([r["tau"]], [r["test_curve"][str(r["tau"])] * 100], color=line.get_color(), s=18, zorder=3)
    ax.set_xlabel("tau"); ax.set_ylabel("test accuracy (%)")
    ax.set_title("Post-hoc correction of ERM vs tau (dots: tau chosen on validation)")
    ax.legend(); fig.tight_layout()
    out = f"{decoupled.BASE}/results/figures"
    os.makedirs(out, exist_ok=True)
    fig.savefig(f"{out}/tau_compare.png", dpi=150)
    plt.show()
