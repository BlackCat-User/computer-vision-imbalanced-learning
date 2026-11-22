import glob, json, os, re, sys
import numpy as np, matplotlib.pyplot as plt

BASE = "/content/drive/MyDrive/Colab Notebooks/cv-longtail"
sys.path.insert(0, BASE)
OUT = f"{BASE}/results/figures"
LABEL = {"ce": "ERM", "taunorm": "tau-norm", "crt": "cRT", "balanced_softmax": "Balanced softmax",
         "posthoc_la": "Post-hoc LA (tau on val)", "posthoc_la1": "Post-hoc LA (tau=1)",
         "cos_ce": "Cosine + CE", "cos_ldam": "Cosine + LDAM", "cos_ldam_drw": "Cosine + LDAM-DRW"}
ORDER = list(LABEL)
KEYS = ["overall", "many", "medium", "few"]


def lab(m): return LABEL.get(m, m)
def sd(v): v = np.asarray(v); return float(v.std(ddof=1)) if len(v) > 1 else 0.0


def load():
    runs = {}
    for f in sorted(glob.glob(f"{BASE}/results/cifar100_*_rho100_seed*.json")):
        m = re.match(r"cifar100_(.+)_rho100_seed(\d+)\.json$", os.path.basename(f))
        if m: runs.setdefault(m.group(1), {})[int(m.group(2))] = json.load(open(f))
    return dict(sorted(runs.items(), key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else 99))


def _save(name):
    os.makedirs(OUT, exist_ok=True)
    plt.tight_layout(); plt.savefig(f"{OUT}/c100_{name}.png", dpi=150); plt.show()


# 1. validation curve vs tau (like your first figure), one panel per tuned method
def fig_val_tau(runs):
    ms = [m for m in ["taunorm", "posthoc_la"] if m in runs]
    if not ms: return
    fig, ax = plt.subplots(1, len(ms), figsize=(6 * len(ms), 4), squeeze=False)
    for a, m in zip(ax[0], ms):
        for s, r in sorted(runs[m].items()):
            t = [float(k) for k in r["val_curve"]]; v = [x * 100 for x in r["val_curve"].values()]
            l, = a.plot(t, v, label=f"seed {s}")
            k = min(r["val_curve"], key=lambda k: abs(float(k) - r["tau"]))
            a.scatter([r["tau"]], [r["val_curve"][k] * 100], color=l.get_color(), zorder=3)
        a.set_xlabel("tau"); a.set_ylabel("val balanced acc (%)"); a.legend()
        a.set_title(f"{lab(m)}: validation (990 images, 99 classes) vs tau, dots = chosen")
    _save("val_tau")


# 2. test accuracy vs tau (like your second figure), recomputed from the ERM backbones
def tau_sweep(seeds=(0, 1, 2)):
    cache = f"{OUT}/c100_tau_sweep.json"
    if os.path.exists(cache): return json.load(open(cache))
    import torch
    from decoupled100 import load_model, get_data, extract
    from losses import log_prior
    _, _, tl, counts, _ = get_data(); lp = log_prior(counts)
    taus = [round(t, 2) for t in np.arange(0, 4.01, 0.1)]
    res = {"taus": taus, "norm": [], "la": []}
    for s in seeds:
        model = load_model(s); ft, yt = extract(model, tl)
        W, b = model.fc.weight.detach().cpu(), model.fc.bias.detach().cpu()
        n = W.norm(dim=1, keepdim=True)
        res["norm"].append([((ft @ (W / n ** t).T).argmax(1) == yt).float().mean().item() * 100 for t in taus])
        res["la"].append([((ft @ W.T + b - t * lp).argmax(1) == yt).float().mean().item() * 100 for t in taus])
    os.makedirs(OUT, exist_ok=True); json.dump(res, open(cache, "w")); return res


def fig_test_tau(runs):
    sw = tau_sweep(); t = sw["taus"]; plt.figure(figsize=(7.5, 4.5))
    for key, m, name in [("norm", "taunorm", "weight normalisation (tau-norm)"),
                         ("la", "posthoc_la", "logit adjustment")]:
        a = np.array(sw[key]); mu, s_ = a.mean(0), a.std(0)
        l, = plt.plot(t, mu, label=f"{name} (n={len(a)})")
        plt.fill_between(t, mu - s_, mu + s_, color=l.get_color(), alpha=0.2)
        if m in runs:
            plt.scatter([r["tau"] for r in runs[m].values()], [r["overall"] * 100 for r in runs[m].values()],
                        color=l.get_color(), s=18, zorder=3)
    plt.xlabel("tau"); plt.ylabel("test accuracy (%)"); plt.legend()
    plt.title("CIFAR-100-LT post-hoc correction of ERM vs tau (dots: tau chosen on validation)")
    _save("test_tau")


# 3. accuracy by class-frequency bin (classes 0-9 = most frequent ... 90-99 = rarest)
def _bins(rs): return np.array([r["per_class"] for r in rs.values()]).reshape(len(rs), 10, 10).mean(2) * 100


def fig_binned(runs):
    plt.figure(figsize=(9, 4.5)); x = np.arange(10)
    for m, rs in runs.items():
        b = _bins(rs); plt.errorbar(x, b.mean(0), yerr=[sd(b[:, j]) for j in range(10)],
                                    marker="o", capsize=2, label=lab(m))
    plt.xticks(x, [f"{10*j}-{10*j+9}" for j in range(10)]); plt.xlabel("class index (head -> tail)")
    plt.ylabel("test accuracy (%)"); plt.title("Accuracy by class-frequency bin (mean ± std over seeds)")
    plt.legend(fontsize=8); _save("by_bin")
    if "ce" in runs:
        plt.figure(figsize=(9, 4.5)); base = _bins(runs["ce"]).mean(0)
        for m, rs in runs.items():
            if m != "ce": plt.plot(x, _bins(rs).mean(0) - base, marker="o", label=lab(m))
        plt.axhline(0, color="k", lw=0.8); plt.xticks(x, [f"{10*j}-{10*j+9}" for j in range(10)])
        plt.xlabel("class index (head -> tail)"); plt.ylabel("gain over ERM (points)")
        plt.title("Where each method gains or loses against ERM"); plt.legend(fontsize=8); _save("gain_by_bin")


# 4. head/tail trade-off: many vs few accuracy
def fig_tradeoff(runs):
    plt.figure(figsize=(6.5, 5))
    for m, rs in runs.items():
        a = np.array([[r["many"], r["few"]] for r in rs.values()]) * 100
        l = plt.scatter(a[:, 0], a[:, 1], s=12, alpha=0.4)
        plt.errorbar(a[:, 0].mean(), a[:, 1].mean(), xerr=sd(a[:, 0]), yerr=sd(a[:, 1]), fmt="o",
                     color=l.get_facecolor()[0][:3], capsize=3)
        plt.annotate(lab(m), (a[:, 0].mean(), a[:, 1].mean()), fontsize=7, xytext=(4, 4), textcoords="offset points")
    plt.xlabel("many-shot accuracy (%)"); plt.ylabel("few-shot accuracy (%)")
    plt.title("Head vs tail trade-off (small dots = seeds, big dot = mean)"); _save("tradeoff")


# 5. ranking by overall accuracy
def fig_ranking(runs):
    rows = sorted(((np.mean([r["overall"] for r in rs.values()]) * 100, m) for m, rs in runs.items()))
    plt.figure(figsize=(7, 0.45 * len(rows) + 1.5))
    for i, (mu, m) in enumerate(rows):
        v = [r["overall"] * 100 for r in runs[m].values()]
        plt.scatter(v, [i] * len(v), c="k", s=12, zorder=3); plt.errorbar(mu, i, xerr=sd(v), fmt="o", capsize=3)
    if "ce" in runs: plt.axvline(np.mean([r["overall"] for r in runs["ce"].values()]) * 100, ls="--", c="gray", label="ERM")
    plt.yticks(range(len(rows)), [lab(m) for _, m in rows]); plt.xlabel("overall test accuracy (%)")
    plt.title("CIFAR-100-LT overall accuracy (dots = seeds)"); plt.legend(); _save("ranking")


# 6. learning curves from the per-epoch logs
def fig_curves():
    logs = {}
    for f in sorted(glob.glob(f"{BASE}/logs/cifar100_*_rho100_seed*.json")):
        m = re.match(r"cifar100_(.+)_rho100_seed\d+\.json$", os.path.basename(f))
        if m: logs.setdefault(m.group(1), []).append(json.load(open(f)))
    if not logs: print("no cifar100 logs"); return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for m, hs in logs.items():
        n = min(len(h) for h in hs); loss = np.array([[r["loss"] for r in h[:n]] for h in hs])
        ax[0].plot(range(n), loss.mean(0), label=f"{lab(m)} (n={len(hs)})")
        pts = {}
        for h in hs:
            for r in h:
                if "test_acc" in r: pts.setdefault(r["epoch"], []).append(r["test_acc"] * 100)
        e = sorted(pts); mu = np.array([np.mean(pts[k]) for k in e]); s_ = np.array([np.std(pts[k]) for k in e])
        l, = ax[1].plot(e, mu, marker="o", ms=3, label=lab(m)); ax[1].fill_between(e, mu - s_, mu + s_, color=l.get_color(), alpha=0.2)
    ax[0].set_yscale("log"); ax[0].set_title("train loss"); ax[1].set_title("test accuracy (logged, never used for selection)")
    for a in ax: a.set_xlabel("epoch"); a.legend(fontsize=8)
    _save("curves")


# 7. classifier weight norms vs class frequency
def fig_norms(runs):
    sets = {}
    if "taunorm" in runs: sets["ERM"] = np.array([r["w_norms"] for r in runs["taunorm"].values()])
    if "crt" in runs: sets["cRT"] = np.array([r["w_norms"] for r in runs["crt"].values()])
    if not sets: return
    counts = next(iter(runs.get("ce", runs[next(iter(runs))]).values()))["counts"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for n, a in sets.items():
        ax.plot(range(100), a.mean(0), label=n); ax.fill_between(range(100), a.mean(0) - a.std(0), a.mean(0) + a.std(0), alpha=0.2)
    ax.set_xlabel("class index (head -> tail)"); ax.set_ylabel("||w_k||")
    ax2 = ax.twinx(); ax2.plot(range(100), counts, "k:"); ax2.set_yscale("log"); ax2.set_ylabel("train images (log)")
    ax.set_title("Classifier weight norms vs class frequency"); ax.legend(); _save("norms")


def make_all():
    runs = load(); print({lab(m): sorted(r) for m, r in runs.items()})
    for fn, args in [(fig_val_tau, (runs,)), (fig_test_tau, (runs,)), (fig_binned, (runs,)), (fig_tradeoff, (runs,)),
                     (fig_ranking, (runs,)), (fig_curves, ()), (fig_norms, (runs,))]:
        try: fn(*args)
        except Exception as e: print(f"{fn.__name__} failed: {type(e).__name__}: {e}")
