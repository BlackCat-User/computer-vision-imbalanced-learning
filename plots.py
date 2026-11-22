import glob, json, os, re
import numpy as np
import matplotlib.pyplot as plt

BASE = "/content/drive/MyDrive/cv-longtail"
OUT = f"{BASE}/results/figures"
CLASSES = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
LABEL = {"ce": "ERM", "reweighted_ce": "Reweighted CE", "focal": "Focal", "cb": "CB",
         "ldam": "LDAM-DRW (linear)", "la": "Logit adj.", "balanced_softmax": "Balanced softmax",
         "taunorm": "tau-norm", "crt": "cRT", "posthoc_la1": "Post-hoc LA (tau=1)",
         "posthoc_la": "Post-hoc LA (tau on val)", "cos_ce": "Cosine + CE",
         "cos_ldam": "Cosine + LDAM", "cos_ldam_drw": "LDAM-DRW (cosine)"}
ORDER = ["ce", "reweighted_ce", "focal", "cb", "ldam", "la", "balanced_softmax", "taunorm", "crt",
         "posthoc_la1", "posthoc_la", "cos_ce", "cos_ldam", "cos_ldam_drw"]
KEYS = ["overall", "many", "medium", "few"]
VAL_COUNTS_0 = 4990   # first class count of the 12,306-image training split


def lab(m): return LABEL.get(m, m)
def sd(v): v = np.asarray(v); return float(v.std(ddof=1)) if len(v) > 1 else 0.0


def on_val_split(r):
    """True if the run used the 12,306-image split (config says so, or the counts do)."""
    return r["config"].get("split") == "val" or r["counts"][0] == VAL_COUNTS_0


def load_runs(imb=100):
    """method -> {seed: result}; only untagged runs trained on the 12,306 split"""
    runs = {}
    for f in sorted(glob.glob(f"{BASE}/results/cifar10_*_rho{imb}_seed*.json")):
        m = re.match(rf"cifar10_(.+)_rho{imb}_seed(\d+)\.json$", os.path.basename(f))
        if not m: continue
        r = json.load(open(f))
        if not on_val_split(r): continue
        runs.setdefault(m.group(1), {})[int(m.group(2))] = r
    return dict(sorted(runs.items(), key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else 99))


def _save(name):
    os.makedirs(OUT, exist_ok=True)
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}.png", dpi=150); plt.show()


def fig_per_class(runs):
    ms = list(runs); w = 0.8 / len(ms); x = np.arange(10)
    plt.figure(figsize=(11, 4.5))
    for i, m in enumerate(ms):
        a = np.array([r["per_class"] for r in runs[m].values()]) * 100
        plt.bar(x + (i - (len(ms) - 1) / 2) * w, a.mean(0), w, yerr=[sd(a[:, c]) for c in range(10)],
                capsize=2, label=f"{lab(m)} (n={len(a)})")
    plt.xticks(x, CLASSES, rotation=45); plt.ylabel("test accuracy (%)")
    plt.title("Per-class accuracy, CIFAR-10-LT rho=100 (head -> tail), mean ± std over seeds")
    plt.legend(); _save("per_class")


def fig_groups(runs):
    ms = list(runs); w = 0.8 / len(ms); x = np.arange(len(KEYS))
    plt.figure(figsize=(8, 4.5)); rng = np.random.RandomState(0)
    for i, m in enumerate(ms):
        a = np.array([[r[k] for k in KEYS] for r in runs[m].values()]) * 100
        pos = x + (i - (len(ms) - 1) / 2) * w
        plt.bar(pos, a.mean(0), w, yerr=[sd(a[:, j]) for j in range(len(KEYS))], capsize=3,
                label=f"{lab(m)} (n={len(a)})", alpha=0.8)
        for j in range(len(KEYS)):
            plt.scatter(pos[j] + rng.uniform(-w / 4, w / 4, len(a)), a[:, j], c="k", s=10, zorder=3)
    plt.xticks(x, KEYS); plt.ylabel("test accuracy (%)")
    plt.title("Overall and many/medium/few (dots = individual seeds)"); plt.legend(); _save("groups")


def fig_curves(imb=100):
    logs = {}
    for f in sorted(glob.glob(f"{BASE}/logs/cifar10_*_rho{imb}_seed*.json")):
        m = re.match(rf"cifar10_(.+?)_rho{imb}_seed\d+", os.path.basename(f))
        logs.setdefault(m.group(1), []).append(json.load(open(f)))
    logs = {m: [h for h in hs if len(h) > 1] for m, hs in logs.items()}
    logs = {m: hs for m, hs in logs.items() if hs}
    if not logs: print("no logs yet: curves need runs started after the run.py patch"); return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for m, hs in logs.items():
        n = min(len(h) for h in hs); ep = np.arange(n)
        loss = np.array([[r["loss"] for r in h[:n]] for h in hs])
        tra = np.array([[r["train_acc"] for r in h[:n]] for h in hs]) * 100
        te = [[(r["epoch"], r["test_acc"] * 100) for r in h[:n] if "test_acc" in r] for h in hs]
        k = min(len(t) for t in te); tx = [e for e, _ in te[0][:k]]; ty = np.array([[a for _, a in t[:k]] for t in te])
        ax[0].plot(ep, loss.mean(0), label=f"{lab(m)} (n={len(hs)})")
        ax[0].fill_between(ep, loss.mean(0) - loss.std(0), loss.mean(0) + loss.std(0), alpha=0.2)
        l, = ax[1].plot(ep, tra.mean(0), ls="--", alpha=0.6)
        ax[1].plot(tx, ty.mean(0), color=l.get_color(), label=f"{lab(m)} test")
        ax[1].fill_between(tx, ty.mean(0) - ty.std(0), ty.mean(0) + ty.std(0), color=l.get_color(), alpha=0.2)
        ax[2].plot(ep, [r["lr"] for r in hs[0][:n]], label=lab(m))
    ax[0].set_yscale("log"); ax[0].set_title("train loss"); ax[0].set_xlabel("epoch")
    ax[1].set_title("accuracy (dashed: train, solid: test)"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("%")
    ax[2].set_yscale("log"); ax[2].set_title("learning rate"); ax[2].set_xlabel("epoch")
    for a in ax: a.legend()
    _save("learning_curves")


def fig_tau(runs):
    if "taunorm" not in runs: return
    plt.figure(figsize=(6, 4))
    for s, r in runs["taunorm"].items():
        t = [float(k) for k in r["val_curve"]]; a = [v * 100 for v in r["val_curve"].values()]
        l, = plt.plot(t, a, label=f"seed {s}")
        plt.scatter([r["tau"]], [r["val_curve"][str(r["tau"])] * 100], color=l.get_color(), zorder=3)
    plt.xlabel("tau"); plt.ylabel("val balanced acc (%)")
    plt.title("tau-norm: validation (100 images) vs tau, dots = chosen"); plt.legend(); _save("tau_curve")


def fig_norms(runs, imb=100):
    import torch   # imported here so the other figures work without it
    sets = {}
    ce = []
    for s in runs.get("ce", {}):
        path = f"{BASE}/checkpoints/cifar10_ce_rho{imb}_seed{s}.pt"
        if not os.path.exists(path):
            print("weight norms: missing checkpoint", path); continue
        ck = torch.load(path, map_location="cpu")
        ce.append(ck["model"]["fc.weight"].norm(dim=1).numpy())
    if ce: sets["ERM"] = np.array(ce)
    # cRT results only have w_norms if the runner saved them; skip otherwise
    if "crt" in runs and all("w_norms" in r for r in runs["crt"].values()):
        sets["cRT"] = np.array([r["w_norms"] for r in runs["crt"].values()])
    if not sets: print("weight norms: nothing to plot"); return
    counts = next(iter(runs[next(iter(runs))].values()))["counts"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for n, a in sets.items():
        ax.errorbar(range(10), a.mean(0), yerr=a.std(0), marker="o", capsize=3, label=n)
    ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES, rotation=45); ax.set_ylabel("||w_k||")
    ax2 = ax.twinx(); ax2.plot(range(10), counts, "k:", label="train count"); ax2.set_yscale("log")
    ax2.set_ylabel("train images per class (log)")
    ax.set_title("Classifier weight norms vs class frequency"); ax.legend(loc="upper center"); _save("weight_norms")


def tables(runs):
    L = ["## Mean ± std over seeds (test accuracy %, sample std)", "",
         "| method | seeds | overall | many | medium | few |", "|---|---|---|---|---|---|"]
    for m, rs in runs.items():
        c = [f"{np.mean([r[k] for r in rs.values()]) * 100:.2f} ± {sd([r[k] * 100 for r in rs.values()]):.2f}" for k in KEYS]
        L.append(f"| {lab(m)} | {len(rs)} | " + " | ".join(c) + " |")
    L += ["", "## Per seed (overall / many / medium / few)", ""]
    for m, rs in runs.items():
        for s, r in sorted(rs.items()):
            extra = f" tau={r['tau']}" if "tau" in r else ""
            L.append(f"- {lab(m)} seed {s}: " + " / ".join(f"{r[k] * 100:.2f}" for k in KEYS) + extra)
    if "ce" in runs:
        L += ["", "## Paired difference vs ERM (same seed = same backbone), percentage points", "",
              "| method | overall | many | medium | few |", "|---|---|---|---|---|"]
        for m, rs in runs.items():
            if m == "ce": continue
            seeds = sorted(set(rs) & set(runs["ce"]))
            if not seeds: continue
            c = []
            for k in KEYS:
                d = np.array([(rs[s][k] - runs["ce"][s][k]) * 100 for s in seeds])
                c.append(f"{d.mean():+.2f} ± {sd(d):.2f}")
            L.append(f"| {lab(m)} (n={len(seeds)}) | " + " | ".join(c) + " |")
    t = "\n".join(L); os.makedirs(OUT, exist_ok=True)
    open(f"{BASE}/results/tables.md", "w").write(t); print(t)


def make_all(imb=100):
    runs = load_runs(imb)
    print({lab(m): sorted(rs) for m, rs in runs.items()})
    if not runs: print("no results found in", f"{BASE}/results"); return
    tables(runs); fig_per_class(runs); fig_groups(runs)
    fig_curves(imb); fig_tau(runs); fig_norms(runs, imb)
