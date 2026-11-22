import glob, json, os, re
import numpy as np, matplotlib.pyplot as plt

BASE = "/content/drive/MyDrive/Colab Notebooks/cv-longtail"
OUT = f"{BASE}/results/figures"
KEYS = ["overall", "many", "medium", "few"]

def load():
    runs = {}
    for f in sorted(glob.glob(f"{BASE}/results/cifar100_*_rho100_seed*.json")):
        m = re.match(r"cifar100_(.+)_rho100_seed(\d+)\.json$", os.path.basename(f))
        if m:  # tagged runs (debug) do not match and are skipped
            runs.setdefault(m.group(1), {})[int(m.group(2))] = json.load(open(f))
    return runs

def make_all():
    runs = load(); os.makedirs(OUT, exist_ok=True)
    ms = list(runs); w = 0.8 / len(ms); x = np.arange(4); rng = np.random.RandomState(0)
    plt.figure(figsize=(8, 4.5))
    lines = ["| method | seeds | overall | many | medium | few |", "|---|---|---|---|---|---|"]
    for i, m in enumerate(ms):
        a = np.array([[r[k] for k in KEYS] for r in runs[m].values()]) * 100
        sd = a.std(0, ddof=1) if len(a) > 1 else np.zeros(4)
        pos = x + (i - (len(ms) - 1) / 2) * w
        plt.bar(pos, a.mean(0), w, yerr=sd, capsize=3, label=f"{m} (n={len(a)})", alpha=0.8)
        for j in range(4):
            plt.scatter(pos[j] + rng.uniform(-w/4, w/4, len(a)), a[:, j], c="k", s=10, zorder=3)
        lines.append(f"| {m} | {len(a)} | " + " | ".join(f"{a.mean(0)[j]:.2f} ± {sd[j]:.2f}" for j in range(4)) + " |")
    plt.xticks(x, KEYS); plt.ylabel("test accuracy (%)")
    plt.title("CIFAR-100-LT (rho=100): overall and many/medium/few"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/c100_groups.png", dpi=150); plt.show()
    t = "\n".join(lines); open(f"{BASE}/results/tables_c100.md", "w").write(t); print(t)
