import glob, json, os, shutil, sys
import numpy as np

BASE = "/content/drive/MyDrive/Colab Notebooks/cv-longtail"
sys.path.insert(0, BASE)
from aggregate_tables import load, lab, sd, KEYS

OUT = f"{BASE}/report_assets"
NAME = {"cifar10": "CIFAR-10-LT", "cifar100": "CIFAR-100-LT"}


def facts(ds, runs):
    r = runs["ce"][0]; c = r["counts"]
    f = {"train": sum(c), "head": c[0], "tail": c[-1], "C": len(c), "split": r["config"].get("split")}
    f["many"] = sum(n > 100 for n in c); f["few"] = sum(n < 20 for n in c)
    f["med"] = len(c) - f["many"] - f["few"]
    for m in ("taunorm", "posthoc_la"):
        if m in runs:
            ks = [float(k) for k in next(iter(runs[m].values()))["val_curve"]]
            f[m + "_grid"] = (min(ks), max(ks), round(ks[1] - ks[0], 2))
    return f


def mean(runs, m, k): return float(np.mean([r[k] for r in runs[m].values()])) * 100


def results_text(ds, runs):
    ms = list(runs); erm = mean(runs, "ce", "overall")
    best = max(ms, key=lambda m: mean(runs, m, "overall"))
    bf = max(ms, key=lambda m: mean(runs, m, "few"))
    pos, flat = [], []
    for m in ms:
        if m == "ce": continue
        seeds = sorted(set(runs[m]) & set(runs["ce"]))
        d = [(runs[m][s]["overall"] - runs["ce"][s]["overall"]) * 100 for s in seeds]
        (pos if all(x > 0 for x in d) else flat).append((lab(m), np.mean(d)))
    wins = ", ".join(f"{n} ({d:+.1f})" for n, d in sorted(pos, key=lambda t: -t[1]))
    t = (f"On {NAME[ds]}, ERM reaches {erm:.1f}\\% overall accuracy, with "
         f"{mean(runs, 'ce', 'many'):.1f}\\% on many-shot, {mean(runs, 'ce', 'medium'):.1f}\\% on medium-shot "
         f"and {mean(runs, 'ce', 'few'):.1f}\\% on few-shot classes (Table~\\ref{{tab:{ds}-main}}). "
         f"The best method by mean overall accuracy is {lab(best)} with {mean(runs, best, 'overall'):.1f}\\% "
         f"({mean(runs, best, 'overall') - erm:+.1f} points over ERM). The best few-shot accuracy is "
         f"{mean(runs, bf, 'few'):.1f}\\%, obtained by {lab(bf)}, against {mean(runs, 'ce', 'few'):.1f}\\% for ERM. ")
    if wins:
        t += (f"Methods that beat ERM on every seed in overall accuracy (paired differences in points, "
              f"Table~\\ref{{tab:{ds}-paired}}): {wins}. ")
    if flat:
        t += ("The following do not beat ERM on every seed and should be read as no clear gain: "
              + ", ".join(f"{n} ({d:+.1f})" for n, d in flat) + ". ")
    return t


def make(figs=None):
    os.makedirs(f"{OUT}/figures", exist_ok=True); os.makedirs(f"{OUT}/tables", exist_ok=True)
    data = {ds: load(ds) for ds in NAME}
    for ds, runs in data.items():
        assert "ce" in runs, f"no ERM results for {ds}"
        for m, rs in runs.items():
            if len(rs) != 3: print(f"WARNING {ds}/{m}: {len(rs)} seeds")
            sp = {r["config"].get("split") for r in rs.values()}
            if len(sp) != 1: print(f"WARNING {ds}/{m}: mixed splits {sp}")
        for fn in ("main.tex", "paired.tex"):
            src = f"{BASE}/results/tables_{ds}/{fn}"
            if os.path.exists(src): shutil.copy(src, f"{OUT}/tables/{ds}_{fn}")
            else: print("missing", src, "-> run aggregate_tables.make_all first")
    F = {ds: facts(ds, data[ds]) for ds in NAME}
    a, b = F["cifar10"], F["cifar100"]

    def grid(f, m):
        g = f.get(m + "_grid"); return f"$[{g[0]:g},{g[1]:g}]$ in steps of {g[2]:g}" if g else "n/a"

    setup = rf"""\section{{Experimental Setup}}
\paragraph{{Datasets.}} We build CIFAR-10-LT and CIFAR-100-LT by subsampling the training sets with an exponential profile
$n_i = \lfloor n_{{\max}}\,\rho^{{-i/(C-1)}}\rfloor$ and imbalance ratio $\rho=100$, with $n_{{\max}}=5000$ and $500$ respectively.
CIFAR-10-LT initially contains 12,406 images, with class counts ranging from 5,000 to 50. We hold out 10 images per class for validation, leaving {a['train']:,} training images, with counts ranging from {a['head']:,} to {a['tail']:,}.
CIFAR-100-LT keeps {b['head']} and {b['tail']} images in the largest and smallest classes, {b['train']} images in total.
The test sets stay balanced (1000 images per class for CIFAR-10, 100 for CIFAR-100).
Classes are grouped as many-shot ($>100$ training images), medium-shot (20 to 100) and few-shot ($<20$).
On CIFAR-100-LT this gives {b['many']} / {b['med']} / {b['few']} classes; on CIFAR-10-LT we use the fixed groups
$\{{0,1,2\}}$, $\{{3,4,5,6\}}$, $\{{7,8,9\}}$ (the standard thresholds would leave the few-shot group empty).
\paragraph{{Backbone and training.}} All methods use a ResNet-32 trained from scratch with SGD (momentum 0.9, weight decay $2\times10^{{-4}}$),
batch size 128, 200 epochs, base learning rate 0.1 with linear warmup over the first 5 epochs and a $\times0.01$ decay at epochs 160 and 180.
Training augmentation is random crop (padding 4) and horizontal flip, with per-channel normalisation; test images are only normalised.
All end-to-end trained models use the same base architecture, data augmentation and 200-epoch training recipe. Decoupled methods reuse the corresponding ERM checkpoint: cRT performs an additional classifier-training stage, whereas post-hoc methods require no further training.
\paragraph{{Methods.}} Baselines: ERM (cross-entropy), inverse-frequency reweighted cross-entropy, focal loss ($\gamma=2$),
class-balanced loss ($\beta=0.9999$) and balanced softmax. Two-stage methods on the trained ERM backbone: cRT
(classifier re-initialised and retrained for 20 epochs with class-balanced sampling, backbone and batch-norm frozen,
cosine learning-rate decay from 0.1) and $\tau$-norm ($w_k/\|w_k\|^{{\tau}}$). Post-hoc logit adjustment ($z_k - \tau\log\pi_k$) on the ERM
logits, with $\tau=1$ or $\tau$ tuned on validation. Cosine-head variants: cosine classifier with scale $s=30$ trained with cross-entropy,
with the LDAM margin (maximum margin 0.5), and with LDAM plus deferred re-weighting from epoch 160 (LDAM-DRW).
\paragraph{{Tuning and protocol.}} Test accuracy is reported at the final epoch; the test set is never used for any selection.
Tuned methods choose $\tau$ by balanced accuracy on a held-out validation set: on CIFAR-10-LT the 100 validation images above,
on CIFAR-100-LT 990 training images that are outside the long-tailed subset (10 per class for 99 classes, class 0 has no spare images).
The $\tau$ grid is {grid(a, 'taunorm')} for $\tau$-norm and {grid(a, 'posthoc_la')} for post-hoc LA on CIFAR-10-LT, and
{grid(b, 'taunorm')} and {grid(b, 'posthoc_la')} on CIFAR-100-LT; ties are broken by the median of the best values.
Every configuration is run with seeds 0, 1, 2, which change the initialisation and batch order but not the long-tailed split.
We report mean $\pm$ sample standard deviation. Two-stage and post-hoc methods reuse the ERM backbone of the same seed, so we also report
paired differences against ERM.
"""
    res = r"\section{Results}" + "\n"
    for ds in NAME:
        res += rf"""\subsection{{{NAME[ds]}}}
{results_text(ds, data[ds])}

\begin{{table}}[t]\centering\small
\caption{{{NAME[ds]}, imbalance 100. Test accuracy (\%), mean $\pm$ std over 3 seeds; bold = best mean.}}\label{{tab:{ds}-main}}
\input{{tables/{ds}_main.tex}}\end{{table}}
\begin{{table}}[t]\centering\small
\caption{{{NAME[ds]}: paired difference against ERM of the same seed (percentage points).}}\label{{tab:{ds}-paired}}
\input{{tables/{ds}_paired.tex}}\end{{table}}
"""
    res += r"""\paragraph{Caveats.} The standard deviation covers initialisation and batch order only, as the split is fixed.
With three seeds, differences below about one to two points are within noise and we do not claim significance.
"""
    open(f"{OUT}/setup.tex", "w").write(setup); open(f"{OUT}/results.tex", "w").write(res)

    wanted = figs or {
        "c100_groups": "c100_groups", "c100_gain_by_bin": "c100_gain_by_bin", "c100_tradeoff": "c100_tradeoff",
        "c100_val_tau": "c100_val_tau", "c100_test_tau": "c100_test_tau", "c100_ranking": "c100_ranking",
        "per_class": "c10_per_class", "groups": "c10_groups", "tau_curve": "c10_tau_curve", "weight_norms": "c10_weight_norms"}
    for src, dst in wanted.items():
        p = f"{BASE}/results/figures/{src}.png"
        if os.path.exists(p): shutil.copy(p, f"{OUT}/figures/{dst}.png"); print("figure ok:", dst)
        else: print("figure MISSING:", src)
    print("\nwritten to", OUT, sorted(os.listdir(OUT)))
  
