import csv, glob, json, os, re
import numpy as np

BASE = "/content/drive/MyDrive/Colab Notebooks/cv-longtail"
KEYS = ["overall", "many", "medium", "few"]
LABEL = {"ce": "ERM", "reweighted_ce": "Reweighted CE", "focal": "Focal", "cb": "Class-balanced",
         "balanced_softmax": "Balanced softmax", "ldam": "LDAM-DRW (linear)",
         "posthoc_la": "Post-hoc LA (tau on val)", "posthoc_la1": "Post-hoc LA (tau=1)",
         "taunorm": "tau-norm", "crt": "cRT",
         "cos_ce": "Cosine + CE", "cos_ldam": "Cosine + LDAM", "cos_ldam_drw": "Cosine + LDAM-DRW"}
FAMILY = {"ce": "Baseline", "reweighted_ce": "Reweighting", "focal": "Reweighting", "cb": "Reweighting",
          "balanced_softmax": "Logit adj. (loss)", "ldam": "Margin",
          "posthoc_la": "Post-hoc", "posthoc_la1": "Post-hoc", "taunorm": "Decoupled", "crt": "Decoupled",
          "cos_ce": "Cosine head", "cos_ldam": "Cosine head", "cos_ldam_drw": "Cosine head"}
ORDER = list(LABEL)


def lab(m): return LABEL.get(m, m)
def sd(v): v = np.asarray(v, float); return float(v.std(ddof=1)) if len(v) > 1 else 0.0


def load(ds, imb=100):
    runs = {}
    for f in sorted(glob.glob(f"{BASE}/results/{ds}_*_rho{imb}_seed*.json")):
        m = re.match(rf"{ds}_(.+)_rho{imb}_seed(\d+)\.json$", os.path.basename(f))
        if m: runs.setdefault(m.group(1), {})[int(m.group(2))] = json.load(open(f))   # tagged/debug runs don't match
    return dict(sorted(runs.items(), key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else 99))


def md(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] +
                     ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def latex(header, rows, align):
    conv = lambda c: re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", str(c)).replace("±", r"$\pm$").replace("tau", r"$\tau$")
    L = [r"\begin{tabular}{" + align + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    L += [" & ".join(conv(c) for c in r) + r" \\" for r in rows] + [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def extra(r):
    a = np.array(r["per_class"]); C = len(a); k = max(1, C // 10)
    return float(np.sort(a)[:k].mean()) * 100, int((a < (0.10 if C >= 100 else 0.50)).sum())


def make_all(ds="cifar100", imb=100):
    runs = load(ds, imb)
    if not runs: print("no results for", ds); return
    out = f"{BASE}/results/tables_{ds}"; os.makedirs(out, exist_ok=True)
    S = {m: {k: (np.mean([r[k] for r in rs.values()]) * 100, sd([r[k] * 100 for r in rs.values()])) for k in KEYS}
         for m, rs in runs.items()}
    best = {k: max(S, key=lambda m: S[m][k][0]) for k in KEYS}
    cell = lambda m, k: (lambda s: f"**{s}**" if m == best[k] else s)(f"{S[m][k][0]:.2f} ± {S[m][k][1]:.2f}")
    sec = []

    rows = [[FAMILY.get(m, "-"), lab(m), len(runs[m])] + [cell(m, k) for k in KEYS] for m in runs]
    H = ["Family", "Method", "n", "Overall", "Many", "Medium", "Few"]
    sec.append("## 1. Test accuracy (%), mean ± sample std over seeds (bold = best mean per column)\n\n" + md(H, rows))
    with open(f"{out}/main.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "family", "n"] + [f"{k}_{s}" for k in KEYS for s in ("mean", "std")])
        for m in runs: w.writerow([m, FAMILY.get(m, "-"), len(runs[m])] + [round(v, 3) for k in KEYS for v in S[m][k]])
    open(f"{out}/main.tex", "w").write(latex(H, rows, "llcrrrr"))

    if "ce" in runs:
        rows = []
        for m, rs in runs.items():
            if m == "ce": continue
            seeds = sorted(set(rs) & set(runs["ce"]))
            if not seeds: continue
            d = {k: np.array([(rs[s][k] - runs["ce"][s][k]) * 100 for s in seeds]) for k in KEYS}
            wins = sum(rs[s]["overall"] > runs["ce"][s]["overall"] for s in seeds)
            rows.append([lab(m), len(seeds)] + [f"{d[k].mean():+.2f} ± {sd(d[k]):.2f}" for k in KEYS] + [f"{wins}/{len(seeds)}"])
        H2 = ["Method", "n", "Δ overall", "Δ many", "Δ medium", "Δ few", "beats ERM (overall)"]
        sec.append("## 2. Paired difference vs ERM, percentage points (same seed)\n\n" + md(H2, rows))
        open(f"{out}/paired.tex", "w").write(latex(H2, rows, "lcrrrrc"))

    L = []
    for m, rs in runs.items():
        for s, r in sorted(rs.items()):
            L.append(f"- {lab(m)} seed {s}: " + " / ".join(f"{r[k] * 100:.2f}" for k in KEYS) +
                     (f" (tau={r['tau']})" if "tau" in r else ""))
    sec.append("## 3. Per seed (overall / many / medium / few)\n\n" + "\n".join(L))

    rows = []
    for m, rs in runs.items():
        e = np.array([extra(r) for r in rs.values()])
        rows.append([lab(m), f"{e[:, 0].mean():.2f} ± {sd(e[:, 0]):.2f}", f"{e[:, 1].mean():.1f} ± {sd(e[:, 1]):.1f}"])
    thr = "10" if len(next(iter(next(iter(runs.values())).values()))["per_class"]) >= 100 else "50"
    sec.append("## 4. Tail robustness\n\n" + md(["Method", "worst-decile class accuracy (%)", f"classes below {thr}% accuracy"], rows))

    if len(next(iter(next(iter(runs.values())).values()))["per_class"]) == 100:
        B = {m: np.array([r["per_class"] for r in rs.values()]).reshape(len(rs), 10, 10).mean(2).mean(0) * 100
             for m, rs in runs.items()}
        hb = [f"{10 * j}-{10 * j + 9}" for j in range(10)]
        sec.append("## 5. Accuracy by class bin (head -> tail), mean over seeds\n\n" +
                   md(["Method"] + hb, [[lab(m)] + [f"{v:.1f}" for v in B[m]] for m in B]))
        if "ce" in B:
            sec.append("## 6. Gain over ERM by class bin, points\n\n" +
                       md(["Method"] + hb, [[lab(m)] + [f"{v:+.1f}" for v in B[m] - B["ce"]] for m in B if m != "ce"]))

    rows = [[lab(m), ", ".join(f"{runs[m][s]['tau']}" for s in sorted(runs[m])),
             f"{np.mean([r['tau'] for r in runs[m].values()]):.2f}"]
            for m in runs if all("tau" in r for r in runs[m].values())]
    if rows: sec.append("## 7. Tuned tau (seeds 0, 1, 2)\n\n" + md(["Method", "tau per seed", "mean"], rows))

    notes = """## Notes for the report
- Test set is balanced, so overall accuracy equals balanced accuracy (mean per-class accuracy).
- Seeds change only initialisation and batch order. The LT split is fixed, so std does not include split variability.
- cRT, tau-norm and post-hoc LA reuse the ERM backbone of the same seed. Their seeds are not independent of ERM's, so use the paired table (2) for comparisons with ERM.
- With n=3 a sample std is rough. Call differences under about 1 to 2 points noise, and do not claim significance.
- Tuned-tau methods choose tau on a small validation set. CIFAR-100 uses 990 held-out training images (99 classes).
- Never mix CIFAR-10 (split: val) and CIFAR-100 (split: lt_full) rows in one table."""
    rep = f"# {ds}-LT, imbalance {imb}\n\n" + "\n\n".join(sec) + "\n\n" + notes
    open(f"{out}/report_{ds}.md", "w").write(rep); print(rep)
