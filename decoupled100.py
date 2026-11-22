import sys, os, json, math
import numpy as np, torch, torch.nn.functional as F
import torchvision, torchvision.transforms as T

BASE = "/content/drive/MyDrive/Colab Notebooks/cv-longtail"
sys.path.insert(0, BASE)
from run import ArrayDS, set_seed
from run100 import NORM
from models.resnet import ResNet32
from engine import predict
from utils import compute_metrics, make_groups

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(seed):
    rj = f"{BASE}/results/cifar100_ce_rho100_seed{seed}.json"
    assert os.path.exists(rj), f"CIFAR-100 ERM seed {seed} not finished"
    assert json.load(open(rj))["config"].get("split") == "lt_full", "wrong split"
    ck = torch.load(f"{BASE}/checkpoints/cifar100_ce_rho100_seed{seed}.pt", map_location=DEV)
    assert ck["epoch"] == 199, f"seed {seed} stopped at epoch {ck['epoch']}"
    m = ResNet32(100).to(DEV); m.load_state_dict(ck["model"]); return m.eval()


def get_data():
    tr = torchvision.datasets.CIFAR100(f"{BASE}/data", train=True, download=True)
    te = torchvision.datasets.CIFAR100(f"{BASE}/data", train=False, download=True)
    sp = json.load(open(f"{BASE}/splits/c100lt_rho100_seed0.json"))
    idx, counts = sp["indices"], sp["counts"]
    t = np.array(tr.targets); used = set(idx); rng = np.random.RandomState(0); val = []
    for c in range(100):
        pool = [i for i in np.where(t == c)[0] if i not in used]
        if len(pool) >= 10: val += rng.choice(pool, 10, replace=False).tolist()
    assert not set(val) & used
    print("val images:", len(val), "| classes covered:", len(val) // 10)
    te_tf = T.Compose([T.ToTensor(), NORM])
    tr_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), NORM])
    train_ds = ArrayDS(tr.data, tr.targets, idx, tr_tf)
    val_ds = ArrayDS(tr.data, tr.targets, val, te_tf)
    test_ds = ArrayDS(te.data, te.targets, list(range(len(te))), te_tf)
    mk = lambda ds, sampler=None, bs=256: torch.utils.data.DataLoader(
        ds, batch_size=bs, sampler=sampler, shuffle=False, num_workers=2, pin_memory=True)
    return train_ds, mk(val_ds), mk(test_ds), counts, mk


@torch.no_grad()
def extract(model, loader):
    fs, ys = [], []
    for x, y in loader:
        fs.append(model.features(x.to(DEV)).cpu()); ys.append(y)
    return torch.cat(fs), torch.cat(ys)


def bal_acc(p, y):
    return float(np.mean([(p[y == c] == c).float().mean().item() for c in y.unique().tolist()]))


def save_result(name, config, counts, preds, targets, extra=None):
    res = {"config": {**config, "split": "lt_full"}, "counts": counts,
           **compute_metrics(preds.numpy(), targets.numpy(), make_groups(counts))}
    if extra: res.update(extra)
    out = f"{BASE}/results/{name}.json"
    json.dump(res, open(out + ".tmp", "w")); os.replace(out + ".tmp", out)
    print(name, "| overall:", round(res["overall"], 4), "many/med/few:",
          round(res["many"], 3), round(res["medium"], 3), round(res["few"], 3), "| tau:", res.get("tau", "-"))


def tau_norm(seed):
    name = f"cifar100_taunorm_rho100_seed{seed}"
    if os.path.exists(f"{BASE}/results/{name}.json"): print("already done:", name); return
    set_seed(seed); model = load_model(seed)
    _, vl, tl, counts, _ = get_data()
    fv, yv = extract(model, vl); ft, yt = extract(model, tl)
    W = model.fc.weight.detach().cpu(); norms = W.norm(dim=1, keepdim=True)
    logits = lambda f, tau: f @ (W / norms ** tau).T
    taus = [round(t, 2) for t in np.arange(0, 2.01, 0.1)]
    va = [bal_acc(logits(fv, t).argmax(1), yv) for t in taus]
    top = [t for t, a in zip(taus, va) if a >= max(va) - 1e-9]; best = top[len(top) // 2]
    print(f"seed {seed} val acc per tau:", {t: round(a, 3) for t, a in zip(taus, va)}, "-> tau", best)
    cfg = {"dataset": "cifar100", "imbalance": 100, "seed": seed, "loss": "taunorm", "tau": best}
    save_result(name, cfg, counts, logits(ft, best).argmax(1), yt,
                {"tau": best, "val_curve": {str(t): a for t, a in zip(taus, va)}, "w_norms": norms.squeeze(1).tolist()})


def crt(seed, epochs=20, lr=0.1, wd=2e-4, bs=128):
    name = f"cifar100_crt_rho100_seed{seed}"
    if os.path.exists(f"{BASE}/results/{name}.json"): print("already done:", name); return
    set_seed(seed); model = load_model(seed)
    train_ds, _, tl, counts, mk = get_data()
    w = torch.tensor([1.0 / counts[y] for y in train_ds.targets])
    loader = mk(train_ds, sampler=torch.utils.data.WeightedRandomSampler(w, len(train_ds), replacement=True), bs=bs)
    for p in model.parameters(): p.requires_grad = False
    model.fc.reset_parameters()
    for p in model.fc.parameters(): p.requires_grad = True
    opt = torch.optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    for ep in range(epochs):
        model.eval()
        for g in opt.param_groups: g["lr"] = 0.5 * lr * (1 + math.cos(math.pi * ep / epochs))
        tot, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(DEV), y.to(DEV)
            with torch.no_grad(): f = model.features(x)
            loss = F.cross_entropy(model.fc(f), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(y); n += len(y)
        if ep % 5 == 0 or ep == epochs - 1: print(f"crt seed{seed} epoch {ep} loss={tot/n:.4f}")
    pt, yt = predict(model, tl, DEV)
    cfg = {"dataset": "cifar100", "imbalance": 100, "seed": seed, "loss": "crt", "epochs": epochs, "lr": lr, "wd": wd}
    save_result(name, cfg, counts, pt, yt, {"w_norms": model.fc.weight.detach().norm(dim=1).cpu().tolist()})
