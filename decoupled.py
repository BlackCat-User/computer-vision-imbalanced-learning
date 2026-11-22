import sys, os, json, math
import numpy as np, torch, torch.nn.functional as F
import torchvision, torchvision.transforms as T

BASE = "/content/drive/MyDrive/Colab Notebooks/cv-longtail"
sys.path.insert(0, BASE)
from run import ArrayDS, GROUPS_C10, NORM, set_seed
from models.resnet import ResNet32
from engine import predict
from utils import compute_metrics

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(seed, imbalance=100):
    rj = f"{BASE}/results/cifar10_ce_rho{imbalance}_seed{seed}.json"
    assert os.path.exists(rj), f"ERM seed {seed} is not finished yet"
    assert json.load(open(rj))["config"].get("split") == "val", \
        f"ERM seed {seed} was not trained on the 12,306 split"
    ck = torch.load(f"{BASE}/checkpoints/cifar10_ce_rho{imbalance}_seed{seed}.pt", map_location=DEV)
    m = ResNet32(10).to(DEV)
    m.load_state_dict(ck["model"])
    return m.eval()


def get_data(imbalance=100):
    tr = torchvision.datasets.CIFAR10(f"{BASE}/data", train=True, download=True)
    te = torchvision.datasets.CIFAR10(f"{BASE}/data", train=False, download=True)
    sp = json.load(open(f"{BASE}/splits/c10lt_rho{imbalance}_seed0_val.json"))
    idx, counts, val = sp["train_indices"], sp["train_counts"], sp["val_indices"]
    assert len(idx) == 12306 and not set(idx) & set(val), "split problem"
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
    model.eval(); fs, ys = [], []
    for x, y in loader:
        fs.append(model.features(x.to(DEV)).cpu()); ys.append(y)
    return torch.cat(fs), torch.cat(ys)


def bal_acc(preds, y):
    return float(np.mean([(preds[y == c] == c).float().mean().item() for c in y.unique().tolist()]))


def save_result(name, config, counts, preds, targets, extra=None):
    res = {"config": {**config, "split": "val"}, "counts": counts,
           **compute_metrics(preds.numpy(), targets.numpy(), GROUPS_C10)}
    if extra: res.update(extra)
    out = f"{BASE}/results/{name}.json"
    with open(out + ".tmp", "w") as f: json.dump(res, f)
    os.replace(out + ".tmp", out)
    print(name, "| overall:", round(res["overall"], 4), "many/med/few:",
          round(res["many"], 3), round(res["medium"], 3), round(res["few"], 3),
          "| tau:", res.get("tau", "-"))


def tau_norm(seed, imbalance=100):
    name = f"cifar10_taunorm_rho{imbalance}_seed{seed}"
    if os.path.exists(f"{BASE}/results/{name}.json"): print("already done:", name); return
    set_seed(seed)
    model = load_model(seed, imbalance)
    _, val_loader, test_loader, counts, _ = get_data(imbalance)
    fv, yv = extract(model, val_loader); ft, yt = extract(model, test_loader)
    W = model.fc.weight.detach().cpu()
    norms = W.norm(dim=1, keepdim=True)
    logits = lambda f, tau: f @ (W / norms ** tau).T
    taus = [round(t, 2) for t in np.arange(0, 2.01, 0.1)]
    va = [bal_acc(logits(fv, t).argmax(1), yv) for t in taus]
    top = [t for t, a in zip(taus, va) if a >= max(va) - 1e-9]
    best = top[len(top) // 2]
    print(f"seed {seed} val balanced acc per tau:", dict(zip(taus, [round(a, 3) for a in va])), "-> tau", best)
    cfg = {"dataset": "cifar10", "imbalance": imbalance, "seed": seed, "loss": "taunorm", "tau": best}
    save_result(name, cfg, counts, logits(ft, best).argmax(1), yt,
                {"tau": best, "val_curve": {str(t): a for t, a in zip(taus, va)},
                 "w_norms": norms.squeeze(1).tolist()})


def crt(seed, imbalance=100, epochs=20, lr=0.1, wd=2e-4, bs=128):
    name = f"cifar10_crt_rho{imbalance}_seed{seed}"
    if os.path.exists(f"{BASE}/results/{name}.json"): print("already done:", name); return
    set_seed(seed)
    model = load_model(seed, imbalance)
    train_ds, val_loader, test_loader, counts, mk = get_data(imbalance)
    w = torch.tensor([1.0 / counts[y] for y in train_ds.targets])
    sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(train_ds), replacement=True)
    loader = mk(train_ds, sampler=sampler, bs=bs)
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
    pv, yv = predict(model, val_loader, DEV); pt, yt = predict(model, test_loader, DEV)
    cfg = {"dataset": "cifar10", "imbalance": imbalance, "seed": seed, "loss": "crt",
           "epochs": epochs, "lr": lr, "wd": wd}
    save_result(name, cfg, counts, pt, yt,
                {"val_bal_acc": bal_acc(pv, yv),
                 "w_norms": model.fc.weight.detach().norm(dim=1).cpu().tolist()})
