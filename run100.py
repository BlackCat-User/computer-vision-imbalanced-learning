import sys, os, json, time
import torch, torchvision, torchvision.transforms as T

BASE = "/content/drive/MyDrive/Colab Notebooks/cv-longtail"
sys.path.insert(0, BASE)
from run import ArrayDS, set_seed
from models.resnet import ResNet32
from engine import train_one_epoch, predict, save_ckpt, load_ckpt, lr_at
from utils import compute_metrics
from losses import get_loss

NORM = T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))


def run(config):
    name = f"cifar100_{config['loss']}_rho{config['imbalance']}_seed{config['seed']}"
    if config.get("tag"): name += f"_{config['tag']}"
    out_json, ckpt = f"{BASE}/results/{name}.json", f"{BASE}/checkpoints/{name}.pt"
    log_path = f"{BASE}/logs/{name}.json"; os.makedirs(f"{BASE}/logs", exist_ok=True)
    if os.path.exists(out_json): print("already done:", name); return

    set_seed(config["seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = torchvision.datasets.CIFAR100(f"{BASE}/data", train=True, download=True)
    te = torchvision.datasets.CIFAR100(f"{BASE}/data", train=False, download=True)
    sp = json.load(open(f"{BASE}/splits/c100lt_rho{config['imbalance']}_seed0.json"))
    idx, counts = sp["indices"], sp["counts"]
    print("train size:", len(idx), "| counts head/tail:", counts[0], counts[-1])
    groups = {"many": [c for c, n in enumerate(counts) if n > 100],
              "few": [c for c, n in enumerate(counts) if n < 20]}
    groups["medium"] = [c for c in range(100) if c not in groups["many"] + groups["few"]]

    tr_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), NORM])
    te_tf = T.Compose([T.ToTensor(), NORM])
    mk = lambda ds, sh, bs: torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=sh, num_workers=2, pin_memory=True)
    train_loader = mk(ArrayDS(tr.data, tr.targets, idx, tr_tf), True, config["batch_size"])
    test_loader = mk(ArrayDS(te.data, te.targets, list(range(len(te))), te_tf), False, 256)

    model = ResNet32(100).to(dev)
    criterion = get_loss(config["loss"], counts, **config.get("loss_kwargs", {}))
    opt = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=config["wd"])
    start, hist = 0, []
    if os.path.exists(ckpt):
        start = load_ckpt(ckpt, model, opt, dev); print(f"resuming from epoch {start}")
        if os.path.exists(log_path): hist = [h for h in json.load(open(log_path)) if h["epoch"] < start]

    for epoch in range(start, config["epochs"]):
        t0 = time.time(); lr = lr_at(epoch, base_lr=config["lr"])
        for g in opt.param_groups: g["lr"] = lr
        st = train_one_epoch(model, train_loader, criterion, opt, dev, epoch)
        save_ckpt(ckpt, model, opt, epoch)
        rec = {"epoch": epoch, "lr": lr, "loss": st["loss"], "train_acc": st["accuracy"]}
        msg = f"epoch {epoch} lr={lr:.5f} loss={st['loss']:.4f} train_acc={st['accuracy']:.3f} ({time.time()-t0:.1f}s)"
        if epoch % 10 == 0 or epoch == config["epochs"] - 1:
            p, y = predict(model, test_loader, dev)
            rec["test_acc"] = (p == y).float().mean().item()   # logging only
            msg += f" test_acc={rec['test_acc']:.4f}"
        hist.append(rec); json.dump(hist, open(log_path, "w")); print(msg)

    preds, targets = predict(model, test_loader, dev)
    res = {"config": {**config, "split": "lt_full"}, "counts": counts,
           **compute_metrics(preds.numpy(), targets.numpy(), groups)}
    json.dump(res, open(out_json + ".tmp", "w")); os.replace(out_json + ".tmp", out_json)
    print("saved", name, "| overall:", round(res["overall"], 4),
          "many/med/few:", round(res["many"], 3), round(res["medium"], 3), round(res["few"], 3))
