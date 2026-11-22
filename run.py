import sys, os, json, random, time
import numpy as np, torch
from PIL import Image
import torchvision, torchvision.transforms as T

BASE = "/content/drive/MyDrive/Colab Notebooks/cv-longtail"
sys.path.insert(0, BASE)
from models.resnet import ResNet32
from engine import train_one_epoch, predict, save_ckpt, load_ckpt, lr_at
from utils import compute_metrics, make_lt_indices
from losses import get_loss

GROUPS_C10 = {"many": [0, 1, 2], "medium": [3, 4, 5, 6], "few": [7, 8, 9]}
NORM = T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))


class ArrayDS(torch.utils.data.Dataset):
    def __init__(self, data, targets, indices, transform):
        self.data = data[indices]
        self.targets = [int(targets[i]) for i in indices]
        self.transform = transform
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.data[i])), self.targets[i]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def run(config):
    tag = config.get("tag", "")
    name = f"{config['dataset']}_{config['loss']}_rho{config['imbalance']}_seed{config['seed']}"
    if tag:
        name += f"_{tag}"
    out_json = f"{BASE}/results/{name}.json"
    ckpt_path = f"{BASE}/checkpoints/{name}.pt"
    if os.path.exists(out_json):
        print("already done:", name); return

    set_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data: split is fixed (split seed 0), the training seed only changes init/order
    train_raw = torchvision.datasets.CIFAR10(f"{BASE}/data", train=True, download=True)
    test_raw = torchvision.datasets.CIFAR10(f"{BASE}/data", train=False, download=True)
    split_path = f"{BASE}/splits/c10lt_rho{config['imbalance']}_seed0_val.json"
    sp = json.load(open(split_path))
    idx, counts = sp["train_indices"], sp["train_counts"]
    print("train size:", len(idx), "| counts:", counts)

    tr_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), NORM])
    te_tf = T.Compose([T.ToTensor(), NORM])
    train_ds = ArrayDS(train_raw.data, train_raw.targets, idx, tr_tf)
    test_ds = ArrayDS(test_raw.data, test_raw.targets, list(range(len(test_raw))), te_tf)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config["batch_size"],
                                               shuffle=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False,
                                              num_workers=2, pin_memory=True)

    model = ResNet32(10).to(device)
    criterion = get_loss(config["loss"], counts, **config.get("loss_kwargs", {}))
    opt = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9,
                          weight_decay=config["wd"])

    start = 0
    if os.path.exists(ckpt_path):
        start = load_ckpt(ckpt_path, model, opt, device)
        print(f"resuming from epoch {start}")

    for epoch in range(start, config["epochs"]):
        t0 = time.time()
        lr = lr_at(epoch, base_lr=config["lr"])
        for g in opt.param_groups:
            g["lr"] = lr
        st = train_one_epoch(model, train_loader, criterion, opt, device, epoch)
        save_ckpt(ckpt_path, model, opt, epoch)
        msg = f"epoch {epoch} lr={lr:.5f} loss={st['loss']:.4f} train_acc={st['accuracy']:.3f} ({time.time()-t0:.1f}s)"
        if epoch % 10 == 0 or epoch == config["epochs"] - 1:
            p, y = predict(model, test_loader, device)
            msg += f" test_acc={(p == y).float().mean().item():.4f}"
        print(msg)

    preds, targets = predict(model, test_loader, device)
    res = {"config": {**config, "split": "val"}, "counts": counts,
           **compute_metrics(preds.numpy(), targets.numpy(), GROUPS_C10)}
    with open(out_json + ".tmp", "w") as f:
        json.dump(res, f)
    os.replace(out_json + ".tmp", out_json)
    print("saved", out_json, "| overall:", round(res["overall"], 4),
          "many/med/few:", round(res["many"], 3), round(res["medium"], 3), round(res["few"], 3))
