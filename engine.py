
import torch


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):

    model.train()

    if hasattr(criterion, "set_epoch"):
        criterion.set_epoch(epoch)

    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:

        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(logits, targets)

        loss.backward()

        optimizer.step()

        batch_size = targets.size(0)

        running_loss += loss.item() * batch_size

        predictions = logits.argmax(dim=1)

        correct += (predictions == targets).sum().item()
        total += batch_size

    return {
        "loss": running_loss / total,
        "accuracy": correct / total
    }


def evaluate(model, loader, criterion, device):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, targets in loader:

            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)

            loss = criterion(logits, targets)

            batch_size = targets.size(0)

            running_loss += loss.item() * batch_size

            predictions = logits.argmax(dim=1)

            correct += (predictions == targets).sum().item()
            total += batch_size

    return {
        "loss": running_loss / total,
        "accuracy": correct / total
    }


def lr_at(epoch, base_lr=0.1, warmup=5, steps=(160, 180), gamma=0.01):
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    lr = base_lr
    for s in steps:
        if epoch >= s:
            lr *= gamma
    return lr


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, targets = [], []
    for images, y in loader:
        preds.append(model(images.to(device)).argmax(1).cpu())
        targets.append(y)
    return torch.cat(preds), torch.cat(targets)


def save_ckpt(path, model, optimizer, epoch):
    import os
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": optimizer.state_dict(),
                "epoch": epoch}, tmp)
    os.replace(tmp, path)


def load_ckpt(path, model, optimizer, device):
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["opt"])
    return ck["epoch"] + 1   # next epoch to run
