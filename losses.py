import torch
import torch.nn.functional as F
from functools import partial


def ce_loss(logits, targets):
    log_probs = F.log_softmax(logits, dim=1)
    picked = log_probs[torch.arange(len(targets)), targets]
    return -picked.mean()


def inverse_freq_weights(class_counts):
    counts = torch.tensor(class_counts, dtype=torch.float)
    w = 1.0 / counts
    return w / w.sum() * len(counts)


def cb_weights(class_counts, beta=0.9999):
    counts = torch.tensor(class_counts, dtype=torch.float)
    effective_num = (1.0 - beta ** counts) / (1.0 - beta)
    w = 1.0 / effective_num
    return w / w.sum() * len(counts)


def weighted_ce_loss(logits, targets, weights):
    log_probs = F.log_softmax(logits, dim=1)
    picked = log_probs[torch.arange(len(targets)), targets]
    w = weights.to(logits.device)[targets]
    return -(w * picked).sum() / w.sum()


def focal_loss(logits, targets, gamma=2.0):
    log_probs = F.log_softmax(logits, dim=1)
    log_pt = log_probs[torch.arange(len(targets)), targets]
    pt = log_pt.exp()
    return (-(1 - pt) ** gamma * log_pt).mean()


class LDAMLoss:
    def __init__(self, class_counts, max_m=0.5, s=30.0, drw_start=None, beta=0.9999):
        counts = torch.tensor(class_counts, dtype=torch.float)
        m = 1.0 / counts ** 0.25
        self.m = m * (max_m / m.max())
        self.s = s
        self.drw_start = drw_start
        self.drw_weights = cb_weights(class_counts, beta)
        self.weights = None

    def set_epoch(self, epoch):
        on = self.drw_start is not None and epoch >= self.drw_start
        self.weights = self.drw_weights if on else None

    def __call__(self, logits, targets):
        m = self.m.to(logits.device)[targets]
        onehot = F.one_hot(targets, logits.shape[1]).bool()
        adjusted = torch.where(onehot, logits - m[:, None], logits)
        w = None if self.weights is None else self.weights.to(logits.device)
        return F.cross_entropy(self.s * adjusted, targets, weight=w)


def get_loss(name, class_counts, **kwargs):
    if name == "ce":
        return ce_loss
    if name == "reweighted_ce":
        return partial(weighted_ce_loss, weights=inverse_freq_weights(class_counts))
    if name == "focal":
        return partial(focal_loss, gamma=kwargs.get("gamma", 2.0))
    if name == "cb":
        return partial(weighted_ce_loss,
                       weights=cb_weights(class_counts, kwargs.get("beta", 0.9999)))
    if name == "ldam":
        return LDAMLoss(class_counts, **kwargs)
    raise ValueError(f"unknown loss: {name}")


def log_prior(class_counts):
    counts = torch.tensor(class_counts, dtype=torch.float)
    return torch.log(counts / counts.sum())


def logit_adjusted_loss(logits, targets, lp, tau=1.0):
    return F.cross_entropy(logits + tau * lp.to(logits.device), targets)


def posthoc_adjust(logits, class_counts, tau=1.0):
    return logits - tau * log_prior(class_counts).to(logits.device)


_get_loss_base = get_loss

def get_loss(name, class_counts, **kwargs):
    if name == "la":
        return partial(logit_adjusted_loss, lp=log_prior(class_counts),
                       tau=kwargs.get("tau", 1.0))
    if name == "balanced_softmax":
        return partial(logit_adjusted_loss, lp=log_prior(class_counts), tau=1.0)
    return _get_loss_base(name, class_counts, **kwargs)
