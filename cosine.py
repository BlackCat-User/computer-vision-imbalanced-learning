"""Cosine classifier head and LDAM(-DRW) loss for the cosine-head experiments.

CosineHead: z_k = scale * cos(w_k, f)   (features and class weights normalised to norm 1)
LDAMLoss:   Cao et al. (2019) on z = scale * cos. margin_k = max_m * n_k^-1/4 / max_j n_j^-1/4
            (rarest class gets max_m), subtracted from the true-class cosine, i.e. scale * margin in z-space.
            DRW: from epoch drw_start the class weights are switched on (Algorithm 1: inverse frequency).
Methods (the name becomes the result file, e.g. cifar10_cos_ldam_drw_rho100_seed0.json):
  cos_ce        cosine head + plain cross-entropy
  cos_ldam      cosine head + LDAM margin
  cos_ldam_drw  cosine head + LDAM margin + deferred re-weighting from epoch 160  (= LDAM-DRW)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import ce_loss, inverse_freq_weights, cb_weights


class CosineHead(nn.Module):
    def __init__(self, in_features, num_classes, scale=30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        self.scale = scale
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, z):
        return self.scale * F.linear(F.normalize(z, dim=1), F.normalize(self.weight, dim=1))


class LDAMLoss:
    """Takes z = scale * cos (the CosineHead output); `s` must equal the head scale."""
    def __init__(self, class_counts, max_m=0.5, s=30.0, drw_start=None, drw_weights="inverse", beta=0.9999):
        counts = torch.tensor(class_counts, dtype=torch.float)
        m = 1.0 / counts ** 0.25
        self.m = m * (max_m / m.max())
        self.s = s
        self.drw_start = drw_start
        self.drw_w = (inverse_freq_weights(class_counts) if drw_weights == "inverse"
                      else cb_weights(class_counts, beta))
        self.weights = None

    def set_epoch(self, epoch):        # called once per epoch by the training loop
        on = self.drw_start is not None and epoch >= self.drw_start
        self.weights = self.drw_w if on else None

    def __call__(self, logits, targets):
        m = self.m.to(logits.device)[targets]
        onehot = F.one_hot(targets, logits.shape[1]).bool()
        adjusted = torch.where(onehot, logits - self.s * m[:, None], logits)
        w = None if self.weights is None else self.weights.to(logits.device)
        return F.cross_entropy(adjusted, targets, weight=w)


METHODS = {
    "cos_ce": ("ce", {}),
    "cos_ldam": ("ldam", {}),
    "cos_ldam_drw": ("ldam", {"drw_start": 160}),
}


def run(R, config):
    """Run run.run() (module R) with a cosine head and the loss of config["loss"] (a METHODS name)."""
    method = config["loss"]
    loss_name, kw = METHODS[method]
    scale = config.get("scale", 30.0)
    orig_model, orig_loss = R.ResNet32, R.get_loss

    def model_fn(n):
        m = orig_model(n)
        in_f = getattr(m.fc, "in_features", m.fc.weight.shape[1])
        m.fc = CosineHead(in_f, n, scale=scale)
        return m

    def loss_fn(name, counts, **kwargs):
        return ce_loss if loss_name == "ce" else LDAMLoss(counts, s=scale, **kw)

    R.ResNet32, R.get_loss = model_fn, loss_fn
    try:
        R.run({**config, "head": "cosine", "scale": scale})
    finally:
        R.ResNet32, R.get_loss = orig_model, orig_loss
