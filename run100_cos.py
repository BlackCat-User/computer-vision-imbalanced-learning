import sys
BASE = "/content/drive/MyDrive/Colab Notebooks/cv-longtail"
sys.path.insert(0, BASE)
import run100, cosine
from losses import ce_loss

def run(config):
    """config["loss"] is one of cosine.METHODS: cos_ce, cos_ldam, cos_ldam_drw (same logic as cosine.run)."""
    loss_name, kw = cosine.METHODS[config["loss"]]
    scale = config.get("scale", 30.0)
    orig_model, orig_loss = run100.ResNet32, run100.get_loss

    def model_fn(n, *a, **k):
        m = orig_model(n)
        in_f = getattr(m.fc, "in_features", m.fc.weight.shape[1])
        m.fc = cosine.CosineHead(in_f, n, scale=scale)
        return m

    def loss_fn(name, counts, **kwargs):
        return ce_loss if loss_name == "ce" else cosine.LDAMLoss(counts, s=scale, **kw)

    run100.ResNet32, run100.get_loss = model_fn, loss_fn
    try:
        run100.run({**config, "head": "cosine", "scale": scale})
    finally:
        run100.ResNet32, run100.get_loss = orig_model, orig_loss
