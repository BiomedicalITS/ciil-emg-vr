try:
    import torch
    from lightning.pytorch import seed_everything

    __SEED = 310
    seed_everything(__SEED)

    ACCELERATOR = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
except ImportError:
    pass

PEUDO_LABELS_PORT = 5111
PREDS_PORT = 5112
PREDS_IP = "127.0.0.1"
