import torch
from .soap.soap import SOAP

def create_optimizer_kwargs(
    optimizer_name: str,
) -> dict:
    if optimizer_name == "adam":
        return {
            "optimizer_class": torch.optim.Adam,
        }
    
    elif optimizer_name == "soap":
        return {
            "optimizer_class": SOAP,
            "optimizer_kwargs": {
                "precondition_frequency": 10,
                "normalize_grads": False,
            }
        }
    elif optimizer_name == "muon":
        return {
            "muon_cfg": {
                "muon_lr": 2e-2,
                "muon_weight_decay": 1e-2,
                "adam_betas": (0.9, 0.95),
                "adam_weight_decay": 1e-2,
            }
        }
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")