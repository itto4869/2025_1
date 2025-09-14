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
                # Base LR for Muon; linearly decays via SB3 ratio coupling
                # to match Adam's schedule shape.
                "muon_lr": 3.0e-4,
                # Set floor to 0.0 to allow pure linear decay to zero if desired.
                "muon_lr_floor": 0.0,
                "muon_weight_decay": 0.0,
                # Align Adam with SB3-style defaults
                "adam_betas": (0.9, 0.999),
                "adam_weight_decay": 0.0,
            }
        }
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def atari_adam_lr_schedule(progress: float) -> float:
    return 2.5e-4 * progress

def atari_soap_lr_schedule(progress: float) -> float:
    return 3.0e-4 * progress