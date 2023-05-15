import torch

from third_party.patch2pix.utils.common.plotting import plot_matches


def model_params(model):
    return sum(param.numel() for param in model.parameters()) / 1e6

def dict_to_deivce(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch