from third_party.patch2pix.utils.common.plotting import plot_matches

def model_params(model):
    return sum(param.numel() for param in model.parameters()) / 1e6
