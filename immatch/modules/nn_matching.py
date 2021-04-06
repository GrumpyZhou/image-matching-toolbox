import numpy as np

def mutual_nn_matching_torch(desc1, desc2, threshold=None):
    import torch
    device = desc1.device
    similarity = desc1 @ desc2.t()
    if threshold:
        # Incase of the descriptors are un-normalized, e.g., CAPS.
        norm1 = desc1.norm(dim=1, keepdim=True)
        norm2 = desc2.norm(dim=1, keepdim=True)        
        if (norm1).mean() != 1 or (norm2).mean() != 1:
            similarity = similarity / (norm1 @ norm2.t())
    
    nn12 = similarity.max(dim=1)[1]
    nn21 = similarity.max(dim=0)[1]
    ids1 = torch.arange(0, similarity.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]]).t()
    scores = similarity.max(dim=1)[0][mask]    
    if threshold:
        mask = scores > threshold
        matches = matches[mask]    
        scores = scores[mask]
    return matches, scores

def mutual_nn_matching(desc1, desc2, threshold=None):
    # Cosine distance for normalized descriptors
    similarity = desc1 @ desc2.T
    if threshold:
        # Incase of the descriptors are un-normalized, e.g., CAPS.
        norm1 = np.linalg.norm(desc1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(desc2, axis=1, keepdims=True)
        if np.mean(norm1) != 1 or np.mean(norm2) != 1:
            similarity = similarity / (norm1 @ norm2.T)
        
    nn12 = similarity.argmax(1)
    nn21 = similarity.argmax(0)
    ids1 = np.arange(similarity.shape[0])
    mask = (ids1 == nn21[nn12])
    matches = np.stack([ids1[mask], nn12[mask]]).T
    scores = similarity.max(1)[mask]    
    if threshold:
        mask = scores > threshold
        matches = matches[mask]
        scores = scores[mask]
    return matches, scores