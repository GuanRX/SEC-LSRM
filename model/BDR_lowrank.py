import torch
import torch.nn.functional as F
from .SubspaceClustering_ import idr


def threshold_affinity(matrix, ratio):
    affinity = (matrix.abs() + matrix.T.abs()) * 0.5
    affinity.fill_diagonal_(0)
    edges = affinity.triu(diagonal=1)
    positive = edges[edges > 0]
    if positive.numel() == 0:
        return affinity
    count = max(1, int(ratio * positive.numel()))
    threshold = positive.topk(count).values.min()
    return affinity.masked_fill(affinity < threshold, 0)


def get_sampleMatrix(x, k, max_iter=50, lambda_val=1e-2, gamma=1e-2, zs='z', ratio=1e-3):
    x = F.normalize(x)
    Z, S = idr(
        x.detach().t(),
        k=k,
        maxIter=max_iter,
        lambda_=lambda_val,
        gamma=gamma,
    )
    if zs == 'z':
        M = Z
    else:
        M = S

    return threshold_affinity(M, ratio)
