import torch
import torch.nn.functional as F


def sampling_contrastive_loss(views, matrices, temperature):
    """Combine the within-view and cross-view terms in Eqs. (21)?(23)."""
    features = [F.normalize(view, dim=1) for view in views]
    diagonal = torch.eye(features[0].size(0), dtype=torch.bool, device=features[0].device)
    loss = features[0].new_zeros(())
    for index, (view, matrix) in enumerate(zip(features, matrices)):
        neighbors = (matrix != 0) & ~diagonal
        # Sparse sampling can leave a row without a positive neighbor.
        fallback = ~neighbors.any(dim=1, keepdim=True) & diagonal
        positive = neighbors | fallback
        scores = (view @ view.T) / temperature
        numerator = scores.masked_fill(~positive, -torch.inf).logsumexp(dim=1)
        denominator = scores.masked_fill(diagonal & ~fallback, -torch.inf).logsumexp(dim=1)
        loss = loss + (denominator - numerator).mean()

        for other_index, other_view in enumerate(features):
            if index == other_index:
                continue
            scores = (view @ other_view.T) / temperature
            denominator = scores.masked_fill(neighbors, -torch.inf).logsumexp(dim=1)
            loss = loss + (denominator - scores.diagonal()).mean()
    return loss


def correlation_reduction_loss(views):
    # Embed each view in its block of the concatenated feature space.
    global_features = F.normalize(torch.cat(views, dim=1), dim=1)
    n_nodes = global_features.shape[0]
    total = global_features.new_zeros(())
    offset = 0
    for view in views:
        end = offset + view.shape[1]
        similarity = F.normalize(view, dim=1) @ global_features[:, offset:end].T
        diagonal = similarity.diagonal()
        total = total + (diagonal - 1).square().mean()
        if n_nodes > 1:
            off_diagonal = similarity - torch.diag_embed(diagonal)
            total = total + off_diagonal.square().sum() / (n_nodes * (n_nodes - 1))
        offset = end
    return total
