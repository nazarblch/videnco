import torch
import torch.nn.functional as F
from torch import nn

__all__ = ['InfoNCE', 'info_nce']


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.
    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113
    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): (M, D) Tensor with negative samples (e.g. embeddings of other inputs).
            If None, then the negative keys for a sample are the positive keys for the other samples.
    Returns:
         Value of the InfoNCE Loss.
     Examples:
        >>> batch_size, embedding_size = 32, 128
        >>> loss = InfoNCE()
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(4 * batch_size, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys, temperature=self.temperature, reduction=self.reduction)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean'):
    # Inputs all have 2 dimensions.
    if query.dim() != 2 or positive_key.dim() != 2 or (negative_keys is not None and negative_keys.dim() != 2):
        raise ValueError('query, positive_key and negative_keys should all have 2 dimensions.')

    # Each query sample is paired with exactly one positive key sample.
    if len(query) != len(positive_key):
        raise ValueError('query and positive_key must have the same number of samples.')

    # Embedding vectors should have same number of components.
    if query.shape[1] != positive_key.shape[1] != (positive_key.shape[1] if negative_keys is None else negative_keys.shape[1]):
        raise ValueError('query, positive_key and negative_keys should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)

    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        # Cosine between all query-negative combinations
        negative_logits = query @ transpose(negative_keys)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


class PatchNCELoss(nn.Module):
    def __init__(self, nce_includes_all_negatives_from_batch: bool = False, nce_T: float = 0.1):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.nce_includes_all_negatives_from_batch = nce_includes_all_negatives_from_batch
        self.nce_T = nce_T

    def forward(self, feat_q, feat_k):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[2]
        n_patch = feat_q.shape[1]
        feat_q = F.normalize(feat_q, dim=-1)
        feat_k = F.normalize(feat_k, dim=-1)
        # feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.view(batchSize * n_patch, 1, -1), feat_k.view(batchSize * n_patch, -1, 1))
        l_pos = l_pos.view(batchSize * n_patch, 1)


        # reshape features to batch size
        feat_q = feat_q.view(batchSize, -1, dim)
        feat_k = feat_k.view(batchSize, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=torch.bool)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss


if __name__ == "__main__":

    fq = torch.randn(4, 100, 20)
    fk = torch.randn(4, 100, 20)

    loss = PatchNCELoss().forward(fq, fk)
    print(loss.shape)