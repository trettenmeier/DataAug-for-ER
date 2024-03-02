import torch
import numpy as np


def apply_mixup(embeddings: torch.Tensor, labels: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Applies mixup separated by pairs. The batch size is kept constant. We take half
    of the original data points and all the augmented datapoints.
    """
    alpha = 0.3
    lambda_value = np.random.beta(alpha, alpha)

    # uneven number of datapoints in batch:
    if len(embeddings) % 2 != 0:
        maximum = len(embeddings) - 1
    else:
        maximum = len(embeddings)

    splitting_index = int(len(embeddings) / 2)

    upper = embeddings[:splitting_index]
    lower = embeddings[splitting_index:maximum]

    new = lambda_value * upper + (1 - lambda_value) * lower

    labels_upper = labels[:splitting_index]
    labels_lower = labels[splitting_index:maximum]

    new_labels = lambda_value * labels_upper + (1 - lambda_value) * labels_lower

    return (
        torch.concat([upper, new]),
        torch.concat([labels_upper, new_labels])
    )
