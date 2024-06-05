import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        """
        Focal Loss for binary classification.
        Args:
            alpha (float): The weighting factor for the rare class (usually positive class).
            gamma (float): Focusing parameter to adjust the rate at which easy examples are down-weighted.
            reduction (str): Specifies the reduction to apply to the output: 'none', 'mean', 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute the focal loss between `inputs` and the ground truth `targets`.
        Args:
            inputs (tensor): Tensor of predictions (logits) from the model, of shape [batch_size,].
            targets (tensor): Ground truth labels, of shape [batch_size,].
        """
        # Sigmoid to convert logits to probabilities
        # p_t = torch.sigmoid(inputs)
        p_t = inputs

        # Focal Loss formula
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = torch.where(
            targets == 1, p_t, 1 - p_t
        )  # p_t if targets are 1, else 1-p_t
        loss = (
            ce_loss
            * ((1 - p_t) ** self.gamma)
            * (torch.where(targets == 1, self.alpha, 1 - self.alpha))
        )

        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss.mean()
