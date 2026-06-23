import torch
import numpy as np
import torch.nn.functional as F

class CentroidAlignmentLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, labels, prototypes):
        """
        features: (Batch, 128)
        labels: (Batch, 14) - Multi-hot (0 or 1)
        prototypes: (14, 128)
        """
        # 1. Create the "Ideal Vector" for this patient
        # If labels=[1, 0, 1...], we average Prototype 0 and Prototype 2.
        # Matrix Multiply sums the vectors: (Batch, 14) x (14, 128) -> (Batch, 128)
        target_centroids = torch.matmul(labels.float(), prototypes)

        # 2. Normalize to project onto the unit sphere
        # This creates a target vector in the "middle" of the active diseases
        # Add epsilon to avoid div by zero for healthy patients (all 0s)
        target_centroids = F.normalize(target_centroids + 1e-8, p=2, dim=1)

        # 3. Filter out healthy patients (who have no target centroid)
        # Sum of labels > 0
        has_disease_mask = labels.sum(dim=1) > 0

        if has_disease_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # 4. MSE Loss between Feature and the Calculated Centroid
        valid_features = features[has_disease_mask]
        valid_targets = target_centroids[has_disease_mask]

        # We want Feature ~= Centroid
        loss = F.mse_loss(valid_features, valid_targets)

        return loss

