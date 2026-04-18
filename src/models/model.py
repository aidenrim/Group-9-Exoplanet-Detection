"""
Phase 3: Model
==============

Dual-branch 1D CNN for Kepler planet-candidate classification.

Architecture overview
---------------------

    Global view (201,) --> Branch G --> flatten (1 600,) ---|
                                                            |--> head --> logit (1,)
    Local view  (61,)  --> Branch L --> flatten (480,)   ---|

Each branch is a stack of Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d blocks.
The two flattened feature vectors are concatenated and passed through a small
fully-connected classifier.

Global view captures context (noise, secondary eclipses, variability).
Local view captures transit shape (flat vs U vs V-shaped, symmetry).

"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d.

    Args:
        in_channels:  Number of input feature maps.
        out_channels: Number of output feature maps (filter count).
        kernel_size:  Convolutional kernel width (default 5).
        pool_size:    MaxPool stride/kernel (default 2, halves sequence length).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        pool_size: int = 2,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ExoplanetCNN(nn.Module):
    """
    Dual-branch 1D CNN for exoplanet transit classification.

    Input
    -----
    global_view : (batch, global_len)   phase-folded, baseline-centred
    local_view  : (batch,  local_len)   zoomed transit region

    Output
    ------
    logits : (batch, 1) raw un-activated scores.
             Apply torch.sigmoid() to obtain probabilities, or pass directly
             to BCEWithLogitsLoss during training.

    Args:
        global_len:  Length of the global-view input
        local_len:   Length of the local-view input
        dropout:     Dropout probability in the classifier head (default 0.5).
                     Dropout is the primary regularisation mechanism; it
                     randomly zeroes half the activations each forward pass
                     during training, forcing the network to not rely on any
                     single feature.
    """

    def __init__(
        self,
        global_len: int = 201,
        local_len: int = 61,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        # Global branch  (3 conv blocks)
        #
        # Input length 201 evolves as:
        #   After block 1: floor(201 / 2) = 100   (channels: 1  -> 16)
        #   After block 2: floor(100 / 2) =  50   (channels: 16 -> 32)
        #   After block 3: floor( 50 / 2) =  25   (channels: 32 -> 64)
        #   Flatten: 64 × 25 = 1600
        self.global_branch = nn.Sequential(
            ConvBlock(1,  16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
        )

        # Local branch  (2 conv blocks)
        #
        # Input length 61 evolves as:
        #   After block 1: floor(61 / 2) = 30   (channels: 1  -> 16)
        #   After block 2: floor(30 / 2) = 15   (channels: 16 -> 32)
        #   Flatten: 32 × 15 = 480
        self.local_branch = nn.Sequential(
            ConvBlock(1,  16),
            ConvBlock(16, 32),
        )

        # Compute the flattened sizes.
        with torch.no_grad():
            dummy_g = torch.zeros(1, 1, global_len)
            dummy_l = torch.zeros(1, 1, local_len)
            g_flat = self.global_branch(dummy_g).flatten(1).shape[1]
            l_flat = self.local_branch(dummy_l).flatten(1).shape[1]

        combined_size = g_flat + l_flat   # 1600 + 480 = 2080 by default

        # Classifier head
        self.head = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1),      # raw logit (no sigmoid)
        )

    def forward(
        self,
        global_view: torch.Tensor,
        local_view: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            global_view: (batch, global_len)
            local_view:  (batch,  local_len)

        Returns:
            logits: (batch, 1)
        """
        g = self.global_branch(global_view.unsqueeze(1))   # (B, 64, 25)
        l = self.local_branch(local_view.unsqueeze(1))     # (B, 32, 15)

        g = g.flatten(1)    # (B, 1 600)
        l = l.flatten(1)    # (B,   480)

        combined = torch.cat([g, l], dim=1)   # (B, 2 080)
        return self.head(combined)            # (B, 1)

    def predict_proba(
        self,
        global_view: torch.Tensor,
        local_view: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return calibrated probabilities in [0, 1].

        Convenience wrapper around forward() + sigmoid for inference.
        Not used during training (BCEWithLogitsLoss applies sigmoid internally).
        """
        return torch.sigmoid(self.forward(global_view, local_view))
