"""
Phase 3: Model
==============

Dual-branch 1D CNN for Kepler planet-candidate classification.

Architecture overview
---------------------

    Global view (201,) ──► Branch G ──► flatten (1 600,) ──┐
                                                            ├──► head ──► logit (1,)
    Local  view  (61,) ──► Branch L ──►  flatten  (480,) ──┘

Each branch is a stack of Conv1d → BatchNorm1d → ReLU → MaxPool1d blocks.
The two flattened feature vectors are concatenated and passed through a small
fully-connected classifier.

Why two branches?
-----------------
The global view (201 bins across the full phase range) gives the network
context: how noisy is this lightcurve?  Is there a secondary eclipse at
phase ±0.5 that would flag an eclipsing binary?  Are there ellipsoidal
brightness variations indicating a stellar companion?

The local view (61 bins zoomed into ±2 transit durations) gives fine
morphology: is the transit flat-bottomed (planet limb darkening profile)
or V-shaped (grazing eclipsing binary)?  Is ingress/egress symmetric?

A single branch over the full phase range would require extremely narrow
convolutional kernels to resolve transit shape, while losing sensitivity
to the broader contextual signals.  Two branches with different effective
resolutions handle both scales cleanly — the same motivation behind the
AstroNet architecture (Shallue & Vanderburg 2018).

Why BatchNorm?
--------------
BatchNorm normalises the distribution of each feature map across the batch
after every convolution.  This prevents the "internal covariate shift"
problem where changing weights in early layers shift the input distribution
seen by later layers, forcing them to constantly re-adapt.  Practically,
BatchNorm allows training with higher learning rates and makes the network
less sensitive to weight initialisation.

Why raw logits (no sigmoid in forward)?
----------------------------------------
BCEWithLogitsLoss fuses the sigmoid and the binary cross-entropy into a
single numerically-stable operation using the log-sum-exp trick.  Calling
sigmoid() explicitly in the model and then using BCELoss is equivalent but
more prone to vanishing gradients near 0 and 1.  Probability predictions
at inference time use .predict_proba(), which applies sigmoid explicitly.

Dynamic flattened-size computation
-----------------------------------
Rather than hardcoding the flattened sizes (1 600 and 480), __init__ runs
a dummy tensor of the correct length through each branch and measures the
output.  This means the model still assembles correctly if GLOBAL_BINS or
LOCAL_BINS in preprocess.py are ever changed.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """
    Conv1d → BatchNorm1d → ReLU → MaxPool1d.

    Uses 'same' padding (padding = kernel_size // 2) so the convolution
    does not shrink the sequence length; all length reduction comes from
    MaxPool1d, making the size arithmetic easy to reason about.

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
                padding=kernel_size // 2,   # 'same' padding
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class ExoplanetCNN(nn.Module):
    """
    Dual-branch 1D CNN for exoplanet transit classification.

    Input
    -----
    global_view : (batch, global_len)   phase-folded, baseline-centred
    local_view  : (batch,  local_len)   zoomed transit region

    Output
    ------
    logits : (batch, 1)   raw un-activated scores.
             Apply torch.sigmoid() to obtain probabilities, or pass directly
             to BCEWithLogitsLoss during training.

    Args:
        global_len:  Length of the global-view input (must match GLOBAL_BINS
                     in preprocess.py, default 201).
        local_len:   Length of the local-view input (must match LOCAL_BINS
                     in preprocess.py, default 61).
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

        # ------------------------------------------------------------------
        # Global branch  (3 conv blocks)
        #
        # Input length 201 evolves as:
        #   After block 1: floor(201 / 2) = 100   (channels: 1  → 16)
        #   After block 2: floor(100 / 2) =  50   (channels: 16 → 32)
        #   After block 3: floor( 50 / 2) =  25   (channels: 32 → 64)
        #   Flatten: 64 × 25 = 1 600
        # ------------------------------------------------------------------
        self.global_branch = nn.Sequential(
            ConvBlock(1,  16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
        )

        # ------------------------------------------------------------------
        # Local branch  (2 conv blocks)
        #
        # Input length 61 evolves as:
        #   After block 1: floor(61 / 2) = 30   (channels: 1  → 16)
        #   After block 2: floor(30 / 2) = 15   (channels: 16 → 32)
        #   Flatten: 32 × 15 = 480
        # ------------------------------------------------------------------
        self.local_branch = nn.Sequential(
            ConvBlock(1,  16),
            ConvBlock(16, 32),
        )

        # ------------------------------------------------------------------
        # Dynamically compute the flattened sizes.
        # ------------------------------------------------------------------
        with torch.no_grad():
            dummy_g = torch.zeros(1, 1, global_len)
            dummy_l = torch.zeros(1, 1, local_len)
            g_flat = self.global_branch(dummy_g).flatten(1).shape[1]
            l_flat = self.local_branch(dummy_l).flatten(1).shape[1]

        combined_size = g_flat + l_flat   # 1 600 + 480 = 2 080 by default

        # ------------------------------------------------------------------
        # Classifier head
        # ------------------------------------------------------------------
        self.head = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1),          # raw logit; no sigmoid here
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
        # unsqueeze(1) adds the channel dimension: (B, L) → (B, 1, L)
        g = self.global_branch(global_view.unsqueeze(1))   # (B, 64, 25)
        l = self.local_branch(local_view.unsqueeze(1))     # (B, 32, 15)

        # Flatten spatial dimension: (B, C, L) → (B, C*L)
        g = g.flatten(1)    # (B, 1 600)
        l = l.flatten(1)    # (B,   480)

        combined = torch.cat([g, l], dim=1)   # (B, 2 080)
        return self.head(combined)             # (B, 1)

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
