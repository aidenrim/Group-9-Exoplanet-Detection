from __future__ import annotations

import torch
from torch import nn


class TransitCNN(nn.Module):
    """
    A simple 1D CNN for classifying light curves as containing a transit or not. The architecture consists of three convolutional layers
    followed by a fully connected layer. The model is designed to take in light curves of a spoecified number of bins with 512 bins by default.
    The convolutional layers use ReLU activations, which is when the output of the convolution is passed through a non-linear activation function
    that sets all negative values to zero and keeps positive values unchanged.

    ReLU is used in this project because it helps the model learn complex patterns in the data while avoid issues like
    vanishing gradients, which are when the gradients become very small during training, making it difficult for the model to learn.
    """
    def __init__(self, input_bins: int = 512) -> None:
        """
        Initializes the TransitCNN model.


        Args:
            input_bins (int): The number of bins in the input light curve. Default is 512. This should match the number of bins used in the
            data preprocessing step.
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        flattened = 64 * (input_bins // 8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, 128),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model. The input tensor `x` is expected to have the shape (batch_size, 1, input_bins),
        where `input_bins` is the number of bins in the input light curve. The output is a tensor of shape where each element represents the logit (the row output of model)
        before applying the sigmoid function, which can be interpreted as the unnormalized log probability of the
        positive class (transit present). To get the probability of a transit, you would apply the sigmoid function to the output logits.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, input_bins
        Returns:
            torch.Tensor: Output tensor of shape (batch_size,) containing the logits for each input light curve.
        """
        x = self.features(x)
        return self.classifier(x).squeeze(-1)

