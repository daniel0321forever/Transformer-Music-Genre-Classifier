import torch
from torch import nn


class ResBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.features = dim
        self.kernel_size = 3
        self.padding = (self.kernel_size + 1) // 2 - 1

        self.res1 = nn.Sequential(
            nn.Conv1d(self.features, self.features,
                      self.kernel_size, padding=self.padding),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(self.features, self.features,
                      self.kernel_size, padding=self.padding),
            nn.LeakyReLU(),
        )

        self.res2 = nn.Sequential(
            nn.Conv1d(self.features, self.features,
                      self.kernel_size, padding=self.padding),
            nn.LeakyReLU(),
            nn.Conv1d(self.features, self.features,
                      self.kernel_size, padding=self.padding),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        res1 = self.res1(x)
        x = x + res1

        res2 = self.res2(x)
        y = x + res2

        return y


class ResCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.cat = 10

        self.frames = 431
        self.bins = 20
        self.latent_dim = 256
        self.reduced_dim = self.latent_dim // 4
        self.iter = 1

        # pooling
        self.pool_kernel_size = 3

        self.resblock = nn.Sequential(
            nn.Conv1d(self.bins, self.latent_dim, kernel_size=3, padding=1),
            ResBlock(dim=self.latent_dim),
        )

        self.conv13_0 = nn.Sequential(
            nn.Conv1d(self.latent_dim, self.reduced_dim, 1),
            nn.GELU(),
            nn.Dropout(),
            nn.Conv1d(self.reduced_dim, self.reduced_dim,
                      kernel_size=3, padding=1),
        )

        self.conv15_0 = nn.Sequential(
            nn.Conv1d(self.latent_dim, self.reduced_dim, 1),
            nn.GELU(),
            nn.Dropout(),
            nn.Conv1d(self.reduced_dim, self.reduced_dim,
                      kernel_size=5, padding=2),
        )

        self.conv1_0 = nn.Sequential(
            nn.Conv1d(self.latent_dim, self.reduced_dim, 1),
            nn.GELU(),
        )

        self.convM1_0 = nn.Sequential(
            nn.MaxPool1d(kernel_size=self.pool_kernel_size,
                         padding=1, dilation=1, stride=1),
            nn.Conv1d(self.latent_dim, self.reduced_dim, 1),
            nn.GELU(),
        )

        self.flatten = nn.Sequential(
            nn.AvgPool1d(kernel_size=3),
            nn.Conv1d(self.reduced_dim * 4, 1, kernel_size=1),
        )

        # could use globel pooling to get one
        in_features = (
            self.frames - self.pool_kernel_size) // self.pool_kernel_size + 1

        self.fc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.GELU(),
            nn.Linear(in_features=in_features, out_features=self.cat),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = (batch, bin, frames)
        x = self.resblock(x)  # x = (batch, features=128, frames=431)

        for i in range(self.iter):
            x1 = self.conv1_0(x)  # x = (batch, features=32, frames=431)
            x3 = self.conv13_0(x)  # x = (batch, features=32, frames=431)
            x5 = self.conv15_0(x)  # x = (batch, features=32, frames=431)
            x_M1 = self.convM1_0(x)  # x = (batch, features=32, frames=431)

            # concat
            # x = (batch, features=128, frames=431)
            x = torch.concat([x1, x3, x5, x_M1], dim=1)

        x_flat = self.flatten(x)  # x = (batch, 1, pooled_frames=143)
        x_flat = x_flat.squeeze(dim=1)  # x = (batch, pooled_frames=143)

        y = self.fc(x_flat)  # x = (batch, cat=10)
        return y
