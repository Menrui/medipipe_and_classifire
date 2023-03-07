import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_length: int,
        num_classes: int,
        mlp_dim: int = 512,
        drop_rate: float = 0.2,
    ):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_length, mlp_dim),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, x) -> torch.Tensor:
        output = self.fc(x)
        return output
