import torch
import torch.nn as nn


class MLPEmbedding(nn.Module):
    def __init__(self, num_users: int, num_items: int, emb_dim: int = 32):
        super().__init__()
        self.u_emb = nn.Embedding(num_users, emb_dim)
        self.i_emb = nn.Embedding(num_items, emb_dim)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        u = self.u_emb(user_idx)
        i = self.i_emb(item_idx)
        x = torch.cat([u, i], dim=1)
        return self.fc(x).squeeze()