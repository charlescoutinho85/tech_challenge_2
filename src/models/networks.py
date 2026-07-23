"""Arquiteturas de rede neural para o modelo de recomendação."""

import torch
import torch.nn as nn


class MLPEmbedding(nn.Module):
    """MLP com embeddings de usuário e item para previsão de rating."""

    def __init__(self, num_users: int, num_items: int, emb_dim: int = 32):
        """Inicializa as camadas de embedding e a rede totalmente conectada.

        Args:
            num_users: Número de usuários distintos.
            num_items: Número de itens distintos.
            emb_dim: Dimensão dos vetores de embedding.
        """
        super().__init__()
        self.u_emb = nn.Embedding(num_users, emb_dim)
        self.i_emb = nn.Embedding(num_items, emb_dim)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """Executa o forward pass, prevendo o rating para o par usuário/item.

        Args:
            user_idx: Índices de usuário.
            item_idx: Índices de item.

        Returns:
            Rating previsto para cada par usuário/item.
        """
        u = self.u_emb(user_idx)
        i = self.i_emb(item_idx)
        x = torch.cat([u, i], dim=1)
        return self.fc(x).squeeze()
