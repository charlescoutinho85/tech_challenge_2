"""Testes para a arquitetura de rede neural MLPEmbedding."""

import torch

from src.models.networks import MLPEmbedding


def test_forward_output_shape_matches_batch_size() -> None:
    model = MLPEmbedding(num_users=5, num_items=5, emb_dim=4)
    user_idx = torch.tensor([0, 1, 2])
    item_idx = torch.tensor([0, 1, 2])

    output = model(user_idx, item_idx)

    assert output.shape == (3,)


def test_forward_is_deterministic_with_fixed_seed() -> None:
    torch.manual_seed(42)
    model_a = MLPEmbedding(num_users=5, num_items=5, emb_dim=4)

    torch.manual_seed(42)
    model_b = MLPEmbedding(num_users=5, num_items=5, emb_dim=4)

    user_idx = torch.tensor([0, 1])
    item_idx = torch.tensor([0, 1])

    torch.testing.assert_close(model_a(user_idx, item_idx), model_b(user_idx, item_idx))
