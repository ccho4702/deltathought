import torch

from deltaomni.semantic_tokens import SemanticTokenBottleneck, assignment_statistics


def test_semantic_token_bottleneck_supports_soft_and_hard_interfaces() -> None:
    model = SemanticTokenBottleneck(
        input_dim=8,
        hidden_dim=16,
        token_count=2,
        codebook_size=6,
        classes=4,
        num_heads=4,
    )
    delta = torch.randn(3, 5, 8)

    soft = model(delta, modality_index=1, temperature=0.7, hard=False)
    hard = model(delta, modality_index=1, temperature=0.7, hard=True)

    assert soft.tokens.shape == hard.tokens.shape == (3, 2, 16)
    assert soft.class_logits.shape == hard.class_logits.shape == (3, 4)
    assert soft.assignment_probabilities.shape == (3, 2, 6)
    assert torch.allclose(soft.assignment_probabilities.sum(dim=-1), torch.ones(3, 2))
    hard.class_logits.sum().backward()
    assert model.assignment[-1].weight.grad is not None


def test_assignment_statistics_detect_codebook_collapse() -> None:
    uniform = torch.full((4, 1, 4), 0.25)
    collapsed = torch.zeros(4, 1, 4)
    collapsed[..., 0] = 1

    uniform_stats = assignment_statistics(uniform)
    collapsed_stats = assignment_statistics(collapsed)

    assert uniform_stats["effective_codes"] > collapsed_stats["effective_codes"]
