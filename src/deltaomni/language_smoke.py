from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from deltaomni.backbones import load_backbone_config
from deltaomni.language import DeltaLanguageProjector, FrozenCausalCaptionBackend
from deltaomni.provenance import audit
from deltaomni.train_sanity import _atomic_json


def run(config_path: Path, provenance_path: Path) -> dict:
    config = load_backbone_config(config_path)
    provenance = audit(provenance_path)
    device = torch.device(config.device)
    backend = FrozenCausalCaptionBackend(
        config.language,
        config.cache_dir,
        device,
        provenance,
    )
    torch.manual_seed(42)
    projector = DeltaLanguageProjector(768, backend.hidden_size).to(device)
    anchor = torch.randn(1, config.language_smoke_anchor_tokens, 768, device=device)
    delta = torch.randn(1, config.video.delta_tokens, 768, device=device) * 0.1
    prompt = "Describe only the visual change: "
    caption = "An object moves to the right."
    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=config.language_smoke_learning_rate,
    )
    prefix = projector(anchor, delta, modality_index=1)
    initial_loss = backend.caption_loss(prefix, prompt, caption)
    for _ in range(config.language_smoke_steps):
        optimizer.zero_grad(set_to_none=True)
        prefix = projector(anchor, delta, modality_index=1)
        loss = backend.caption_loss(prefix, prompt, caption)
        loss.backward()
        optimizer.step()
    prefix = projector(anchor, delta, modality_index=1)
    final_loss = backend.caption_loss(prefix, prompt, caption)
    gradients = [
        parameter.grad
        for parameter in projector.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    checks = {
        "loss_finite": bool(torch.isfinite(initial_loss) and torch.isfinite(final_loss)),
        "loss_decreased": bool(final_loss < initial_loss),
        "projector_has_gradients": bool(gradients),
        "projector_gradients_finite": all(torch.isfinite(gradient).all() for gradient in gradients),
        "language_model_frozen": not any(
            parameter.requires_grad for parameter in backend.parameters()
        ),
        "prefix_shape_valid": prefix.shape == (
            1,
            config.language_smoke_anchor_tokens + config.video.delta_tokens,
            backend.hidden_size,
        ),
    }
    return {
        "model_id": config.language.model_id,
        "revision": config.language.revision,
        "hidden_size": backend.hidden_size,
        "prefix_shape": list(prefix.shape),
        "smoke_steps": config.language_smoke_steps,
        "initial_caption_loss": float(initial_loss.detach()),
        "final_caption_loss": float(final_loss.detach()),
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test delta prefix to frozen causal LM")
    parser.add_argument("--config", type=Path, default=Path("configs/backbones.yaml"))
    parser.add_argument("--provenance", type=Path, default=Path("configs/provenance.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/reports/language_smoke.json"))
    args = parser.parse_args()
    report = run(args.config, args.provenance)
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
