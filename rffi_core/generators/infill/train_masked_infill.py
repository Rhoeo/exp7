"""Train the P3 local masked-infill smoke model without classifier feedback."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rffi_core.generators.infill.data import MaskedTokenWindowDataset
from rffi_core.generators.infill.context_prior import BidirectionalTransitionPrior
from rffi_core.generators.infill.models import build_infill_model, masked_cross_entropy


ROOT = Path(__file__).resolve().parents[3]


def sha256_file(path: Path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def fused_masked_logits(logits, targets, input_tokens, indicator, prior, weight):
    mask = targets != -100
    model_logits = logits[mask]
    prior_logits = prior.scores(input_tokens, indicator).to(model_logits.device)
    if prior_logits.shape != model_logits.shape:
        raise RuntimeError("transition-prior and masked-logit shapes differ")
    return model_logits + float(weight) * prior_logits, targets[mask]


def evaluate_loss(model, loader, device, prior, prior_weight):
    model.eval()
    total_loss = 0.0
    total_targets = 0
    correct = 0
    top5_correct = 0
    with torch.inference_mode():
        for batch in loader:
            tokens = batch["tokens"].to(device)
            targets = batch["targets"].to(device)
            indicator = batch["target_indicator"].to(device)
            global_positions = batch["global_positions"].to(device)
            device_labels = batch["device_label"].to(device)
            logits = model(tokens, device_labels, global_positions, indicator)
            masked_logits, masked_targets = fused_masked_logits(
                logits, targets, tokens, indicator, prior, prior_weight
            )
            loss = torch.nn.functional.cross_entropy(masked_logits, masked_targets)
            count = int(masked_targets.numel())
            total_loss += float(loss) * count
            total_targets += count
            correct += int((masked_logits.argmax(dim=1) == masked_targets).sum())
            top5_correct += int(
                (masked_logits.topk(5, dim=1).indices == masked_targets[:, None])
                .any(dim=1)
                .sum()
            )
    return {
        "loss": total_loss / max(1, total_targets),
        "masked_targets": total_targets,
        "top1_accuracy": correct / max(1, total_targets),
        "top5_accuracy": top5_correct / max(1, total_targets),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/generator/wifib_masked_infill_v1.yaml"
    )
    args = parser.parse_args()
    config_path = resolve(ROOT, args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    device = torch.device(config["training"]["device"])
    data = config["data"]
    common = {
        "token_cache_path": resolve(ROOT, data["token_cache"]),
        "window_index_path": resolve(ROOT, data["window_index"]),
        "split_config_path": resolve(ROOT, data["split_config"]),
        "candidate_graph_path": resolve(ROOT, data["candidate_graph"]),
        "context_length": config["model"]["context_length"],
        "seed": seed,
        "neighbor_corruption_probability": config["training"][
            "neighbor_corruption_probability"
        ],
        "transition_supported": config["training"]["transition_supported"],
        "candidate_top_k": config["training"]["candidate_top_k"],
    }
    train_dataset = MaskedTokenWindowDataset(
        role=data["train_role"],
        max_samples=config["training"]["max_train_samples"],
        **common,
    )
    validation_dataset = MaskedTokenWindowDataset(
        role=data["validation_role"],
        max_samples=config["training"]["max_validation_samples"],
        **common,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    model = build_infill_model(config["model"]).to(device)
    prior = BidirectionalTransitionPrior(
        train_dataset.graph.transition_counts,
        train_dataset.graph.arrays["usage_counts"],
        alpha=config["training"]["transition_prior_alpha"],
    ).to(device)
    prior_weight = config["training"]["transition_prior_weight"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    output_dir = resolve(ROOT, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_loss = float("inf")
    started = time.time()
    for epoch in range(int(config["training"]["epochs"])):
        train_dataset.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        train_targets = 0
        for batch in train_loader:
            tokens = batch["tokens"].to(device)
            targets = batch["targets"].to(device)
            indicator = batch["target_indicator"].to(device)
            global_positions = batch["global_positions"].to(device)
            device_labels = batch["device_label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens, device_labels, global_positions, indicator)
            masked_logits, masked_targets = fused_masked_logits(
                logits, targets, tokens, indicator, prior, prior_weight
            )
            loss = torch.nn.functional.cross_entropy(masked_logits, masked_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(masked_targets.numel())
            train_loss += float(loss.detach()) * count
            train_targets += count
        validation = evaluate_loss(model, validation_loader, device, prior, prior_weight)
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(1, train_targets),
            "train_masked_targets": train_targets,
            "validation": validation,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        checkpoint = {
            "schema_version": "wifib-masked-infill-checkpoint-v1",
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": config["model"],
            "config": config,
            "config_sha256": sha256_file(config_path),
            "label_map": train_dataset.label_map,
            "classifier_feedback": False,
            "policy_gate_attack_feedback": False,
            "final_test_used": False,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            torch.save(checkpoint, output_dir / "best.pt")
    summary = {
        "status": "smoke_complete",
        "run_mode": config["run_mode"],
        "seconds": time.time() - started,
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "history": history,
        "best_validation_loss": best_loss,
        "classifier_feedback": False,
        "policy_gate_attack_feedback": False,
        "final_test_used": False,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
