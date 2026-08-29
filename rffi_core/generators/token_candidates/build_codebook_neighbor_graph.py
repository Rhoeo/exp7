"""Build the frozen-codebook candidate prior without using final-test records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from rffi_core.generators.token_candidates.token_candidate_graph import (
    TokenCandidateGraph,
    mutual_neighbor_mask,
    nearest_neighbor_arrays,
    undirected_component_sizes,
)


ROOT = Path(__file__).resolve().parents[3]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_codebook(checkpoint_path: Path):
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    embeddings = checkpoint["model_state"]["quantizer.embedding.weight"]
    return embeddings.detach().cpu().numpy().astype(np.float32), checkpoint


def generator_train_indices(window_index: Path) -> np.ndarray:
    selected = []
    with window_index.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "generator_train":
                selected.append(int(row["cache_index"]))
    return np.asarray(selected, dtype=np.int64)


def token_statistics(tokens, indices, codebook_size: int, chunk_rows: int = 256):
    usage = np.zeros(codebook_size, dtype=np.int64)
    transitions = np.zeros((codebook_size, codebook_size), dtype=np.int64)
    for start in range(0, len(indices), chunk_rows):
        batch_indices = indices[start : start + chunk_rows]
        batch = np.asarray(tokens[batch_indices], dtype=np.int64)
        usage += np.bincount(batch.reshape(-1), minlength=codebook_size)
        pair_ids = batch[:, :-1] * codebook_size + batch[:, 1:]
        transitions += np.bincount(
            pair_ids.reshape(-1), minlength=codebook_size * codebook_size
        ).reshape(codebook_size, codebook_size)
    return usage, transitions


def percentile_dict(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default="runs/stage_g1/frozen/wifib_v1/vqvae_p1_k1024.pt"
    )
    parser.add_argument(
        "--token-cache",
        default=r"E:\data_cache\rffi_v1\tokens\wifib_vq_p1_k1024\tokens.npy",
    )
    parser.add_argument(
        "--window-index", default=r"E:\data_cache\rffi_v1\wifib\window_index.csv"
    )
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument(
        "--output",
        default="artifacts/token_graph/wifib_vq_p1_k1024_neighbors.npz",
    )
    args = parser.parse_args()

    checkpoint_path = (ROOT / args.checkpoint).resolve()
    token_path = Path(args.token_cache).resolve()
    index_path = Path(args.window_index).resolve()
    output_path = (ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings, checkpoint = load_codebook(checkpoint_path)
    neighbors, distances = nearest_neighbor_arrays(embeddings, args.top_k)
    mutual = mutual_neighbor_mask(neighbors)
    train_indices = generator_train_indices(index_path)
    tokens = np.load(str(token_path), mmap_mode="r", allow_pickle=False)
    usage, transitions = token_statistics(tokens, train_indices, embeddings.shape[0])
    row_indices = np.arange(embeddings.shape[0])[:, None]
    forward = transitions[row_indices, neighbors]
    backward = transitions[neighbors, row_indices]
    metadata = {
        "schema_version": "wifib-token-candidate-prior-v2",
        "graph_role": "static coarse prior only",
        "codebook_size": int(embeddings.shape[0]),
        "latent_dim": int(embeddings.shape[1]),
        "top_k": int(args.top_k),
        "top_k_subsets": [16, 32],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "token_cache": str(token_path),
        "token_cache_sha256": sha256_file(token_path),
        "source_split_for_usage_and_transitions": "generator_train",
        "source_records": int(len(train_indices)),
        "final_test_used_for_graph_statistics": False,
        "decoder_context_note": (
            "Current frozen patch=1 polyphase decoder is pointwise, but this graph "
            "is deliberately not treated as waveform-validity proof."
        ),
        "runtime_required_filters": [
            "masked-infill relative probability",
            "actual decoded waveform RF precheck",
        ],
    }
    np.savez_compressed(
        str(output_path),
        neighbors=neighbors,
        latent_distances=distances,
        mutual_neighbors=mutual,
        usage_counts=usage,
        forward_transition_counts=forward,
        backward_transition_counts=backward,
        transition_counts=transitions,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    graph = TokenCandidateGraph.load(output_path)
    component_16 = undirected_component_sizes(neighbors[:, :16])
    component_32 = undirected_component_sizes(neighbors)
    usage_prob = usage.astype(np.float64) / max(1, usage.sum())
    nonzero = usage_prob[usage_prob > 0]
    perplexity = float(np.exp(-np.sum(nonzero * np.log(nonzero))))
    diagnostics = {
        "schema_version": "wifib-token-graph-diagnostics-v1",
        "graph": str(output_path),
        "graph_sha256": sha256_file(output_path),
        "metadata": metadata,
        "shape": list(neighbors.shape),
        "active_codes": int(np.count_nonzero(usage)),
        "usage_perplexity": perplexity,
        "usage_count_distribution": percentile_dict(usage),
        "latent_distance_top16": percentile_dict(distances[:, :16]),
        "latent_distance_top32": percentile_dict(distances),
        "mutual_fraction_top16": float(mutual[:, :16].mean()),
        "mutual_fraction_top32": float(mutual.mean()),
        "transition_supported_fraction_top16": float(
            ((forward[:, :16] + backward[:, :16]) > 0).mean()
        ),
        "transition_supported_fraction_top32": float(
            ((forward + backward) > 0).mean()
        ),
        "components_top16": component_16,
        "components_top32": component_32,
        "self_edges": int(np.sum(neighbors == np.arange(graph.codebook_size)[:, None])),
        "candidate_prior_only": True,
        "waveform_validity_proven": False,
    }
    gate_passed = (
        graph.validate()
        and diagnostics["active_codes"] == graph.codebook_size
        and diagnostics["self_edges"] == 0
        and component_32[0] == graph.codebook_size
    )
    report_dir = ROOT / "reports/next_stage"
    report_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = report_dir / "token_graph_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = [
        "# P2 Token Candidate Prior Diagnostics",
        "",
        "- Gate: **%s**" % ("PASS" if gate_passed else "FAIL"),
        "- Graph shape: `%s`" % (tuple(neighbors.shape),),
        "- Active generator-train codes: %d/%d" % (diagnostics["active_codes"], graph.codebook_size),
        "- Usage perplexity: %.2f" % perplexity,
        "- Mutual edge fraction (top-16/top-32): %.3f / %.3f"
        % (diagnostics["mutual_fraction_top16"], diagnostics["mutual_fraction_top32"]),
        "- Transition-supported edge fraction (top-16/top-32): %.3f / %.3f"
        % (
            diagnostics["transition_supported_fraction_top16"],
            diagnostics["transition_supported_fraction_top32"],
        ),
        "- Largest undirected component (top-16/top-32): %d / %d"
        % (component_16[0], component_32[0]),
        "",
        "This artifact is a static coarse prior. It is not a proof that a replacement is waveform-valid; P3 relative-probability and P4 actual-decoding checks remain mandatory.",
    ]
    summary_path = report_dir / "token_graph_diagnostics.md"
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")

    p0 = json.loads((report_dir / "current_state_manifest.json").read_text(encoding="utf-8"))
    stage_report = {
        "schema_version": "rffi-next-stage-report-v1",
        "stage": "P2_candidate_prior_graph",
        "git_commit": None,
        "config_hash": hashlib.sha256(
            json.dumps(vars(args), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "data_split_hash": json.loads(
            (ROOT / "configs/data/wifib_next_stage_splits.json").read_text(encoding="utf-8")
        )["role_assignment_sha256"],
        "checkpoint_hash": p0["artifacts"]["vq_codec"]["sha256"],
        "seeds": [],
        "source_count": int(len(train_indices)),
        "victim_query_budget": 0,
        "edit_budget": 0,
        "metrics": diagnostics,
        "gate": {
            "status": "PASS" if gate_passed else "FAIL",
            "reason": (
                "Candidate prior is complete, deterministic, active, connected, and self-edge free."
                if gate_passed
                else "Candidate prior integrity checks failed."
            ),
        },
        "artifacts": {
            "graph": str(output_path),
            "diagnostics_json": str(diagnostics_path),
            "diagnostics_markdown": str(summary_path),
        },
        "next_allowed_stage": "P3_masked_infill" if gate_passed else None,
        "prohibited_actions": [
            "treat static graph membership as waveform validity",
            "fall back to unrestricted codebook replacement",
            "access policy_gate attack outcomes",
            "access final_test signal data",
            "train PPO",
        ],
    }
    stage_path = report_dir / "p2_stage_report.json"
    stage_path.write_text(
        json.dumps(stage_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"graph": str(output_path), "gate": stage_report["gate"]}, indent=2))
    if not gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
