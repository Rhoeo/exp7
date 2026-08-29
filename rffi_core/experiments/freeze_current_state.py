"""Create the immutable P0 provenance snapshot for the next RFFI stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = Path(r"E:\data_cache\rffi_v1")
REPORT_DIR = ROOT / "reports" / "next_stage"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def describe_file(path: Path, role: str, status: str) -> dict:
    return {
        "path": str(path),
        "role": role,
        "status": status,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def combined_hash(items: list[dict]) -> str:
    digest = hashlib.sha256()
    for item in sorted(items, key=lambda value: value["path"]):
        digest.update(item["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(item["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def source_tree() -> tuple[list[dict], str]:
    paths: list[Path] = []
    for pattern in (
        "rffi_core/**/*.py",
        "tests/**/*.py",
        "configs/**/*.json",
        "configs/**/*.yaml",
        "configs/**/*.yml",
    ):
        paths.extend(ROOT.glob(pattern))
    items = []
    for path in sorted(set(paths)):
        if "__pycache__" in path.parts:
            continue
        item = describe_file(path, "source_or_config", "current")
        item["relative_path"] = path.relative_to(ROOT).as_posix()
        items.append(item)
    digest = hashlib.sha256()
    for item in sorted(items, key=lambda value: value["relative_path"]):
        digest.update(item["relative_path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(item["sha256"].encode("ascii"))
        digest.update(b"\n")
    return items, digest.hexdigest()


def git_state() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=str(ROOT),
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"is_repository": True, "commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"is_repository": False, "commit": None, "dirty": None}


def disk_state(root: str) -> dict:
    usage = shutil.disk_usage(root)
    return {
        "root": root,
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def verify_embedded_hashes(artifacts: dict[str, dict]) -> list[dict]:
    checks = []
    frozen_models = json.loads(
        (ROOT / "runs/stage_g0/frozen/wifib_v1/frozen_models.json").read_text(
            encoding="utf-8"
        )
    )
    frozen_tokenizer = json.loads(
        (ROOT / "runs/stage_g1/frozen/wifib_v1/frozen_tokenizer.json").read_text(
            encoding="utf-8"
        )
    )
    expected = {
        "victim_a": frozen_models["models"]["victim_a"]["checkpoint_sha256"],
        "evaluator_b": frozen_models["models"]["evaluator_b"]["checkpoint_sha256"],
        "vq_codec": frozen_tokenizer["checkpoint_sha256"],
        "data_config": frozen_models["data"]["config_sha256"],
        "manifest_checksums": frozen_models["data"]["manifest_checksums_sha256"],
    }
    for name, expected_hash in expected.items():
        actual_hash = artifacts[name]["sha256"]
        checks.append(
            {
                "name": name,
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "passed": actual_hash == expected_hash,
            }
        )
    return checks


def build_manifest(test_count: int, test_status: str, test_command: str) -> dict:
    artifact_specs = {
        "data_config": (
            ROOT / "configs/data/rffi_data_v1.json",
            "frozen_data_config",
            "frozen",
        ),
        "manifest_checksums": (
            CACHE_ROOT / "manifests/manifest_checksums.json",
            "data_manifest_checksums",
            "frozen",
        ),
        "dataset_audit": (
            CACHE_ROOT / "manifests/dataset_audit.json",
            "data_audit",
            "frozen",
        ),
        "wifib_cache_metadata": (
            CACHE_ROOT / "wifib/cache_metadata.json",
            "iq_cache_metadata",
            "frozen",
        ),
        "wifib_cache_validation": (
            CACHE_ROOT / "wifib/cache_validation.json",
            "iq_cache_validation",
            "frozen",
        ),
        "token_cache_metadata": (
            CACHE_ROOT / "tokens/wifib_vq_p1_k1024/metadata.json",
            "token_cache_metadata",
            "frozen",
        ),
        "victim_a": (
            ROOT / "runs/stage_g0/frozen/wifib_v1/victim_a.pt",
            "attack_visible_victim",
            "frozen",
        ),
        "evaluator_b": (
            ROOT / "runs/stage_g0/frozen/wifib_v1/evaluator_b.pt",
            "preliminary_identity_auditor_only",
            "frozen",
        ),
        "frozen_models_record": (
            ROOT / "runs/stage_g0/frozen/wifib_v1/frozen_models.json",
            "model_provenance",
            "frozen",
        ),
        "vq_codec": (
            ROOT / "runs/stage_g1/frozen/wifib_v1/vqvae_p1_k1024.pt",
            "high_fidelity_reconstruction_codec",
            "frozen_not_ppo_ready",
        ),
        "frozen_codec_record": (
            ROOT / "runs/stage_g1/frozen/wifib_v1/frozen_tokenizer.json",
            "codec_provenance",
            "frozen",
        ),
        "rfgpt_pilot": (
            ROOT / "runs/stage_g3/pilot_rfgpt_4k/best.pt",
            "teacher_forced_diagnostic_and_local_baseline_only",
            "frozen_no_go_long_rollout",
        ),
        "rfgpt_pilot_config": (
            ROOT / "runs/stage_g3/pilot_rfgpt_4k/run_config.json",
            "diagnostic_config",
            "frozen",
        ),
        "progress_report": (
            ROOT / "reports/EXPERIMENT_PROGRESS_REPORT_2026-08-28.md",
            "historical_report",
            "frozen",
        ),
        "g1_report": (
            ROOT / "reports/stage_g1/WiFiB_Stage_G1_Report.md",
            "historical_report",
            "frozen",
        ),
        "g4_report": (
            ROOT / "reports/stage_g4/WiFiB_Stage_G4_Hardness_Report.md",
            "historical_report",
            "frozen_diagnostic_only",
        ),
        "expert_execution_spec": (
            ROOT / "RFFI_NEXT_STAGE_CODEX_EXECUTION_SPEC_2026-08-28.md",
            "expert_input",
            "reference",
        ),
        "next_stage_plan": (
            ROOT / "reports/next_stage/NEXT_STAGE_IMPLEMENTATION_PLAN.md",
            "approved_execution_plan",
            "current",
        ),
    }
    missing = [str(path) for path, _, _ in artifact_specs.values() if not path.is_file()]
    artifacts = {
        name: describe_file(path, role, status)
        for name, (path, role, status) in artifact_specs.items()
        if path.is_file()
    }
    verification = verify_embedded_hashes(artifacts) if not missing else []
    tree_items, tree_hash = source_tree()
    checkpoint_items = [
        artifacts[name]
        for name in ("victim_a", "evaluator_b", "vq_codec", "rfgpt_pilot")
        if name in artifacts
    ]
    gate_passed = (
        not missing
        and test_status.lower() == "pass"
        and test_count >= 17
        and all(item["passed"] for item in verification)
    )
    return {
        "schema_version": "rffi-next-stage-report-v1",
        "stage": "P0_current_state_freeze",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_state()["commit"],
        "git": git_state(),
        "source_tree_hash": tree_hash,
        "config_hash": artifacts.get("data_config", {}).get("sha256"),
        "data_split_hash": artifacts.get("manifest_checksums", {}).get("sha256"),
        "checkpoint_hash": combined_hash(checkpoint_items),
        "seeds": [20260828, 20260830],
        "source_count": None,
        "victim_query_budget": 0,
        "edit_budget": 0,
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "torch_cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu_count": os.cpu_count(),
            "disks": [disk_state("D:\\"), disk_state("E:\\")],
        },
        "tests": {
            "status": test_status.lower(),
            "count": test_count,
            "command": test_command,
        },
        "metrics": {
            "frozen_hash_checks_passed": sum(item["passed"] for item in verification),
            "frozen_hash_checks_total": len(verification),
            "source_tree_files": len(tree_items),
        },
        "artifacts": artifacts,
        "source_tree": tree_items,
        "embedded_hash_verification": verification,
        "missing_artifacts": missing,
        "historical_final_test_access": {
            "accessed_before_P0": True,
            "evidence": "Frozen G0 provenance contains baseline final_test metrics.",
            "used_for_checkpoint_selection": False,
            "policy_from_P0_onward": "sealed; do not recompute or use for selection until final protocol is frozen",
        },
        "formal_conclusions": {
            "vq_codec": "PASS as high-fidelity reconstruction codec only",
            "causal_rfgpt_long_rollout": "NO-GO",
            "sparse_token_editing": "PARTIAL PASS",
            "ppo_readiness": "NOT REACHED; P7 prohibited",
        },
        "gate": {
            "status": "PASS" if gate_passed else "FAIL",
            "reason": (
                "Required provenance artifacts exist, embedded hashes match, and 17 tests pass."
                if gate_passed
                else "Missing artifacts, hash mismatch, or tests did not pass."
            ),
        },
        "next_allowed_stage": "P1_data_roles_and_policy_gate" if gate_passed else None,
        "prohibited_actions": [
            "access final_test signal data or recompute final_test metrics",
            "scale causal RF-GPT long-rollout generation",
            "train or implement PPO",
            "use Evaluator B in proposal, reward, state, or online filtering",
            "integrate ManyTx into current training",
        ],
    }


def write_summary(manifest: dict) -> None:
    gate = manifest["gate"]
    history = manifest["historical_final_test_access"]
    lines = [
        "# P0 Current State Summary",
        "",
        "## Outcome",
        "",
        "- Gate: **%s**" % gate["status"],
        "- Reason: %s" % gate["reason"],
        "- Git commit: `%s` (workspace is not a Git repository)"
        % manifest["git_commit"],
        "- Source tree hash: `%s`" % manifest["source_tree_hash"],
        "- Combined checkpoint hash: `%s`" % manifest["checkpoint_hash"],
        "- Tests: %s/%s" % (manifest["tests"]["count"], manifest["tests"]["status"]),
        "",
        "## Frozen interpretation",
        "",
        "- VQ P1/K1024: PASS only as a high-fidelity reconstruction codec.",
        "- Causal RF-GPT full 2,048-token rollout: NO-GO; diagnostic/local baseline only.",
        "- Sparse token editing: PARTIAL PASS; proposal superiority is not established.",
        "- PPO readiness: NOT REACHED; P7 remains prohibited.",
        "",
        "## Data boundary",
        "",
        "- Historical final-test access before P0: `%s`." % history["accessed_before_P0"],
        "- It was not used for checkpoint selection, but the G0 provenance contains baseline metrics.",
        "- From P0 onward the final-test signal data is sealed until all methods and protocols are frozen.",
        "- P1 must obtain policy Gate samples from the untouched buffer, not from historically seen reward_validation.",
        "",
        "## Environment",
        "",
        "- Python: `%s`" % manifest["environment"]["python_executable"],
        "- PyTorch: `%s`" % manifest["environment"]["torch_version"],
        "- CUDA available: `%s`" % manifest["environment"]["cuda_available"],
        "- Next allowed stage: `%s`" % manifest["next_allowed_stage"],
        "",
        "Full machine-readable provenance is in `current_state_manifest.json`.",
    ]
    (REPORT_DIR / "current_state_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-count", type=int, required=True)
    parser.add_argument("--test-status", choices=("pass", "fail"), required=True)
    parser.add_argument("--test-command", required=True)
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.test_count, args.test_status, args.test_command)
    output = REPORT_DIR / "current_state_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_summary(manifest)
    print(json.dumps({"output": str(output), "gate": manifest["gate"]}, indent=2))
    if manifest["gate"]["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
