#!/usr/bin/env python
"""Encode the aligned WiFi-B cache once with a frozen VQ tokenizer."""

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rffi_core.data.build_data_manifest import load_json
from rffi_core.generators.vqvae.models import build_reconstruction_model
from rffi_core.generators.vqvae.train_reconstruction import choose_device


def sha256_file(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-config", default="configs/data/rffi_data_v1.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model = build_reconstruction_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    if model.quantizer is None:
        raise ValueError("checkpoint is not a VQ tokenizer")
    device = choose_device(args.device)
    model.to(device).eval()
    config = load_json(Path(args.data_config).resolve())
    cache_dir = Path(config["cache_root"]).resolve() / "wifib"
    iq_path = cache_dir / "iq_window_2048_float32.npy"
    power_path = cache_dir / "power_window_2048_float32.npy"
    index_path = cache_dir / "window_index.csv"
    iq_cache = np.load(str(iq_path), mmap_mode="r", allow_pickle=False)
    power_cache = np.load(str(power_path), mmap_mode="r", allow_pickle=False)
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        index_rows = list(csv.DictReader(handle))
    if len(index_rows) != iq_cache.shape[0] or len(power_cache) != iq_cache.shape[0]:
        raise ValueError("WiFi-B caches and index have different row counts")
    token_count = iq_cache.shape[2] // int(checkpoint["model_config"].get("patch_size", 4))
    codebook_size = int(checkpoint["model_config"]["codebook_size"])
    dtype = np.uint16 if codebook_size <= np.iinfo(np.uint16).max + 1 else np.uint32
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tokens.npy"
    partial_path = output_dir / "tokens.partial.npy"
    if output_path.exists() or partial_path.exists():
        raise FileExistsError("refusing to overwrite an existing token cache")
    token_cache = np.lib.format.open_memmap(
        str(partial_path),
        mode="w+",
        dtype=dtype,
        shape=(iq_cache.shape[0], token_count),
    )
    histogram = np.zeros(codebook_size, dtype=np.int64)
    started = time.time()
    with torch.inference_mode():
        for start in range(0, iq_cache.shape[0], args.batch_size):
            end = min(iq_cache.shape[0], start + args.batch_size)
            iq = np.asarray(iq_cache[start:end], dtype=np.float32).copy()
            powers = np.asarray(power_cache[start:end], dtype=np.float32)
            iq /= np.sqrt(np.maximum(powers, 1e-12))[:, None, None]
            inputs = torch.from_numpy(iq).to(device=device, non_blocking=True)
            indices = model.encode_code_indices(inputs).cpu().numpy().astype(dtype)
            token_cache[start:end] = indices
            histogram += np.bincount(indices.reshape(-1), minlength=codebook_size)
            if start == 0 or end == iq_cache.shape[0] or end % 8192 < args.batch_size:
                print("encoded %d/%d" % (end, iq_cache.shape[0]), flush=True)
    token_cache.flush()
    del token_cache
    partial_path.replace(output_path)
    split_counts = Counter(row["split"] for row in index_rows)
    probabilities = histogram.astype(np.float64) / max(1, int(histogram.sum()))
    active = probabilities > 0
    perplexity = float(np.exp(-np.sum(probabilities[active] * np.log(probabilities[active]))))
    metadata = {
        "schema_version": "wifib-rf-token-cache-v1",
        "status": "complete",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "source_iq_cache": str(iq_path),
        "source_power_cache": str(power_path),
        "source_index": str(index_path),
        "tokens_path": str(output_path),
        "tokens_sha256": sha256_file(output_path),
        "shape": list(np.load(str(output_path), mmap_mode="r").shape),
        "dtype": np.dtype(dtype).name,
        "codebook_size": codebook_size,
        "active_codes": int(active.sum()),
        "perplexity": perplexity,
        "maximum_code_fraction": float(probabilities.max()),
        "split_counts": dict(sorted(split_counts.items())),
        "normalization": "per-frame unit complex power before encoding",
        "seconds": time.time() - started,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print("token cache: %s" % output_path, flush=True)
    print(
        "active %d/%d, perplexity %.2f" % (active.sum(), codebook_size, perplexity),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

