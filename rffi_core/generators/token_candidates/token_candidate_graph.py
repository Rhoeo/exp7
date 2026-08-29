"""Read and validate the versioned static token candidate graph."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def nearest_neighbor_arrays(embeddings: np.ndarray, top_k: int):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [codes, latent_dim]")
    code_count = embeddings.shape[0]
    if top_k <= 0 or top_k >= code_count:
        raise ValueError("top_k must be positive and smaller than code count")
    # Use direct broadcasting instead of BLAS matmul. The local Conda runtime
    # links PyTorch and NumPy to different OpenMP builds, and importing a
    # checkpoint before a NumPy BLAS call otherwise aborts the process.
    delta = embeddings[:, None, :] - embeddings[None, :, :]
    distances = np.sum(delta * delta, axis=2)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1, kind="stable")[:, :top_k]
    selected = np.take_along_axis(distances, neighbors, axis=1)
    return neighbors.astype(np.int32), np.sqrt(selected).astype(np.float32)


def mutual_neighbor_mask(neighbors: np.ndarray) -> np.ndarray:
    neighbors = np.asarray(neighbors)
    code_count, top_k = neighbors.shape
    mask = np.zeros((code_count, top_k), dtype=np.bool_)
    neighbor_sets = [set(int(value) for value in row) for row in neighbors]
    for source in range(code_count):
        for position, target in enumerate(neighbors[source]):
            mask[source, position] = source in neighbor_sets[int(target)]
    return mask


def undirected_component_sizes(neighbors: np.ndarray) -> list[int]:
    code_count = int(neighbors.shape[0])
    adjacency = [set() for _ in range(code_count)]
    for source, row in enumerate(neighbors):
        for target in row:
            target = int(target)
            adjacency[source].add(target)
            adjacency[target].add(source)
    seen = set()
    sizes = []
    for start in range(code_count):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for target in adjacency[node]:
                if target not in seen:
                    seen.add(target)
                    stack.append(target)
        sizes.append(size)
    return sorted(sizes, reverse=True)


class TokenCandidateGraph:
    """Static coarse candidate prior; runtime probability/RF filters remain required."""

    REQUIRED_ARRAYS = (
        "neighbors",
        "latent_distances",
        "mutual_neighbors",
        "usage_counts",
        "forward_transition_counts",
        "backward_transition_counts",
    )

    def __init__(self, arrays: dict[str, np.ndarray], metadata: dict):
        self.arrays = arrays
        self.metadata = metadata
        self.validate()

    @classmethod
    def load(cls, path):
        path = Path(path)
        with np.load(str(path), allow_pickle=False) as payload:
            arrays = {name: payload[name] for name in cls.REQUIRED_ARRAYS}
            if "transition_counts" in payload.files:
                arrays["transition_counts"] = payload["transition_counts"]
            metadata = json.loads(str(payload["metadata_json"].item()))
        return cls(arrays, metadata)

    @property
    def codebook_size(self):
        return int(self.arrays["neighbors"].shape[0])

    @property
    def top_k(self):
        return int(self.arrays["neighbors"].shape[1])

    def validate(self):
        missing = set(self.REQUIRED_ARRAYS).difference(self.arrays)
        if missing:
            raise ValueError("candidate graph missing arrays: %s" % sorted(missing))
        neighbors = np.asarray(self.arrays["neighbors"])
        if neighbors.ndim != 2:
            raise ValueError("neighbors must have shape [codes, top_k]")
        code_count, top_k = neighbors.shape
        if code_count < 2 or top_k < 1:
            raise ValueError("candidate graph is empty")
        if np.any(neighbors < 0) or np.any(neighbors >= code_count):
            raise ValueError("candidate indices are out of range")
        if np.any(neighbors == np.arange(code_count)[:, None]):
            raise ValueError("candidate graph contains self edges")
        if any(len(set(int(value) for value in row)) != top_k for row in neighbors):
            raise ValueError("candidate graph contains duplicate neighbors")
        for name in (
            "latent_distances",
            "mutual_neighbors",
            "forward_transition_counts",
            "backward_transition_counts",
        ):
            if self.arrays[name].shape != neighbors.shape:
                raise ValueError("%s shape differs from neighbors" % name)
        if self.arrays["usage_counts"].shape != (code_count,):
            raise ValueError("usage_counts shape differs from code count")
        if "transition_counts" in self.arrays and self.arrays[
            "transition_counts"
        ].shape != (code_count, code_count):
            raise ValueError("transition_counts shape differs from code count")
        if not np.all(np.isfinite(self.arrays["latent_distances"])):
            raise ValueError("latent distances must be finite")
        return True

    def candidates(
        self,
        token: int,
        top_k: int | None = None,
        mutual_only: bool = False,
        transition_supported: bool = False,
    ) -> np.ndarray:
        token = int(token)
        if token < 0 or token >= self.codebook_size:
            raise IndexError(token)
        limit = self.top_k if top_k is None else min(int(top_k), self.top_k)
        if limit <= 0:
            return np.empty((0,), dtype=np.int32)
        values = self.arrays["neighbors"][token, :limit]
        keep = np.ones(values.shape, dtype=np.bool_)
        if mutual_only:
            keep &= self.arrays["mutual_neighbors"][token, :limit]
        if transition_supported:
            counts = (
                self.arrays["forward_transition_counts"][token, :limit]
                + self.arrays["backward_transition_counts"][token, :limit]
            )
            keep &= counts > 0
        return np.asarray(values[keep], dtype=np.int32)

    @property
    def transition_counts(self):
        if "transition_counts" not in self.arrays:
            raise RuntimeError("this graph version does not contain full transition counts")
        return self.arrays["transition_counts"]
