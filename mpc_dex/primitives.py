"""PCA coordination priors extracted from supplied joint trajectories.

This module supplies no rotation primitive. A useful rotation prior requires
trajectories that actually exhibit rotation, with their origin recorded by the
caller. PCA alone does not establish that a trajectory is a manipulation skill.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class PrimitiveBasis:
    """Joint mean and column-orthonormal PCA basis in actuator joint order."""

    mean: np.ndarray
    basis: np.ndarray
    coefficient_min: np.ndarray
    coefficient_max: np.ndarray
    explained_variance_ratio: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "mean", "basis", "coefficient_min", "coefficient_max",
            "explained_variance_ratio",
        ):
            array = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain only finite values")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if self.mean.shape != (16,):
            raise ValueError("mean must have 16 actuator-ordered joint values")
        if self.basis.ndim != 2 or self.basis.shape[0] != 16:
            raise ValueError("basis must have shape (16, components)")
        d = self.basis.shape[1]
        if not 1 <= d <= 16:
            raise ValueError("a primitive needs between 1 and 16 components")
        for name in ("coefficient_min", "coefficient_max", "explained_variance_ratio"):
            if getattr(self, name).shape != (d,):
                raise ValueError(f"{name} must match the number of components")
        if np.any(self.coefficient_min >= self.coefficient_max):
            raise ValueError("every coefficient must have a nonzero extraction range")
        if not np.allclose(self.basis.T @ self.basis, np.eye(d), atol=1e-7, rtol=1e-7):
            raise ValueError("basis columns must be orthonormal")
        if np.any(self.explained_variance_ratio <= 0) or self.explained_variance_ratio.sum() > 1 + 1e-7:
            raise ValueError("invalid explained variance ratios")
        # JSON round-trip makes an independent copy and forbids object payloads,
        # NaN and non-serializable provenance before the file can be written.
        metadata = json.loads(json.dumps(self.metadata, allow_nan=False))
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")
        provenance = metadata.get("provenance", {})
        if not isinstance(provenance, dict) or not isinstance(provenance.get("source"), str) or not provenance["source"].strip():
            raise ValueError("primitive metadata requires provenance.source")
        object.__setattr__(self, "metadata", metadata)

    @property
    def components(self) -> int:
        return self.basis.shape[1]

    @classmethod
    def from_trajectories(
        cls,
        trajectories: np.ndarray,
        explained_variance: float = 0.9,
        *,
        provenance: Mapping[str, Any],
    ) -> "PrimitiveBasis":
        """Extract the smallest PCA basis meeting the variance threshold.

        Input is (frames, 16), in radians and actuator joint order. The caller
        must supply provenance including a nonempty ``source`` description or
        path. Additional keys can record behavior, policy/model hashes, joint
        order, object dimensions and collection settings. A hash of the actual
        canonical float64 extraction array is recorded automatically.
        """
        q = np.asarray(trajectories, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 16 or q.shape[0] < 2:
            raise ValueError("trajectories must have shape (at least 2 frames, 16)")
        if not np.all(np.isfinite(q)):
            raise ValueError("trajectories must contain only finite joint positions")
        if np.all(q == q[0]):
            raise ValueError("constant trajectories cannot define a motion primitive")
        if not np.isfinite(explained_variance) or not 0 < explained_variance <= 1:
            raise ValueError("explained_variance must be in (0, 1]")
        mean = q.mean(axis=0)
        centered = q - mean
        _, singular_values, vectors = np.linalg.svd(centered, full_matrices=False)
        cutoff = np.finfo(np.float64).eps * max(q.shape) * singular_values[0]
        rank = int(np.count_nonzero(singular_values > cutoff))
        if rank == 0:
            raise ValueError("constant trajectories cannot define a motion primitive")
        variance = singular_values[:rank] ** 2
        ratios = variance / variance.sum()
        cumulative = ratios.cumsum()
        cumulative[-1] = 1.0
        count = int(np.searchsorted(cumulative, explained_variance)) + 1
        basis = vectors[:count].T.copy()
        # Sign fixes aid comparison and reproducible serialization; PCA signs
        # have no effect on the represented coordination subspace.
        for column in range(count):
            pivot = np.argmax(np.abs(basis[:, column]))
            if basis[pivot, column] < 0:
                basis[:, column] *= -1
        coefficients = centered @ basis
        metadata = {
            "schema_version": 1,
            "method": "mean_centered_joint_trajectory_pca",
            "provenance": dict(provenance),
            "extraction": {
                "frames": len(q),
                "joints": 16,
                "units": "radians",
                "requested_explained_variance": float(explained_variance),
                "retained_explained_variance": float(ratios[:count].sum()),
                "trajectories_float64_le_sha256": hashlib.sha256(
                    np.ascontiguousarray(q, dtype="<f8").tobytes()
                ).hexdigest(),
            },
        }
        return cls(
            mean, basis, coefficients.min(axis=0), coefficients.max(axis=0),
            ratios[:count], metadata,
        )

    def project(self, joint_positions: np.ndarray) -> np.ndarray:
        q = np.asarray(joint_positions, dtype=np.float64)
        if q.ndim < 1 or q.shape[-1] != 16 or not np.all(np.isfinite(q)):
            raise ValueError("joint positions must be finite with last dimension 16")
        return (q - self.mean) @ self.basis

    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        z = np.asarray(coefficients, dtype=np.float64)
        if z.ndim < 1 or z.shape[-1] != self.components or not np.all(np.isfinite(z)):
            raise ValueError("coefficients must be finite and match the basis")
        return self.mean + z @ self.basis.T

    def save(self, path: str | Path) -> None:
        """Write arrays and JSON text; no Python objects or pickle payloads."""
        metadata = json.dumps(self.metadata, sort_keys=True, allow_nan=False)
        with Path(path).open("wb") as stream:
            np.savez_compressed(
                stream, mean=self.mean, basis=self.basis,
                coefficient_min=self.coefficient_min,
                coefficient_max=self.coefficient_max,
                explained_variance_ratio=self.explained_variance_ratio,
                metadata_json=np.array(metadata),
            )

    @classmethod
    def load(cls, path: str | Path) -> "PrimitiveBasis":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            if metadata.get("schema_version") != 1:
                raise ValueError("unsupported primitive schema version")
            return cls(
                data["mean"], data["basis"], data["coefficient_min"],
                data["coefficient_max"], data["explained_variance_ratio"], metadata,
            )
