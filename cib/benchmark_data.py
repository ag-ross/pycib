"""
Deterministic benchmark datasets for scaling and solver development.

In this module, small, reproducible CIB matrices are provided for:
  - correctness regression (slow path vs fast path),
  - smoke benchmarking (throughput reporting), and
  - solver-mode integration tests (enumeration vs sampling).

The datasets are generated in pure Python with deterministic RNG seeding so that
results are stable across runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from cib.core import CIBMatrix


@dataclass(frozen=True)
class BenchmarkSpec:
    """
    Specification of a benchmark dataset.
    """

    name: str
    n_descriptors: int
    n_states: int
    seed: int


def _make_uniform_descriptors(*, n_descriptors: int, n_states: int) -> Dict[str, List[str]]:
    descriptors: Dict[str, List[str]] = {}
    for i in range(int(n_descriptors)):
        descriptors[f"D{i:02d}"] = [f"S{j}" for j in range(int(n_states))]
    return descriptors


def _populate_random_impacts(
    matrix: CIBMatrix,
    *,
    seed: int,
    impact_min: int = -3,
    impact_max: int = 3,
) -> None:
    """
    Random off-diagonal impacts are populated.

    Notes:
    - Values are drawn as integers in [impact_min, impact_max].
    - Diagonal judgement sections are omitted (CIB convention) and are therefore
      left unset.
    """
    rng = np.random.default_rng(int(seed))
    descs = list(matrix.descriptors.keys())
    for src in descs:
        for tgt in descs:
            if src == tgt:
                continue
            for src_state in matrix.descriptors[src]:
                for tgt_state in matrix.descriptors[tgt]:
                    v = int(rng.integers(int(impact_min), int(impact_max) + 1))
                    if v == 0:
                        continue
                    matrix.set_impact(src, src_state, tgt, tgt_state, float(v))


def benchmark_specs() -> Sequence[BenchmarkSpec]:
    """
    Benchmark specifications are returned.
    """
    return (
        BenchmarkSpec(name="B1_tiny_6x3", n_descriptors=6, n_states=3, seed=11_001),
        BenchmarkSpec(name="B2_medium_12x3", n_descriptors=12, n_states=3, seed=11_002),
        BenchmarkSpec(name="B3_medium_hard_18x3", n_descriptors=18, n_states=3, seed=11_003),
    )


def make_benchmark_matrix(spec: BenchmarkSpec) -> CIBMatrix:
    """
    Create a deterministic benchmark matrix according to a specification.
    """
    desc = _make_uniform_descriptors(
        n_descriptors=int(spec.n_descriptors), n_states=int(spec.n_states)
    )
    m = CIBMatrix(desc)
    _populate_random_impacts(m, seed=int(spec.seed))
    return m


def benchmark_matrix_b1() -> CIBMatrix:
    """
    Return the tiny sanity benchmark matrix (6 descriptors × 3 states).
    """
    spec = BenchmarkSpec(name="B1_tiny_6x3", n_descriptors=6, n_states=3, seed=11_001)
    return make_benchmark_matrix(spec)


def benchmark_matrix_b2() -> CIBMatrix:
    """
    Return the medium enumeration benchmark matrix (12 descriptors × 3 states).
    """
    spec = BenchmarkSpec(name="B2_medium_12x3", n_descriptors=12, n_states=3, seed=11_002)
    return make_benchmark_matrix(spec)


def benchmark_matrix_b3() -> CIBMatrix:
    """
    Return the medium-hard benchmark matrix (18 descriptors × 3 states).
    """
    spec = BenchmarkSpec(name="B3_medium_hard_18x3", n_descriptors=18, n_states=3, seed=11_003)
    return make_benchmark_matrix(spec)

