"""
Model reduction utilities for high-cardinality descriptors.

In this module, state-binning helpers are provided so that a reduced CIB matrix
can be constructed. The reduction is intended as a pragmatic aid for scaling.

Note: binning and aggregation do not guarantee preservation of the original
consistent-scenario set. Validation is recommended on tractable subproblems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np

from cib.core import CIBMatrix, Scenario


AggregationRule = Literal["mean", "max_abs", "max_signed", "weighted_mean"]


def _ordered_unique(items: Iterable[str]) -> Tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for x in items:
        xs = str(x)
        if xs in seen:
            continue
        seen.add(xs)
        out.append(xs)
    return tuple(out)


def bin_states(
    matrix: CIBMatrix,
    *,
    mapping: Mapping[str, Mapping[str, str]],
) -> CIBMatrix:
    """
    Create a reduced matrix by mapping fine states to binned states.

    Args:
        matrix: Original CIB matrix.
        mapping: Descriptor -> (fine_state -> bin_label) mapping.

    Returns:
        A new `CIBMatrix` with binned descriptors and aggregated impacts.

    Raises:
        ValueError: If mappings are invalid or incomplete.
    """
    new_desc: Dict[str, list[str]] = {}
    for d, states in matrix.descriptors.items():
        if d not in mapping:
            new_desc[d] = list(states)
            continue
        m = mapping[d]
        for s in states:
            if s not in m:
                raise ValueError(
                    f"Mapping for descriptor {d!r} was incomplete: state {s!r} was missing"
                )
        new_desc[d] = list(_ordered_unique(m[s] for s in states))
    return CIBMatrix(new_desc)


def reduce_matrix(
    matrix: CIBMatrix,
    *,
    mapping: Mapping[str, Mapping[str, str]],
    aggregation: AggregationRule = "weighted_mean",
    weights: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> CIBMatrix:
    """
    Reduce a `CIBMatrix` by binning states and aggregating impacts.

    Args:
        matrix: Original CIB matrix.
        mapping: Descriptor -> (fine_state -> bin_label) mapping.
        aggregation: Aggregation rule used for collapsing fine impacts.
        weights: Optional descriptor -> (fine_state -> weight) mapping, required
            when aggregation is "weighted_mean".

    Returns:
        Reduced `CIBMatrix`.
    """
    reduced = bin_states(matrix, mapping=mapping)
    agg = str(aggregation)

    # Precompute reverse maps: descriptor -> bin_label -> fine_states
    fine_by_bin: Dict[str, Dict[str, list[str]]] = {}
    for d, states in matrix.descriptors.items():
        if d not in mapping:
            fine_by_bin[d] = {s: [s] for s in states}
            continue
        bmap = mapping[d]
        out: Dict[str, list[str]] = {}
        for s in states:
            out.setdefault(str(bmap[s]), []).append(str(s))
        fine_by_bin[d] = out

    for src_desc in reduced.descriptors.keys():
        for tgt_desc in reduced.descriptors.keys():
            if src_desc == tgt_desc:
                continue
            for src_bin in reduced.descriptors[src_desc]:
                src_fine = fine_by_bin[src_desc][src_bin]
                for tgt_bin in reduced.descriptors[tgt_desc]:
                    tgt_fine = fine_by_bin[tgt_desc][tgt_bin]

                    vals: list[float] = []
                    wts: list[float] = []
                    for s_src in src_fine:
                        for s_tgt in tgt_fine:
                            v = float(matrix.get_impact(src_desc, s_src, tgt_desc, s_tgt))
                            vals.append(v)
                            if weights is not None and src_desc in weights:
                                wts.append(float(weights[src_desc].get(s_src, 1.0)))
                            else:
                                wts.append(1.0)

                    if not vals:
                        continue

                    value: float
                    if agg == "mean":
                        value = float(np.mean(np.asarray(vals, dtype=np.float64)))
                    elif agg == "max_abs":
                        a = np.asarray(vals, dtype=np.float64)
                        value = float(a[np.argmax(np.abs(a))])
                    elif agg == "max_signed":
                        value = float(np.max(np.asarray(vals, dtype=np.float64)))
                    elif agg == "weighted_mean":
                        if weights is None:
                            raise ValueError("weights must be provided for aggregation='weighted_mean'")
                        w = np.asarray(wts, dtype=np.float64)
                        a = np.asarray(vals, dtype=np.float64)
                        if float(w.sum()) <= 0.0:
                            raise ValueError(
                                f"Non-zero weights were required for descriptor {src_desc!r}"
                            )
                        value = float(np.average(a, weights=w))
                    else:
                        raise ValueError("aggregation is not recognised")

                    if np.isclose(value, 0.0):
                        continue
                    reduced.set_impact(src_desc, src_bin, tgt_desc, tgt_bin, float(value))

    return reduced


def map_scenario_to_reduced(
    scenario: Scenario,
    *,
    reduced_matrix: CIBMatrix,
    mapping: Mapping[str, Mapping[str, str]],
) -> Scenario:
    """
    Map a scenario defined on a fine matrix to a reduced-matrix scenario.
    """
    sdict = scenario.to_dict()
    out: Dict[str, str] = {}
    for d, states in reduced_matrix.descriptors.items():
        if d not in sdict:
            raise ValueError(f"Descriptor {d!r} was missing from scenario")
        fine = str(sdict[d])
        if d in mapping:
            out[d] = str(mapping[d][fine])
        else:
            out[d] = fine
    return Scenario(out, reduced_matrix)

