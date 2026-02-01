from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class RelevanceSpec:
    """
    Minimal DAG parent specification: parents[child] = {parent1, parent2, ...}.

    The specification is used to scale multiplier-derived constraints.

    The following conventions are used:

    - For a multiplier key (i <- j), the parent relationship is interpreted as: j is a parent of i.
    - If weights are provided, weights[(i, j)] is used for (i <- j).
    """

    parents: Mapping[str, Set[str]]
    weights: Optional[Mapping[Tuple[str, str], float]] = None


def relevance_weight(
    *,
    child: str,
    parent: str,
    spec: RelevanceSpec,
    default_weight: float = 0.0,
) -> float:
    """
    Return a relevance weight for a directed pair (child <- parent).

    If an explicit weight is provided, it is used. Otherwise, membership in
    spec.parents is used as a binary selector with default_weight for absent edges.
    """
    child = str(child)
    parent = str(parent)
    default_weight = float(default_weight)

    if spec.weights is not None:
        w = spec.weights.get((child, parent))
        if w is not None:
            return float(w)

    if parent in spec.parents.get(child, set()):
        return 1.0
    return float(default_weight)


def topological_order(nodes: Sequence[str], parents: Mapping[str, Set[str]]) -> List[str]:
    """
    Kahn topological sort.
    """
    nodes = [str(n) for n in nodes]
    parent_map: Dict[str, Set[str]] = {str(n): set(parents.get(str(n), set())) for n in nodes}

    out: List[str] = []
    ready = [n for n in nodes if not parent_map.get(n)]
    ready.sort()
    while ready:
        n = ready.pop(0)
        out.append(n)
        for m in nodes:
            if n in parent_map.get(m, set()):
                parent_map[m].remove(n)
                if not parent_map[m] and m not in out and m not in ready:
                    ready.append(m)
                    ready.sort()

    if len(out) != len(nodes):
        raise ValueError("Graph has at least one cycle")
    return out

