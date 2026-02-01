"""
Global sensitivity analysis.

Research-oriented utilities are provided for the following question:
  - Which drivers (initial conditions, cyclic states, rule activations, etc.)
    are most associated with outcomes (final states, transitions, numeric summaries)?

Design notes are as follows:
  - The core API is kept explicit, and custom DriverSpec/OutcomeSpec may be provided.
  - Default drivers/outcomes are provided that operate without access to internal simulation objects.
  - When optional per-run diagnostics are provided (e.g. threshold rules applied), they are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from cib.pathway import TransformationPathway
from cib.rare_events import event_rate_diagnostics

Scalar = Union[float, int, str, bool]
FeatureKind = Literal["numeric", "binary", "categorical"]


@dataclass(frozen=True)
class DriverSpec:
    """
    A driver is a per-run feature extracted from a path (or diagnostics).
    """

    name: str
    kind: FeatureKind
    extract: Callable[[TransformationPathway, Mapping[str, Any]], Scalar]


@dataclass(frozen=True)
class OutcomeSpec:
    """
    An outcome is a per-run response extracted from a path (or diagnostics).
    """

    name: str
    kind: FeatureKind
    extract: Callable[[TransformationPathway, Mapping[str, Any]], Scalar]


@dataclass(frozen=True)
class ImportanceSummary:
    name: str
    estimate: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class OutcomeSensitivity:
    outcome: str
    driver_importance: Tuple[ImportanceSummary, ...]
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class GlobalSensitivityReport:
    """
    A container for global sensitivity outputs is provided.
    """

    n_runs: int
    driver_names: Tuple[str, ...]
    outcome_names: Tuple[str, ...]
    outcome_sensitivity: Tuple[OutcomeSensitivity, ...]
    rare_outcome_warnings: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()


def _as_float(x: Scalar) -> float:
    if isinstance(x, (bool, np.bool_)):
        return float(bool(x))
    if isinstance(x, (int, np.integer)):
        return float(int(x))
    if isinstance(x, (float, np.floating)):
        return float(x)
    raise ValueError(f"Expected numeric scalar, got {type(x)}")


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Spearman rank correlation is computed (robust fallback without SciPy).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size == 0:
        return 0.0
    # Rank with average ties.
    def _rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, a.size + 1, dtype=float)
        # Tie handling: average ranks within equal values.
        v = a[order]
        i = 0
        while i < v.size:
            j = i + 1
            while j < v.size and v[j] == v[i]:
                j += 1
            if j - i > 1:
                avg = float(np.mean(ranks[order[i:j]]))
                ranks[order[i:j]] = avg
            i = j
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    rx -= float(np.mean(rx))
    ry -= float(np.mean(ry))
    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(rx * ry) / denom)


def _one_hot_columns(values: Sequence[Scalar], *, prefix: str) -> Tuple[np.ndarray, Tuple[str, ...]]:
    cats = sorted({str(v) for v in values})
    if not cats:
        return np.zeros((len(values), 0), dtype=float), tuple()
    X = np.zeros((len(values), len(cats)), dtype=float)
    idx_by_cat = {c: i for i, c in enumerate(cats)}
    for i, v in enumerate(values):
        j = int(idx_by_cat[str(v)])
        X[i, j] = 1.0
    names = tuple(f"{prefix}={c}" for c in cats)
    return X, names


def _design_matrix(
    driver_values: Mapping[str, Sequence[Scalar]],
    driver_kinds: Mapping[str, FeatureKind],
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """
    Driver columns are converted into a numeric design matrix.
    """
    cols: List[np.ndarray] = []
    names: List[str] = []
    n: Optional[int] = None
    for dname, vals in driver_values.items():
        if n is None:
            n = len(vals)
        if len(vals) != n:
            raise ValueError("All driver columns must have the same length")
        kind = driver_kinds[dname]
        if kind in {"numeric", "binary"}:
            cols.append(np.asarray([_as_float(v) for v in vals], dtype=float)[:, None])
            names.append(str(dname))
        else:
            X, col_names = _one_hot_columns(vals, prefix=str(dname))
            cols.append(X)
            names.extend(list(col_names))
    if n is None:
        return np.zeros((0, 0), dtype=float), tuple()
    if not cols:
        return np.zeros((n, 0), dtype=float), tuple()
    X = np.concatenate(cols, axis=1)
    return X, tuple(names)


def _ridge_fit_predict(X: np.ndarray, y: np.ndarray, *, alpha: float = 1e-6) -> np.ndarray:
    """
    Closed-form ridge regression fitting and prediction are performed (no sklearn dependency).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("Bad X/y shapes")
    if X.shape[1] == 0:
        return np.full_like(y, float(np.mean(y)))
    # Standardize columns for stability.
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    sd[sd == 0.0] = 1.0
    Xs = (X - mu) / sd
    if not np.all(np.isfinite(Xs)) or not np.all(np.isfinite(y)):
        return np.full_like(y, float(np.mean(y)))
    ys = y - float(np.mean(y))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
        A = Xs.T @ Xs
        A = A + float(alpha) * np.eye(A.shape[0])
        b = Xs.T @ ys
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        return np.full_like(y, float(np.mean(y)))
    try:
        beta = np.linalg.solve(A, b)
    except Exception:
        return np.full_like(y, float(np.mean(y)))
    if not np.all(np.isfinite(beta)):
        return np.full_like(y, float(np.mean(y)))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
        pred = Xs @ beta + float(np.mean(y))
    if not np.all(np.isfinite(pred)):
        return np.full_like(y, float(np.mean(y)))
    return np.asarray(pred, dtype=float)


def _logistic_sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class _LogisticModel:
    mu: np.ndarray
    sd: np.ndarray
    beta: np.ndarray
    intercept: float


def _logistic_fit(
    X: np.ndarray,
    y01: np.ndarray,
    *,
    alpha: float = 1e-6,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> Optional[_LogisticModel]:
    """
    Regularised logistic regression fitting is performed by Newton/IRLS.

    The intercept is not regularised.
    """
    X = np.asarray(X, dtype=float)
    y01 = np.asarray(y01, dtype=float)
    if X.ndim != 2 or y01.ndim != 1 or X.shape[0] != y01.shape[0]:
        raise ValueError("Bad X/y shapes")
    if X.shape[1] == 0:
        return None
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y01)):
        return None

    y01 = np.asarray((y01 > 0.5), dtype=float)
    p_hat = float(np.mean(y01))

    # Degenerate labels are treated as intercept-only.
    if p_hat <= 0.0 or p_hat >= 1.0:
        return _LogisticModel(
            mu=np.zeros((X.shape[1],), dtype=float),
            sd=np.ones((X.shape[1],), dtype=float),
            beta=np.zeros((X.shape[1],), dtype=float),
            intercept=float(np.log((p_hat + 1e-12) / (1.0 - p_hat + 1e-12))),
        )

    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    sd[sd == 0.0] = 1.0
    Xs = (X - mu) / sd
    if not np.all(np.isfinite(Xs)):
        return None

    beta = np.zeros((Xs.shape[1],), dtype=float)
    intercept = float(np.log(p_hat / (1.0 - p_hat)))
    lam = float(alpha)

    for _ in range(int(max_iter)):
        with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
            z = Xs @ beta + intercept
        if not np.all(np.isfinite(z)):
            return None
        p = _logistic_sigmoid(z)
        w = p * (1.0 - p)
        w = np.maximum(w, 1e-12)
        r = y01 - p

        with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
            g_beta = Xs.T @ r - lam * beta
        g_int = float(np.sum(r))
        if not np.all(np.isfinite(g_beta)) or not np.isfinite(g_int):
            return None

        Xw = Xs * w[:, None]
        with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
            H = Xs.T @ Xw + lam * np.eye(Xs.shape[1])
        H_int = float(np.sum(w))
        with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
            H_beta_int = Xs.T @ w
        if not np.all(np.isfinite(H)) or not np.isfinite(H_int) or not np.all(np.isfinite(H_beta_int)):
            return None

        aug = np.zeros((Xs.shape[1] + 1, Xs.shape[1] + 1), dtype=float)
        aug[0, 0] = H_int
        aug[0, 1:] = H_beta_int
        aug[1:, 0] = H_beta_int
        aug[1:, 1:] = H
        rhs = np.concatenate(([g_int], g_beta))

        if not np.all(np.isfinite(aug)) or not np.all(np.isfinite(rhs)):
            return None
        try:
            delta = np.linalg.solve(aug, rhs)
        except Exception:
            return None

        step_int = float(delta[0])
        step_beta = delta[1:]
        if not np.all(np.isfinite(step_beta)) or not np.isfinite(step_int):
            return None

        intercept_new = intercept + step_int
        beta_new = beta + step_beta

        if float(np.linalg.norm(step_beta)) + abs(step_int) < float(tol):
            intercept, beta = intercept_new, beta_new
            break
        intercept, beta = intercept_new, beta_new

    return _LogisticModel(
        mu=np.asarray(mu, dtype=float),
        sd=np.asarray(sd, dtype=float),
        beta=np.asarray(beta, dtype=float),
        intercept=float(intercept),
    )


def _logistic_predict_proba(model: _LogisticModel, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Xs = (X - model.mu) / model.sd
    with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
        z = Xs @ model.beta + float(model.intercept)
    if not np.all(np.isfinite(z)):
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    p = _logistic_sigmoid(z)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return np.asarray(p, dtype=float)


def _logloss(y01: np.ndarray, p: np.ndarray) -> float:
    y01 = np.asarray((np.asarray(y01, dtype=float) > 0.5), dtype=float)
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(y01 * np.log(p) + (1.0 - y01) * np.log(1.0 - p)))


def _permutation_importance_mse(
    X: np.ndarray,
    y: np.ndarray,
    *,
    rng: np.random.Generator,
    alpha: float = 1e-6,
) -> np.ndarray:
    """
    Permutation importance is computed by the increase in MSE after one feature column is permuted.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.shape[0] != y.shape[0]:
        raise ValueError("Bad X/y shapes")
    if X.shape[1] == 0:
        return np.zeros((0,), dtype=float)
    pred0 = _ridge_fit_predict(X, y, alpha=alpha)
    mse0 = float(np.mean((pred0 - y) ** 2))
    imps = np.zeros((X.shape[1],), dtype=float)
    for j in range(X.shape[1]):
        Xp = X.copy()
        perm = rng.permutation(X.shape[0])
        Xp[:, j] = Xp[perm, j]
        pred = _ridge_fit_predict(Xp, y, alpha=alpha)
        mse = float(np.mean((pred - y) ** 2))
        imps[j] = float(mse - mse0)
    return imps


def _permutation_importance_logloss(
    X: np.ndarray,
    y01: np.ndarray,
    *,
    rng: np.random.Generator,
    alpha: float = 1e-6,
) -> np.ndarray:
    """
    Permutation importance is computed by the increase in log-loss after one feature column is permuted.
    """
    X = np.asarray(X, dtype=float)
    y01 = np.asarray((np.asarray(y01, dtype=float) > 0.5), dtype=float)
    if X.shape[0] != y01.shape[0]:
        raise ValueError("Bad X/y shapes")
    if X.shape[1] == 0:
        return np.zeros((0,), dtype=float)

    model = _logistic_fit(X, y01, alpha=alpha)
    if model is None:
        p0 = np.full((X.shape[0],), float(np.mean(y01)), dtype=float)
    else:
        p0 = _logistic_predict_proba(model, X)
    base = _logloss(y01, p0)

    imps = np.zeros((X.shape[1],), dtype=float)
    for j in range(X.shape[1]):
        Xp = X.copy()
        perm = rng.permutation(X.shape[0])
        Xp[:, j] = Xp[perm, j]
        if model is None:
            pp = np.full((X.shape[0],), float(np.mean(y01)), dtype=float)
        else:
            pp = _logistic_predict_proba(model, Xp)
        loss = _logloss(y01, pp)
        imps[j] = float(loss - base)
    return imps


def _bootstrap_ci(values: np.ndarray, *, q_low: float = 0.025, q_high: float = 0.975) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (0.0, 0.0)
    lo = float(np.quantile(values, float(q_low)))
    hi = float(np.quantile(values, float(q_high)))
    return (lo, hi)


def _extract_tables(
    paths: Sequence[TransformationPathway],
    *,
    drivers: Sequence[DriverSpec],
    outcomes: Sequence[OutcomeSpec],
    per_run_context: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, List[Scalar]], Dict[str, List[Scalar]], Dict[str, FeatureKind], Dict[str, FeatureKind]]:
    if len(per_run_context) != len(paths):
        raise ValueError("per_run_context must align with paths")
    driver_values: Dict[str, List[Scalar]] = {d.name: [] for d in drivers}
    outcome_values: Dict[str, List[Scalar]] = {o.name: [] for o in outcomes}
    driver_kinds = {d.name: d.kind for d in drivers}
    outcome_kinds = {o.name: o.kind for o in outcomes}

    for p, ctx in zip(paths, per_run_context):
        for d in drivers:
            driver_values[d.name].append(d.extract(p, ctx))
        for o in outcomes:
            outcome_values[o.name].append(o.extract(p, ctx))

    return driver_values, outcome_values, driver_kinds, outcome_kinds


def _default_driver_specs_dynamic(
    *,
    scenario_mode: Literal["realized", "equilibrium"],
    cyclic_descriptor_names: Optional[Sequence[str]],
    include_initial: bool,
) -> List[DriverSpec]:
    drivers: List[DriverSpec] = []

    if include_initial:
        def _init_state(desc: str) -> DriverSpec:
            return DriverSpec(
                name=f"init:{desc}",
                kind="categorical",
                extract=lambda p, _ctx, d=str(desc): str(p.scenarios_for_mode(scenario_mode)[0].to_dict().get(d)),
            )

        # Initial state drivers are created later after we know descriptor names;
        # placeholder handled in factory below.
        drivers.append(
            DriverSpec(
                name="__INIT_STATE_PLACEHOLDER__",
                kind="categorical",
                extract=lambda _p, _ctx: "",
            )
        )

    if cyclic_descriptor_names:
        for cd in cyclic_descriptor_names:
            drivers.append(
                DriverSpec(
                    name=f"final:{cd}",
                    kind="categorical",
                    extract=lambda p, _ctx, d=str(cd): str(p.scenarios_for_mode(scenario_mode)[-1].to_dict().get(d)),
                )
            )

    # Threshold application drivers (if diagnostics provide them).
    drivers.append(
        DriverSpec(
            name="any_threshold_applied",
            kind="binary",
            extract=lambda _p, ctx: bool(ctx.get("any_threshold_applied", False)),
        )
    )
    return drivers


def _expand_init_state_placeholders(
    drivers: List[DriverSpec],
    *,
    paths: Sequence[TransformationPathway],
    scenario_mode: Literal["realized", "equilibrium"],
) -> List[DriverSpec]:
    if not drivers:
        return []
    has_placeholder = any(d.name == "__INIT_STATE_PLACEHOLDER__" for d in drivers)
    if not has_placeholder:
        return drivers
    if not paths:
        return [d for d in drivers if d.name != "__INIT_STATE_PLACEHOLDER__"]
    descs = list(paths[0].scenarios_for_mode(scenario_mode)[0].to_dict().keys())
    out: List[DriverSpec] = [d for d in drivers if d.name != "__INIT_STATE_PLACEHOLDER__"]
    for d in descs:
        out.append(
            DriverSpec(
                name=f"init:{d}",
                kind="categorical",
                extract=lambda p, _ctx, dd=str(d): str(p.scenarios_for_mode(scenario_mode)[0].to_dict().get(dd)),
            )
        )
    return out


def _default_outcome_specs_dynamic(
    *,
    scenario_mode: Literal["realized", "equilibrium"],
    key_descriptors: Optional[Sequence[str]] = None,
    max_key_descriptors: int = 10,
    numeric_mappings: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> List[OutcomeSpec]:
    outcomes: List[OutcomeSpec] = []

    # Final state categorical outcomes for key descriptors.
    outcomes.append(
        OutcomeSpec(
            name="__FINAL_STATE_PLACEHOLDER__",
            kind="categorical",
            extract=lambda _p, _ctx: "",
        )
    )

    # Total Hamming flips across periods as a numeric outcome.
    def _n_flips(p: TransformationPathway, _ctx: Mapping[str, Any]) -> float:
        scen = p.scenarios_for_mode(scenario_mode)
        total = 0
        for a, b in zip(scen, scen[1:]):
            da = a.to_dict()
            db = b.to_dict()
            total += sum(1 for k in da.keys() if da.get(k) != db.get(k))
        return float(total)

    outcomes.append(OutcomeSpec(name="transition:hamming_flips_total", kind="numeric", extract=_n_flips))

    # Numeric final-state outcome if a mapping is provided.
    if numeric_mappings:
        for desc, mapping in numeric_mappings.items():
            outcomes.append(
                OutcomeSpec(
                    name=f"final_numeric:{desc}",
                    kind="numeric",
                    extract=lambda p, _ctx, d=str(desc), m=dict(mapping): float(
                        m[str(p.scenarios_for_mode(scenario_mode)[-1].to_dict().get(d))]
                    ),
                )
            )

    # Allow expanding final-state placeholders later once descriptor names are known.
    return outcomes


def _expand_final_state_placeholders(
    outcomes: List[OutcomeSpec],
    *,
    paths: Sequence[TransformationPathway],
    scenario_mode: Literal["realized", "equilibrium"],
    key_descriptors: Optional[Sequence[str]],
    max_key_descriptors: int,
) -> List[OutcomeSpec]:
    if not outcomes:
        return []
    has_placeholder = any(o.name == "__FINAL_STATE_PLACEHOLDER__" for o in outcomes)
    if not has_placeholder:
        return outcomes
    if not paths:
        return [o for o in outcomes if o.name != "__FINAL_STATE_PLACEHOLDER__"]
    all_descs = list(paths[0].scenarios_for_mode(scenario_mode)[-1].to_dict().keys())
    if key_descriptors is not None:
        chosen = [d for d in key_descriptors if d in all_descs]
    else:
        chosen = all_descs[: max(0, int(max_key_descriptors))]
    out: List[OutcomeSpec] = [o for o in outcomes if o.name != "__FINAL_STATE_PLACEHOLDER__"]
    for d in chosen:
        out.append(
            OutcomeSpec(
                name=f"final:{d}",
                kind="categorical",
                extract=lambda p, _ctx, dd=str(d): str(p.scenarios_for_mode(scenario_mode)[-1].to_dict().get(dd)),
            )
        )
    return out


def compute_global_sensitivity_dynamic(
    paths: Sequence[TransformationPathway],
    *,
    outcomes: Optional[Sequence[OutcomeSpec]] = None,
    drivers: Optional[Sequence[DriverSpec]] = None,
    bootstrap: int = 200,
    seed: int = 123,
    scenario_mode: Literal["realized", "equilibrium"] = "realized",
    diagnostics_by_run: Optional[Sequence[Mapping[str, Any]]] = None,
    cyclic_descriptor_names: Optional[Sequence[str]] = None,
    key_descriptors: Optional[Sequence[str]] = None,
    numeric_mappings: Optional[Mapping[str, Mapping[str, float]]] = None,
    ridge_alpha: float = 1e-6,
) -> GlobalSensitivityReport:
    """
    Global sensitivity for dynamic pathway ensembles is computed.

    Notes:
    - If `diagnostics_by_run` includes `threshold_rules_applied` lists (as produced by
      `DynamicCIB.simulate_path(..., diagnostics=...)`), then a binary driver
      `any_threshold_applied` is set per run.
    """
    if not paths:
        raise ValueError("paths cannot be empty")
    n = int(len(paths))

    # Prepare per-run context.
    ctxs: List[Dict[str, Any]] = []
    for i in range(n):
        ctx: Dict[str, Any] = {}
        if diagnostics_by_run is not None:
            d = diagnostics_by_run[i]
            applied = d.get("threshold_rules_applied")
            any_applied = False
            if isinstance(applied, list):
                any_applied = any(bool(x) for x in applied)
            ctx["any_threshold_applied"] = bool(any_applied)
            ctx["threshold_rules_applied"] = applied
        ctxs.append(ctx)

    d_specs = list(drivers) if drivers is not None else _default_driver_specs_dynamic(
        scenario_mode=scenario_mode,
        cyclic_descriptor_names=cyclic_descriptor_names,
        include_initial=True,
    )
    d_specs = _expand_init_state_placeholders(d_specs, paths=paths, scenario_mode=scenario_mode)

    o_specs = list(outcomes) if outcomes is not None else _default_outcome_specs_dynamic(
        scenario_mode=scenario_mode,
        key_descriptors=key_descriptors,
        numeric_mappings=numeric_mappings,
    )
    o_specs = _expand_final_state_placeholders(
        o_specs,
        paths=paths,
        scenario_mode=scenario_mode,
        key_descriptors=key_descriptors,
        max_key_descriptors=10,
    )

    driver_values, outcome_values, driver_kinds, outcome_kinds = _extract_tables(
        paths, drivers=d_specs, outcomes=o_specs, per_run_context=ctxs
    )

    # Build design matrix.
    X, X_names = _design_matrix(driver_values, driver_kinds)

    rng = np.random.default_rng(int(seed))
    bootstrap = max(1, int(bootstrap))

    outcome_sens: List[OutcomeSensitivity] = []
    rare_warn: List[str] = []
    notes: List[str] = []
    if X.shape[1] == 0:
        notes.append("No drivers produced any numeric features; importance is empty.")
    if int(X.shape[1]) >= 5000:
        notes.append(
            f"A large number of feature columns was produced by one-hot expansion (n_features={int(X.shape[1])})."
        )
    if int(bootstrap) >= 500:
        notes.append(f"A large bootstrap count was requested (bootstrap={int(bootstrap)}).")

    for oname, ovals in outcome_values.items():
        okind = outcome_kinds[oname]

        # Outcome conversion is performed for model fitting and permutation importance.
        y: np.ndarray
        y_names: Tuple[str, ...]
        if okind in {"numeric", "binary"}:
            y = np.asarray([_as_float(v) for v in ovals], dtype=float)
            y_names = (str(oname),)
        else:
            y_oh, y_cols = _one_hot_columns(ovals, prefix=str(oname))
            # For categorical outcomes, we analyze each one-vs-rest column separately.
            # Each column is a binary label (0/1).
            for col_idx, col_name in enumerate(y_cols):
                yy = y_oh[:, col_idx].astype(float)
                # Rare outcome warning.
                k = int(yy.sum())
                if k > 0 and k < max(5, int(0.01 * n)):
                    diag = event_rate_diagnostics(k=k, n=n)
                    if diag.is_under_sampled:
                        rare_warn.append(
                            f"Outcome {col_name!r} is rare (k={k}, n={n}); CI=[{diag.interval.lower:.3f},{diag.interval.upper:.3f}]"
                        )

                if X.shape[1] == 0:
                    outcome_sens.append(
                        OutcomeSensitivity(outcome=col_name, driver_importance=tuple(), notes=("no drivers",))
                    )
                    continue

                # Bootstrap permutation importances are computed.
                boot_imps: List[np.ndarray] = []
                for _b in range(bootstrap):
                    idx = rng.integers(0, n, size=n)
                    imp = _permutation_importance_logloss(
                        X[idx, :], yy[idx], rng=rng, alpha=float(ridge_alpha)
                    )
                    boot_imps.append(imp)
                boot = np.stack(boot_imps, axis=0)  # (B, p)
                est = np.median(boot, axis=0)

                items: List[ImportanceSummary] = []
                for j, dname in enumerate(X_names):
                    lo, hi = _bootstrap_ci(boot[:, j])
                    items.append(
                        ImportanceSummary(
                            name=str(dname),
                            estimate=float(est[j]),
                            ci_low=float(lo),
                            ci_high=float(hi),
                        )
                    )
                items.sort(key=lambda it: float(it.estimate), reverse=True)
                outcome_sens.append(
                    OutcomeSensitivity(
                        outcome=col_name,
                        driver_importance=tuple(items[: min(30, len(items))]),
                    )
                )
            continue

        # Scalar/binary outcome path:
        if okind == "binary":
            k = int(np.sum(y))
            if k > 0 and k < max(5, int(0.01 * n)):
                diag = event_rate_diagnostics(k=k, n=n)
                if diag.is_under_sampled:
                    rare_warn.append(
                        f"Outcome {oname!r} is rare (k={k}, n={n}); CI=[{diag.interval.lower:.3f},{diag.interval.upper:.3f}]"
                    )

        if X.shape[1] == 0:
            outcome_sens.append(
                OutcomeSensitivity(outcome=str(oname), driver_importance=tuple(), notes=("no drivers",))
            )
            continue

        # Screening: Spearman corr with each numeric/one-hot driver column.
        screen = np.asarray([_spearman_corr(X[:, j], y) for j in range(X.shape[1])], dtype=float)

        boot_imps = []
        for _b in range(bootstrap):
            idx = rng.integers(0, n, size=n)
            if okind == "binary":
                imp = _permutation_importance_logloss(
                    X[idx, :], y[idx], rng=rng, alpha=float(ridge_alpha)
                )
            else:
                imp = _permutation_importance_mse(
                    X[idx, :], y[idx], rng=rng, alpha=float(ridge_alpha)
                )
            boot_imps.append(imp)
        boot = np.stack(boot_imps, axis=0)
        est = np.median(boot, axis=0)

        items = []
        for j, dname in enumerate(X_names):
            lo, hi = _bootstrap_ci(boot[:, j])
            items.append(
                ImportanceSummary(
                    name=str(dname),
                    estimate=float(est[j]),
                    ci_low=float(lo),
                    ci_high=float(hi),
                )
            )
        items.sort(key=lambda it: float(it.estimate), reverse=True)
        outcome_sens.append(
            OutcomeSensitivity(
                outcome=str(oname),
                driver_importance=tuple(items[: min(30, len(items))]),
                notes=(f"screening_spearman_abs_top={float(np.max(np.abs(screen))):.3f}",),
            )
        )

    return GlobalSensitivityReport(
        n_runs=n,
        driver_names=tuple(d.name for d in d_specs),
        outcome_names=tuple(o.name for o in o_specs),
        outcome_sensitivity=tuple(outcome_sens),
        rare_outcome_warnings=tuple(rare_warn),
        notes=tuple(notes),
    )


def compute_global_sensitivity_attractors(
    mc_result: Any,
    *,
    top_k: int = 20,
) -> GlobalSensitivityReport:
    """
    A report for static Monte Carlo attractor discovery is computed.

    Note: a single attractor discovery result does not define sensitivity unless multiple
    results are provided across varying assumptions. A diagnostics summary of discovered
    attractors and undersampling warnings for rare attractors is provided.
    """
    counts = getattr(mc_result, "counts", None)
    ranked = getattr(mc_result, "attractor_keys_ranked", None)
    diag = getattr(mc_result, "diagnostics", {})
    if counts is None or ranked is None:
        raise ValueError("mc_result must look like MonteCarloAttractorResult")

    n = int(diag.get("n_completed_runs", 0)) or int(diag.get("runs", 0)) or int(sum(int(v) for v in counts.values()))
    if n <= 0:
        n = int(sum(int(v) for v in counts.values()))

    warnings: List[str] = []
    outcomes: List[OutcomeSensitivity] = []
    kmax = min(int(top_k), len(ranked))
    for key in ranked[:kmax]:
        c = int(counts.get(key, 0))
        name = f"attractor:{key.kind}:{key.value}"
        if n > 0:
            ev = event_rate_diagnostics(k=c, n=n)
            if ev.is_under_sampled and c > 0:
                warnings.append(
                    f"{name} is rare (k={c}, n={n}); CI=[{ev.interval.lower:.3f},{ev.interval.upper:.3f}]"
                )
        outcomes.append(OutcomeSensitivity(outcome=name, driver_importance=tuple(), notes=("no_drivers",)))

    return GlobalSensitivityReport(
        n_runs=int(n),
        driver_names=tuple(),
        outcome_names=tuple(o.outcome for o in outcomes),
        outcome_sensitivity=tuple(outcomes),
        rare_outcome_warnings=tuple(warnings),
        notes=("A static attractor result summary is provided only; multiple results are required for sensitivity across assumptions.",),
    )

