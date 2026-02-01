from __future__ import annotations

from cib.core import CIBMatrix
from cib.cyclic import CyclicDescriptor
from cib.dynamic import DynamicCIB
from cib.sensitivity import compute_global_sensitivity_dynamic


def test_global_sensitivity_dynamic_detects_cyclic_driver() -> None:
    # Cycle <-> Y coordination system; Y tracks Cycle each period.
    descriptors = {"Cycle": ["Low", "High"], "Y": ["Low", "High"]}
    m = CIBMatrix(descriptors)

    # Cycle influences Y.
    m.set_impact("Cycle", "Low", "Y", "Low", 2.0)
    m.set_impact("Cycle", "Low", "Y", "High", -2.0)
    m.set_impact("Cycle", "High", "Y", "Low", -2.0)
    m.set_impact("Cycle", "High", "Y", "High", 2.0)
    # Y influences Cycle (coordination equilibrium per period).
    m.set_impact("Y", "Low", "Cycle", "Low", 2.0)
    m.set_impact("Y", "Low", "Cycle", "High", -2.0)
    m.set_impact("Y", "High", "Cycle", "Low", -2.0)
    m.set_impact("Y", "High", "Cycle", "High", 2.0)

    dyn = DynamicCIB(m, periods=[1, 2, 3, 4])
    dyn.add_cyclic_descriptor(
        CyclicDescriptor(
            name="Cycle",
            transition={
                "Low": {"Low": 0.8, "High": 0.2},
                "High": {"High": 0.8, "Low": 0.2},
            },
        )
    )

    # Variation is induced by the stochastic cyclic transitions across runs.
    paths = dyn.simulate_ensemble(initial={"Cycle": "Low", "Y": "Low"}, n_runs=120, base_seed=123)

    rep = compute_global_sensitivity_dynamic(
        paths,
        cyclic_descriptor_names=["Cycle"],
        key_descriptors=["Y"],
        bootstrap=50,
        seed=123,
    )

    # We expect at least one categorical final outcome for Y to be present.
    assert any(o.startswith("final:Y") for o in rep.outcome_names)
    # And at least one importance list should be non-empty.
    assert any(len(os.driver_importance) > 0 for os in rep.outcome_sensitivity)

