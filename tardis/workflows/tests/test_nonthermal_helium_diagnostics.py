"""Verification of read-only diagnostics, not of physical helium rates."""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest

from tardis.workflows.nonthermal_helium_diagnostics import (
    diagnose_level_relaxation,
)


@pytest.mark.parametrize(
    (
        "input_values",
        "candidate_values",
        "input_ion",
        "candidate_ion",
        "classification",
        "projected_bad",
        "introduced_bad",
    ),
    [
        (
            [0.0, 0.0],
            [0.0, 0.0],
            3e-23,
            4e-23,
            "positive_ion_without_level_distribution",
            2,
            2,
        ),
        (
            [np.nan, np.nan],
            [np.nan, np.nan],
            3e-23,
            4e-23,
            "nonfinite_input_and_candidate",
            2,
            0,
        ),
        ([1.0, 0.0], [np.nan, np.nan], 1.0, 1.0, "nonfinite_candidate", 0, 0),
        ([np.nan, np.nan], [1.0, 0.0], 1.0, 1.0, "nonfinite_input", 0, 0),
        ([1.0, 0.0], [0.0, 0.0], 1.0, 0.0, "finite", 0, 0),
        (
            [np.nan, np.nan],
            [np.nan, np.nan],
            0.0,
            0.0,
            "nonfinite_input_and_candidate",
            0,
            0,
        ),
        ([0.0, 0.0], [0.0, 0.0], 0.0, 0.0, "finite", 0, 0),
        ([1.0, 0.0], [1.0, 0.0], 1.0, None, "missing_parent_ion", 2, 2),
        ([1.0, 0.0], [1.0, 0.0], 1.0, np.nan, "nonfinite_parent_density", 2, 2),
    ],
)
def test_level_audit_classifies_empty_and_nonfinite_distributions(
    input_values: list[float],
    candidate_values: list[float],
    input_ion: float,
    candidate_ion: float | None,
    classification: str,
    projected_bad: int,
    introduced_bad: int,
) -> None:
    """Count undefined fractions and distinguish creation from propagation.

    Regime: two Li I levels, shell 53, alpha=0.5. The exact limiting cases
    follow from 0/0, NaN times a positive total, fallback to [1, 0], and the
    zero-ion mask. Counts are integers with no numerical tolerance. A missing
    parent is distinct from an existing zero ion; NaN sums must not be
    described as a physical zero. This is numerical diagnostic verification.
    """
    level_index = pd.MultiIndex.from_tuples(
        [(3, 0, 0), (3, 0, 1)],
        names=["atomic_number", "ion_number", "level_number"],
    )
    ion_index = level_index.droplevel("level_number").unique()
    current_levels = pd.DataFrame(input_values, index=level_index, columns=[53])
    candidate_levels = pd.DataFrame(
        candidate_values, index=level_index, columns=[53]
    )
    current_ions = pd.DataFrame([input_ion], index=ion_index, columns=[53])
    candidate_ions = pd.DataFrame(
        [candidate_ion if candidate_ion is not None else 0.0],
        index=ion_index,
        columns=[53],
    )
    if candidate_ion is None:
        candidate_ions = candidate_ions.iloc[:0]
    snapshots = [current_levels, candidate_levels, current_ions, candidate_ions]
    originals = [frame.copy(deep=True) for frame in snapshots]

    report = diagnose_level_relaxation(*snapshots)

    row = report.loc[(3, 0, 53)]
    assert row["classification"] == classification
    assert row["projected_nonfinite_level_count"] == projected_bad
    assert row["introduced_nonfinite_level_count"] == introduced_bad
    assert bool(row["candidate_parent_present"]) == (candidate_ion is not None)
    if np.isnan(input_values).all():
        assert row["input_nonfinite_level_count"] == 2
        assert np.isnan(row["input_level_sum_strict"])
        npt.assert_equal(row["input_level_sum_skipna"], 0.0)
    for snapshot, original in zip(snapshots, originals, strict=True):
        pdt.assert_frame_equal(snapshot, original)


def test_level_audit_aligns_ion_and_shell_labels() -> None:
    """Locate the undefined Li I distribution despite reversed ion/shell order.

    All levels and parent densities are finite. Only Li I in shell 53 has
    zero levels and a positive parent density, so two new NaNs are expected
    there and nowhere else. Exact integer counts verify label alignment.
    """
    level_index = pd.MultiIndex.from_tuples(
        [(3, 0, 1), (2, 0, 0), (3, 0, 0)],
        names=["atomic_number", "ion_number", "level_number"],
    )
    levels = pd.DataFrame(
        [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0]],
        index=level_index,
        columns=[25, 53],
    )
    ions = pd.DataFrame(
        [[1.0, 1.0], [1.0, 1.0]],
        index=pd.MultiIndex.from_tuples(
            [(3, 0), (2, 0)], names=["atomic_number", "ion_number"]
        ),
        columns=[53, 25],
    )
    report = diagnose_level_relaxation(levels, levels, ions, ions)
    affected = report.loc[report["affected"]]
    assert affected.index.tolist() == [(3, 0, 53)]
    npt.assert_array_equal(affected["introduced_nonfinite_level_count"], [2])


def test_level_audit_uses_actual_adaptive_factor() -> None:
    """A full ion step to zero clears NaNs that half damping would retain."""
    index = pd.MultiIndex.from_tuples(
        [(3, 0, 0), (3, 0, 1)],
        names=["atomic_number", "ion_number", "level_number"],
    )
    levels = pd.DataFrame(np.nan, index=index, columns=[53])
    ions = pd.DataFrame(
        [1.0], index=index.droplevel("level_number").unique(), columns=[53]
    )
    fixed = diagnose_level_relaxation(levels, levels, ions, ions * 0)
    adaptive = diagnose_level_relaxation(
        levels,
        levels,
        ions,
        ions * 0,
        ion_relaxation_factors=pd.DataFrame([1.0], index=[3], columns=[53]),
    )
    npt.assert_equal(
        fixed.loc[(3, 0, 53), "projected_nonfinite_level_count"], 2
    )
    npt.assert_equal(
        adaptive.loc[(3, 0, 53), "projected_nonfinite_level_count"], 0
    )
    npt.assert_equal(adaptive.loc[(3, 0, 53), "relaxed_ion_density"], 0.0)
