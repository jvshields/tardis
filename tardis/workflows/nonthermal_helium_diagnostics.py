"""Inspect retained helium coupling states without running a plasma solver.

This module imports no TARDIS modules, so it can also be loaded with
``runpy.run_path`` in a notebook that still holds an older workflow instance.
"""

import numpy as np
import pandas as pd


def diagnose_level_relaxation(
    current_levels: pd.DataFrame,
    candidate_levels: pd.DataFrame,
    current_ions: pd.DataFrame,
    candidate_ions: pd.DataFrame,
    relaxation: float = 0.5,
    ion_relaxation_factors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Audit a selective level update using snapshots, without changing them.

    Parameters
    ----------
    current_levels, candidate_levels : pandas.DataFrame
        Input and unrelaxed candidate level densities in cm^-3, indexed by
        atomic number, ion number, and level number; columns are shell labels.
    current_ions, candidate_ions : pandas.DataFrame
        Corresponding ion densities in cm^-3, indexed by atomic and ion number.
    relaxation : float, optional
        Fixed ion-population damping fraction when explicit factors are absent.
    ion_relaxation_factors : pandas.DataFrame, optional
        Actual per-element factors, indexed by atomic number with shell
        columns. Supply these when auditing an adaptive coupling update.

    Returns
    -------
    pandas.DataFrame
        One row per represented candidate ion and shell. Reports parent-ion
        availability, level sums and nonfinite counts in each snapshot, the
        predicted nonfinite count after selective relaxation, and whether that
        update creates nonfinite levels from finite inputs. ``classification``
        distinguishes missing parents, contaminated inputs/candidates, and
        finite zero distributions with a positive ion total. Sums ending in
        ``skipna`` deliberately reproduce pandas' all-missing-to-zero sum;
        ``strict`` sums remain NaN if any level is nonfinite. This audit can
        identify a propagation mechanism, not reconstruct the first corrupt
        iteration from an already contaminated snapshot.
    """
    ion_names = ["atomic_number", "ion_number"]
    columns = candidate_levels.columns
    current_levels = current_levels.reindex(
        index=candidate_levels.index, columns=columns
    )
    current_ions = current_ions.reindex(columns=columns)
    candidate_ions = candidate_ions.reindex(columns=columns)
    groups = candidate_levels.groupby(level=ion_names).sum().index
    level_groups = candidate_levels.index.droplevel("level_number")
    report_index = pd.MultiIndex.from_tuples(
        [(*ion, shell) for ion in groups for shell in columns],
        names=[*ion_names, "shell"],
    )
    report = pd.DataFrame(index=report_index)
    totals = {}
    for name, levels, ions in (
        ("input", current_levels, current_ions),
        ("candidate", candidate_levels, candidate_ions),
    ):
        finite = pd.DataFrame(
            np.isfinite(levels.to_numpy(dtype=float)),
            index=levels.index,
            columns=columns,
        )
        nonfinite_count = (~finite).groupby(level=ion_names).sum()
        level_sums = levels.groupby(level=ion_names).sum()
        totals[name] = levels.groupby(level=ion_names).transform("sum")
        aligned_ions = ions.reindex(index=groups, columns=columns)
        report[f"{name}_nonfinite_level_count"] = (
            nonfinite_count.to_numpy().ravel()
        )
        report[f"{name}_level_sum_skipna"] = level_sums.to_numpy().ravel()
        report[f"{name}_level_sum_strict"] = (
            level_sums.mask(nonfinite_count > 0).to_numpy().ravel()
        )
        report[f"{name}_ion_density"] = aligned_ions.to_numpy().ravel()
        report[f"{name}_parent_present"] = np.repeat(
            groups.isin(ions.index), len(columns)
        )

    relaxed_ions = (
        relaxation * candidate_ions + (1.0 - relaxation) * current_ions
    )
    if ion_relaxation_factors is not None:
        ion_index = relaxed_ions.index
        factors = pd.DataFrame(
            ion_relaxation_factors.reindex(
                index=ion_index.get_level_values("atomic_number"),
                columns=columns,
            ).to_numpy(),
            index=ion_index,
            columns=columns,
        )
        relaxed_ions = factors * candidate_ions + (1 - factors) * current_ions
    report["relaxed_ion_density"] = (
        relaxed_ions.reindex(index=groups, columns=columns).to_numpy().ravel()
    )
    fractions = candidate_levels.div(
        totals["candidate"].where(totals["candidate"] != 0.0)
    ).mask(
        totals["candidate"] == 0.0,
        current_levels.div(totals["input"].where(totals["input"] != 0.0)),
    )
    aligned_relaxed_ions = pd.DataFrame(
        relaxed_ions.reindex(index=level_groups, columns=columns).to_numpy(),
        index=candidate_levels.index,
        columns=columns,
    )
    projected = (fractions * aligned_relaxed_ions).mask(
        aligned_relaxed_ions == 0.0, 0.0
    )
    projected_bad = pd.DataFrame(
        ~np.isfinite(projected.to_numpy(dtype=float)),
        index=candidate_levels.index,
        columns=columns,
    )
    newly_bad = (
        projected_bad
        & np.isfinite(current_levels)
        & np.isfinite(candidate_levels)
    )
    report["projected_nonfinite_level_count"] = (
        projected_bad.groupby(level=ion_names).sum().to_numpy().ravel()
    )
    report["introduced_nonfinite_level_count"] = (
        newly_bad.groupby(level=ion_names).sum().to_numpy().ravel()
    )
    report["classification"] = np.select(
        [
            ~report["input_parent_present"]
            | ~report["candidate_parent_present"],
            ~np.isfinite(report["input_ion_density"])
            | ~np.isfinite(report["candidate_ion_density"]),
            (report["input_nonfinite_level_count"] > 0)
            & (report["candidate_nonfinite_level_count"] > 0),
            report["candidate_nonfinite_level_count"] > 0,
            report["input_nonfinite_level_count"] > 0,
            (report["input_level_sum_skipna"] == 0.0)
            & (report["candidate_level_sum_skipna"] == 0.0)
            & (report["relaxed_ion_density"] > 0.0),
        ],
        [
            "missing_parent_ion",
            "nonfinite_parent_density",
            "nonfinite_input_and_candidate",
            "nonfinite_candidate",
            "nonfinite_input",
            "positive_ion_without_level_distribution",
        ],
        default="finite",
    )
    report["affected"] = (report["classification"] != "finite") | (
        report["projected_nonfinite_level_count"] > 0
    )
    return report
