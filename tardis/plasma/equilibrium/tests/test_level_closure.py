"""Physical closure requirements independent of nonlinear solver status."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from tardis.plasma.equilibrium.evaluator import (
    PlasmaEquilibriumEvaluator,
    calculate_nlte_level_population_residual,
)
from tardis.plasma.equilibrium.inputs import (
    BoundBoundMatrixRates,
    ContinuumRateCoefficients,
    LevelEquationRates,
)
from tardis.plasma.equilibrium.solver import PlasmaEquilibriumSolver
from tardis.plasma.equilibrium.tests.test_evaluator import (
    toy_evaluator as toy_evaluator,  # noqa: PLC0414 - re-export shared pytest fixture
)


@pytest.mark.parametrize("rate_scale", [1e-20, 1.0, 1e20])
@pytest.mark.parametrize("fallback_stalls", [False, True])
def test_positive_stalled_level_solve_recovers_rate_balance(
    toy_evaluator: PlasmaEquilibriumEvaluator,
    monkeypatch: pytest.MonkeyPatch,
    rate_scale: float,
    fallback_stalls: bool,
) -> None:
    """Recover two-level SE irrespective of the absolute rate timescale.

    Verification: with upward/downward rates s, 2s, ionization s from both
    levels, and Case B recombination into the excited state, the excited
    equation is s*(b0 - 3*b1 + 1)=0, hence b0=b1=1/2. Tolerance 1e-8
    follows the required relative kinetic closure, not raw inverse seconds.
    The stopped optimizer boundary is controlled; recovery uses real SciPy.
    """
    evaluator = toy_evaluator
    continuum = ContinuumRateCoefficients(
        np.full(2, rate_scale),
        np.zeros(2),
        np.full(2, rate_scale / 1e9),
        np.zeros(2),
        np.zeros(2),
    )
    rates = BoundBoundMatrixRates(
        2,
        np.array([0, 1]),
        np.array([1, 0]),
        np.zeros(2),
        np.zeros(2),
        rate_scale * np.array([1.0, 2.0]),
        np.array([-1, -1]),
    )
    monkeypatch.setattr(
        "tardis.plasma.equilibrium.evaluator.root",
        lambda *args, **kwargs: SimpleNamespace(
            x=np.array([0.8, 0.2]), success=False, nfev=1
        ),
    )
    arguments = (
        0,
        1e9,
        1e4,
        np.array([0.5, 0.5]),
        continuum,
        evaluator.shell_number_densities[0],
        evaluator.sobolev_inputs[0],
        rates,
    )
    if fallback_stalls:
        monkeypatch.setattr(
            "tardis.plasma.equilibrium.evaluator.least_squares",
            lambda *args, **kwargs: SimpleNamespace(
                x=np.array([0.8, 0.2]), success=True, nfev=1
            ),
        )
        with pytest.raises(ValueError, match="relative level residual"):
            evaluator._calculate_level_solution(*arguments)
        assert not evaluator._level_solution_cache
        return
    result = evaluator._calculate_level_solution(*arguments)
    assert_allclose(result[0], [0.5, 0.5], rtol=1e-8, atol=0.0)


def test_equilibrium_terminal_validation_rejects_unclosed_levels() -> None:
    """Exact charge balance cannot compensate for a 20% kinetic imbalance.

    This is acceptance wiring, not a kinetic solver verification. The reported
    relative residual is independently prescribed while charge is exactly zero.
    """

    class UnclosedEvaluator:
        def evaluate(
            self,
            electron_density: npt.NDArray[np.float64],
            electron_temperature: npt.NDArray[np.float64],
            level_seed: pd.DataFrame,
        ) -> SimpleNamespace:
            return SimpleNamespace(
                electron_residual=pd.Series([0.0]),
                normalized_population=level_seed,
                relative_level_residual=pd.DataFrame([[0.0], [0.2]]),
            )

    with pytest.raises(RuntimeError, match=r"level.*residual"):
        PlasmaEquilibriumSolver(fixed_temperature_ratio=0.9).solve(
            UnclosedEvaluator(),
            [1e9],
            [9000.0],
            pd.DataFrame([[0.5], [0.5]]),
            [1e10],
            [1e4],
        )


def test_final_level_closure_uses_absolute_populations_and_continuum_ratio(
    toy_evaluator: PlasmaEquilibriumEvaluator,
) -> None:
    """Rebuild the physical final rates, not trial fractions or trial beta.

    Two levels with b=[0.8,0.2], n_ion/n_neutral=0.25, ionization 1 s^-1,
    Case B recombination 1 s^-1, and radiative up/down rates beta, 2beta.
    The final optical depth is exactly 7, and the excited balance is
    0.4*beta+0.05 with gross flow 1.2*beta+0.45. The trial seed [0.5,0.5]
    and its eliminated continuum ratio differ deliberately. Atol=0 ensures
    the positive imbalance cannot be hidden; rtol=1e-12 allows roundoff.
    """
    evaluator = toy_evaluator
    evaluator.sobolev_inputs = (
        replace(evaluator.sobolev_inputs[0], tau_coefficient=np.array([1e-9])),
    )
    levels = pd.DataFrame(
        [[8e9], [2e9]], index=evaluator.level_population_index, columns=[0]
    )
    ions = pd.DataFrame(
        [[1e10], [2.5e9]],
        index=pd.MultiIndex.from_tuples([(1, 0), (1, 1)]),
        columns=[0],
    )
    continuum = ContinuumRateCoefficients(
        np.ones(2), np.zeros(2), np.full(2, 1e-9), np.zeros(2), np.zeros(2)
    )
    state = SimpleNamespace(
        fractions=np.array([[0.5, 0.5]]),
        species_index=evaluator.level_population_index,
        continuum_rate_coefficients=(continuum,),
        bound_bound_rates=(None,),
    )
    residual = evaluator._calculate_final_residual(
        pd.Series([1e9]),
        np.array([1e4]),
        levels,
        state,
        normalize_residual=True,
        ion_population=ions,
    )
    beta = -np.expm1(-7.0) / 7.0
    denominator = (
        1.2 * beta
        + 0.45
        + 64
        * np.finfo(float).eps
        / evaluator.level_residual_tolerance
        * (1.25 + 2.4 * beta)
    )
    ground_denominator = (
        0.8
        + 1.2 * beta
        + 64
        * np.finfo(float).eps
        / evaluator.level_residual_tolerance
        * (1.25 + 2.4 * beta)
    )
    assert_allclose(
        residual.to_numpy()[:, 0],
        [
            (-0.8 - 0.4 * beta) / ground_denominator,
            (0.4 * beta + 0.05) / denominator,
        ],
        rtol=1e-12,
        atol=0.0,
    )


def test_rate_normalization_has_zero_turnover_limit(
    toy_evaluator: PlasmaEquilibriumEvaluator,
) -> None:
    """Absent transitions imply zero kinetic imbalance, not 0/0.

    Verification is the exact no-flow limiting case: zero ionization and
    bound-bound rates produce zero continuum ratio and source. Recombination
    coefficients are positive but no upper ions are populated. Any normalized
    bound distribution conserves population. Absolute tolerance is zero.
    """
    empty_integer = np.array([], dtype=np.int64)
    empty_float = np.array([], dtype=np.float64)
    rates = BoundBoundMatrixRates(
        2,
        empty_integer,
        empty_integer,
        empty_float,
        empty_float,
        empty_float,
        empty_integer,
    )
    residual, _, ratio = calculate_nlte_level_population_residual(
        np.array([0.6, 0.4]),
        LevelEquationRates(np.zeros(2), np.ones(2), np.zeros((2, 2))),
        toy_evaluator.rate_matrix_solver,
        None,
        None,
        (1, 0),
        toy_evaluator.shell_number_densities[0],
        toy_evaluator.sobolev_inputs[0],
        bound_bound_rates=rates,
        normalize_residual=True,
    )
    assert_allclose(residual, 0.0, atol=0.0)
    assert ratio == 0.0


def test_final_closure_retains_ground_kinetics_when_excited_levels_close(
    toy_evaluator: PlasmaEquilibriumEvaluator,
) -> None:
    """Ground-only continuum imbalance must not disappear into normalization.

    With no upward pumping, all neutral atoms are in the ground state and
    the excited equation is exactly closed. Ground ionization is 1 s^-1
    while recombination supplies only 0.25 s^-1 per neutral atom. Its net
    -0.75 s^-1 and gross 1.25 s^-1 must remain in final kinetic validation.
    Verification uses these analytic flows with roundoff-only rtol=1e-14.
    """
    evaluator = toy_evaluator
    evaluator.case_b_species = frozenset()
    evaluator.j_blues.iloc[:, :] = 0.0
    levels = pd.DataFrame(
        [[1e10], [0.0]], index=evaluator.level_population_index, columns=[0]
    )
    ions = pd.DataFrame(
        [[1e10], [2.5e9]],
        index=pd.MultiIndex.from_tuples([(1, 0), (1, 1)]),
        columns=[0],
    )
    continuum = ContinuumRateCoefficients(
        np.array([1.0, 0.0]),
        np.zeros(2),
        np.array([1e-9, 0.0]),
        np.zeros(2),
        np.zeros(2),
    )
    state = SimpleNamespace(
        fractions=np.array([[1.0, 0.0]]),
        species_index=evaluator.level_population_index,
        continuum_rate_coefficients=(continuum,),
        bound_bound_rates=(None,),
    )
    actual = evaluator._calculate_final_residual(
        pd.Series([1e9]),
        np.array([1e4]),
        levels,
        state,
        normalize_residual=True,
        ion_population=ions,
    )
    scale = 1.25 * (
        1 + 64 * np.finfo(float).eps / evaluator.level_residual_tolerance
    )
    assert_allclose(
        actual.to_numpy()[:, 0], [-0.75 / scale, 0.0], rtol=1e-14, atol=0.0
    )
