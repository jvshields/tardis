from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from tardis.plasma.equilibrium.solver import PlasmaEquilibriumSolver


class AnalyticEvaluator:
    """Evaluator with a prescribed density and temperature root."""

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(
        self,
        electron_density: np.ndarray,
        electron_temperature: np.ndarray,
        level_seed: pd.DataFrame,
    ) -> SimpleNamespace:
        self.calls += 1
        columns = level_seed.columns
        normalized_population = level_seed / level_seed.sum(axis=0)
        return SimpleNamespace(
            electron_residual=pd.Series(
                electron_density / 2.0e9 - 1.0, index=columns
            ),
            fractional_heating=pd.Series(
                electron_temperature / 8000.0 - 1.0, index=columns
            ),
            normalized_population=normalized_population,
        )


def test_equilibrium_solver_finds_normalized_density_temperature_root() -> None:
    """Solve electron and thermal residuals and rebuild with accepted levels."""
    evaluator = AnalyticEvaluator()
    level_seed = pd.DataFrame([[0.8], [0.2]], columns=[0])

    solution = PlasmaEquilibriumSolver().solve(
        evaluator,
        np.array([1.0e9]),
        np.array([6000.0]),
        level_seed,
        np.array([1.0e10]),
        np.array([1.0e4]),
    )

    npt.assert_allclose(solution.electron_density, [2.0e9], rtol=1e-10)
    npt.assert_allclose(solution.electron_temperature, [8000.0], rtol=1e-10)
    npt.assert_allclose(solution.evaluation.electron_residual, 0.0, atol=1e-10)
    npt.assert_allclose(solution.evaluation.fractional_heating, 0.0, atol=1e-10)
    assert evaluator.calls > 2


def test_equilibrium_solver_reports_residual_evaluation_progress(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Report nonlinear residual evaluations without changing the solution.

    Claim
    -----
    A long equilibrium solve exposes its residual-evaluation count and
    least-squares completion through INFO logging.

    Regime
    ------
    One shell has independent linear density and temperature residuals.

    Verification
    ------------
    The test checks orchestration messages only; the separate analytic-root
    test verifies the numerical result.
    """
    caplog.set_level("INFO", logger="tardis.plasma.equilibrium.solver")
    evaluator = AnalyticEvaluator()

    PlasmaEquilibriumSolver().solve(
        evaluator,
        np.array([1.0e9]),
        np.array([6000.0]),
        pd.DataFrame([[0.8], [0.2]], columns=[0]),
        np.array([1.0e10]),
        np.array([1.0e4]),
    )

    assert "Starting plasma equilibrium residual evaluation 1" in caplog.text
    assert "Finished plasma equilibrium residual evaluation 1" in caplog.text
    assert (
        "Finished plasma equilibrium least-squares optimization" in caplog.text
    )
    assert "Finished plasma equilibrium terminal validation" in caplog.text


def test_equilibrium_solver_supplies_shell_local_jacobian_sparsity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restrict finite differences to independent shell-local variables.

    Claim
    -----
    Each shell's charge and thermal residuals depend only on that shell's
    electron density and electron temperature.

    Regime
    ------
    Three shells ordered as ``(density, temperature)`` variable pairs.

    Verification
    ------------
    The exact sparsity matrix passed to SciPy contains three independent
    two-by-two diagonal blocks and no cross-shell entries.
    """
    captured_arguments = {}

    def capture_least_squares(
        residual: object,
        initial_guess: np.ndarray,
        **kwargs: object,
    ) -> SimpleNamespace:
        del residual
        captured_arguments.update(kwargs)
        return SimpleNamespace(
            success=True,
            message="test solution",
            x=initial_guess,
            nfev=1,
            njev=1,
            cost=0.0,
            status=1,
        )

    monkeypatch.setattr(
        "tardis.plasma.equilibrium.solver.least_squares",
        capture_least_squares,
    )
    shell_columns = pd.RangeIndex(3)
    level_seed = pd.DataFrame(np.tile([[0.8], [0.2]], 3), columns=shell_columns)

    PlasmaEquilibriumSolver().solve(
        AnalyticEvaluator(),
        np.full(3, 2.0e9),
        np.full(3, 8000.0),
        level_seed,
        np.full(3, 1.0e10),
        np.full(3, 1.0e4),
    )

    expected_sparsity = np.zeros((6, 6), dtype=bool)
    for shell_idx in range(3):
        variable_slice = slice(2 * shell_idx, 2 * shell_idx + 2)
        expected_sparsity[variable_slice, variable_slice] = True
    npt.assert_array_equal(
        captured_arguments["jac_sparsity"].toarray(), expected_sparsity
    )
