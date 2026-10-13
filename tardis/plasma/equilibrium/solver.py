"""Coordinate normalized electron-density and temperature equilibrium roots."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import block_diag

logger = logging.getLogger(__name__)


class EquilibriumEvaluator(Protocol):
    """Evaluator interface consumed by :class:`PlasmaEquilibriumSolver`."""

    def evaluate(
        self,
        trial_electron_density: npt.NDArray[np.float64],
        electron_temperature: npt.NDArray[np.float64],
        level_seed: pd.DataFrame,
    ) -> EquilibriumEvaluation:
        """Return electron and fractional-heating residuals."""


class EquilibriumEvaluation(Protocol):
    """Minimum evaluator output required by the normalized root solver."""

    electron_residual: pd.Series | None
    fractional_heating: pd.Series | None
    normalized_population: pd.DataFrame


@dataclass(frozen=True)
class PlasmaEquilibriumSolution:
    """Accepted dimensional state and terminal evaluator output."""

    electron_density: npt.NDArray[np.float64]
    electron_temperature: npt.NDArray[np.float64]
    evaluation: EquilibriumEvaluation


class PlasmaEquilibriumSolver:
    """Solve an evaluator in normalized density-temperature coordinates."""

    def __init__(
        self,
        minimum_electron_temperature: float = 1500.0,
        maximum_temperature_ratio: float = 1.5,
        maximum_function_evaluations: int = 100,
        residual_tolerance: float = 1.0e-7,
    ) -> None:
        """Configure physical bounds and nonlinear-solver effort."""
        self.minimum_electron_temperature = minimum_electron_temperature
        self.maximum_temperature_ratio = maximum_temperature_ratio
        self.maximum_function_evaluations = maximum_function_evaluations
        self.residual_tolerance = residual_tolerance

    def solve(
        self,
        evaluator: EquilibriumEvaluator,
        initial_electron_density: npt.ArrayLike,
        initial_electron_temperature: npt.ArrayLike,
        level_seed: pd.DataFrame,
        maximum_electron_density: npt.ArrayLike,
        radiation_temperature: npt.ArrayLike,
    ) -> PlasmaEquilibriumSolution:
        """Solve charge and thermal residuals with fixed rate inputs.

        Parameters
        ----------
        evaluator : EquilibriumEvaluator
            Evaluator whose rate inputs remain fixed during this solve.
        initial_electron_density : array-like
            Initial electron density in inverse cubic centimeters.
        initial_electron_temperature : array-like
            Initial electron temperature in kelvin.
        level_seed : pandas.DataFrame
            Initial selected-species normalized level populations.
        maximum_electron_density : array-like
            Full-ionization electron-density scale.
        radiation_temperature : array-like
            Radiation-temperature scale for the thermal unknown.

        Returns
        -------
        PlasmaEquilibriumSolution
            Accepted dimensional state and terminal evaluation.
        """
        initial_density = np.asarray(initial_electron_density, dtype=np.float64)
        initial_temperature = np.asarray(
            initial_electron_temperature, dtype=np.float64
        )
        maximum_density = np.asarray(maximum_electron_density, dtype=np.float64)
        radiation_temperature = np.asarray(
            radiation_temperature, dtype=np.float64
        )
        shell_count = len(initial_density)
        solve_start_time = perf_counter()
        logger.info(
            "Starting plasma equilibrium solve for %d shells (%d variables; "
            "maximum function evaluations %d).",
            shell_count,
            2 * shell_count,
            self.maximum_function_evaluations,
        )
        initial_guess = np.empty(2 * shell_count)
        initial_guess[::2] = initial_density / maximum_density
        initial_guess[1::2] = initial_temperature / radiation_temperature
        lower_bound = np.empty_like(initial_guess)
        lower_bound[::2] = np.finfo(float).eps
        lower_bound[1::2] = (
            self.minimum_electron_temperature / radiation_temperature
        )
        upper_bound = np.empty_like(initial_guess)
        upper_bound[::2] = 1.0
        upper_bound[1::2] = self.maximum_temperature_ratio
        initial_guess = np.clip(initial_guess, lower_bound, upper_bound)
        jacobian_sparsity = block_diag(
            [np.ones((2, 2), dtype=bool)] * shell_count,
            format="csr",
        )
        residual_evaluation_count = 0

        def residual(
            candidate: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            nonlocal residual_evaluation_count
            residual_evaluation_count += 1
            evaluation_idx = residual_evaluation_count
            evaluation_start_time = perf_counter()
            logger.info(
                "Starting plasma equilibrium residual evaluation %d.",
                evaluation_idx,
            )
            evaluation = evaluator.evaluate(
                maximum_density * candidate[::2],
                radiation_temperature * candidate[1::2],
                level_seed,
            )
            if (
                evaluation.electron_residual is None
                or evaluation.fractional_heating is None
            ):
                raise ValueError(
                    "The evaluator must provide electron and thermal residuals."
                )
            result = np.empty_like(candidate)
            result[::2] = np.asarray(
                evaluation.electron_residual, dtype=np.float64
            )
            result[1::2] = np.asarray(
                evaluation.fractional_heating, dtype=np.float64
            )
            logger.info(
                "Finished plasma equilibrium residual evaluation %d in %.3f "
                "s (maximum absolute residual %.3e).",
                evaluation_idx,
                perf_counter() - evaluation_start_time,
                float(np.max(np.abs(result))),
            )
            return result

        least_squares_start_time = perf_counter()
        root = least_squares(
            residual,
            initial_guess,
            bounds=(lower_bound, upper_bound),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=self.maximum_function_evaluations,
            method="trf",
            jac_sparsity=jacobian_sparsity,
        )
        logger.info(
            "Finished plasma equilibrium least-squares optimization in %.3f "
            "s (residual evaluations=%d, nfev=%d, njev=%s, cost=%.3e, "
            "status=%d).",
            perf_counter() - least_squares_start_time,
            residual_evaluation_count,
            root.nfev,
            root.njev,
            root.cost,
            root.status,
        )
        if not root.success:
            raise RuntimeError(
                "Plasma equilibrium solve failed: " + root.message
            )
        electron_density = maximum_density * root.x[::2]
        electron_temperature = radiation_temperature * root.x[1::2]
        terminal_validation_start_time = perf_counter()
        seed_evaluation = evaluator.evaluate(
            electron_density,
            electron_temperature,
            level_seed,
        )
        terminal_evaluation = evaluator.evaluate(
            electron_density,
            electron_temperature,
            seed_evaluation.normalized_population,
        )
        terminal_residual = np.concatenate(
            [
                np.asarray(
                    terminal_evaluation.electron_residual,
                    dtype=np.float64,
                ),
                np.asarray(
                    terminal_evaluation.fractional_heating,
                    dtype=np.float64,
                ),
            ]
        )
        if not np.all(np.isfinite(terminal_residual)):
            raise RuntimeError(
                "Plasma equilibrium solve returned nonfinite residuals."
            )
        maximum_residual = float(np.max(np.abs(terminal_residual)))
        if maximum_residual > self.residual_tolerance:
            raise RuntimeError(
                "Plasma equilibrium solve did not close its normalized "
                f"residuals: {maximum_residual:.3e} > "
                f"{self.residual_tolerance:.3e}."
            )
        logger.info(
            "Finished plasma equilibrium terminal validation in %.3f s "
            "after %d total evaluator calls (maximum residual %.3e; total "
            "solve time %.3f s).",
            perf_counter() - terminal_validation_start_time,
            residual_evaluation_count + 2,
            maximum_residual,
            perf_counter() - solve_start_time,
        )
        return PlasmaEquilibriumSolution(
            electron_density,
            electron_temperature,
            terminal_evaluation,
        )
