"""Compose high-energy transport with coupled nonthermal helium equilibrium."""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy import units as u

from tardis import constants as const
from tardis.energy_input.gamma_ray_transport import (
    calculate_ejecta_velocity_volume,
)
from tardis.io.atom_data.parse_atom_data import parse_atom_data
from tardis.opacities.macro_atom.macroatom_solver import (
    BoundBoundMacroAtomSolver,
)
from tardis.opacities.opacity_solver import OpacitySolver
from tardis.opacities.opacity_state import OpacityState
from tardis.opacities.tau_sobolev import (
    SOBOLEV_COEFFICIENT,
    calculate_beta_sobolev,
    calculate_sobolev_line_opacity,
)
from tardis.plasma.assembly import PlasmaSolverFactory
from tardis.plasma.equilibrium.evaluator import (
    PlasmaEquilibriumEvaluator,
    calculate_lte_populations,
)
from tardis.plasma.equilibrium.inputs import (
    FixedRateState,
    ShellNumberDensity,
    SobolevInputs,
)
from tardis.plasma.equilibrium.ion_populations import IonPopulationSolver
from tardis.plasma.equilibrium.rate_matrix import (
    AnalyticIonRateMatrix,
    RateMatrix,
)
from tardis.plasma.equilibrium.rates import (
    AnalyticPhotoionizationRateSolver,
    CollisionalIonizationRateSolver,
    ThermalCollisionalRateSolver,
)
from tardis.plasma.equilibrium.rates.heating_cooling_rates import (
    BoundFreeThermalRates,
    CollisionalBoundThermalRates,
    CollisionalIonizationThermalRates,
    FreeFreeThermalRates,
)
from tardis.plasma.equilibrium.rates.radiative_rates import RadiativeRatesSolver
from tardis.plasma.equilibrium.solver import (
    PlasmaEquilibriumSolution,
    PlasmaEquilibriumSolver,
)
from tardis.plasma.equilibrium.thermal_balance import ThermalBalanceSolver
from tardis.plasma.nonthermal.pynonthermal_solver import (
    NonthermalPlasmaInput,
    NonthermalPlasmaState,
)
from tardis.plasma.radiation_field import DilutePlanckianRadiationField
from tardis.transport.montecarlo.configuration import montecarlo_globals
from tardis.transport.montecarlo.modes.classic.solver import (
    MCTransportSolverClassic,
)
from tardis.workflows.high_energy.deposition import (
    DepositionState,
    prepare_deposition_state,
)
from tardis.workflows.high_energy.tardis_he_workflow import (
    TARDISHEWorkflow,
    TARDISHEWorkflowResult,
)
from tardis.workflows.standard_tardis_workflow import StandardTARDISWorkflow

logger = logging.getLogger(__name__)


class NonthermalRateSolver(Protocol):
    """Nonthermal rate provider used by the coupled plasma solver."""

    def solve(
        self, plasma_input: NonthermalPlasmaInput
    ) -> NonthermalPlasmaState:
        """Return fixed rates for one candidate plasma state."""


class OpticalPlasma(Protocol):
    """Mutable optical-plasma fields required by helium coupling."""

    electron_densities: pd.Series
    t_electrons: npt.ArrayLike
    ion_number_density: pd.DataFrame
    level_number_density: pd.DataFrame
    number_density: pd.DataFrame

    def update(self, **kwargs: object) -> None:
        """Commit an accepted plasma state."""


class OpticalSimulationState(Protocol):
    """Optical model fields required by helium coupling."""

    t_radiative: npt.ArrayLike
    time_explosion: u.Quantity
    v_inner: u.Quantity
    v_outer: u.Quantity


class OpticalWorkflow(Protocol):
    """Optical workflow state consumed by the coupled plasma solver."""

    plasma_solver: OpticalPlasma
    simulation_state: OpticalSimulationState

    def run(self) -> None:
        """Run the optical Monte Carlo iteration."""

    def build_helium_evaluator(
        self,
        plasma_state: CoupledPlasmaState,
        fixed_rates: FixedRateState,
        bound_bound_species_filter: tuple[int, int] | None,
    ) -> PlasmaEquilibriumEvaluator:
        """Build an evaluator from the current continuum estimators."""


@dataclass(frozen=True)
class HighEnergySettings:
    """Arguments for one frozen high-energy transport calculation."""

    time_start: float
    time_end: float
    number_of_packets: int
    time_steps: int
    time_space: str
    seed: int
    positronium_fraction: float
    spectrum_bins: int
    grey_opacity: float = -1.0

    def as_run_arguments(self) -> dict[str, object]:
        """Return names accepted by :meth:`TARDISHEWorkflow.run`."""
        return {
            "time_start": self.time_start,
            "time_end": self.time_end,
            "number_of_packets": self.number_of_packets,
            "time_steps": self.time_steps,
            "time_space": self.time_space,
            "seed": self.seed,
            "fp": self.positronium_fraction,
            "spectrum_bins": self.spectrum_bins,
            "grey_opacity": self.grey_opacity,
        }


@dataclass(frozen=True)
class NonthermalCouplingSettings:
    """Convergence and relaxation settings for the outer Picard iteration."""

    maximum_iterations: int = 20
    relative_tolerance: float = 1.0e-3
    absolute_population_tolerance: float = 1.0e-20
    relaxation: float = 0.5


@dataclass(frozen=True)
class CoupledPlasmaState:
    """Candidate plasma fields refreshed between nonthermal solves."""

    electron_density: npt.NDArray[np.float64]
    electron_temperature: npt.NDArray[np.float64]
    ion_number_density: pd.DataFrame
    level_number_density: pd.DataFrame


@dataclass(frozen=True)
class NonthermalCouplingResult:
    """Converged plasma, nonthermal, and iteration diagnostic state."""

    plasma_state: CoupledPlasmaState
    nonthermal_state: NonthermalPlasmaState
    equilibrium_solution: PlasmaEquilibriumSolution
    convergence_history: pd.DataFrame


EvaluatorFactory = Callable[
    [
        CoupledPlasmaState,
        FixedRateState,
        OpticalWorkflow,
        tuple[int, int] | None,
    ],
    PlasmaEquilibriumEvaluator,
]


class NonthermalHeliumPlasmaSolver:
    """Iterate pynonthermal outside frozen evaluator equilibrium solves."""

    def __init__(
        self,
        nonthermal_solver: NonthermalRateSolver,
        evaluator_factory: EvaluatorFactory | None = None,
        equilibrium_solver: PlasmaEquilibriumSolver | None = None,
        settings: NonthermalCouplingSettings = NonthermalCouplingSettings(),
        bound_bound_species_filter: tuple[int, int] | None = (2, 0),
    ) -> None:
        """Configure coupled nonthermal and equilibrium owners.

        Parameters
        ----------
        nonthermal_solver : NonthermalRateSolver
            Provider of fixed nonthermal rates for candidate plasma states.
        evaluator_factory : EvaluatorFactory, optional
            Override builder receiving the candidate state, fixed rates,
            optical workflow, and optional bound-bound species filter. When
            omitted, the optical workflow owns evaluator construction.
        equilibrium_solver : PlasmaEquilibriumSolver, optional
            Normalized charge and thermal-balance solver.
        settings : NonthermalCouplingSettings, optional
            Picard convergence and relaxation settings.
        bound_bound_species_filter : tuple[int, int] or None, optional
            Atomic number and zero-based ion number whose bound-bound inputs
            the evaluator factory should retain. ``None`` requests no line
            filtering. The helium workflow defaults to He I, ``(2, 0)``.
        """
        self.nonthermal_solver = nonthermal_solver
        self.evaluator_factory = evaluator_factory
        self.equilibrium_solver = (
            equilibrium_solver or PlasmaEquilibriumSolver()
        )
        self.bound_bound_species_filter = bound_bound_species_filter
        self.settings = settings
        self.deposition_state: DepositionState | None = None
        self.last_result: NonthermalCouplingResult | None = None

    @staticmethod
    def _maximum_relative_change(
        candidate: npt.ArrayLike,
        previous: npt.ArrayLike,
        floor: float,
    ) -> float:
        """Return a floored componentwise relative change."""
        candidate = np.asarray(candidate, dtype=np.float64)
        previous = np.asarray(previous, dtype=np.float64)
        return float(
            np.max(
                np.abs(candidate - previous)
                / np.maximum(np.abs(previous), floor)
            )
        )

    @staticmethod
    def _rate_change(
        candidate: FixedRateState,
        previous: FixedRateState | None,
    ) -> float:
        """Return the largest fixed ionization or heating change."""
        if previous is None:
            return np.inf
        changes = []
        for candidate_rate, previous_rate in (
            (candidate.ionization_rates, previous.ionization_rates),
            (candidate.heating_rate_density, previous.heating_rate_density),
        ):
            if candidate_rate is None and previous_rate is None:
                continue
            if candidate_rate is None or previous_rate is None:
                return np.inf
            changes.append(
                NonthermalHeliumPlasmaSolver._maximum_relative_change(
                    candidate_rate,
                    previous_rate,
                    np.finfo(float).tiny,
                )
            )
        return max(changes, default=0.0)

    def _initial_state(
        self, optical_workflow: OpticalWorkflow
    ) -> CoupledPlasmaState:
        """Copy the current optical plasma state."""
        plasma = optical_workflow.plasma_solver
        return CoupledPlasmaState(
            np.asarray(plasma.electron_densities, dtype=np.float64),
            np.asarray(plasma.t_electrons, dtype=np.float64),
            plasma.ion_number_density.copy(deep=True),
            plasma.level_number_density.copy(deep=True),
        )

    def solve(
        self, optical_workflow: OpticalWorkflow
    ) -> NonthermalCouplingResult:
        """Converge and commit one optical iteration's helium plasma state."""
        if self.deposition_state is None:
            raise RuntimeError("High-energy deposition has not been prepared.")
        current = self._initial_state(optical_workflow)
        plasma = optical_workflow.plasma_solver
        shell_columns = plasma.number_density.columns
        logger.info(
            "Solving coupled nonthermal helium plasma in all %d shells: %s",
            len(shell_columns),
            shell_columns.tolist(),
        )
        maximum_electron_density = (
            plasma.number_density.multiply(
                plasma.number_density.index.to_numpy(dtype=float), axis=0
            )
            .sum()
            .to_numpy(dtype=float)
        )
        complete_radiation_temperature = np.asarray(
            optical_workflow.simulation_state.t_radiative, dtype=np.float64
        )
        previous_fixed_rates = None
        convergence_rows = []
        terminal_nonthermal = None
        terminal_solution = None

        for iteration in range(1, self.settings.maximum_iterations + 1):
            nonthermal_state = self.nonthermal_solver.solve(
                NonthermalPlasmaInput(
                    self.deposition_state.rate_density,
                    current.electron_density / u.cm**3,
                    current.electron_temperature * u.K,
                    current.ion_number_density,
                )
            )
            if self.evaluator_factory is None:
                evaluator = optical_workflow.build_helium_evaluator(
                    current,
                    nonthermal_state.fixed_rates,
                    self.bound_bound_species_filter,
                )
            else:
                evaluator = self.evaluator_factory(
                    current,
                    nonthermal_state.fixed_rates,
                    optical_workflow,
                    self.bound_bound_species_filter,
                )
            he_i_level_seed = current.level_number_density.loc[(2, 0)].copy(
                deep=True
            )
            level_seed = he_i_level_seed.divide(
                he_i_level_seed.sum(axis=0), axis="columns"
            )
            if not np.isfinite(level_seed.to_numpy(dtype=float)).all():
                raise ValueError(
                    "He I level populations cannot provide a normalized "
                    "equilibrium seed."
                )
            solution = self.equilibrium_solver.solve(
                evaluator,
                current.electron_density,
                current.electron_temperature,
                level_seed,
                maximum_electron_density,
                complete_radiation_temperature,
            )
            evaluation = solution.evaluation
            if evaluation.ion_population is None:
                raise RuntimeError(
                    "The evaluator did not solve ion populations."
                )
            candidate = CoupledPlasmaState(
                solution.electron_density,
                solution.electron_temperature,
                evaluation.ion_population,
                evaluation.absolute_level_population,
            )
            density_change = self._maximum_relative_change(
                candidate.electron_density,
                current.electron_density,
                np.finfo(float).tiny,
            )
            temperature_change = self._maximum_relative_change(
                candidate.electron_temperature,
                current.electron_temperature,
                np.finfo(float).tiny,
            )
            level_change = self._maximum_relative_change(
                candidate.level_number_density,
                current.level_number_density,
                self.settings.absolute_population_tolerance,
            )
            candidate_helium = candidate.ion_number_density.xs(
                2, level="atomic_number"
            )
            current_helium = current.ion_number_density.xs(
                2, level="atomic_number"
            )
            candidate_he_ii_fraction = candidate_helium.loc[1] / (
                candidate_helium.sum(axis=0)
            )
            current_he_ii_fraction = current_helium.loc[1] / (
                current_helium.sum(axis=0)
            )
            he_ii_fraction_change = float(
                np.max(
                    np.abs(candidate_he_ii_fraction - current_he_ii_fraction)
                )
            )
            nonthermal_change = self._rate_change(
                nonthermal_state.fixed_rates, previous_fixed_rates
            )
            maximum_change = max(
                density_change,
                temperature_change,
                he_ii_fraction_change,
                level_change,
                nonthermal_change,
            )
            convergence_rows.append(
                {
                    "iteration": iteration,
                    "electron_density": density_change,
                    "electron_temperature": temperature_change,
                    "he_ii_fraction": he_ii_fraction_change,
                    "level_population": level_change,
                    "nonthermal_rates": nonthermal_change,
                    "maximum_change": maximum_change,
                }
            )
            logger.info(
                "Nonthermal helium iteration %d/%d: maximum change %.3e",
                iteration,
                self.settings.maximum_iterations,
                maximum_change,
            )
            terminal_nonthermal = nonthermal_state
            terminal_solution = solution
            if maximum_change < self.settings.relative_tolerance:
                current = candidate
                break
            relaxation = self.settings.relaxation
            current = CoupledPlasmaState(
                relaxation * candidate.electron_density
                + (1.0 - relaxation) * current.electron_density,
                relaxation * candidate.electron_temperature
                + (1.0 - relaxation) * current.electron_temperature,
                relaxation * candidate.ion_number_density
                + (1.0 - relaxation) * current.ion_number_density,
                relaxation * candidate.level_number_density
                + (1.0 - relaxation) * current.level_number_density,
            )
            previous_fixed_rates = nonthermal_state.fixed_rates
        else:
            worst = convergence_rows[-1]["maximum_change"]
            raise RuntimeError(
                "Nonthermal helium coupling did not converge after "
                f"{self.settings.maximum_iterations} iterations; "
                f"maximum relative change={worst:.3e}."
            )

        if terminal_nonthermal is None or terminal_solution is None:
            raise RuntimeError("No nonthermal helium iteration was performed.")
        plasma.update(
            electron_densities=pd.Series(
                current.electron_density, index=shell_columns
            ),
            ion_number_density=current.ion_number_density,
            level_number_density=current.level_number_density,
            link_t_rad_t_electron=(
                current.electron_temperature / complete_radiation_temperature
            ),
        )
        history = pd.DataFrame(convergence_rows).set_index("iteration")
        self.last_result = NonthermalCouplingResult(
            current,
            terminal_nonthermal,
            terminal_solution,
            history,
        )
        return self.last_result


class NonthermalHeliumOpticalWorkflow(StandardTARDISWorkflow):
    """Classic optical workflow specialized for helium coupling."""

    plasma_coupling_solver: NonthermalHeliumPlasmaSolver

    def __init__(
        self,
        configuration: object,
        enable_virtual_packet_logging: bool = False,
        log_level: str | None = None,
        specific_log_level: str | None = None,
        show_progress_bars: bool = True,
        show_convergence_plots: bool = False,
        convergence_plots_kwargs: Mapping[str, object] | None = None,
        csvy: bool = False,
        continuum_interaction_species: Collection[str] = ("He I", "He II"),
        case_b_species: Collection[tuple[int, int]] = (),
    ) -> None:
        """Initialize classic transport and helium equilibrium.

        The standard workflow initialization is reused for its simulation and
        convergence state. The plasma retains helium photoionization data for
        analytic equilibrium rates, while packet transport uses the classic
        bound-bound macro-atom implementation.

        Parameters
        ----------
        configuration : object
            TARDIS configuration.
        enable_virtual_packet_logging : bool, optional
            Enable virtual-packet logging.
        log_level : str, optional
            Global workflow log level.
        specific_log_level : str, optional
            Component-specific workflow log level.
        show_progress_bars : bool, optional
            Display progress bars.
        show_convergence_plots : bool, optional
            Display convergence plots.
        convergence_plots_kwargs : mapping, optional
            Convergence-plot configuration.
        csvy : bool, optional
            Read the ejecta model from CSVY.
        continuum_interaction_species : collection of str, optional
            Species whose photoionization data are exposed to the equilibrium
            evaluator. These transitions are not enabled in packet transport.
        case_b_species : collection of tuple[int, int], optional
            Species for which direct ground-state recombination is suppressed.
            The default is Case A for helium.
        """
        super().__init__(
            configuration,
            enable_virtual_packet_logging=enable_virtual_packet_logging,
            log_level=log_level,
            specific_log_level=specific_log_level,
            show_progress_bars=show_progress_bars,
            show_convergence_plots=show_convergence_plots,
            convergence_plots_kwargs=convergence_plots_kwargs,
            csvy=csvy,
        )
        self.case_b_species = frozenset(case_b_species)
        montecarlo_globals.CONTINUUM_PROCESSES_ENABLED = False

        # ``StandardTARDISWorkflow`` prepares its atomic data for classic
        # transport. Atomic data preparation is intentionally one-shot, so a
        # fresh owner is required for the continuum selection below.
        self.atom_data = parse_atom_data(configuration)

        radiation_field = DilutePlanckianRadiationField(
            self.simulation_state.t_radiative,
            self.simulation_state.dilution_factor,
        )
        self.simulation_state.radiation_field_state = radiation_field
        elemental_number_density = (
            self.simulation_state.calculate_elemental_number_density(
                self.atom_data.atom_data.mass
            )
        )
        factory = PlasmaSolverFactory(self.atom_data, configuration)
        factory.helium_treatment = "none"
        factory.line_interaction_type = "macroatom"
        factory.continuum_interaction_species = list(
            continuum_interaction_species
        )
        factory.prepare_factory(
            self.simulation_state.abundance.index,
            "tardis.plasma.properties.property_collections",
            configuration,
            allow_continuum=True,
        )
        self.plasma_solver_factory = factory
        self.plasma_solver = factory.assemble(
            elemental_number_density,
            radiation_field,
            self.simulation_state.time_explosion,
            self.simulation_state._electron_densities,
        )
        self.plasma_solver.freeze(
            "electron_densities",
            "ion_number_density",
            "level_number_density",
        )

        self.opacity_solver = OpacitySolver(
            "macroatom", configuration.plasma.disable_line_scattering
        )
        self.macro_atom_solver = BoundBoundMacroAtomSolver(
            self.atom_data.levels,
            self.atom_data.lines,
            line_interaction_type="macroatom",
        )
        self.transport_state = None
        self.transport_solver = MCTransportSolverClassic.from_config(
            configuration,
            packet_source=self.simulation_state.packet_source,
            enable_virtual_packet_logging=enable_virtual_packet_logging,
        )
        self.transport_solver.line_interaction_type = "macroatom"

    def _current_plasma_state(self) -> CoupledPlasmaState:
        """Return an immutable copy of the current continuum plasma state."""
        return CoupledPlasmaState(
            np.asarray(self.plasma_solver.electron_densities, dtype=np.float64),
            np.asarray(self.plasma_solver.t_electrons, dtype=np.float64),
            self.plasma_solver.ion_number_density.copy(deep=True),
            self.plasma_solver.level_number_density.copy(deep=True),
        )

    def _update_sobolev_state(self) -> None:
        """Refresh line optical depths after accepting a plasma state."""
        self._tau_sobolev = calculate_sobolev_line_opacity(
            self.atom_data.lines,
            self.plasma_solver.level_number_density,
            self.plasma_solver.time_explosion,
            self.plasma_solver.stimulated_emission_factor,
        )
        self._beta_sobolev = calculate_beta_sobolev(self._tau_sobolev)

    def _build_helium_evaluator(
        self,
        plasma_state: CoupledPlasmaState,
        fixed_rates: FixedRateState | None,
        bound_bound_species_filter: tuple[int, int] | None,
        maximum_electron_density: npt.NDArray[np.float64],
    ) -> PlasmaEquilibriumEvaluator:
        """Build one helium evaluator with analytic bound-free rates."""
        plasma = self.plasma_solver
        shell_columns = plasma_state.ion_number_density.columns
        shell_positions = plasma.number_density.columns.get_indexer(
            shell_columns
        )
        if np.any(shell_positions < 0):
            raise ValueError(
                "Coupled shell labels are absent from the optical plasma."
            )
        equilibrium_species = (2, 0)
        level_index = plasma.level_number_density.index
        equilibrium_levels = plasma.atomic_data.levels.loc[level_index]
        photoionization_data = plasma.photo_ion_cross_sections.loc[
            plasma.photo_ion_cross_sections.index.isin(
                plasma.level2continuum_idx.index
            )
        ]
        level2continuum_edge_idx = plasma.level2continuum_idx
        species_level_positions = np.flatnonzero(
            (level_index.get_level_values("atomic_number") == 2)
            & (level_index.get_level_values("ion_number") == 0)
        )
        shell_number_densities = tuple(
            ShellNumberDensity(
                float(plasma.number_density.loc[2, shell]),
                plasma_state.level_number_density[shell].to_numpy(
                    dtype=np.float64
                ),
                species_level_positions,
            )
            for shell in shell_columns
        )

        all_line_index = plasma.lines.index
        all_line_species = all_line_index.droplevel(
            ["level_number_lower", "level_number_upper"]
        )
        if bound_bound_species_filter is None:
            line_positions = np.arange(len(all_line_index), dtype=np.int64)
        else:
            line_positions = np.flatnonzero(
                np.asarray(
                    all_line_species.isin([bound_bound_species_filter]),
                    dtype=bool,
                )
            )
        if len(line_positions) == 0:
            raise ValueError(
                "No bound-bound lines match "
                f"bound_bound_species_filter={bound_bound_species_filter}."
            )
        lines = plasma.lines.iloc[line_positions]
        line_index = lines.index
        lower_line_positions = plasma.lines_lower_level_index[line_positions]
        upper_line_positions = plasma.lines_upper_level_index[line_positions]
        sobolev_input = SobolevInputs(
            lower_line_positions,
            upper_line_positions,
            plasma.g.iloc[lower_line_positions].to_numpy(),
            plasma.g.iloc[upper_line_positions].to_numpy(),
            plasma.metastability.iloc[upper_line_positions].to_numpy(),
            np.asarray(
                line_index.droplevel(
                    ["level_number_lower", "level_number_upper"]
                ).isin([equilibrium_species]),
                dtype=bool,
            ),
            (
                lines.wavelength_cm.to_numpy()
                * lines.f_lu.to_numpy()
                * SOBOLEV_COEFFICIENT
                * plasma.time_explosion.to_value(u.s)
            ),
            np.arange(len(lines), dtype=np.int64),
            line_index,
        )
        collision_strengths = plasma.atomic_data.yg_data
        if bound_bound_species_filter is not None:
            collision_strengths = collision_strengths.loc[
                (
                    bound_bound_species_filter[0],
                    bound_bound_species_filter[1],
                    slice(None),
                    slice(None),
                ),
                :,
            ]
        rate_matrix_solver = RateMatrix(
            RadiativeRatesSolver(lines),
            ThermalCollisionalRateSolver(
                equilibrium_levels,
                lines,
                plasma.atomic_data.collision_data_temperatures,
                collision_strengths,
                collision_strengths_type="cmfgen",
            ),
            equilibrium_levels,
        )
        selected_number_density = plasma.number_density.loc[:, shell_columns]
        electron_density = pd.Series(
            plasma_state.electron_density, index=shell_columns
        )
        lte_ion_population, lte_level_population = calculate_lte_populations(
            plasma.thermal_phi_lte.loc[:, shell_columns],
            plasma.thermal_lte_partition_function.loc[:, shell_columns],
            selected_number_density,
            electron_density,
            plasma.thermal_lte_level_boltzmann_factor.loc[:, shell_columns],
            equilibrium_levels,
        )

        ion_rate_matrix = AnalyticIonRateMatrix(
            AnalyticPhotoionizationRateSolver(photoionization_data),
            CollisionalIonizationRateSolver(photoionization_data),
            fixed_rates=fixed_rates,
        )
        ion_population_solver = IonPopulationSolver(ion_rate_matrix)

        collisional_index = rate_matrix_solver.electron_rate_solver.all_collisional_strengths_index
        lower_level_index = pd.MultiIndex.from_arrays(
            [
                collisional_index.get_level_values("atomic_number"),
                collisional_index.get_level_values("ion_number"),
                collisional_index.get_level_values("level_number_lower"),
            ],
            names=["atomic_number", "ion_number", "level_number"],
        )
        upper_level_index = pd.MultiIndex.from_arrays(
            [
                collisional_index.get_level_values("atomic_number"),
                collisional_index.get_level_values("ion_number"),
                collisional_index.get_level_values("level_number_upper"),
            ],
            names=["atomic_number", "ion_number", "level_number"],
        )
        collisional_line_frequencies = (
            equilibrium_levels.energy.loc[upper_level_index].to_numpy()
            - equilibrium_levels.energy.loc[lower_level_index].to_numpy()
        ) / const.h.cgs.value
        radiation_field = DilutePlanckianRadiationField(
            plasma.dilute_planckian_radiation_field.temperature[
                shell_positions
            ],
            plasma.dilute_planckian_radiation_field.dilution_factor[
                shell_positions
            ],
        )
        thermal_balance_solver = ThermalBalanceSolver(
            BoundFreeThermalRates(photoionization_data),
            FreeFreeThermalRates(),
            CollisionalIonizationThermalRates(photoionization_data),
            CollisionalBoundThermalRates(
                pd.DataFrame({"nu": collisional_line_frequencies})
            ),
        )
        thermal_balance_arguments = {
            "free_free_heating_estimator": pd.Series(0.0, index=shell_columns),
            "radiation_field": radiation_field,
        }
        return PlasmaEquilibriumEvaluator(
            photoionization_data,
            level2continuum_edge_idx,
            None,
            None,
            None,
            equilibrium_levels,
            plasma.ionization_data,
            rate_matrix_solver,
            pd.DataFrame(
                np.asarray(plasma.j_blues)[line_positions][:, shell_positions],
                index=line_index,
                columns=shell_columns,
            ),
            shell_number_densities,
            tuple(sobolev_input for _ in shell_columns),
            level_index,
            equilibrium_species,
            selected_number_density,
            maximum_electron_density,
            ion_population_solver=ion_population_solver,
            ion_population_arguments={
                "radiation_field": radiation_field,
                "elemental_number_density": selected_number_density,
                "lte_level_population": lte_level_population,
                "lte_ion_population": lte_ion_population,
                "estimated_ion_population": plasma_state.ion_number_density,
                "partition_function": plasma.thermal_lte_partition_function.loc[
                    :, shell_columns
                ],
                "boltzmann_factor": plasma.thermal_lte_level_boltzmann_factor.loc[
                    :, shell_columns
                ],
            },
            thermal_balance_solver=thermal_balance_solver,
            thermal_balance_arguments=thermal_balance_arguments,
            radiation_field=radiation_field,
            fixed_rates=fixed_rates,
            case_b_species=self.case_b_species,
        )

    def build_helium_evaluator(
        self,
        plasma_state: CoupledPlasmaState,
        fixed_rates: FixedRateState,
        bound_bound_species_filter: tuple[int, int] | None,
    ) -> PlasmaEquilibriumEvaluator:
        """Build an analytic bound-free evaluator for one Picard iteration."""
        maximum_electron_density = (
            self.plasma_solver.number_density.multiply(
                self.plasma_solver.number_density.index.to_numpy(dtype=float),
                axis=0,
            )
            .sum()
            .to_numpy(dtype=float)
        )
        return self._build_helium_evaluator(
            plasma_state,
            fixed_rates,
            bound_bound_species_filter,
            maximum_electron_density,
        )

    def solve_plasma(self, estimated_radfield_properties: object) -> None:
        """Update classic radiation estimators, then converge helium."""
        super().solve_plasma(estimated_radfield_properties)
        self.plasma_coupling_solver.solve(self)
        self._update_sobolev_state()

    def solve_opacity(self) -> dict[str, object]:
        """Solve line opacity and classic bound-bound macro-atom states."""
        self._update_sobolev_state()
        tau_sobolev = self._tau_sobolev
        beta_sobolev = self._beta_sobolev
        if self.opacity_solver.disable_line_scattering:
            tau_sobolev = pd.DataFrame(
                np.zeros_like(tau_sobolev),
                index=tau_sobolev.index,
                columns=tau_sobolev.columns,
            )
            beta_sobolev = pd.DataFrame(
                np.ones_like(beta_sobolev),
                index=beta_sobolev.index,
                columns=beta_sobolev.columns,
            )
        opacity_state = OpacityState(
            self.plasma_solver.electron_densities,
            np.asarray(self.plasma_solver.t_electrons, dtype=np.float64),
            self.atom_data.lines.nu,
            tau_sobolev,
            beta_sobolev,
            continuum_state=None,
        )
        macro_atom_state = self.macro_atom_solver.solve(
            self.plasma_solver.j_blues,
            opacity_state.beta_sobolev,
            self.plasma_solver.stimulated_emission_factor,
        )
        return {
            "opacity_state": opacity_state,
            "macro_atom_state": macro_atom_state,
        }

    def solve_montecarlo(
        self,
        opacity_states: dict[str, object],
        no_of_real_packets: int,
        no_of_virtual_packets: int = 0,
    ) -> npt.NDArray[np.float64]:
        """Run classic Monte Carlo transport."""
        classic_transport_plasma = SimpleNamespace(
            continuum_interaction_species=pd.MultiIndex.from_tuples(
                [], names=["atomic_number", "ion_number"]
            )
        )
        self.transport_state = self.transport_solver.initialize_transport_state(
            self.simulation_state,
            opacity_states["opacity_state"],
            opacity_states["macro_atom_state"],
            classic_transport_plasma,
            no_of_real_packets,
            no_of_virtual_packets=no_of_virtual_packets,
            iteration=self.completed_iterations,
        )
        virtual_packet_energies = self.transport_solver.run(
            self.transport_state,
            show_progress_bars=self.show_progress_bars,
        )
        output_energy = self.transport_state.packet_collection.output_energies
        if np.sum(output_energy < 0) == len(output_energy):
            logger.critical("No r-packet escaped through the outer boundary.")
        return virtual_packet_energies


class NonthermalHeliumWorkflow:
    """Own high-energy and coupled optical nonthermal-helium calculations."""

    def __init__(
        self,
        optical_workflow: NonthermalHeliumOpticalWorkflow,
        high_energy_workflow: TARDISHEWorkflow,
        plasma_coupling_solver: NonthermalHeliumPlasmaSolver,
        high_energy_settings: HighEnergySettings,
    ) -> None:
        """Compose the optical, high-energy, and plasma-coupling owners."""
        self.optical_workflow = optical_workflow
        self.high_energy_workflow = high_energy_workflow
        self.plasma_coupling_solver = plasma_coupling_solver
        self.high_energy_settings = high_energy_settings
        self.high_energy_result: TARDISHEWorkflowResult | None = None
        self.deposition_state: DepositionState | None = None

    def run(self) -> None:
        """Run high-energy transport once, then the coupled optical workflow."""
        optical_state = self.optical_workflow.simulation_state
        high_energy_state = self.high_energy_workflow.simulation_state
        for boundary_name in ("v_inner", "v_outer"):
            optical_boundary = getattr(optical_state, boundary_name).to_value(
                u.cm / u.s
            )
            high_energy_boundary = getattr(
                high_energy_state, boundary_name
            ).to_value(u.cm / u.s)
            if (
                optical_boundary.shape != high_energy_boundary.shape
                or not np.allclose(
                    optical_boundary,
                    high_energy_boundary,
                    rtol=1.0e-12,
                    atol=0.0,
                )
            ):
                raise ValueError(
                    "High-energy and optical shell velocity boundaries do "
                    f"not match for {boundary_name}."
                )
        self.high_energy_result = self.high_energy_workflow.run(
            **self.high_energy_settings.as_run_arguments()
        )
        time_bin_edges, effective_times = self.high_energy_workflow.get_times(
            self.high_energy_settings.time_start,
            self.high_energy_settings.time_end,
            self.high_energy_settings.time_space,
            self.high_energy_settings.time_steps,
        )
        time_bin_edges = time_bin_edges * u.d
        effective_times = (effective_times * u.d).to_value(u.s)
        velocity_volume = calculate_ejecta_velocity_volume(
            self.high_energy_workflow.simulation_state
        )
        shell_volume = (
            velocity_volume[:, np.newaxis]
            * effective_times[np.newaxis, :] ** 3
            * u.cm**3
        )
        shell_index = self.optical_workflow.plasma_solver.number_density.columns
        self.deposition_state = prepare_deposition_state(
            self.high_energy_result,
            time_bin_edges,
            optical_state.time_explosion,
            shell_volume,
            shell_index,
        )
        self.plasma_coupling_solver.deposition_state = self.deposition_state
        self.optical_workflow.plasma_coupling_solver = (
            self.plasma_coupling_solver
        )
        self.optical_workflow.run()
