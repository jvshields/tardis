from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest
from astropy import units as u

from tardis.plasma.equilibrium.inputs import FixedRateState
from tardis.plasma.equilibrium.solver import PlasmaEquilibriumSolution
from tardis.plasma.nonthermal.pynonthermal_solver import (
    NonthermalPlasmaState,
)
from tardis.workflows.high_energy.deposition import DepositionState
from tardis.workflows.nonthermal_helium_workflow import (
    CoupledPlasmaState,
    HighEnergySettings,
    NonthermalCouplingSettings,
    NonthermalHeliumOpticalWorkflow,
    NonthermalHeliumPlasmaSolver,
    NonthermalHeliumWorkflow,
)
from tardis.workflows.standard_tardis_workflow import StandardTARDISWorkflow


class FakeNonthermalSolver:
    """Return rates that track the supplied electron density."""

    def __init__(self) -> None:
        self.electron_densities: list[float] = []

    def solve(self, plasma_input: object) -> NonthermalPlasmaState:
        electron_densities = plasma_input.electron_number_density.to_value(
            u.cm**-3
        )
        electron_density = float(electron_densities[0])
        self.electron_densities.append(electron_density)
        shell_columns = plasma_input.ion_number_density.columns
        ion_index = pd.MultiIndex.from_tuples(
            [(2, 0, 0, 1)],
            names=[
                "atomic_number",
                "ion_number",
                "ion_number_source",
                "ion_number_destination",
            ],
        )
        return NonthermalPlasmaState(
            FixedRateState(
                ionization_rates=pd.DataFrame(
                    [electron_densities / 1.0e9],
                    index=ion_index,
                    columns=shell_columns,
                ),
                heating_rate_density=pd.Series(
                    electron_densities / 1.0e9, index=shell_columns
                ),
            ),
            pd.DataFrame(columns=shell_columns),
        )


class FixedEquilibriumSolver:
    """Return the same known physical equilibrium on every call."""

    def __init__(self) -> None:
        self.level_seeds: list[pd.DataFrame] = []

    def solve(
        self, *args: object, **kwargs: object
    ) -> PlasmaEquilibriumSolution:
        del kwargs
        level_seed = args[3]
        self.level_seeds.append(level_seed)
        shell_columns = level_seed.columns
        shell_count = len(shell_columns)
        ion_index = pd.MultiIndex.from_tuples(
            [(2, 0), (2, 1)], names=["atomic_number", "ion_number"]
        )
        level_index = pd.MultiIndex.from_tuples(
            [(2, 0, 0), (2, 0, 1)],
            names=["atomic_number", "ion_number", "level_number"],
        )
        evaluation = SimpleNamespace(
            ion_population=pd.DataFrame(
                np.tile([[6.0e8], [4.0e8]], shell_count),
                index=ion_index,
                columns=shell_columns,
            ),
            absolute_level_population=pd.DataFrame(
                np.tile([[4.0e8], [2.0e8]], shell_count),
                index=level_index,
                columns=shell_columns,
            ),
        )
        return PlasmaEquilibriumSolution(
            np.full(shell_count, 4.0e8),
            np.full(shell_count, 8000.0),
            evaluation,
        )


class FakePlasma:
    """Minimal mutable optical plasma used by the coupling test."""

    def __init__(self, shell_count: int = 1) -> None:
        shell_columns = pd.RangeIndex(shell_count)
        self.electron_densities = pd.Series(
            np.arange(shell_count) * 1.0e8 + 2.0e8, index=shell_columns
        )
        self.t_electrons = np.arange(shell_count) * 1000.0 + 6000.0
        ion_index = pd.MultiIndex.from_tuples(
            [(2, 0), (2, 1)], names=["atomic_number", "ion_number"]
        )
        self.ion_number_density = pd.DataFrame(
            np.tile([[8.0e8], [2.0e8]], shell_count),
            index=ion_index,
            columns=shell_columns,
        )
        level_index = pd.MultiIndex.from_tuples(
            [(2, 0, 0), (2, 0, 1)],
            names=["atomic_number", "ion_number", "level_number"],
        )
        self.level_number_density = pd.DataFrame(
            np.tile([[6.0e8], [2.0e8]], shell_count),
            index=level_index,
            columns=shell_columns,
        )
        self.number_density = pd.DataFrame(
            np.tile([[0.0], [1.0e9]], shell_count),
            index=[1, 2],
            columns=shell_columns,
        )

    def update(self, **kwargs: object) -> None:
        for name, value in kwargs.items():
            setattr(self, name, value)


def test_nonthermal_rates_refresh_only_between_equilibrium_solves() -> None:
    """Refresh pynonthermal across Picard steps and commit convergence."""
    provider = FakeNonthermalSolver()
    equilibrium_solver = FixedEquilibriumSolver()
    factory_species_filters: list[tuple[int, int] | None] = []
    optical_workflow = SimpleNamespace(
        plasma_solver=FakePlasma(),
        simulation_state=SimpleNamespace(t_radiative=np.array([1.0e4])),
    )

    def evaluator_factory(
        state: object,
        fixed_rates: object,
        workflow: object,
        bound_bound_species_filter: tuple[int, int] | None,
    ) -> SimpleNamespace:
        del state, fixed_rates, workflow
        factory_species_filters.append(bound_bound_species_filter)
        return SimpleNamespace()

    coupling_solver = NonthermalHeliumPlasmaSolver(
        provider,
        evaluator_factory,
        equilibrium_solver=equilibrium_solver,
        bound_bound_species_filter=(2, 0),
        settings=NonthermalCouplingSettings(
            maximum_iterations=4,
            relative_tolerance=1.0e-12,
            relaxation=1.0,
        ),
    )
    coupling_solver.deposition_state = DepositionState(
        1.0 * u.s,
        0,
        pd.Index([0]),
        np.array([1.0]) * u.erg / u.s / u.cm**3,
        np.array([0.0]) * u.erg / u.s / u.cm**3,
        np.array([1.0]) * u.erg / u.s / u.cm**3,
    )

    result = coupling_solver.solve(optical_workflow)

    npt.assert_array_equal(provider.electron_densities, [2.0e8, 4.0e8, 4.0e8])
    npt.assert_array_equal(result.plasma_state.electron_density, [4.0e8])
    npt.assert_array_equal(
        optical_workflow.plasma_solver.electron_densities, [4.0e8]
    )
    assert len(result.convergence_history) == 3
    assert factory_species_filters == [(2, 0)] * 3
    for level_seed in equilibrium_solver.level_seeds:
        assert level_seed.index.equals(pd.Index([0, 1], name="level_number"))
        npt.assert_allclose(level_seed.sum(axis=0), 1.0)


def test_nonthermal_solver_uses_workflow_evaluator_factory_by_default() -> None:
    """Keep continuum-estimator ownership inside the optical workflow.

    Claim
    -----
    A caller does not need to reconstruct continuum mappings in a notebook.

    Regime
    ------
    One-shell deterministic coupling with no explicit evaluator factory.

    Verification
    ------------
    The coupling solver requests every evaluator from the optical workflow.
    """
    provider = FakeNonthermalSolver()
    equilibrium_solver = FixedEquilibriumSolver()
    factory_calls = []
    optical_workflow = SimpleNamespace(
        plasma_solver=FakePlasma(),
        simulation_state=SimpleNamespace(t_radiative=np.array([1.0e4])),
    )

    def build_helium_evaluator(
        state: object,
        fixed_rates: object,
        bound_bound_species_filter: tuple[int, int] | None,
    ) -> SimpleNamespace:
        factory_calls.append((state, fixed_rates, bound_bound_species_filter))
        return SimpleNamespace()

    optical_workflow.build_helium_evaluator = build_helium_evaluator
    coupling_solver = NonthermalHeliumPlasmaSolver(
        provider,
        evaluator_factory=None,
        equilibrium_solver=equilibrium_solver,
        settings=NonthermalCouplingSettings(
            maximum_iterations=4,
            relative_tolerance=1.0e-12,
            relaxation=1.0,
        ),
    )
    coupling_solver.deposition_state = DepositionState(
        1.0 * u.s,
        0,
        pd.Index([0]),
        np.array([1.0]) * u.erg / u.s / u.cm**3,
        np.array([0.0]) * u.erg / u.s / u.cm**3,
        np.array([1.0]) * u.erg / u.s / u.cm**3,
    )

    coupling_solver.solve(optical_workflow)

    assert len(factory_calls) == 3
    assert all(call[2] == (2, 0) for call in factory_calls)


def test_nonthermal_solver_updates_all_shells() -> None:
    """The coupled solve evaluates and commits every simulation shell.

    Claim
    -----
    Every shell participates in the coupled nonthermal equilibrium solve.

    Regime
    ------
    A deterministic 55-shell plasma spans the former development cutoff.

    Verification
    ------------
    The evaluator receives all physical shell labels and every shell equals
    the exact equilibrium state returned by the test solver.
    """
    provider = FakeNonthermalSolver()
    equilibrium_solver = FixedEquilibriumSolver()
    plasma = FakePlasma(shell_count=55)
    factory_shell_columns = []
    optical_workflow = SimpleNamespace(
        plasma_solver=plasma,
        simulation_state=SimpleNamespace(
            t_radiative=np.linspace(1.0e4, 1.1e4, 55)
        ),
    )

    def evaluator_factory(
        state: CoupledPlasmaState,
        fixed_rates: object,
        workflow: object,
        bound_bound_species_filter: tuple[int, int] | None,
    ) -> SimpleNamespace:
        del fixed_rates, workflow, bound_bound_species_filter
        factory_shell_columns.append(state.ion_number_density.columns.copy())
        return SimpleNamespace()

    coupling_solver = NonthermalHeliumPlasmaSolver(
        provider,
        evaluator_factory,
        equilibrium_solver=equilibrium_solver,
        settings=NonthermalCouplingSettings(
            maximum_iterations=4,
            relative_tolerance=1.0e-12,
            relaxation=1.0,
        ),
    )
    coupling_solver.deposition_state = DepositionState(
        1.0 * u.s,
        0,
        pd.RangeIndex(55),
        np.ones(55) * u.erg / u.s / u.cm**3,
        np.zeros(55) * u.erg / u.s / u.cm**3,
        np.ones(55) * u.erg / u.s / u.cm**3,
    )

    result = coupling_solver.solve(optical_workflow)

    assert all(
        columns.equals(pd.RangeIndex(55)) for columns in factory_shell_columns
    )
    npt.assert_array_equal(
        result.plasma_state.electron_density, np.full(55, 4.0e8)
    )
    npt.assert_array_equal(
        result.plasma_state.electron_temperature, np.full(55, 8000.0)
    )
    pdt.assert_frame_equal(
        result.plasma_state.ion_number_density,
        pd.DataFrame(
            np.tile([[6.0e8], [4.0e8]], 55),
            index=result.plasma_state.ion_number_density.index,
            columns=pd.RangeIndex(55),
        ),
    )
    pdt.assert_frame_equal(
        result.plasma_state.level_number_density,
        pd.DataFrame(
            np.tile([[4.0e8], [2.0e8]], 55),
            index=result.plasma_state.level_number_density.index,
            columns=pd.RangeIndex(55),
        ),
    )


class FakeHighEnergyWorkflow:
    """Return a deterministic flight-path deposition result."""

    def __init__(self, outer_velocity: float = 2.0e8) -> None:
        self.simulation_state = SimpleNamespace(
            v_inner=np.array([1.0e8]) * u.cm / u.s,
            v_outer=np.array([outer_velocity]) * u.cm / u.s,
        )
        self.run_arguments: dict[str, object] | None = None

    def run(self, **kwargs: object) -> SimpleNamespace:
        self.run_arguments = kwargs
        return SimpleNamespace(
            gamma_ray_deposition_estimator=pd.DataFrame(
                [[2.0]], index=[0], columns=[86400.0]
            ),
            positron_energy=pd.DataFrame([[0.0]], index=[0], columns=[86400.0]),
        )

    def get_times(
        self,
        time_start: float,
        time_end: float,
        time_space: str,
        time_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        del time_start, time_end, time_space, time_steps
        return np.array([1.0, 2.0]), np.array([1.5])


class FakeOpticalWorkflow:
    """Record execution after a coupling solver has been attached."""

    def __init__(self) -> None:
        self.plasma_solver = FakePlasma()
        self.simulation_state = SimpleNamespace(
            t_radiative=np.array([1.0e4]),
            time_explosion=1.5 * u.d,
            v_inner=np.array([1.0e8]) * u.cm / u.s,
            v_outer=np.array([2.0e8]) * u.cm / u.s,
        )
        self.ran = False

    def run(self) -> None:
        assert self.plasma_coupling_solver.deposition_state is not None
        self.ran = True


def make_high_energy_settings() -> HighEnergySettings:
    """Return the smallest complete high-energy argument set."""
    return HighEnergySettings(
        time_start=1.0,
        time_end=2.0,
        number_of_packets=1,
        time_steps=1,
        time_space="linear",
        seed=1963,
        positronium_fraction=0.0,
        spectrum_bins=1,
    )


def test_composed_workflow_runs_high_energy_once_and_attaches_coupling() -> (
    None
):
    """Prepare aligned deposition before entering optical transport."""
    optical_workflow = FakeOpticalWorkflow()
    high_energy_workflow = FakeHighEnergyWorkflow()
    coupling_solver = SimpleNamespace(deposition_state=None)
    workflow = NonthermalHeliumWorkflow(
        optical_workflow,
        high_energy_workflow,
        coupling_solver,
        make_high_energy_settings(),
    )

    workflow.run()

    assert optical_workflow.ran
    assert high_energy_workflow.run_arguments["seed"] == 1963
    npt.assert_allclose(
        coupling_solver.deposition_state.rate_density.to_value(
            u.erg / u.s / u.cm**3
        ),
        [2.0],
    )


def test_composed_workflow_rejects_mismatched_velocity_grid() -> None:
    """Do not transfer deposition between physically different shells."""
    workflow = NonthermalHeliumWorkflow(
        FakeOpticalWorkflow(),
        FakeHighEnergyWorkflow(outer_velocity=3.0e8),
        SimpleNamespace(deposition_state=None),
        make_high_energy_settings(),
    )

    with np.testing.assert_raises_regex(ValueError, "v_outer"):
        workflow.run()


def test_specialized_optical_workflow_uses_bound_bound_macro_atom() -> None:
    """Keep continuum state out of the classic macro-atom opacity solve.

    Claim
    -----
    Nonthermal helium coupling does not require continuum packet processes.

    Regime
    ------
    One deterministic opacity update with a bound-bound macro-atom solver.

    Verification
    ------------
    The macro-atom solver receives only the three classic bound-bound inputs.
    """
    macro_atom_arguments = []
    workflow = object.__new__(NonthermalHeliumOpticalWorkflow)
    workflow.plasma_solver = SimpleNamespace(
        j_blues=pd.DataFrame([[1.0]]),
        stimulated_emission_factor=np.ones((1, 1)),
        electron_densities=pd.Series([1.0]),
        t_electrons=np.array([1.0]),
    )
    workflow.atom_data = SimpleNamespace(
        lines=pd.DataFrame(
            {"nu": [1.0]},
            index=pd.Index([0]),
        )
    )
    workflow.opacity_solver = SimpleNamespace(disable_line_scattering=False)
    workflow._update_sobolev_state = lambda: None
    workflow._tau_sobolev = pd.DataFrame([[1.0]])
    workflow._beta_sobolev = pd.DataFrame([[1.0]])
    workflow.macro_atom_solver = SimpleNamespace(
        solve=lambda *args: (macro_atom_arguments.append(args), object())[1]
    )

    workflow.solve_opacity()

    assert len(macro_atom_arguments[0]) == 3


def test_specialized_optical_workflow_disables_continuum_transport() -> None:
    """Present an empty continuum selection to classic packet transport."""
    initialized_with = []
    transport_state = SimpleNamespace(
        packet_collection=SimpleNamespace(output_energies=np.array([1.0]))
    )
    workflow = object.__new__(NonthermalHeliumOpticalWorkflow)
    workflow.simulation_state = object()
    workflow.completed_iterations = 0
    workflow.show_progress_bars = False
    workflow.transport_solver = SimpleNamespace(
        initialize_transport_state=lambda *args, **kwargs: (
            initialized_with.append(args[3]),
            transport_state,
        )[1],
        run=lambda *args, **kwargs: np.array([2.0]),
    )

    virtual_packet_energies = workflow.solve_montecarlo(
        {"opacity_state": object(), "macro_atom_state": object()},
        no_of_real_packets=1,
    )

    assert initialized_with[0].continuum_interaction_species.empty
    npt.assert_array_equal(virtual_packet_energies, [2.0])


def test_specialized_optical_workflow_couples_after_classic_plasma_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Converge nonthermal helium after the classic radiation-field update."""
    call_order = []
    workflow = object.__new__(NonthermalHeliumOpticalWorkflow)
    estimated_radfield_properties = object()
    monkeypatch.setattr(
        StandardTARDISWorkflow,
        "solve_plasma",
        lambda owner, estimated: call_order.append(
            ("classic", owner, estimated)
        ),
    )
    workflow._update_sobolev_state = lambda: call_order.append(("sobolev",))
    workflow.plasma_coupling_solver = SimpleNamespace(
        solve=lambda owner: (
            call_order.append(("nonthermal", owner)),
            object(),
        )[1]
    )

    workflow.solve_plasma(estimated_radfield_properties)

    assert [call[0] for call in call_order] == [
        "classic",
        "nonthermal",
        "sobolev",
    ]
    assert call_order[0][2] is estimated_radfield_properties
