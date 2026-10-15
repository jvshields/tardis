Nonthermal helium workflow
==========================

The nonthermal helium workflow composes three independent calculation owners:

* :class:`~tardis.workflows.high_energy.tardis_he_workflow.TARDISHEWorkflow`
  runs once and supplies flight-path gamma-ray deposition plus positron energy;
* :class:`~tardis.plasma.nonthermal.pynonthermal_solver.PynonthermalHeliumSolver`
  converts a candidate plasma state into fixed helium excitation, ionization,
  and heating rates; and
* :class:`~tardis.plasma.equilibrium.evaluator.PlasmaEquilibriumEvaluator`
  solves the optical plasma while those nonthermal rates remain frozen.

The outer Picard iteration refreshes pynonthermal only after an evaluator solve
has finished. It therefore never constructs a Spencer--Fano system inside a
SciPy residual evaluation. High-energy Monte Carlo transport is also frozen
throughout the optical calculation.

The optical calculation uses continuum packet processes with direct macro-atom
traversal. Each activated macro atom follows its normalized internal
transitions until deactivation; the Type IIP absorbing Markov-chain matrix is
not constructed. The first optical transport uses analytic continuum rates to
initialize its packet state. Subsequent plasma solves use the packet
photoionization and heating estimators, while the nonthermal rates remain fixed
within each coupled evaluator solve. Bound--bound transitions without CMFGEN
collision strengths retain the existing van Regemorter approximation for
their collisional macro-atom branches.

Helium-only continuum atomic data do not require an H I ground-state entry;
hydrogen Case B suppression applies only when that entry is present.

Rate conventions
----------------

The fixed bound--bound channel has units of inverse seconds and enters as

.. math::

   R_{ij} = \beta_{ij} R_{ij,\mathrm{rad}}
          + n_e C_{ij,\mathrm{thermal}}
          + R_{ij,\mathrm{nonthermal}}.

Consequently, a nonthermal rate is not multiplied by an electron density or a
Sobolev escape probability. The same stage-averaged He I ionization
coefficient must be supplied to both ``PlasmaEquilibriumEvaluator`` and the
ion rate matrix used by its ion population solver. Following the first optical
transport, that matrix uses estimated photoionization rates. The current
first-version approximation applies the nonthermal coefficient uniformly to
every He I bound level.

``FixedRateState.heating_rate_density`` is in
``erg s^-1 cm^-3``. It is added once to the heating side of the thermal balance
and is also returned separately as
``PlasmaEquilibriumEvaluation.fixed_heating_rate_density``.

Deposition
----------

The optical epoch selects the containing high-energy time bin. Gamma
deposition comes from ``gamma_ray_deposition_estimator``, which already has
units of ``erg s^-1 cm^-3``. Positron energy is divided by the selected bin
width and homologous shell volume before it is added. Shell labels and time
bin edges are validated explicitly. The high-energy and optical inner and
outer shell velocities must also match; deposition is not silently remapped
between different physical grids. The final time bin is not selected silently.
The standalone workflow reconstructs the authoritative time-bin edges and
effective times from its own high-energy settings; it does not change the
output interface of ``TARDISHEWorkflow``.

The complete composition is passed to pynonthermal. Do not multiply the
deposition by the helium number fraction. Zero-deposition shells bypass the
Spencer--Fano solve and return exact zero rates and heating.

The current flight-path estimator includes Compton energy loss and
photoabsorption but omits pair-creation deposition. Accordingly,
``DepositionState.includes_pair_creation`` is ``False`` and the prepared field
should not yet be described as fully total gamma deposition.

Optional dependency and level mapping
-------------------------------------

Pynonthermal is imported lazily and is not an unconditional TARDIS dependency.
Constructing a solver for a positive-deposition shell without pynonthermal
installed raises an actionable ``ImportError``. The implementation was
validated with pynonthermal 2026.4.27.

The default excitation mapping is the 37-level helium superlevel ordering used
by the Boyle-model experiments. A different atomic dataset must supply an
explicit destination mapping. Each mapping entry is validated to have
nonnegative branching fractions that sum to one. Scientific production runs
must additionally verify the energy, statistical weight, and quantum-number
identity of their chosen helium levels before constructing the evaluator.

Spencer--Fano cost grows steeply with both the energy-grid size and the number
of included ion stages. ``PynonthermalHeliumSolver`` therefore exposes
``number_of_energy_points`` and ``minimum_ion_number_fraction``. The latter is
applied independently in every shell; zero retains the complete composition,
while positive He I and He II populations are retained regardless of the
cutoff. Any nonzero cutoff or reduced energy grid is a numerical approximation
that should be checked for convergence of heating, excitation, and ionization
fractions for the model being calculated.

Case A and Case B
-----------------

``PlasmaEquilibriumEvaluator.case_b_species`` controls ground-state
recombination suppression per species. Hydrogen remains Case B by default to
preserve the Type IIP workflow. Helium is Case A unless ``(2, 0)`` is supplied
explicitly. This selection is scientific input and should be recorded with a
run rather than chosen to optimize agreement with a comparison model.

Python orchestration
--------------------

:class:`~tardis.workflows.nonthermal_helium_workflow.NonthermalHeliumWorkflow`
accepts a specialized
:class:`~tardis.workflows.nonthermal_helium_workflow.NonthermalHeliumOpticalWorkflow`,
a high-energy workflow, a ``NonthermalHeliumPlasmaSolver``, and typed
high-energy settings. The specialized optical workflow owns the additional
post-plasma coupling step; ``StandardTARDISWorkflow`` remains unchanged. The
specialized workflow constructs both the evaluator and its ion population
solver from the same ``FixedRateState``. It receives the coupling solver's
``bound_bound_species_filter``. The default is He I, ``(2, 0)``, so the
workflow can restrict its line table, mean intensities, and Sobolev inputs
before constructing the otherwise species-agnostic evaluator. Passing
``None`` requests the complete bound-bound dataset.

The composed workflow performs the following sequence:

#. run high-energy transport once;
#. prepare the epoch- and shell-aligned deposition rate density;
#. run the normal optical iterations;
#. after each optical plasma update, iterate pynonthermal and the equilibrium
   evaluator to the configured density, temperature, He II fraction, level,
   ionization-rate, and heating-rate tolerances; and
#. commit only the accepted coupled state to the optical plasma.

The initial interface is Python-only. Configuration-schema support and a
reference atomic-data mapping beyond the Boyle 37-level ordering remain future
work.
