import pandas as pd
import numpy as np
from scipy.optimize import root

from tardis.plasma.properties.base import ProcessingPlasmaProperty
from tardis.plasma.properties.nlte_excitation_data import NLTEExcitationData

__all__ = [
    "NLTERateEquationSolver",
]


class NLTERateEquationSolver(ProcessingPlasmaProperty):
    outputs = ("ion_number_density_nlte", "electron_densities_nlte")

    def calculate(
        self,
        gamma,
        alpha_sp,
        alpha_stim,
        coll_ion_coeff,
        coll_recomb_coeff,
        partition_function,
        levels,
        level_boltzmann_factor,
        phi,
        rate_matrix_index,
        number_density,
        nlte_excitation_species,
    ):
        """Calculates ion number densities and electron densities using NLTE ionization.

        Parameters
        ----------
        gamma : pandas.DataFrame
            The rate coefficient for radiative ionization.
        alpha_sp : pandas.DataFrame
            The rate coefficient for spontaneous recombination.
        alpha_stim : pandas.DataFrame
            The rate coefficient for stimulated recombination.
        coll_ion_coeff : pandas.DataFrame
            The rate coefficient for collisional ionization in the Seaton
            approximation.
        coll_recomb_coeff : pandas.DataFrame
            The rate coefficient for collisional recombination.
        partition_function : pandas.DataFrame
            General partition function. Indexed by atomic number, ion number.
        levels : MultiIndex
            (atomic_number, ion_number, level_number)
            Index of filtered atomic data.
        level_boltzmann_factor : pandas.DataFrame
            General Boltzmann factor.
        phi : pandas.DataFrame
            Saha Factors.
        rate_matrix_index : MultiIndex
            (atomic_number, ion_number, treatment type)
            If ion is treated in LTE or nebular ionization, 3rd index is "lte_ion",
            if treated in NLTE ionization, 3rd index is "nlte_ion".
        number_density : pandas.DataFrame
            Number density in each shell for each species.
        nlte_excitation_species : list
            List of species treated in NLTE excitation.

        Returns
        -------
        ion_number_densities_nlte : pandas.DataFrame
            Number density with NLTE ionization treatment.
        electron_densities_nlte : Series
            Electron density with NLTE ionization treatment.
        """

        # nlte_data = NLTEExcitationData(atomic_data.lines, nlte_excitation_species) - will be used in a future PR
        (
            total_photo_ion_coefficients,
            total_rad_recomb_coefficients,
            total_coll_ion_coefficients,
            total_coll_recomb_coefficients,
        ) = self.prepare_ion_recomb_coefficients_nlte_ion(
            gamma,
            alpha_sp,
            alpha_stim,
            coll_ion_coeff,
            coll_recomb_coeff,
            partition_function,
            levels,
            level_boltzmann_factor,
        )
        # TODO: call prepare_bound_bound_rate_matrix if there are NLTE excitation species
        initial_electron_densities = number_density.sum(axis=0)
        atomic_numbers = (
            rate_matrix_index.get_level_values("atomic_number")
            .unique()
            .drop("n_e")
        )  # dropping the n_e index, as rate_matrix_index's first index is (atomic_numbers, "n_e")

        index = rate_matrix_index.droplevel("level_number").drop("n_e")
        ion_number_density_nlte = pd.DataFrame(
            0.0, index=index, columns=phi.columns
        )
        electron_densities_nlte = pd.Series(0.0, index=phi.columns)

        for shell in phi.columns:
            solution_vector = self.prepare_solution_vector(
                number_density[shell],
                rate_matrix_index,
            )
            first_guess = self.prepare_first_guess(
                rate_matrix_index,
                atomic_numbers,
                number_density[shell],
                initial_electron_densities[shell],
            )
            solution = root(
                self.population_objective_function,
                first_guess,
                args=(
                    atomic_numbers,
                    phi[shell],
                    solution_vector,
                    rate_matrix_index,
                    total_photo_ion_coefficients[shell],
                    total_rad_recomb_coefficients[shell],
                    total_coll_ion_coefficients[shell],
                    total_coll_recomb_coefficients[shell],
                ),
                jac=True,
            )
            assert solution.success
            ion_number_density_nlte[shell] = solution.x[:-1]
            electron_densities_nlte[shell] = solution.x[-1]
        # TODO: change the jacobian and rate matrix to use shell id and get coefficients from the attribute of the class.
        return ion_number_density_nlte, electron_densities_nlte

    @staticmethod
    def calculate_rate_matrix(
        atomic_numbers,
        phi_shell,
        electron_density,
        rate_matrix_index,
        total_photo_ion_coefficients,
        total_rad_recomb_coefficients,
        total_coll_ion_coefficients,
        total_coll_recomb_coefficients,
    ):
        """

        Parameters
        ----------
        phi_shell : pandas.DataFrame
            Saha Factors in the current shell
        electron_density : float
            Guess for electron density in the current shell
        rate_matrix_index : pandas.MultiIndex
            Index used for constructing the rate matrix
        total_photo_ion_coefficients : pandas.DataFrame
            Photo ionization coefficients
        total_rad_recomb_coefficients : pandas.DataFrame
            Radiative recombination coefficients (should get multiplied by electron density)
        total_coll_ion_coefficients : pandas.DataFrame
            Collisional ionization coefficients (should get multiplied by electron density)
        total_coll_recomb_coefficients : pandas.DataFrame
            Collisional recombination coefficients (should get multiplied by electron density^2)

        Returns
        -------
        pandas.DataFrame
            Rate matrix used for NLTE solver.
        """
        rate_matrix = pd.DataFrame(
            0.0, columns=rate_matrix_index, index=rate_matrix_index
        )
        total_rad_recomb_coefficients = (
            total_rad_recomb_coefficients * electron_density
        )
        total_coll_ion_coefficients = (
            total_coll_ion_coefficients * electron_density
        )
        total_coll_recomb_coefficients = (
            total_coll_recomb_coefficients * electron_density**2
        )
        for atomic_number in atomic_numbers:
            ion_numbers = rate_matrix.loc[atomic_number].index.get_level_values(
                "ion_number"
            )
            phi_block = phi_shell.loc[atomic_number]
            rate_matrix_block = NLTERateEquationSolver.lte_rate_matrix_block(
                phi_block, electron_density
            )

            nlte_ion_numbers = ion_numbers[
                rate_matrix.loc[atomic_number].index.get_level_values(
                    "level_number"
                )
                == "nlte_ion"
            ]
            # >>> lte_ion_numbers is for future use in NLTE excitation treatment
            lte_ion_numbers = ion_numbers[
                rate_matrix.loc[atomic_number].index.get_level_values(
                    "level_number"
                )
                == "lte_ion"
            ]
            # <<<
            for ion_number in nlte_ion_numbers:
                rate_matrix_block = NLTERateEquationSolver.set_nlte_ion_rate(
                    rate_matrix_block,
                    atomic_number,
                    ion_number,
                    total_rad_recomb_coefficients.loc[(atomic_number,)],
                    total_photo_ion_coefficients.loc[(atomic_number,)],
                    total_coll_ion_coefficients.loc[(atomic_number,)],
                    total_coll_recomb_coefficients.loc[(atomic_number,)],
                )
            rate_matrix.loc[
                (atomic_number, slice(None)), (atomic_number)
            ] = rate_matrix_block

        charge_conservation_row = (
            NLTERateEquationSolver.prepare_charge_conservation_row(
                atomic_numbers
            )
        )
        rate_matrix.loc[("n_e", slice(None))] = charge_conservation_row
        return rate_matrix

    @staticmethod
    def set_nlte_ion_rate(
        rate_matrix_block,
        atomic_number,
        ion_number,
        total_rad_recomb_coefficients,
        total_photo_ion_coefficients,
        total_coll_ion_coefficients,
        total_coll_recomb_coefficients,
    ):
        """Calculates the row for the species treated in NLTE ionization

        Parameters
        ----------
        rate_matrix_block : numpy.array
            The diagonal block corresponding to current atomic number.
        atomic_number : int
            Current atomic number
        ion_number : int
            Current ion number
        total_rad_recomb_coefficients : pandas.DataFrame
            Rad. recomb. coefficients for current atomic number
        total_photo_ion_coefficients : pandas.DataFrame
            Photo ionization coefficients for current atomic number
        total_coll_ion_coefficients : pandas.DataFrame
            Collisional ionization coefficients for current atomic number
        total_coll_recomb_coefficients : pandas.DataFrame
            Collisional recombination coefficients for current atomic number

        Returns
        -------
        numpy.array
            Rate matrix block with a changed row for NLTE ionization treatment
        """
        ion_coefficients = (
            total_photo_ion_coefficients + total_coll_ion_coefficients
        )
        recomb_coefficients = (
            total_rad_recomb_coefficients + total_coll_recomb_coefficients
        )
        if atomic_number != ion_number:
            ion_coeff_matrix_ion_row = NLTERateEquationSolver.ion_matrix(
                ion_coefficients, atomic_number, ion_number
            )
            recomb_coeff_matrix_ion_row = NLTERateEquationSolver.recomb_matrix(
                recomb_coefficients, atomic_number, ion_number
            )
            rate_matrix_block[ion_number, :] = (
                ion_coeff_matrix_ion_row + recomb_coeff_matrix_ion_row
            )
        return rate_matrix_block

    @staticmethod
    def lte_rate_matrix_block(phi_block, electron_density):
        """Creates the generic LTE block for rate matrix.

        Parameters
        ----------
        phi_block : pandas.DataFrame
            Saha Factors for current atomic number
        electron_density : float
            Current guess for electron density

        Returns
        -------
        numpy.array
            LTE block for rate matrix
        """
        lte_rate_vector_block = -1.0 * np.hstack([*phi_block.values, -1.0])
        lte_rate_matrix_block = np.diag(lte_rate_vector_block)
        n_e_initial = np.ones(len(phi_block)) * electron_density
        n_e_matrix = np.diag(n_e_initial, 1)
        lte_rate_matrix_block += n_e_matrix
        lte_rate_matrix_block[-1, :] = 1.0
        return lte_rate_matrix_block

    @staticmethod
    def prepare_phi(phi):
        """
        Makes sure that phi does not have any 0 entries.
        """
        phi[phi == 0.0] = 1.0e-10 * phi[phi > 0.0].min().min()
        return phi

    @staticmethod
    def recomb_matrix(recomb_coefficients, atomic_number, ion_number):
        """Constructs a recombination rate matrix from the recombination rates.

        Parameters
        ----------
        recomb_coefficients : pandas.DataFrame
            Recombination coefficients.
        atomic_number : int64
            Current atomic number. Used for the dimension of a square matrix.
        ion_number : int64
            Current ion number. Used for returning the correct row.

        Returns
        -------
        numpy.ndarray
        """
        offdiag = np.zeros(atomic_number)
        index = recomb_coefficients.index
        for i in index:
            offdiag[i] = recomb_coefficients.loc[i]
        diag = np.hstack([np.zeros(1), -offdiag])
        return (np.diag(diag) + np.diag(offdiag, k=1))[ion_number, :]

    @staticmethod
    def ion_matrix(ion_coefficients, atomic_number, ion_number):
        """Constructs an ionization rate matrix from the ionization rates.

        Parameters
        ----------
        ion_coefficients : pandas.DataFrame
            Recombination coefficients.
        atomic_number : int64
            Current atomic number. Used for the dimension of a square matrix.
        ion_number : int64
            Current ion number. Used for returning the correct row.

        Returns
        -------
        numpy.ndarray
        """
        offdiag = np.zeros(atomic_number)
        index = ion_coefficients.index
        for i in index:
            offdiag[i] = ion_coefficients.loc[i]
        diag = np.hstack([-offdiag, np.zeros(1)])
        return (np.diag(diag) + np.diag(offdiag, k=-1))[ion_number, :]

    @staticmethod
    def prepare_charge_conservation_row(atomic_numbers):
        """Prepares the last row of the rate_matrix. This row corresponds to the charge
        density equation."""
        charge_conservation_row = []
        for atomic_number in atomic_numbers:
            charge_conservation_row.append(np.arange(0, atomic_number + 1))
        charge_conservation_row = np.hstack([*charge_conservation_row, -1])
        # TODO needs to be modified for use in nlte_excitation
        return charge_conservation_row

    @staticmethod
    def prepare_ion_recomb_coefficients_nlte_ion(
        gamma,
        alpha_sp,
        alpha_stim,
        coll_ion_coeff,
        coll_recomb_coeff,
        partition_function,
        levels,
        level_boltzmann_factor,
    ):
        """
        Prepares the ionization and recombination coefficients by grouping them for
        ion numbers.

        Parameters
        ----------
        gamma : pandas.DataFrame
            The rate coefficient for radiative ionization.
        alpha_sp : pandas.DataFrame
            The rate coefficient for spontaneous recombination.
        alpha_stim : pandas.DataFrame
            The rate coefficient for stimulated recombination.
        coll_ion_coeff : pandas.DataFrame
            The rate coefficient for collisional ionization in the Seaton
            approximation.
        coll_recomb_coeff : pandas.DataFrame
            The rate coefficient for collisional recombination.
        partition_function : pandas.DataFrame
            General partition function. Indexed by atomic number, ion number.
        levels : MultiIndex
            (atomic_number, ion_number, level_number)
            Index of filtered atomic data.
        level_boltzmann_factor : pandas.DataFrame
            General Boltzmann factor.
        Returns
        -------
        total_photo_ion_coefficients
            Photoionization coefficients grouped by atomic number and ion number.
        total_rad_recomb_coefficients
            Radiative recombination coefficients grouped by atomic number and ion number.
        total_coll_ion_coefficients
            Collisional ionization coefficients grouped by atomic number and ion number.
        total_coll_recomb_coefficients
            Collisional recombination coefficients grouped by atomic number and ion number.
        """
        indexer = pd.Series(
            np.arange(partition_function.shape[0]),
            index=partition_function.index,
        )
        _ion2level_idx = indexer.loc[levels.droplevel("level_number")].values
        partition_function_broadcast = partition_function.values[_ion2level_idx]
        level_population_fraction = pd.DataFrame(
            level_boltzmann_factor.values / partition_function_broadcast,
            index=levels,
        )
        total_photo_ion_coefficients = (
            (level_population_fraction.loc[gamma.index] * gamma)
            .groupby(level=("atomic_number", "ion_number"))
            .sum()
        )

        total_rad_recomb_coefficients = (
            (alpha_sp + alpha_stim)
            .groupby(level=["atomic_number", "ion_number"])
            .sum()
        )
        total_coll_ion_coefficients = (
            (
                level_population_fraction.loc[coll_ion_coeff.index]
                * coll_ion_coeff
            )
            .groupby(level=("atomic_number", "ion_number"))
            .sum()
        )
        total_coll_recomb_coefficients = (
            (coll_recomb_coeff)
            .groupby(level=("atomic_number", "ion_number"))
            .sum()
        )
        return (
            total_photo_ion_coefficients,
            total_rad_recomb_coefficients,
            total_coll_ion_coefficients,
            total_coll_recomb_coefficients,
        )

    @staticmethod
    def jacobian_matrix(
        atomic_numbers,
        populations,
        rate_matrix,
        rate_matrix_index,
        total_rad_recomb_coefficients,
        total_coll_ion_coefficients,
        total_coll_recomb_coefficients,
    ):
        """Creates the jacobian matrix used for NLTE ionization solver

        Parameters
        ----------
        populations : numpy.array
            Ion populations, electron density
        rate_matrix : pandas.DataFrame
            Rate matrix used for NLTE solver.
        rate_matrix_index : MultiIndex
            (atomic_number, ion_number, treatment type)
            If ion is treated in LTE or nebular ionization, 3rd index is "lte_ion",
            if treated in NLTE ionization, 3rd index is "nlte_ion".
        total_rad_recomb_coefficients : pandas.DataFrame
            Radiative recombination coefficients grouped by atomic number and ion number.
        total_coll_ion_coefficients : pandas.DataFrame
            Collisional ionization coefficients(should get multiplied by electron density).
        total_coll_recomb_coefficients : pandas.DataFrame
            Collisional recombination coefficients(should get multiplied by electron density).

        Returns
        -------
        numpy.array
            Jacobian matrix used for NLTE ionization solver
        """
        # TODO: for future use, can be vectorized.
        index = 0
        jacobian_matrix = rate_matrix.copy().values
        jacobian_matrix[:-1, -1] = populations[1:]
        for atomic_number in atomic_numbers:
            for i in range(index, index + atomic_number):
                if rate_matrix_index[i][2] == "nlte_ion":
                    jacobian_matrix[
                        i, -1
                    ] = NLTERateEquationSolver.deriv_matrix_block(
                        atomic_number,
                        total_rad_recomb_coefficients.loc[(atomic_number,)],
                        total_coll_ion_coefficients.loc[(atomic_number,)],
                        total_coll_recomb_coefficients.loc[(atomic_number,)],
                        populations[index : index + atomic_number + 1],
                        populations[-1],
                    )[
                        i - index
                    ]
            index += atomic_number + 1
            jacobian_matrix[index - 1, -1] = 0  # number conservation row
        return jacobian_matrix

    @staticmethod
    def deriv_matrix_block(
        atomic_number,
        total_rad_recomb_coefficients,
        total_coll_ion_coefficients,
        total_coll_recomb_coefficients,
        current_ion_number_densities,
        current_electron_density,
    ):
        """Calculates the dot product of the derivative of rate matrix and ion number densities+electron density column.

        Parameters
        ----------
        atomic_number : int64
            Current atomic number
        total_rad_recomb_coefficients : pandas.DataFrame
            Radiative recombination coefficients grouped by atomic number and ion number.
        total_coll_ion_coefficients : pandas.DataFrame
            Collisional ionization coefficients.
        total_coll_recomb_coefficients : pandas.DataFrame
            Collisional recombination coefficients.
        current_ion_number_densities : numpy.array
            Current ion number densities for the current atomic number.
        current_electron_density : float64
            Current electron density

        Returns
        -------
        numpy.array
            Returns the part of the last column of the jacobian matrix, corresponding to atomic number.
        """
        ion_numbers = np.arange(0, atomic_number)
        radiative_rate_coeff_matrix = NLTERateEquationSolver.recomb_matrix(
            total_rad_recomb_coefficients, atomic_number, ion_numbers
        )
        coll_recomb_matrix = (
            NLTERateEquationSolver.recomb_matrix(
                total_coll_recomb_coefficients, atomic_number, ion_numbers
            )
            * current_electron_density
            * 2
        )
        coll_ion_coeff_matrix = NLTERateEquationSolver.ion_matrix(
            total_coll_ion_coefficients, atomic_number, ion_numbers
        )
        deriv_matrix = (
            radiative_rate_coeff_matrix
            + coll_ion_coeff_matrix
            + coll_recomb_matrix
        )
        return np.dot(deriv_matrix, current_ion_number_densities)

    def prepare_first_guess(
        self,
        rate_matrix_index,
        atomic_numbers,
        number_density,
        electron_density,
    ):
        """Constructs a first guess for ion number densities and electron density, where all species are singly ionized.

        Parameters
        ----------
        rate_matrix_index : pandas.MultiIndex
            (atomic_number, ion_number, treatment type)
        atomic_numbers : numpy.array
            All atomic numbers present in the plasma.
        number_density : pandas.DataFrame
            Number density of present species.
        electron_density : float
            Current value of electron density.

        Returns
        -------
        numpy.array
            Guess for ion number densities and electron density.
        """
        first_guess = pd.Series(0.0, index=rate_matrix_index)
        for atomic_number in atomic_numbers:
            first_guess.loc[(atomic_number, 1)][0] = number_density.loc[
                atomic_number
            ]
        # TODO: After the first iteration, the new guess can be the old solution.
        first_guess = first_guess.values
        first_guess[-1] = electron_density
        return first_guess

    def population_objective_function(
        self,
        populations,
        atomic_numbers,
        phi,
        solution_vector,
        rate_matrix_index,
        total_photo_ion_coefficients,
        total_rad_recomb_coefficients,
        total_coll_ion_coefficients,
        total_coll_recomb_coefficients,
    ):
        """Main set of equations for the NLTE ionization solver.

        To solve the statistical equilibrium equations, we need to find the root
        of the objective function A*x - B, where x are the populations,
        A is the matrix of rates, and B is the solution vector.

        Parameters
        ----------
        populations : numpy.array
            Current values of ion number densities and electron density.
        atomic_numbers : numpy.array
            All atomic numbers present in the plasma.
        phi : pandas.DataFrame
            Saha Factors of the current shell.
        solution_vector : numpy.array
            Solution vector for the set of equations.
        rate_matrix_index : pandas.MultiIndex
            (atomic_number, ion_number, treatment type)
            If ion is treated in LTE or nebular ionization, 3rd index is "lte_ion",
            if treated in NLTE ionization, 3rd index is "nlte_ion".
        total_photo_ion_coefficients : pandas.DataFrame
            Photo ion. coefficients for current atomic number
        total_rad_recomb_coefficients : pandas.DataFrame
            Radiative recombination coefficients for current atomic number
        total_coll_ion_coefficients : pandas.DataFrame
            Collisional ionization coefficients for current atomic number
        total_coll_recomb_coefficients : pandas.DataFrame
            Coll. recomb. coefficients for current atomic number
        Returns
        -------
        (numpy.array, numpy.array)
            Returns the objective function and jacobian of the rate matrix in a tuple.
        """
        electron_density = populations[-1]
        rate_matrix = self.calculate_rate_matrix(
            atomic_numbers,
            phi,
            electron_density,
            rate_matrix_index,
            total_photo_ion_coefficients,
            total_rad_recomb_coefficients,
            total_coll_ion_coefficients,
            total_coll_recomb_coefficients,
        )
        jacobian_matrix = self.jacobian_matrix(
            atomic_numbers,
            populations,
            rate_matrix,
            rate_matrix_index,
            total_rad_recomb_coefficients,
            total_coll_ion_coefficients,
            total_coll_recomb_coefficients,
        )
        return (
            np.dot(rate_matrix.values, populations) - solution_vector,
            jacobian_matrix,
        )

    def solution_vector_block(self, number_density, needed_number_of_elements):
        """Block of the solution vector for the current atomic number.

        Block for the solution vector has the form (0, 0, ..., 0, number_density).
        Length is equal to the number of present ions and number of
        levels, if treated with NLTE excitation.

        Parameters
        ----------
        number_density : float
            Number density of the current atomic number.
        needed_number_of_elements : int
            Number of needed elements in the solution vector for current atomic number.

        Returns
        -------
        numpy.array
            Block of the solution vector corresponding to the current atomic number.
        """
        solution_vector = np.zeros(needed_number_of_elements)
        solution_vector[-1] = number_density
        return solution_vector

    def prepare_solution_vector(self, number_density, rate_matrix_index):
        """Constructs the solution vector for the NLTE ionization solver set of equations by combining
        all solution verctor blocks.

        Parameters
        ----------
        number_density : pandas.DataFrame
            Number densities of all present species.
        rate_matrix_index : pandas.MultiIndex
            (atomic_number, ion_number, treatment type)

        Returns
        -------
        numpy.array
            Solution vector for the NLTE ionization solver.
        """
        atomic_numbers = number_density.index
        solution_array = []
        for atomic_number in atomic_numbers:
            needed_number_of_elements = (
                rate_matrix_index.get_level_values("atomic_number")
                == atomic_number
            ).sum()
            solution_array.append(
                self.solution_vector_block(
                    number_density.loc[atomic_number],
                    needed_number_of_elements,
                )
            )
        solution_vector = np.hstack(solution_array + [0])
        return solution_vector

    @staticmethod
    def prepare_bound_bound_rate_matrix(
        number_of_levels,
        lines_index,
        r_ul_index,
        r_ul_matrix,
        r_lu_index,
        r_lu_matrix,
        beta_sobolev,
    ):
        """Calculates a matrix with bound-bound rates for NLTE excitation treatment.

        Parameters
        ----------
        number_of_levels : int
            Number of levels for the specified species.
        lines_index : numpy.array
            Index of lines in nlte_data.
        r_ul_index : numpy.array
            Index used for r_ul matrix
        r_ul_matrix : numpy.array
            Matrix with the rates(upper to lower transition) of bound-bound interaction(DOES NOT INCLUDE THE BETA SOBOLEVS)
            (number_of_levels, number_of_levels, number_of_shells)
        r_lu_index : numpy.array
            Index used for r_lu matrix
        r_lu_matrix : numpy.array
        r_lu_matrix : numpy.array
            Matrix with the rates(lower to upper transition) of bound-bound interaction(DOES NOT INCLUDE THE BETA SOBOLEVS)
            (number_of_levels, number_of_levels, number_of_shells)
        beta_sobolev : pandas.DataFrame
            Beta Sobolev factors.
        Returns
        -------
        numpy.array (number of levels, number of levels)
            Matrix with excitation-deexcitation rates(should be added to NLTE rate matrix for excitation treatment).
            NOTE: only works with ONE ion number treated in NLTE excitation AT ONCE.
        """
        number_of_shells = beta_sobolev.shape[1]
        try:
            beta_sobolev_filtered = beta_sobolev.iloc[lines_index]
        except AttributeError:
            beta_sobolev_filtered = beta_sobolev
        r_ul_matrix_reshaped = r_ul_matrix.reshape(
            (number_of_levels**2, number_of_shells)
        )
        r_lu_matrix_reshaped = r_lu_matrix.reshape(
            (number_of_levels**2, number_of_shells)
        )
        r_ul_matrix_reshaped[r_ul_index] *= beta_sobolev_filtered
        r_lu_matrix_reshaped[r_lu_index] *= beta_sobolev_filtered
        rates_matrix_bound_bound = r_ul_matrix + r_lu_matrix
        for i in range(number_of_levels):
            rates_matrix_bound_bound[i, i] = -rates_matrix_bound_bound[
                :, i
            ].sum(axis=0)
        return rates_matrix_bound_bound

    def prepare_r_uls_r_lus(
        number_of_levels,
        number_of_shells,
        j_blues,
        excitation_species,
        nlte_data,
    ):
        """Calculates part of rate matrices for bound bound interactions.

        Parameters
        ----------
        number_of_levels : int
            Number of levels for the NLTE excitation species.
        number_of_shells : int
            Number of shells.
        j_blues : pandas.DataFrame, dtype float
            Mean intensities in the blue wings of the line transitions.
        excitation_species : tuple
            Species treated in NLTE excitation.
        nlte_data : NLTEExcitationData
            Data relevant to NLTE excitation species.

        Returns
        -------
        lines_index : numpy.array
            Index of lines in nlte_data.
        number_of_levels : int
            Number of levels for the specified species.
        r_ul_index : numpy.array
            Index used for r_ul matrix
        r_ul_matrix_reshaped : numpy.array
            Matrix with the rates(upper to lower transition) of bound-bound interaction(DOES NOT INCLUDE THE BETA SOBOLEVS)
        r_lu_index : numpy.array
            Index used for r_lu matrix
        r_lu_matrix_reshaped : numpy.array
            Matrix with the rates(lower to upper transition) of bound-bound interaction(DOES NOT INCLUDE THE BETA SOBOLEVS)
        """
        # number_of_levels = atomic_data_levels.energy.loc[
        #     excitation_species
        # ].count() do this in the solver
        lnl = nlte_data.lines_level_number_lower[excitation_species]
        lnu = nlte_data.lines_level_number_upper[excitation_species]
        (lines_index,) = nlte_data.lines_idx[excitation_species]

        try:
            j_blues_filtered = j_blues.iloc[lines_index]
        except AttributeError:
            j_blues_filtered = j_blues
        A_uls = nlte_data.A_uls[excitation_species]
        B_uls = nlte_data.B_uls[excitation_species]
        B_lus = nlte_data.B_lus[excitation_species]
        r_lu_index = lnu * number_of_levels + lnl
        r_ul_index = lnl * number_of_levels + lnu
        r_ul_matrix = np.zeros(
            (number_of_levels, number_of_levels, number_of_shells),
            dtype=np.float64,
        )
        r_ul_matrix_reshaped = r_ul_matrix.reshape(
            (number_of_levels**2, number_of_shells)
        )
        r_ul_matrix_reshaped[r_ul_index] = (
            A_uls[np.newaxis].T + B_uls[np.newaxis].T * j_blues_filtered
        )
        r_lu_matrix = np.zeros_like(r_ul_matrix)
        r_lu_matrix_reshaped = r_lu_matrix.reshape(
            (number_of_levels**2, number_of_shells)
        )
        r_lu_matrix_reshaped[r_lu_index] = (
            B_lus[np.newaxis].T * j_blues_filtered
        )
        return (
            lines_index,
            r_ul_index,
            r_ul_matrix,
            r_lu_index,
            r_lu_matrix,
        )
        # TODO: beta sobolev needs to be recalculated for each iteration, because it depends on number density

    @staticmethod
    def create_coll_exc_deexc_matrix(
        coll_exc_coefficient,
        coll_deexc_coefficient,
        number_of_levels,
    ):
        """Generates a coefficient matrix from collisional excitation/deexcitation coefficients.

        Needs to be multiplied by electron density when added to the overall rate_matrix.
        Parameters
        ----------
        coll_exc_coefficient : pandas.Series
            Series of collisional excitation coefficients for current (atomic number, ion_number)
            in the current shell.
        coll_deexc_coefficient : pandas.Series
            Series of collisional deexcitation coefficients for (atomic number, ion_number)
            in the current shell.
        number_of_levels : int
            Number of levels for the current atomic number, ion number.

        Returns
        -------
        coeff_matrix : np.array (number of levels, number of levels)
            Square matrix constructed by collisional exc./deexc. coefficients.
        """
        diagonal_exc = np.zeros(number_of_levels)
        diagonal_deexc = np.zeros(number_of_levels)
        col_exc_coefficient_sum_lower = coll_exc_coefficient.groupby(
            "level_number_lower"
        ).sum()
        col_deexc_coefficient_sum_upper = coll_deexc_coefficient.groupby(
            "level_number_upper"
        ).sum()

        diagonal_exc[col_exc_coefficient_sum_lower.index] = (
            -1 * col_exc_coefficient_sum_lower.values
        )
        diagonal_deexc[col_deexc_coefficient_sum_upper.index] = (
            -1 * col_deexc_coefficient_sum_upper.values
        )
        exc_matrix = np.zeros((number_of_levels, number_of_levels))
        deexc_matrix = np.zeros((number_of_levels, number_of_levels))
        exc_matrix[
            (
                coll_exc_coefficient.index.get_level_values(
                    "level_number_upper"
                ),
                coll_exc_coefficient.index.get_level_values(
                    "level_number_lower"
                ),
            )
        ] = coll_exc_coefficient.values
        deexc_matrix[
            (
                coll_exc_coefficient.index.get_level_values(
                    "level_number_lower"
                ),
                coll_exc_coefficient.index.get_level_values(
                    "level_number_upper"
                ),
            )
        ] = coll_deexc_coefficient.values
        np.fill_diagonal(exc_matrix, diagonal_exc)
        np.fill_diagonal(deexc_matrix, diagonal_deexc)
        coeff_matrix = exc_matrix + deexc_matrix
        return coeff_matrix
