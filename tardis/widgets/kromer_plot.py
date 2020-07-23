"""A simple plotting tool to create spectral diagnostics plots similar to those
originally proposed by M. Kromer (see, for example, Kromer et al. 2013, figure
4).
"""
import logging
import numpy as np
import astropy.units as units
import astropy.constants as csts
import pandas as pd
from collections import namedtuple

import astropy.modeling.blackbody as abb

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import matplotlib.cm as cm

import plotly.graph_objects as go


plt.rcdefaults()

logger = logging.getLogger(__name__)

elements = pd.read_csv("elements.csv", names=["symbol", "atomic_num"])
inv_elements = pd.Series(
    elements["symbol"], index=elements["atomic_num"]
).to_dict()

PacketsData = namedtuple("PacketsData", ["virtual", "real"])


class KromerPlotter:
    """A plotter, generating spectral diagnostics plots as proposed by M.
    Kromer.

    With this tool, a specific visualisation of Tardis spectra may be produced.
    It illustrates which elements predominantly contribute to the emission and
    absorption part of the emergent (virtual) packet spectrum.
   
    Notes
    -----
    For this to work, the model must be generated by a Tardis calculation using
    the virtual packet logging capability. This requires a compilation with the
    --with-vpacket-logging flag.

    This way of illustrating the spectral synthesis process was introduced by
    M. Kromer (see e.g. [1]_).

    References
    ----------

    .. [1] Kromer et al. "SN 2010lp - Type Ia Supernova from a Violent Merger
       of Two Carbon-Oxygen White Dwarfs" ApjL, 2013, 778, L18

    """

    def __init__(
        self,
        last_interaction_type,
        last_line_interaction_in_id,
        last_line_interaction_out_id,
        last_interaction_in_nu,
        lines_data,
        packet_nus,
        packet_energies,
        R_phot,
        spectrum_wave,
        spectrum_luminosity,
        t_inner,
        time_of_simulation,
    ):

        # Store the passed values that will be reused by other methods
        self.R_phot = R_phot
        self.spectrum_wave = spectrum_wave
        self.spectrum_luminosity = spectrum_luminosity
        self.t_inner = t_inner
        self.time_of_simulation = time_of_simulation

        # None of the mask is used by plotting functions but only for properties creation
        noint_masks = PacketsData._make(
            [arr == -1 for arr in last_interaction_type]
        )
        escat_masks = PacketsData._make(
            [arr == 1 for arr in last_interaction_type]
        )
        escatonly_masks = PacketsData._make(
            [
                ((line_int_in == -1) * (escat_mask)).astype(np.bool)
                for line_int_in, escat_mask in zip(
                    last_line_interaction_in_id, escat_masks
                )
            ]
        )
        line_masks = PacketsData._make(
            [
                (int_type > -1) * (line_int_in > -1)
                for int_type, line_int_in in zip(
                    last_interaction_type, last_line_interaction_in_id
                )
            ]
        )

        self.lam_noint = PacketsData._make(
            [
                (csts.c.cgs / nus[noint_mask]).to(units.AA)
                for nus, noint_mask in zip(packet_nus, noint_masks)
            ]
        )

        self.lam_escat = PacketsData._make(
            [
                (csts.c.cgs / nus[escatonly_mask]).to(units.AA)
                for nus, escatonly_mask in zip(packet_nus, escatonly_masks)
            ]
        )

        self.weights_escat = PacketsData._make(
            [
                energies[escatonly_mask] / self.time_of_simulation
                for energies, escatonly_mask in zip(
                    packet_energies, escatonly_masks
                )
            ]
        )

        self.weights_noint = PacketsData._make(
            [
                energies[noint_mask] / self.time_of_simulation
                for energies, noint_mask in zip(packet_energies, noint_masks)
            ]
        )

        self.line_in_infos = PacketsData._make(
            [
                lines_data.iloc[line_int_in[line_mask]]
                for line_int_in, line_mask in zip(
                    last_line_interaction_in_id, line_masks
                )
            ]
        )

        self.line_in_nu = PacketsData._make(
            [
                int_nu[line_mask]
                for int_nu, line_mask in zip(last_interaction_in_nu, line_masks)
            ]
        )

        self.line_in_L = PacketsData._make(
            [
                energies[line_mask]
                for energies, line_mask in zip(packet_energies, line_masks)
            ]
        )

        self.line_out_infos = PacketsData._make(
            [
                lines_data.iloc[line_int_out[line_mask]]
                for line_int_out, line_mask in zip(
                    last_line_interaction_out_id, line_masks
                )
            ]
        )

        self.line_out_nu = PacketsData._make(
            [nus[line_mask] for nus, line_mask in zip(packet_nus, line_masks)]
        )

        self.line_out_L = PacketsData._make(
            [
                energies[line_mask]
                for energies, line_mask in zip(packet_energies, line_masks)
            ]
        )

    @classmethod
    def from_simulation(cls, sim):
        return cls(
            last_interaction_type=PacketsData(
                virtual=sim.runner.virt_packet_last_interaction_type,
                real=sim.runner.last_interaction_type[
                    sim.runner.emitted_packet_mask
                ],
            ),
            last_line_interaction_in_id=PacketsData(
                virtual=sim.runner.virt_packet_last_line_interaction_in_id,
                real=sim.runner.last_line_interaction_in_id[
                    sim.runner.emitted_packet_mask
                ],
            ),
            last_line_interaction_out_id=PacketsData(
                virtual=sim.runner.virt_packet_last_line_interaction_out_id,
                real=sim.runner.last_line_interaction_out_id[
                    sim.runner.emitted_packet_mask
                ],
            ),
            last_interaction_in_nu=PacketsData(
                virtual=sim.runner.virt_packet_last_interaction_in_nu
                * units.Hz,
                real=sim.runner.last_interaction_in_nu[
                    sim.runner.emitted_packet_mask
                ]
                * units.Hz,
            ),
            lines_data=sim.plasma.atomic_data.lines.reset_index().set_index(
                "line_id"
            ),
            packet_nus=PacketsData(
                virtual=sim.runner.virt_packet_nus * units.Hz,
                real=sim.runner.output_nu[sim.runner.emitted_packet_mask],
            ),
            packet_energies=PacketsData(
                virtual=sim.runner.virt_packet_energies * units.erg,
                real=sim.runner.output_energy[sim.runner.emitted_packet_mask],
            ),
            R_phot=(sim.model.v_inner[0] * sim.model.time_explosion).to("cm"),
            spectrum_wave=PacketsData(
                virtual=sim.runner.spectrum_virtual.wavelength,
                real=sim.runner.spectrum.wavelength,
            ),
            spectrum_luminosity=PacketsData(
                virtual=sim.runner.spectrum_virtual.luminosity_density_lambda,
                real=sim.runner.spectrum.luminosity_density_lambda,
            ),
            t_inner=sim.model.t_inner,
            time_of_simulation=sim.runner.time_of_simulation,
        )

    @property
    def zmax(self):
        """Maximum atomic number"""
        return self._zmax

    @property
    def cmap(self):
        """Colour map, used to highlight the different atoms"""
        return self._cmap

    @property
    def ax(self):
        """Main axes, containing the emission part of the Kromer plot"""
        return self._ax

    @property
    def pax(self):
        """Secondary axes, containing the absorption part of the Kromer plot"""
        return self._pax

    @property
    def bins(self):
        """frequency binning for the spectral visualisation"""
        return self._bins

    @property
    def xlim(self):
        """wavelength limits"""
        return self._xlim

    @property
    def ylim(self):
        """Flux limits"""
        return self._ylim

    @property
    def twinx(self):
        """switch to decide where to place the absorption part of the Kromer
        plot"""
        return self._twinx

    def get_elements_to_plot(self, mode_idx):
        """produces list of elements to be included in the kromer plot"""
        line_out_infos_within_xlims = self.line_out_infos[mode_idx].loc[
            (self.line_out_infos[mode_idx].wavelength >= self._xlim[0])
            & (self.line_out_infos[mode_idx].wavelength <= self._xlim[1])
        ]
        elements_to_plot = np.c_[
            np.unique(
                line_out_infos_within_xlims.atomic_number.values,
                return_counts=True,
            )
        ]
        if len(elements_to_plot) > self._nelements:
            elements_to_plot = elements_to_plot[
                np.argsort(elements_to_plot[:, 1])[::-1]
            ]
            elements_to_plot = elements_to_plot[: self._nelements]
            elements_to_plot = elements_to_plot[
                np.argsort(elements_to_plot[:, 0])
            ]
        else:
            self._nelements = len(elements_to_plot)
        return elements_to_plot

    def generate_plot(
        self,
        mode="virtual",
        ax=None,
        cmap=cm.jet,
        bins=None,
        xlim=None,
        ylim=None,
        nelements=None,
        twinx=False,
    ):
        """Generate the actual "Kromer" plot

        Parameters
        ----------
        mode: 'real' or 'virtual'
        ax : matplotlib.axes or None
            axes object into which the emission part of the Kromer plot should
            be plotted; if None, a new one is generated (default None)
        cmap : matplotlib.cm.ListedColormap or None
            color map object used for the illustration of the different atomic
            contributions (default matplotlib.cm.jet)
        bins : np.ndarray or None
            array of the wavelength bins used for the illustration of the
            atomic contributions; if None, the same binning as for the stored
            virtual spectrum is used (default None)
        xlim : tuple or array-like or None
            wavelength limits for the display; if None, the x-axis is
            automatically scaled (default None)
        ylim : tuple or array-like or None
            flux limits for the display; if None, the y-axis is automatically
            scaled (default None)
        twinx : boolean
            determines where the absorption part of the Kromer plot is placed,
            if True, the absorption part is attached at the top of the main
            axes box, otherwise it is placed below the emission part (default
            False)

        Returns
        -------
        fig : matplotlib.axes
            Axes object with the plot drawn onto it
        """
        possible_modes = PacketsData._fields
        if mode in possible_modes:
            mode_idx = possible_modes.index(mode)
        else:
            raise ValueError(
                "Invalid value of mode, it can only be 'real' or 'virtual'"
            )

        self._pax = None
        self._zmax = 100

        self._cmap = cmap
        self._ax = ax
        self._ylim = ylim
        self._twinx = twinx

        if nelements == None:
            self._nelements = len(
                np.unique(self.line_out_infos[mode_idx].atomic_number.values)
            )
        else:
            self._nelements = nelements

        if xlim == None:
            self._xlim = [
                np.min(self.spectrum_wave[mode_idx]).value,
                np.max(self.spectrum_wave[mode_idx]).value,
            ]
        else:
            self._xlim = xlim

        if bins is None:
            self._bins = self.spectrum_wave[mode_idx][::-1]
        else:
            self._bins = bins

        elements_to_plot = self.get_elements_to_plot(mode_idx)

        self._axes_handling_preparation()
        self._generate_emission_part(mode_idx, elements_to_plot)
        self._generate_photosphere_part(mode_idx)
        self._generate_and_add_colormap(elements_to_plot)
        self._generate_and_add_legend()
        self._paxes_handling_preparation()
        self._generate_absorption_part(mode_idx, elements_to_plot)
        self._axis_handling_label_rescale()

        return plt.gca()

    def _axes_handling_preparation(self):
        """prepare the main axes; create a new axes if none exists"""

        if self._ax is None:
            self._ax = plt.figure().add_subplot(111)

    def _paxes_handling_preparation(self):
        """prepare the axes for the absorption part of the Kromer plot
        according to the twinx value"""

        if self.twinx:
            self._pax = self._ax.twinx()
        else:
            self._pax = self._ax

    def _generate_emission_part(self, mode, elements):
        """generate the emission part of the Kromer plot"""

        lams = [self.lam_noint[mode], self.lam_escat[mode]]
        weights = [self.weights_noint[mode], self.weights_escat[mode]]
        colors = ["black", "grey"]

        for zi in elements[:, 0]:
            mask = self.line_out_infos[mode].atomic_number.values == zi
            lams.append(
                (csts.c.cgs / (self.line_out_nu[mode][mask])).to(units.AA)
            )
            weights.append(
                self.line_out_L[mode][mask] / self.time_of_simulation
            )
        for ii in range(self._nelements):
            colors.append(self.cmap(float(ii) / float(self._nelements)))

        Lnorm = 0
        for w, lam in zip(weights, lams):
            Lnorm += np.sum(w[(lam >= self.bins[0]) * (lam <= self.bins[-1])])

        lams = [tmp_lam.value for tmp_lam in lams]
        weights = [tmp_wt.value for tmp_wt in weights]
        ret = self.ax.hist(
            lams,
            bins=self.bins.value,
            stacked=True,
            histtype="stepfilled",
            density=True,
            weights=weights,
        )

        for i, col in enumerate(ret[-1]):
            for reti in col:
                reti.set_facecolor(colors[i])
                reti.set_edgecolor(colors[i])
                reti.set_linewidth(0)
                reti.xy[:, 1] *= Lnorm.to("erg / s").value

        self.ax.plot(
            self.spectrum_wave[mode],
            self.spectrum_luminosity[mode],
            color="blue",
            drawstyle="steps-post",
            lw=0.5,
        )

    def _generate_photosphere_part(self, mode):
        """generate the photospheric input spectrum part of the Kromer plot"""

        Lph = (
            abb.blackbody_lambda(self.spectrum_wave[mode], self.t_inner)
            * 4
            * np.pi ** 2
            * self.R_phot ** 2
            * units.sr
        ).to("erg / (AA s)")

        self.ax.plot(self.spectrum_wave[mode], Lph, color="red", ls="dashed")

    def _generate_absorption_part(self, mode, elements):
        """generate the absorption part of the Kromer plot"""

        lams = []
        weights = []
        colors = []

        for zi in elements[:, 0]:
            mask = self.line_in_infos[mode].atomic_number.values == zi
            lams.append((csts.c.cgs / self.line_in_nu[mode][mask]).to(units.AA))
            weights.append(self.line_in_L[mode][mask] / self.time_of_simulation)
        for ii in range(self._nelements):
            colors.append(self.cmap(float(ii) / float(self._nelements)))

        Lnorm = 0
        for w, lam in zip(weights, lams):
            Lnorm -= np.sum(w[(lam >= self.bins[0]) * (lam <= self.bins[-1])])

        lams = [tmp_l.value for tmp_l in lams]
        weights = [tmp_wt.value for tmp_wt in weights]
        ret = self.pax.hist(
            lams,
            bins=self.bins.value,
            stacked=True,
            histtype="stepfilled",
            density=True,
            weights=weights,
        )

        for i, col in enumerate(ret[-1]):
            for reti in col:
                reti.set_facecolor(colors[i])
                reti.set_edgecolor(colors[i])
                reti.set_linewidth(0)
                reti.xy[:, 1] *= Lnorm.to("erg / s").value

    def _generate_and_add_colormap(self, elements):
        """generate the custom color map, linking colours with atomic
        numbers"""

        values = [
            self.cmap(float(i) / float(self._nelements))
            for i in range(self._nelements)
        ]

        custcmap = matplotlib.colors.ListedColormap(values)
        bounds = np.arange(self._nelements) + 0.5
        norm = matplotlib.colors.Normalize(vmin=0, vmax=self._nelements)
        mappable = cm.ScalarMappable(norm=norm, cmap=custcmap)
        mappable.set_array(np.linspace(1, self.zmax + 1, 256))
        labels = [inv_elements[zi].capitalize() for zi in elements[:, 0]]

        mainax = self.ax
        cbar = plt.colorbar(mappable, ax=mainax)
        cbar.set_ticks(bounds)
        cbar.set_ticklabels(labels)

    def _generate_and_add_legend(self):
        """add legend"""

        bpatch = patches.Patch(color="black", label="photosphere")
        gpatch = patches.Patch(color="grey", label="e-scattering")
        bline = lines.Line2D([], [], color="blue", label="virtual spectrum")
        phline = lines.Line2D(
            [], [], color="red", ls="dashed", label="L at photosphere"
        )

        self.ax.legend(handles=[phline, bline, gpatch, bpatch])

    def _axis_handling_label_rescale(self):
        """add axis labels and perform axis scaling"""

        if self.ylim is None:
            self.ax.autoscale(axis="y")
        else:
            self.ax.set_ylim(self.ylim)

        self._ylim = self.ax.get_ylim()

        if self.xlim is None:
            self.ax.autoscale(axis="x")
        else:
            self.ax.set_xlim(self.xlim)

        self._xlim = self.ax.get_xlim()

        if self.twinx:
            self.pax.set_ylim([-self.ylim[-1], -self.ylim[0]])
            self.pax.set_yticklabels([])
        else:
            self.pax.set_ylim([-self.ylim[-1], self.ylim[-1]])
        self.pax.set_xlim(self.xlim)

        self.ax.set_xlabel(r"$\lambda$ [$\mathrm{\AA}$]")
        ylabel = r"$L_{\mathrm{\lambda}}$ [$\mathrm{erg\,s^{-1}\,\AA^{-1}}$]"
        self.ax.set_ylabel(ylabel)
