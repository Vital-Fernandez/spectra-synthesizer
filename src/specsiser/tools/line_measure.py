import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt, rcParams, spines, gridspec
from scipy import integrate

from lmfit import fit_report
from lmfit.models import PolynomialModel
import astropy.units as au
from specutils import Spectrum1D, SpectralRegion
from specutils.manipulation import noise_region_uncertainty
from specutils.fitting import find_lines_derivative

from src.specsiser.data_printing import label_decomposition, PdfPrinter
from src.specsiser.tools.line_fitting import EmissionFitting, gaussian_model, linear_model, c_KMpS

from matplotlib.widgets import SpanSelector

# Parameters configuration: 0) Normalized by flux, 1) Regions wavelengths, 2) Array variable
LOG_COLUMNS = {'wavelength': [False, False, True],
               'intg_flux': [True, False, False],
               'intg_err': [True, False, False],
               'gauss_flux': [True, False, True],
               'gauss_err': [True, False, True],
               'eqw': [False, False, True],
               'eqw_err': [False, False, True],
               'ion': [False, False, True],
               'latexLabel': [False, False, True],
               'blended_label': [False, False, False],
               'w1': [False, True, False],
               'w2': [False, True, False],
               'w3': [False, True, False],
               'w4': [False, True, False],
               'w5': [False, True, False],
               'w6': [False, True, False],
               'peak_wave': [False, False, False],
               'peak_flux': [True, False, False],
               'cont': [True, False, False],
               'std_cont': [True, False, False],
               'm_cont': [True, False, False],
               'n_cont': [True, False, False],
               'snr_line': [False, False, False],
               'snr_cont': [False, False, False],
               'z_line': [False, False, False],
               'amp': [True, False, True],
               'center': [False, False, True],
               'sigma': [False, False, True],
               'amp_err': [True, False, True],
               'center_err': [False, False, True],
               'sigma_err': [False, False, True],
               'v_r': [False, False, True],
               'v_r_err': [False, False, True],
               'sigma_vel': [False, False, True],
               'sigma_vel_err': [False, False, True],
               'observation': [False, False, False],
               'comments': [False, False, False]}

LINELOG_TYPES = {'index': '<U50',
                 'wavelength': '<f8',
                 'intg_flux': '<f8',
                 'intg_err': '<f8',
                 'gauss_flux': '<f8',
                 'gauss_err': '<f8',
                 'eqw': '<f8',
                 'eqw_err': '<f8',
                 'ion': '<U50',
                 'pynebCode': '<f8',
                 'pynebLabel': '<f8',
                 'lineType': '<f8',
                 'latexLabel': '<U50',
                 'blended_label': '<U50',
                 'w1': '<f8',
                 'w2': '<f8',
                 'w3': '<f8',
                 'w4': '<f8',
                 'w5': '<f8',
                 'w6': '<f8',
                 'm_cont': '<f8',
                 'n_cont': '<f8',
                 'cont': '<f8',
                 'std_cont': '<f8',
                 'peak_flux': '<f8',
                 'peak_wave': '<f8',
                 'snr_line': '<f8',
                 'snr_cont': '<f8',
                 'amp': '<f8',
                 'mu': '<f8',
                 'sigma': '<f8',
                 'amp_err': '<f8',
                 'mu_err': '<f8',
                 'sigma_err': '<f8',
                 'v_r': '<f8',
                 'v_r_err': '<f8',
                 'sigma_vel': '<f8',
                 'sigma_err_vel': '<f8',
                 'observation': '<U50',
                 'comments': '<U50',
                 'obsFlux': '<f8',
                 'obsFluxErr': '<f8',
                 'f_lambda': '<f8',
                 'obsInt': '<f8',
                 'obsIntErr': '<f8'}

_LOG_EXPORT = list(set(LOG_COLUMNS.keys()) - set(['ion', 'wavelength',
                                                 'latexLabel',
                                                 'w1', 'w2',
                                                 'w3', 'w4',
                                                 'w5', 'w6', 'observation']))

_MASK_EXPORT = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'observation']

LINEMEASURER_PARAMS = ['pixelWidth',
                       'peak_wave',
                       'peakInt',
                       'intg_flux',
                       'intg_err',
                       'gauss_flux',
                       'gauss_err',
                       'n_cont',
                       'm_cont',
                       'std_cont',
                       'fit_function',
                       'p1',
                       'p1_Err']

PARAMS_CONVERSION = {'intg_flux': 'intg_flux',
                     'intg_err': 'intg_err',
                     'cont': 'cont',
                     'm_cont': 'm_cont',
                     'n_cont': 'n_cont',
                     'std_cont': 'std_cont',
                     'gauss_flux': 'gauss_flux',
                     'gauss_err': 'gauss_err',
                     'eqw': 'eqw',
                     'eqw_err': 'eqw_err'}

WAVE_UNITS_DEFAULT, FLUX_UNITS_DEFAULT = au.AA, au.erg / au.s / au.cm ** 2 / au.AA

VAL_LIST = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
SYB_LIST = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]

FLUX_TEX_TABLE_HEADERS = [r'$Transition$', '$EW(\AA)$', '$F(\lambda)$', '$I(\lambda)$']
FLUX_TXT_TABLE_HEADERS = [r'$Transition$', 'EW', 'EW_error', 'F(lambda)', 'F(lambda)_error', 'I(lambda)', 'I(lambda)_error']

KIN_TEX_TABLE_HEADERS = [r'$Transition$', r'$Comp$', r'$v_{r}\left(\nicefrac{km}{s}\right)$', r'$\sigma_{int}\left(\nicefrac{km}{s}\right)$', r'Flux $(\nicefrac{erg}{cm^{-2} s^{-1} \AA^{-1})}$']
KIN_TXT_TABLE_HEADERS = [r'$Transition$', r'$Comp$', 'v_r', 'v_r_error', 'sigma_int', 'sigma_int_error', 'flux', 'flux_error']

STANDARD_PLOT = {'figure.figsize': (14, 7),
                 'axes.titlesize': 14,
                 'axes.labelsize': 14,
                 'legend.fontsize': 12,
                 'xtick.labelsize': 12,
                 'ytick.labelsize': 12}
STANDARD_AXES = {'xlabel': r'Wavelength $(\AA)$', 'ylabel': r'Flux $(erg\,cm^{-2} s^{-1} \AA^{-1})$'}


def lineslogFile_to_DF(lineslog_address):
    """
    This function attemps several approaches to import a lines log from a sheet or text file lines as a pandas
    dataframe
    :param lineslog_address: String with the location of the input lines log file
    :return lineslogDF: Dataframe with line labels as index and default column headers (wavelength, w1 to w6)
    """

    # Text file
    try:
        lineslogDF = pd.read_csv(lineslog_address, delim_whitespace=True, header=0, index_col=0)
    except ValueError:

        # Excel file
        try:
            lineslogDF = pd.read_excel(lineslog_address, sheet_name=0, header=0, index_col=0)
        except ValueError:
            print(f'- ERROR: Could not open lines log at: {lineslog_address}')

    return lineslogDF


def compute_lineWidth(idx_peak, spec_flux, delta_i, min_delta=2):
    """
    Algororithm to measure emision line width given its peak location
    :param idx_peak:
    :param spec_flux:
    :param delta_i:
    :param min_delta:
    :return:
    """

    i = idx_peak
    while (spec_flux[i] > spec_flux[i + delta_i]) or (np.abs(idx_peak - (i + delta_i)) <= min_delta):
        i += delta_i

    return i


def latex_science_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def kinematic_component_labelling(line_latex_label, comp_ref):

    if len(comp_ref) != 2:
        print(f'-- Warning: Components label for {line_latex_label} is {comp_ref}. Code only prepare for a 2 character description (ex. n1, w2...)')

    number = comp_ref[-1]
    letter = comp_ref[0]

    if letter in ('n', 'w'):
        if letter == 'n':
            comp_label = f'Narrow {number}'
        if letter == 'w':
            comp_label = f'Wide {number}'
    else:
        comp_label = f'{letter}{number}'

    if '-' in line_latex_label:
        lineEmisLabel = line_latex_label.replace(f'-{comp_ref}', '')
    else:
        lineEmisLabel = line_latex_label

    return comp_label, lineEmisLabel


def lineslogFile_to_DF(lineslog_address):
    """
    This function attemps several approaches to import a lines log from a sheet or text file lines as a pandas
    dataframe
    :param lineslog_address: String with the location of the input lines log file
    :return lineslogDF: Dataframe with line labels as index and default column headers (wavelength, w1 to w6)
    """

    # Text file
    try:
        lineslogDF = pd.read_csv(lineslog_address, delim_whitespace=True, header=0, index_col=0)
    except ValueError:

        # Excel file
        try:
            lineslogDF = pd.read_excel(lineslog_address, sheet_name=0, header=0, index_col=0)
        except ValueError:
            print(f'- ERROR: Could not open lines log at: {lineslog_address}')

    return lineslogDF


def save_lineslog(linesDF, file_address):

    with open(file_address, 'wb') as output_file:
        string_DF = linesDF.to_string()
        output_file.write(string_DF.encode('UTF-8'))

    return


class LineMesurer(EmissionFitting):

    def __init__(self, input_wave=None, input_flux=None, input_err=None, linesDF_address=None, redshift=0,
                 normFlux=1, crop_waves=None):

        # Emission model inheritance
        EmissionFitting.__init__(self)

        # Class attributes
        self.wave = None
        self.flux = None
        self.errFlux = None
        self.normFlux = normFlux
        self.redshift = redshift
        self.linesLogAddress = linesDF_address
        self.linesDF = None

        # Start cropping the input spectrum if necessary
        if crop_waves is not None:
            idcs_cropping = (input_wave >= crop_waves[0]) & (input_wave <= crop_waves[1])
            input_wave = input_wave[idcs_cropping]
            input_flux = input_flux[idcs_cropping]
            if input_err is not None:
                input_err = input_err[idcs_cropping]

        # Apply the redshift correction
        if input_wave is not None:
            self.wave_rest = input_wave / (1 + self.redshift)
            if (input_wave is not None) and (input_flux is not None):
                self.wave = input_wave
                self.flux = input_flux # * (1 + self.redshift)
                if input_err is not None:
                    self.errFlux = input_err # * (1 + self.redshift)

        # Normalize the spectrum
        if input_flux is not None:
            self.flux = self.flux / self.normFlux
            if input_err is not None:
                self.errFlux = self.errFlux / self.normFlux

        # Generate empty dataframe to store measurement use cwd as default storing folder
        self.linesLogAddress = linesDF_address
        if self.linesLogAddress is None:
            self.linesDF = pd.DataFrame(columns=LOG_COLUMNS.keys())

        # Otherwise use the one from the user
        else:
            if Path(self.linesLogAddress).is_file():
                self.linesDF = lineslogFile_to_DF(linesDF_address)
            else:
                print(f'-- WARNING: linesLog not found at {self.linesLogAddress}')

        return

    def print_results(self, label=None, show_fit_report=True, show_plot=False, log_scale=True, frame='obs'):

        # Case no line as input: Show the current measurement
        if label is None:
            if self.lineLabel is not None:
                label = self.lineLabel
                output_ref = (f'Input line: {label}\n'
                              f'- Line regions: {self.lineWaves}\n'
                              f'- Spectrum: normalization flux: {self.normFlux}; redshift {self.redshift}\n'
                              f'- Peak: wavelength {self.peak_wave:.2f}; peak intensity {self.peak_flux:.2f}\n'
                              f'- Continuum: slope {self.m_cont:.2e}; intercept {self.n_cont:.2e}\n')

                if self.blended_check:
                    mixtureComponents = np.array(self.blended_label.split('-'))
                else:
                    mixtureComponents = np.array([label], ndmin=1)

                if mixtureComponents.size == 1:
                    output_ref += f'- Intg Eqw: {self.eqw[0]:.2f} +/- {self.eqw_err[0]:.2f}\n'

                output_ref += f'- Intg flux: {self.intg_flux:.3f} +/- {self.intg_err:.3f}\n'

                for i, lineRef in enumerate(mixtureComponents):
                    output_ref += (f'- {lineRef} gaussian fitting:\n'
                                   f'-- Gauss flux: {self.gauss_flux[i]:.3f} +/- {self.gauss_err[i]:.3f}\n'
                                   f'-- Amplitude: {self.amp[i]:.3f} +/- {self.amp_err[i]:.3f}\n'
                                   f'-- Center: {self.center[i]:.2f} +/- {self.center_err[i]:.2f}\n'
                                   f'-- Sigma: {self.sigma[i]:.2f} +/- {self.sigma_err[i]:.2f}\n\n')
            else:
                output_ref = f'- No measurement performed\n'

        # Case with line input: search and show that measurement
        elif self.linesDF is not None:
            if label in self.linesDF.index:
                output_ref = self.linesDF.loc[label].to_string
            else:
                output_ref = f'- WARNING: {label} not found in  lines table\n'
        else:
            output_ref = '- WARNING: Measurement lines log not defined\n'

        # Display the print lmfit report if available
        if show_fit_report:
            if self.fit_output is not None:
                output_ref += f'- LmFit output:\n{fit_report(self.fit_output)}\n'
            else:
                output_ref += f'- LmFit output not available\n'

        # Show the result
        print(output_ref)

        # Display plot
        if show_plot:
            self.plot_fit_components(self.fit_output, log_scale=log_scale, frame=frame)

        return

    def fit_from_wavelengths(self, label, line_wavelengths, user_conf={}, algorithm='lmfit'):

        # For security previous measurement is cleared and a copy of the user configuration is used
        self.clear_fit()
        fit_conf = user_conf.copy()

        # Label the current measurement
        self.lineLabel = label
        self.lineWaves = line_wavelengths

        # Establish spectrum line and continua regions
        idcsEmis, idcsCont = self.define_masks(self.wave_rest, self.flux, self.lineWaves)

        # Integrated line properties
        emisWave, emisFlux = self.wave[idcsEmis], self.flux[idcsEmis]
        contWave, contFlux = self.wave[idcsCont], self.flux[idcsCont]
        err_array = self.errFlux[idcsEmis] if self.errFlux is not None else None
        self.line_properties(emisWave, emisFlux, contWave, contFlux, err_array, bootstrap_size=1000)

        # Check if blended line
        if self.lineLabel in fit_conf:
            self.blended_label = fit_conf[self.lineLabel]
            if '_b' in self.lineLabel:
                self.blended_check = True

        # Check the kinematics import
        self.import_kinematics_from_line(fit_conf, z_cor=1 + self.redshift)

        # Gaussian fitting # TODO Add logic for very small lines
        idcsLine = idcsEmis + idcsCont
        x_array = self.wave[idcsLine]
        y_array = self.flux[idcsLine]
        w_array = 1.0/self.errFlux[idcsLine] if self.errFlux is not None else np.full(x_array.size, 1.0 / self.std_cont)
        self.gauss_lmfit(self.lineLabel, x_array, y_array, w_array, fit_conf, self.linesDF, z_obj=self.redshift)

        # Safe the results to log DF
        self.results_to_database(self.lineLabel, self.linesDF, fit_conf)

        return

    def import_kinematics_from_line(self, user_conf, z_cor):

        # Check if line kinematics are contained in blended line
        if self.blended_label != 'None':
            childs_list = self.blended_label.split('-')
        else:
            childs_list = np.array(self.lineLabel, ndmin=1)

        for child_label in childs_list:
            parent_label = user_conf.get(f'{child_label}_kinem')

            if parent_label is not None:

                # Case we want to copy from previous line and the data is not available
                if (parent_label not in self.linesDF.index) and (not self.blended_check):
                    print(
                        f'-- WARNING: {parent_label} has not been measured. Its kinematics were not copied to {child_label}')

                else:
                    ion_parent, wtheo_parent, latex_parent = label_decomposition(parent_label, scalar_output=True)
                    ion_child, wtheo_child, latex_child = label_decomposition(child_label, scalar_output=True)

                    # Copy v_r and sigma_vel in wavelength units
                    for param_ext in ('center', 'sigma'):
                        param_label_child = f'{child_label}_{param_ext}'

                        # Warning overwritten existing configuration
                        if param_label_child in user_conf:
                            print(f'-- WARNING: {param_label_child} overwritten by {parent_label} kinematics in configuration input')

                        # Case we want to copy from previous line
                        if not self.blended_check:
                            mu_parent = self.linesDF.loc[parent_label, ['center', 'center_err']].values
                            sigma_parent = self.linesDF.loc[parent_label, ['sigma_vel', 'sigma_err']].values

                            if param_ext == 'center':
                                param_value = wtheo_child / wtheo_parent * (mu_parent / z_cor)
                            else:
                                param_value = wtheo_child / wtheo_parent * sigma_parent

                            user_conf[param_label_child] = {'value': param_value[0], 'vary': False}
                            user_conf[f'{param_label_child}_err'] = {'value': param_value[1], 'vary': False}

                        # Case where parent and child are in blended group
                        else:
                            param_label_parent = f'{parent_label}_{param_ext}'
                            param_expr_parent = f'{wtheo_child / wtheo_parent:0.8f}*{param_label_parent}'

                            user_conf[param_label_child] = {'expr': param_expr_parent}

        return

    def continuum_remover(self, noiseRegionLims, intLineThreshold=((4, 4), (1.5, 1.5)), degree=(3, 7)):

        assert self.wave_rest[0] < noiseRegionLims[0] and noiseRegionLims[1] < self.wave_rest[-1], \
            f'Error noise region {self.wave_rest[0]} < {noiseRegionLims[0]} and {noiseRegionLims[1]} < {self.wave_rest[-1]}'

        # Identify high flux regions
        idcs_noiseRegion = (noiseRegionLims[0] <= self.wave_rest) & (self.wave_rest <= noiseRegionLims[1])
        noise_mean, noise_std = self.flux[idcs_noiseRegion].mean(), self.flux[idcs_noiseRegion].std()

        # Perform several continuum fits to improve the line detection
        input_wave, input_flux = self.wave_rest, self.flux
        for i in range(len(intLineThreshold)):
            # Mask line regions
            emisLimit = intLineThreshold[i][0] * (noise_mean + noise_std)
            absoLimit = (noise_mean + noise_std) / intLineThreshold[i][1]
            idcsLineMask = np.where((input_flux >= absoLimit) & (input_flux <= emisLimit))
            wave_masked, flux_masked = input_wave[idcsLineMask], input_flux[idcsLineMask]

            # Perform continuum fits iteratively
            poly3Mod = PolynomialModel(prefix=f'poly_{degree[i]}', degree=degree[i])
            poly3Params = poly3Mod.guess(flux_masked, x=wave_masked)
            poly3Out = poly3Mod.fit(flux_masked, poly3Params, x=wave_masked)

            input_flux = input_flux - poly3Out.eval(x=self.wave_rest) + noise_mean

        return input_flux - noise_mean

    def match_lines(self, obsLineTable, maskDF, lineType='emission', tol=5, blendedLineList=[], detect_check=False,
                    find_line_borders='Auto', include_unknown=False):

        #TODO maybe we should remove not detected from output
        theoLineDF = pd.DataFrame.copy(maskDF)

        # Query the lines from the astropy finder tables # TODO Expand technique for absorption lines
        idcsLineType = obsLineTable['line_type'] == lineType
        idcsLinePeak = np.array(obsLineTable[idcsLineType]['line_center_index'])
        waveObs = self.wave_rest[idcsLinePeak]

        # Theoretical wave values
        waveTheory = theoLineDF.wavelength.values

        # Match the lines with the theoretical emission
        tolerance = np.diff(self.wave_rest).mean() * tol
        theoLineDF['observation'] = 'not detected'
        unidentifiedLine = dict.fromkeys(theoLineDF.columns.values, np.nan)

        for i in np.arange(waveObs.size):

            idx_array = np.where(np.isclose(a=waveTheory, b=waveObs[i], atol=tolerance))

            if len(idx_array[0]) == 0:
                unknownLineLabel = 'xy_{:.0f}A'.format(np.round(waveObs[i]))

                # Scheme to avoid repeated lines
                if (unknownLineLabel not in theoLineDF.index) and detect_check:
                    newRow = unidentifiedLine.copy()
                    newRow.update({'wavelength': waveObs[i], 'w3': waveObs[i] - 5, 'w4': waveObs[i] + 5,
                                   'observation': 'not identified'})
                    theoLineDF.loc[unknownLineLabel] = newRow

            else:
                row_index = theoLineDF.index[theoLineDF.wavelength == waveTheory[idx_array[0][0]]]
                theoLineDF.loc[row_index, 'observation'] = 'detected'
                theoLineLabel = row_index[0]

                # TODO lines like Halpha+[NII] this does not work, we should add exclusion
                if find_line_borders == True:
                    minSeparation = 4 if theoLineLabel in blendedLineList else 2
                    idx_min = compute_lineWidth(idcsLinePeak[i], self.flux, delta_i=-1, min_delta=minSeparation)
                    idx_max = compute_lineWidth(idcsLinePeak[i], self.flux, delta_i=1, min_delta=minSeparation)
                    theoLineDF.loc[row_index, 'w3'] = self.wave_rest[idx_min]
                    theoLineDF.loc[row_index, 'w4'] = self.wave_rest[idx_max]
                else:
                    if find_line_borders == 'Auto':
                        if '_b' not in theoLineLabel:
                            minSeparation = 4 if theoLineLabel in blendedLineList else 2
                            idx_min = compute_lineWidth(idcsLinePeak[i], self.flux, delta_i=-1, min_delta=minSeparation)
                            idx_max = compute_lineWidth(idcsLinePeak[i], self.flux, delta_i=1, min_delta=minSeparation)
                            theoLineDF.loc[row_index, 'w3'] = self.wave_rest[idx_min]
                            theoLineDF.loc[row_index, 'w4'] = self.wave_rest[idx_max]

        if include_unknown is False:
            idcs_unknown = theoLineDF['observation'] == 'not detected'
            theoLineDF.drop(index=theoLineDF.loc[idcs_unknown].index.values, inplace=True)

        # Sort by wavelength
        theoLineDF.sort_values('wavelength', inplace=True)

        # Latex labels
        ion_array, wavelength_array, latexLabel_array = label_decomposition(theoLineDF.index.values)
        theoLineDF['latexLabel'] = latexLabel_array
        theoLineDF['blended_label'] = 'None'

        return theoLineDF

    def line_finder(self, input_flux, noiseWaveLim, intLineThreshold=3, verbose=False):

        assert noiseWaveLim[0] > self.wave_rest[0] or noiseWaveLim[1] < self.wave[-1]

        # Establish noise values
        idcs_noiseRegion = (noiseWaveLim[0] <= self.wave_rest) & (self.wave_rest <= noiseWaveLim[1])
        noise_region = SpectralRegion(noiseWaveLim[0] * WAVE_UNITS_DEFAULT, noiseWaveLim[1] * WAVE_UNITS_DEFAULT)
        flux_threshold = intLineThreshold * input_flux[idcs_noiseRegion].std()

        input_spectrum = Spectrum1D(input_flux * FLUX_UNITS_DEFAULT, self.wave_rest * WAVE_UNITS_DEFAULT)
        input_spectrum = noise_region_uncertainty(input_spectrum, noise_region)
        linesTable = find_lines_derivative(input_spectrum, flux_threshold)

        # Additional tools include
        # from specutils.fitting import find_lines_threshold, find_lines_derivative, fit_generic_continuum

        if verbose:
            print(linesTable)

        return linesTable

    def results_to_database(self, lineLabel, linesDF, fit_conf, export_params=_LOG_EXPORT):

        # Recover label data
        if self.blended_check:
            line_components = self.blended_label.split('-')
        else:
            line_components = np.array([lineLabel], ndmin=1)

        ion, waveRef, latexLabel = label_decomposition(line_components, combined_dict=fit_conf)

        # Loop through the line components
        for i, line in enumerate(line_components):

            # Convert current measurement to a pandas series container
            line_log = pd.Series(index=LOG_COLUMNS.keys())
            line_log['ion', 'wavelength', 'latexLabel'] = ion[i], waveRef[i], latexLabel[i]
            line_log['w1': 'w6'] = self.lineWaves

            # Treat every line
            for param in export_params:

                # Get component parameter
                if LOG_COLUMNS[param][2]:
                    param_value = self.__getattribute__(param)[i]
                else:
                    param_value = self.__getattribute__(param)

                # De normalize
                if LOG_COLUMNS[param][0]:
                    param_value = param_value * self.normFlux

                line_log[param] = param_value

            # Assign line series to dataframe
            linesDF.loc[line] = line_log

        return

    def database_to_attr(self):

        # Conversion parameters
        for name_attr, name_df in PARAMS_CONVERSION.items():
            value_df = self.linesDF.loc[self.lineLabel, name_df]
            self.__setattr__(name_attr, value_df)

        # Gaussian fit parameters
        for param in ('amp', 'center', 'sigma'):
            param_value = np.array([self.linesDF.loc[self.lineLabel, param]])
            self.__setattr__(param, param_value)

            param_sigma = np.array([self.linesDF.loc[self.lineLabel, f'{param}_err']])
            self.__setattr__(f'{param}_err', param_sigma)

        return

    def save_lineslog(self, linesDF, file_address):

        with open(file_address, 'wb') as output_file:
            string_DF = linesDF.to_string()
            output_file.write(string_DF.encode('UTF-8'))

        return

    def plot_spectrum(self, continuumFlux=None, obsLinesTable=None, matchedLinesDF=None, noise_region=None,
                      log_scale=False, plotConf={}, axConf={}, specLabel='Observed spectrum', output_address=None):

        # Plot Configuration
        defaultConf = STANDARD_PLOT.copy()
        defaultConf.update(plotConf)
        rcParams.update(defaultConf)
        fig, ax = plt.subplots()

        # Plot the spectrum
        ax.step(self.wave_rest, self.flux, label=specLabel)

        # Plot the continuum if available
        if continuumFlux is not None:
            ax.step(self.wave_rest, continuumFlux, label='Error Continuum', linestyle=':')

        # Plot astropy detected lines if available
        if obsLinesTable is not None:
            idcs_emission = obsLinesTable['line_type'] == 'emission'
            idcs_linePeaks = np.array(obsLinesTable[idcs_emission]['line_center_index'])
            ax.scatter(self.wave_rest[idcs_linePeaks], self.flux[idcs_linePeaks], label='Detected lines', facecolors='none',
                       edgecolors='tab:purple')

        if matchedLinesDF is not None:
            idcs_foundLines = (matchedLinesDF.observation.isin(('detected', 'not identified'))) & \
                              (matchedLinesDF.wavelength >= self.wave_rest[0]) & \
                              (matchedLinesDF.wavelength <= self.wave_rest[-1])
            if 'latexLabel' in matchedLinesDF:
                lineLatexLabel = matchedLinesDF.loc[idcs_foundLines].latexLabel.values
            else:
                lineLatexLabel = matchedLinesDF.loc[idcs_foundLines].index.values
            lineWave = matchedLinesDF.loc[idcs_foundLines].wavelength.values
            w3, w4 = matchedLinesDF.loc[idcs_foundLines].w3.values, matchedLinesDF.loc[idcs_foundLines].w4.values
            observation = matchedLinesDF.loc[idcs_foundLines].observation.values

            for i in np.arange(lineLatexLabel.size):
                if observation[i] == 'detected':
                    color_area = 'tab:red' if observation[i] == 'not identified' else 'tab:green'
                    ax.axvspan(w3[i], w4[i], alpha=0.25, color=color_area)
                    ax.text(lineWave[i], 0, lineLatexLabel[i], rotation=270)

        if noise_region is not None:
            ax.axvspan(noise_region[0], noise_region[1], alpha=0.15, color='tab:cyan', label='Noise region')

        if log_scale:
            ax.set_yscale('log')

        if self.normFlux != 1:
            if 'ylabel' not in axConf:
                y_label = STANDARD_AXES['ylabel']
                axConf['ylabel'] = y_label.replace('Flux', r'$Flux\,/\,{}$'.format(latex_science_float(self.normFlux)))

        ax.update({**STANDARD_AXES, **axConf})
        ax.legend()

        if output_address is None:
            plt.tight_layout()
            plt.show()
        else:
            plt.savefig(output_address, bbox_inches='tight')

        plt.close(fig)


        return

    def plot_fit_components(self, lmfit_output=None, line_label=None, fig_conf={}, ax_conf={}, output_address=None,
                                  log_scale=False, frame='obs'):

        # Determine line Label:
        # TODO this function should read from lines log
        # TODO this causes issues if vary is false... need a better way to get label
        line_label = line_label if line_label is not None else self.lineLabel
        ion, wave, latexLabel = label_decomposition(line_label, scalar_output=True)

        # Plot Configuration
        defaultConf = STANDARD_PLOT.copy()
        defaultConf.update(fig_conf)
        rcParams.update(defaultConf)

        defaultConf = STANDARD_AXES.copy()
        defaultConf.update(ax_conf)

        # Case in which no emission line is introduced
        if lmfit_output is None:
            fig, ax = plt.subplots()
            ax = [ax]
        else:
            # fig, ax = plt.subplots(nrows=2)
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
            spec_ax = plt.subplot(gs[0])
            grid_ax = plt.subplot(gs[1], sharex=spec_ax)
            ax = [spec_ax, grid_ax]

        if frame == 'obs':
            z_cor = 1
            wave_plot = self.wave
            flux_plot = self.flux
        elif frame == 'rest':
            z_cor = 1 + self.z_line
            wave_plot = self.wave / z_cor
            flux_plot = self.flux * z_cor
        else:
            exit(f'-- Plot with frame name {frame} not recognize. Code will stop.')


        # Establish spectrum line and continua regions
        idcsEmis, idcsContBlue, idcsContRed = self.define_masks(self.wave_rest,
                                                                self.flux,
                                                                self.lineWaves,
                                                                merge_continua=False)
        idcs_plot = (wave_plot[idcsContBlue][0] - 5 <= wave_plot) & (wave_plot <= wave_plot[idcsContRed][-1] + 5)

        # Plot line spectrum
        ax[0].step(wave_plot[idcs_plot], flux_plot[idcs_plot], label=r'Observed spectrum: {}'.format(latexLabel), where='mid')
        ax[0].scatter(self.peak_wave/z_cor, self.peak_flux*z_cor, color='tab:blue', alpha=0.7)

        # Plot selection regions
        ax[0].fill_between(wave_plot[idcsContBlue], 0, flux_plot[idcsContBlue], facecolor='tab:orange', step='mid', alpha=0.2)
        ax[0].fill_between(wave_plot[idcsEmis], 0, flux_plot[idcsEmis], facecolor='tab:green', step='mid', alpha=0.2)
        ax[0].fill_between(wave_plot[idcsContRed], 0, flux_plot[idcsContRed], facecolor='tab:orange', step='mid', alpha=0.2)

        # Axes formatting
        if self.normFlux != 1.0:
            defaultConf['ylabel'] = defaultConf['ylabel'] + " $\\times{{{0:.2g}}}$".format(self.normFlux)

        if log_scale:
            ax[0].set_yscale('log')


        # Plot the Gaussian fit if available
        if lmfit_output is not None:

            # Recover values from fit
            x_in, y_in = lmfit_output.userkws['x'], lmfit_output.data

            # Resample gaussians
            wave_resample = np.linspace(x_in[0], x_in[-1], 200)
            flux_resample = lmfit_output.eval_components(x=wave_resample)

            # Plot input data
            ax[0].scatter(x_in/z_cor, y_in*z_cor, color='tab:red', label='Input data', alpha=0.4)
            ax[0].plot(x_in/z_cor, lmfit_output.best_fit*z_cor, label='Gaussian fit')

            # Plot individual components
            if not self.blended_check:
                contLabel = f'{line_label}_cont_'
            else:
                contLabel = f'{self.blended_label.split("-")[0]}_cont_'

            cont_flux = flux_resample.get(contLabel, 0.0)
            for comp_label, comp_flux in flux_resample.items():
                comp_flux = comp_flux + cont_flux if comp_label != contLabel else comp_flux
                ax[0].plot(wave_resample/z_cor, comp_flux*z_cor, label=f'{comp_label}', linestyle='--')

            # Continuum residual plot:
            residual = (y_in - lmfit_output.best_fit)/self.cont
            ax[1].step(x_in/z_cor, residual*z_cor, where='mid')

            label = r'$\sigma_{Continuum}/\overline{F(linear)}$'
            print('Sigma', self.std_cont, 'norm_sigma', self.std_cont / self.cont)
            y_low, y_high = -self.std_cont / self.cont, self.std_cont / self.cont
            ax[1].fill_between(x_in/z_cor, y_low*z_cor, y_high*z_cor, facecolor='tab:orange', alpha=0.5, label=label)

            # Err residual plot if available:
            if self.errFlux is not None:
                label = r'$\sigma_{Error}/\overline{F(linear)}$'
                err_norm = np.sqrt(self.errFlux[idcs_plot])/self.cont
                ax[1].fill_between(wave_plot[idcs_plot]/z_cor, -err_norm*z_cor, err_norm*z_cor, facecolor='tab:red', alpha=0.5, label=label)

            # Residual plot labeling
            ax[1].set_xlim(ax[0].get_xlim())
            ax[1].set_ylim(2*residual.min(), 2*residual.max())
            ax[1].legend(loc='upper left')
            ax[1].set_ylabel(r'$\frac{F_{obs}}{F_{fit}} - 1$')
            ax[1].set_xlabel(r'Wavelength $(\AA)$')

        ax[0].legend()
        ax[0].update(defaultConf)

        if output_address is None:
            plt.tight_layout()
            plt.show()
        else:
            plt.savefig(output_address, bbox_inches='tight')

        return

    def plot_line_grid(self, linesDF, plotConf={}, ncols=10, nrows=None, output_address=None, log_scale=True, frame='rest'):

        # Line labels to plot
        lineLabels = linesDF.index.values

        # Define plot axes grid size
        if nrows is None:
            nrows = int(np.ceil(lineLabels.size / ncols))
        if 'figure.figsize' not in plotConf:
            nrows = int(np.ceil(lineLabels.size / ncols))
            plotConf['figure.figsize'] = (ncols * 3, nrows * 3)
        n_axes, n_lines = ncols * nrows, lineLabels.size

        if frame == 'obs':
            z_cor = 1
            wave_plot = self.wave
            flux_plot = self.flux
        elif frame == 'rest':
            z_cor = 1 + self.redshift
            wave_plot = self.wave / z_cor
            flux_plot = self.flux * z_cor
        else:
            exit(f'-- Plot with frame name {frame} not recognize. Code will stop.')

        # Figure configuration
        defaultConf = STANDARD_PLOT.copy()
        defaultConf.update(plotConf)
        rcParams.update(defaultConf)

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
        axesList = ax.flatten()

        # Loop through the lines
        for i in np.arange(n_axes):
            if i < n_lines:

                # Line data
                lineLabel = lineLabels[i]
                lineWaves = linesDF.loc[lineLabel, 'w1':'w6'].values
                latexLabel = linesDF.loc[lineLabel, 'latexLabel']


                # Establish spectrum line and continua regions
                idcsEmis, idcsContBlue, idcsContRed = self.define_masks(self.wave_rest,
                                                                        self.flux,
                                                                        lineWaves,
                                                                        merge_continua=False)
                idcs_plot = (wave_plot[idcsContBlue][0] - 5 <= wave_plot) & (
                            wave_plot <= wave_plot[idcsContRed][-1] + 5)

                # Plot observation
                ax_i = axesList[i]
                ax_i.step(wave_plot[idcs_plot], flux_plot[idcs_plot], where='mid')
                ax_i.fill_between(wave_plot[idcsContBlue], 0, flux_plot[idcsContBlue], facecolor='tab:orange', step="mid", alpha=0.2)
                ax_i.fill_between(wave_plot[idcsEmis], 0, flux_plot[idcsEmis], facecolor='tab:blue', step="mid", alpha=0.2)
                ax_i.fill_between(wave_plot[idcsContRed], 0, flux_plot[idcsContRed], facecolor='tab:orange', step="mid", alpha=0.2)

                if set(['m_cont', 'n_cont', 'amp', 'center', 'sigma']).issubset(linesDF.columns):

                    line_params = linesDF.loc[lineLabel, ['m_cont', 'n_cont']].values
                    gaus_params = linesDF.loc[lineLabel, ['amp', 'center', 'sigma']].values

                    # Plot curve fitting
                    if (not pd.isnull(line_params).any()) and (not pd.isnull(gaus_params).any()):

                        wave_resample = np.linspace(self.wave[idcs_plot][0], self.wave[idcs_plot][-1], 500)

                        m_cont, n_cont = line_params /self.normFlux
                        line_resample = linear_model(wave_resample, m_cont, n_cont)

                        amp, mu, sigma = gaus_params
                        amp = amp/self.normFlux
                        gauss_resample = gaussian_model(wave_resample, amp, mu, sigma) + line_resample
                        ax_i.plot(wave_resample/z_cor, gauss_resample*z_cor, '--', color='tab:purple', linewidth=1.50)

                    else:
                        for child in ax_i.get_children():
                            if isinstance(child, spines.Spine):
                                child.set_color('tab:red')

                # Axis format
                ax_i.yaxis.set_major_locator(plt.NullLocator())
                ax_i.yaxis.set_ticklabels([])
                ax_i.xaxis.set_major_locator(plt.NullLocator())
                ax_i.axes.yaxis.set_visible(False)
                ax_i.set_title(latexLabel)

                if log_scale:
                    ax_i.set_yscale('log')

            # Clear not filled axes
            else:
                fig.delaxes(axesList[i])

        if output_address is None:
            plt.tight_layout()
            plt.show()
        else:
            plt.savefig(output_address, bbox_inches='tight')

        plt.close(fig)

        return

    def plot_line_mask_selection(self, linesDF, df_file_address, ncols=10, nrows=None, logscale=True):

        # Update mask file for new location
        # TODO it may be better to clear the linesDF after the plot is done
        self.linesLogAddress = df_file_address
        self.linesDF = pd.DataFrame(columns=LOG_COLUMNS.keys())
        for column in linesDF.columns:
            self.linesDF[column] = linesDF[column]

        # Plot data
        lineLabels = self.linesDF.index.values

        if nrows is None:
            nrows = int(np.ceil(lineLabels.size / ncols))

        # Compute plot grid size
        plotConf = {'figure.figsize': (nrows * 2, 8)}

        # Plot format
        defaultConf = STANDARD_PLOT.copy()
        defaultConf.update(plotConf)
        rcParams.update(defaultConf)

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
        axesList = ax.flatten()
        dict_spanSelec = {}

        # Generate plot
        for i in np.arange(lineLabels.size):
            self.lineWaves = self.linesDF.loc[lineLabels[i], 'w1':'w6'].values
            self.plot_line_region_i(axesList[i], lineLabels[i], self.linesDF, logscale=logscale)
            dict_spanSelec[f'spanner_{i}'] = SpanSelector(axesList[i],
                                                          self.on_select,
                                                          'horizontal',
                                                          useblit=True,
                                                          rectprops=dict(alpha=0.5, facecolor='tab:blue'))

        bpe = fig.canvas.mpl_connect('button_press_event', self.on_click)
        aee = fig.canvas.mpl_connect('axes_enter_event', self.on_enter_axes)

        plt.gca().axes.yaxis.set_ticklabels([])
        plt.tight_layout()
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()

        plt.show()
        plt.close(fig)

        return

    def plot_line_region_i(self, ax, lineLabel, linesDF, limitPeak=5, logscale=False):

        # Plot line region:
        lineWave = linesDF.loc[lineLabel, 'wavelength']

        # Decide type of plot
        non_nan = (~pd.isnull(self.lineWaves)).sum()

        # Incomplete selections
        if non_nan < 6:  # selections

            idcsLinePeak = (lineWave - limitPeak <= self.wave_rest) & (self.wave_rest <= lineWave + limitPeak)
            idcsLineArea = (lineWave - limitPeak * 2 <= self.wave_rest) & (lineWave - limitPeak * 2 <= self.lineWaves[3])
            wavePeak, fluxPeak = self.wave_rest[idcsLinePeak], self.flux[idcsLinePeak]
            waveLine, fluxLine = self.wave_rest[idcsLineArea], self.flux[idcsLineArea]
            idxPeakFlux = np.argmax(fluxPeak)

            ax.step(waveLine, fluxLine)

            if non_nan == 2:
                idx1, idx2 = np.searchsorted(self.wave_rest, self.lineWaves[0:2])
                ax.fill_between(self.wave_rest[idx1:idx2], 0.0, self.flux[idx1:idx2], facecolor='tab:green',
                                step='mid', alpha=0.5)

            if non_nan == 4:
                idx1, idx2, idx3, idx4 = np.searchsorted(self.wave_rest, self.lineWaves[0:4])
                ax.fill_between(self.wave_rest[idx1:idx2], 0.0, self.flux[idx1:idx2], facecolor='tab:green',
                                step='mid', alpha=0.5)
                ax.fill_between(self.wave_rest[idx3:idx4], 0.0, self.flux[idx3:idx4], facecolor='tab:green',
                                step='mid', alpha=0.5)


        # Complete selections
        else:

            # Proceed to measurment

            idcsContLeft = (self.lineWaves[0] <= self.wave_rest) & (self.wave_rest <= self.lineWaves[1])
            idcsContRight = (self.lineWaves[4] <= self.wave_rest) & (self.wave_rest <= self.lineWaves[5])
            idcsLinePeak = (lineWave - limitPeak <= self.wave_rest) & (self.wave_rest <= lineWave + limitPeak)
            idcsLineArea = (self.lineWaves[2] <= self.wave_rest) & (self.wave_rest <= self.lineWaves[3])

            waveCentral, fluxCentral = self.wave_rest[idcsLineArea], self.flux[idcsLineArea]
            wavePeak, fluxPeak = self.wave_rest[idcsLinePeak], self.flux[idcsLinePeak]

            idcsLinePlot = (self.lineWaves[0] - 5 <= self.wave_rest) & (self.wave_rest <= self.lineWaves[5] + 5)
            waveLine, fluxLine = self.wave_rest[idcsLinePlot], self.flux[idcsLinePlot]
            ax.step(waveLine, fluxLine)

            ax.fill_between(waveCentral, 0, fluxCentral, step="pre", alpha=0.4)
            ax.fill_between(self.wave_rest[idcsContLeft], 0, self.flux[idcsContLeft], facecolor='tab:orange', step="pre", alpha=0.2)
            ax.fill_between(self.wave_rest[idcsContRight], 0, self.flux[idcsContRight], facecolor='tab:orange', step="pre", alpha=0.2)

        # Plot format
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.xaxis.set_major_locator(plt.NullLocator())

        ion, wavelength, latexLabel = label_decomposition(lineLabel, scalar_output=True)
        ax.update({'title': latexLabel})
        # ax.set_yscale('log')
        try:
            idxPeakFlux = np.argmax(fluxPeak)
            ax.set_ylim(ymin=np.min(fluxLine) / 5, ymax=fluxPeak[idxPeakFlux] * 1.25)
        except:
            print('Fale peak')

        ax.yaxis.set_ticklabels([])
        ax.axes.yaxis.set_visible(False)

        if logscale:
            ax.set_yscale('log')

        return

    def plot_line_region_i_fit(self, ax, lineLabel, linesDF, limitPeak=5):

        # Plot line region:
        lineWave = linesDF.loc[lineLabel, 'wavelength']

        # Decide type of plot
        non_nan = (~pd.isnull(self.lineWaves)).sum()

        # Incomplete selections
        if non_nan < 6:  # selections

            idcsLinePeak = (lineWave - limitPeak <= self.wave) & (self.wave <= lineWave + limitPeak)
            idcsLineArea = (lineWave - limitPeak * 2 <= self.wave) & (lineWave - limitPeak * 2 <= self.lineWaves[3])
            wavePeak, fluxPeak = self.wave[idcsLinePeak], self.flux[idcsLinePeak]
            waveLine, fluxLine = self.wave[idcsLineArea], self.flux[idcsLineArea]
            idxPeakFlux = np.argmax(fluxPeak)

            ax.step(waveLine, fluxLine)

            if non_nan == 2:
                idx1, idx2 = np.searchsorted(self.wave, self.lineWaves[0:2])
                ax.fill_between(self.wave[idx1:idx2], 0.0, self.flux[idx1:idx2], facecolor='tab:green',
                                step='mid', alpha=0.5)

            if non_nan == 4:
                idx1, idx2, idx3, idx4 = np.searchsorted(self.wave, self.lineWaves[0:4])
                ax.fill_between(self.wave[idx1:idx2], 0.0, self.flux[idx1:idx2], facecolor='tab:green',
                                step='mid', alpha=0.5)
                ax.fill_between(self.wave[idx3:idx4], 0.0, self.flux[idx3:idx4], facecolor='tab:green',
                                step='mid', alpha=0.5)

        # Complete selections
        else:

            # Proceed to measurment
            idcsContLeft = (self.lineWaves[0] <= self.wave) & (self.wave <= self.lineWaves[1])
            idcsContRight = (self.lineWaves[4] <= self.wave) & (self.wave <= self.lineWaves[5])
            idcsLinePeak = (lineWave - limitPeak <= self.wave) & (self.wave <= lineWave + limitPeak)
            idcsLineArea = (self.lineWaves[2] <= self.wave) & (self.wave <= self.lineWaves[3])

            waveCentral, fluxCentral = self.wave[idcsLineArea], self.flux[idcsLineArea]
            wavePeak, fluxPeak = self.wave[idcsLinePeak], self.flux[idcsLinePeak]

            idcsLinePlot = (self.lineWaves[0] - 5 <= self.wave) & (self.wave <= self.lineWaves[5] + 5)
            waveLine, fluxLine = self.wave[idcsLinePlot], self.flux[idcsLinePlot]
            ax.step(waveLine, fluxLine)

            ax.fill_between(waveCentral, 0, fluxCentral, step="pre", alpha=0.4)
            ax.fill_between(self.wave[idcsContLeft], 0, self.flux[idcsContLeft], facecolor='tab:orange', step="pre", alpha=0.2)
            ax.fill_between(self.wave[idcsContRight], 0, self.flux[idcsContRight], facecolor='tab:orange', step="pre", alpha=0.2)
            idxPeakFlux = np.argmax(fluxPeak)

            # Gaussian curve plot
            p1 = linesDF.loc[lineLabel, 'amp':'sigma'].values
            m, n = linesDF.loc[lineLabel, 'm_cont'], linesDF.loc[lineLabel, 'n_cont']
            if (p1[0] is not np.nan) and (p1[0] is not None):
                wave_array = np.linspace(waveLine[0], waveLine[-1], 1000)
                cont_array = m * wave_array + n
                flux_array = gaussian_model(wave_array, p1[0], p1[1], p1[2]) + cont_array
                ax.plot(wave_array, cont_array, ':', color='tab:orange')
                ax.plot(wave_array, flux_array, ':', color='tab:red')

        # Plot format
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.update({'title': f'{linesDF.loc[lineLabel, "latexLabel"]}'})
        ax.set_yscale('log')
        ax.set_ylim(ymin=np.min(fluxLine) / 5, ymax=fluxPeak[idxPeakFlux] * 1.25)
        ax.yaxis.set_ticklabels([])
        ax.axes.yaxis.set_visible(False)

        return

    def table_fluxes(self, lines_df, table_address, pyneb_rc=None, scaleTable=1000):

        # TODO this could be included in sr.print
        tex_address = f'{table_address}'
        txt_address = f'{table_address}.txt'

        # Measure line fluxes
        pdf = PdfPrinter()
        pdf.create_pdfDoc(pdf_type='table')
        pdf.pdf_insert_table(FLUX_TEX_TABLE_HEADERS)

        # Dataframe as container as a txt file
        tableDF = pd.DataFrame(columns=FLUX_TXT_TABLE_HEADERS[1:])

        # Normalization line
        if 'H1_4861A' in lines_df.index:
            flux_Hbeta = lines_df.loc['H1_4861A', 'intg_flux']
        else:
            flux_Hbeta = scaleTable

        obsLines = lines_df.index.values
        for lineLabel in obsLines:

            label_entry = lines_df.loc[lineLabel, 'latexLabel']
            wavelength = lines_df.loc[lineLabel, 'wavelength']
            eqw, eqwErr = lines_df.loc[lineLabel, 'eqw'], lines_df.loc[lineLabel, 'eqw_err']

            flux_intg = lines_df.loc[lineLabel, 'intg_flux'] / flux_Hbeta * scaleTable
            flux_intgErr = lines_df.loc[lineLabel, 'intg_err'] / flux_Hbeta * scaleTable
            flux_gauss = lines_df.loc[lineLabel, 'gauss_flux'] / flux_Hbeta * scaleTable
            flux_gaussErr = lines_df.loc[lineLabel, 'gauss_err'] / flux_Hbeta * scaleTable

            if (lines_df.loc[lineLabel, 'blended_label'] != 'None') and ('_m' not in lineLabel):
                flux, fluxErr = flux_gauss, flux_gaussErr
                label_entry = label_entry + '$_{gauss}$'
            else:
                flux, fluxErr = flux_intg, flux_intgErr

            # Correct the flux
            if pyneb_rc is not None:
                corr = pyneb_rc.getCorrHb(wavelength)
                intensity, intensityErr = flux * corr, fluxErr * corr
                intensity_entry = r'${:0.2f}\,\pm\,{:0.2f}$'.format(intensity, intensityErr)
            else:
                intensity, intensityErr = '-', '-'
                intensity_entry = '-'

            eqw_entry = r'${:0.2f}\,\pm\,{:0.2f}$'.format(eqw, eqwErr)
            flux_entry = r'${:0.2f}\,\pm\,{:0.2f}$'.format(flux, fluxErr)

            # Add row of data
            tex_row_i = [label_entry, eqw_entry, flux_entry, intensity_entry]
            txt_row_i = [label_entry, eqw, eqwErr, flux, fluxErr, intensity, intensityErr]

            lastRow_check = True if lineLabel == obsLines[-1] else False
            pdf.addTableRow(tex_row_i, last_row=lastRow_check)
            tableDF.loc[lineLabel] = txt_row_i[1:]

        if pyneb_rc is not None:

            # Data last rows
            row_Hbetaflux = [r'$H\beta$ $(erg\,cm^{-2} s^{-1} \AA^{-1})$',
                             '',
                             flux_Hbeta,
                             flux_Hbeta * pyneb_rc.getCorr(4861)]

            row_cHbeta = [r'$c(H\beta)$',
                          '',
                          float(pyneb_rc.cHbeta),
                          '']
        else:
            # Data last rows
            row_Hbetaflux = [r'$H\beta$ $(erg\,cm^{-2} s^{-1} \AA^{-1})$',
                             '',
                             flux_Hbeta,
                             '-']

            row_cHbeta = [r'$c(H\beta)$',
                          '',
                          '-',
                          '']

        pdf.addTableRow(row_Hbetaflux, last_row=False)
        pdf.addTableRow(row_cHbeta, last_row=False)
        tableDF.loc[row_Hbetaflux[0]] = row_Hbetaflux[1:] + [''] * 3
        tableDF.loc[row_cHbeta[0]] = row_cHbeta[1:] + [''] * 3

        # Format last rows
        pdf.table.add_hline()
        pdf.table.add_hline()

        # Save the pdf table
        try:
            pdf.generate_pdf(table_address, clean_tex=True)
        except:
            print('-- PDF compilation failure')

        # Save the txt table
        with open(txt_address, 'wb') as output_file:
            string_DF = tableDF.to_string()
            string_DF = string_DF.replace('$', '')
            output_file.write(string_DF.encode('UTF-8'))

        return

    def table_kinematics(self, lines_df, table_address, flux_normalization=1.0):

        # TODO this could be included in sr.print
        tex_address = f'{table_address}'
        txt_address = f'{table_address}.txt'

        # Measure line fluxes
        pdf = PdfPrinter()
        pdf.create_pdfDoc(pdf_type='table')
        pdf.pdf_insert_table(KIN_TEX_TABLE_HEADERS)

        # Dataframe as container as a txt file
        tableDF = pd.DataFrame(columns=KIN_TXT_TABLE_HEADERS[1:])

        obsLines = lines_df.index.values
        for lineLabel in obsLines:

            if not lineLabel.endswith('_b'):
                label_entry = lines_df.loc[lineLabel, 'latexLabel']

                # Establish component:
                blended_check = (lines_df.loc[lineLabel, 'blended_label'] != 'None') and ('_m' not in lineLabel)
                if blended_check:
                    blended_group = lines_df.loc[lineLabel, 'blended_label']
                    comp = 'n1' if lineLabel.count('_') == 1 else lineLabel[lineLabel.rfind('_')+1:]
                else:
                    comp = 'n1'
                comp_label, lineEmisLabel = kinematic_component_labelling(label_entry, comp)

                wavelength = lines_df.loc[lineLabel, 'wavelength']
                v_r, v_r_err =  lines_df.loc[lineLabel, 'v_r':'v_r_err']
                sigma_vel, sigma_vel_err = lines_df.loc[lineLabel, 'sigma_vel':'sigma_vel_err']

                flux_intg = lines_df.loc[lineLabel, 'intg_flux']
                flux_intgErr = lines_df.loc[lineLabel, 'intg_err']
                flux_gauss = lines_df.loc[lineLabel, 'gauss_flux']
                flux_gaussErr = lines_df.loc[lineLabel, 'gauss_err']

                # Format the entries
                vr_entry = r'${:0.1f}\,\pm\,{:0.1f}$'.format(v_r, v_r_err)
                sigma_entry = r'${:0.1f}\,\pm\,{:0.1f}$'.format(sigma_vel, sigma_vel_err)

                if blended_check:
                    flux, fluxErr = flux_gauss, flux_gaussErr
                    label_entry = lineEmisLabel
                else:
                    flux, fluxErr = flux_intg, flux_intgErr

                # Correct the flux
                flux_entry = r'${:0.2f}\,\pm\,{:0.2f}$'.format(flux, fluxErr)

                # Add row of data
                tex_row_i = [label_entry, comp_label, vr_entry, sigma_entry, flux_entry]
                txt_row_i = [lineLabel, comp_label.replace(' ', '_'), v_r, v_r_err, sigma_vel, sigma_vel_err, flux, fluxErr]

                lastRow_check = True if lineLabel == obsLines[-1] else False
                pdf.addTableRow(tex_row_i, last_row=lastRow_check)
                tableDF.loc[lineLabel] = txt_row_i[1:]

        pdf.table.add_hline()

        # Save the pdf table
        try:
            pdf.generate_pdf(tex_address)
        except:
            print('-- PDF compilation failure')

        # Save the txt table
        with open(txt_address, 'wb') as output_file:
            string_DF = tableDF.to_string()
            string_DF = string_DF.replace('$', '')
            output_file.write(string_DF.encode('UTF-8'))

        return

    def on_select(self, Wlow, Whig):

        # Check we are not just clicking on the plot
        if Wlow != Whig:

            # Count number of empty entries to determine next step
            non_nans = (~pd.isnull(self.lineWaves)).sum()

            # Case selecting 1/3 region
            if non_nans == 0:
                self.lineWaves[0] = Wlow
                self.lineWaves[1] = Whig

            # Case selecting 2/3 region
            elif non_nans == 2:
                self.lineWaves[2] = Wlow
                self.lineWaves[3] = Whig
                self.lineWaves = np.sort(self.lineWaves)

            # Case selecting 3/3 region
            elif non_nans == 4:
                self.lineWaves[4] = Wlow
                self.lineWaves[5] = Whig
                self.lineWaves = np.sort(self.lineWaves)

            elif non_nans == 6:
                self.lineWaves = np.sort(self.lineWaves)

                # Caso que se corrija la region de la linea
                if Wlow > self.lineWaves[1] and Whig < self.lineWaves[4]:
                    self.lineWaves[2] = Wlow
                    self.lineWaves[3] = Whig

                # Caso que se corrija el continuum izquierdo
                elif Wlow < self.lineWaves[2] and Whig < self.lineWaves[2]:
                    self.lineWaves[0] = Wlow
                    self.lineWaves[1] = Whig

                # Caso que se corrija el continuum derecho
                elif Wlow > self.lineWaves[3] and Whig > self.lineWaves[3]:
                    self.lineWaves[4] = Wlow
                    self.lineWaves[5] = Whig

                # Case we want to select the complete region
                elif Wlow < self.lineWaves[0] and Whig > self.lineWaves[5]:

                    # # Remove line from dataframe and save it
                    # self.remove_lines_df(self.current_df, self.Current_Label)
                    #
                    # # Save lines log df
                    # self.save_lineslog_dataframe(self.current_df, self.lineslog_df_address)

                    # Clear the selections
                    self.lineWaves = np.array([np.nan] * 6)

                else:
                    print('- WARNING: Unsucessful line selection:')
                    print(f'-- {self.lineLabel}: w_low: {Wlow}, w_high: {Whig}')

            # Check number of measurements after selection
            non_nans = (~pd.isnull(self.lineWaves)).sum()

            # Proceed to re-measurement if possible:
            if non_nans == 6:

                self.linesDF.loc[self.lineLabel, 'w1':'w6'] = self.lineWaves

                idcsLinePeak, idcsContinua = self.define_masks(self.wave_rest, self.flux, self.lineWaves)

                # emisWave, emisFlux = self.wave_rest[idcsLinePeak], self.flux[idcsLinePeak]
                # contWave, contFlux = self.wave_rest[idcsContinua], self.flux[idcsContinua]

                # self.line_properties(emisWave, emisFlux, contWave, contFlux, bootstrap_size=1000)

                # self.line_fit('lmfit', self.lineLabel, idcsLinePeak, idcsContinua, continuum_check=True,
                #               user_conf={})
                # # print(fit_report(self.fit_output))
                # # print(self.fit_params)
                # self.plot_fit_components(self.fit_output)

                self.results_to_database(self.lineLabel, self.linesDF, {}, export_params=['blended_label'])

                self.save_lineslog(self.linesDF, str(self.linesLogAddress))

            # Else delete previous measurent data (except self.lineWaves):
            else:
                for param in LINEMEASURER_PARAMS:
                    self.__setattr__(param, None)

            # Redraw the line measurement
            self.in_ax.clear()
            self.plot_line_region_i(self.in_ax, self.lineLabel, self.linesDF, logscale=False)

            self.in_fig.canvas.draw()

        return

    def on_enter_axes(self, event):

        # TODO we need a better way to intedex than the latex label
        self.in_fig = event.canvas.figure
        self.in_ax = event.inaxes
        idx_line = self.linesDF.latexLabel == self.in_ax.get_title()
        self.lineLabel = self.linesDF.loc[idx_line].index.values[0]
        self.lineWaves = self.linesDF.loc[idx_line, 'w1':'w6'].values[0]

        self.database_to_attr()

        # event.inaxes.patch.set_edgecolor('red')
        event.canvas.draw()

    def on_click(self, event):
        if event.dblclick:
            print(f'{event.button}, {event.x}, {event.y}, {event.xdata}, {event.ydata}')
        # else:
        #     print(f'Wave: {event.xdata}')

    def clear_fit(self):
        super().__init__()


if __name__ == '__main__':

    # Fake data
    pixels_n = 200
    noise_mag = 1.5
    m, n = 0.005, 4.0
    ampTrue, muTrue, sigmaTrue = 20, 5007, 2.3
    areaTrue = np.sqrt(2 * np.pi * sigmaTrue ** 2) * ampTrue
    linelabel, wave_regions = 'O3_5007A', np.array([4960, 4980, 4996, 5015, 5030, 5045])

    red_path = '/Users/Dania/Documents/Proyectos/J0838_cubo/gemini_data/red'

    # Spectrum generation
    wave = np.linspace(4950, 5050, num=200)
    continuum = (m * wave + n)
    noise = np.random.normal(0, noise_mag, pixels_n)
    emLine = gaussian_model(wave, ampTrue, muTrue, sigmaTrue) + continuum
    flux = noise + emLine

    # Call funcions
    lm = LineMesurer(wave, flux, normFlux=10)

    # Perform fit
    lm.fit_from_wavelengths(linelabel, wave_regions)
    lm.print_results(show_fit_report=True, show_plot=True)

    # Comparing flux integration techniques
    idcsLines, idcsContinua = lm.define_masks(wave_regions)
    idcsLine_2, idcsBlueCont, idcsRedCont = lm.define_masks(wave_regions, merge_continua=False)
    lineWave, lineFlux = lm.wave[idcsLines], lm.flux[idcsLines]
    continuaWave, continuaFlux = lm.wave[idcsContinua], lm.flux[idcsContinua]
    lineContinuumFit = lineWave * lm.m_cont + lm.n_cont
    areaSimps = integrate.simps(lineFlux, lineWave) - integrate.simps(lineContinuumFit, lineWave)
    areaTrapz = integrate.trapz(lineFlux, lineWave) - integrate.trapz(lineContinuumFit, lineWave)
    areaIntgPixel = (lm.flux[idcsLines].sum() - lineContinuumFit.sum()) * lm.pixelWidth

    # Print the results
    print(f'True area : {areaTrue}')
    print(f'Simpsons rule: {areaSimps * lm.normFlux}')
    print(f'Trapezoid rule: {areaTrapz * lm.normFlux}')
    print(f'Fit integration: {lm.intg_flux * lm.normFlux} +/- {lm.intg_err * lm.normFlux}')
    print(f'Fit gaussian: {lm.gauss_flux[0] * lm.normFlux} +/- {lm.gauss_err[0] * lm.normFlux}')

    line_snr = np.mean(lineFlux) / np.sqrt(np.mean(np.power(lineFlux, 2)))
    cont_snr = np.mean(continuaFlux) / np.sqrt(np.mean(np.power(continuaFlux, 2)))
    print(f'Line signal to noise gaussian: {line_snr} {lm.snr_line}')
    print(f'Continuum signal to noise gaussian: {cont_snr} {lm.snr_cont}')

    # Lmfit output
    x_in, y_in = lm.fit_output.userkws['x'], lm.fit_output.data
    wave_resample = np.linspace(x_in[0], x_in[-1], 500)
    flux_resample = lm.fit_output.eval_components(x=wave_resample)
    cont_resample = lm.m_cont * wave_resample + lm.n_cont

    fig, ax = plt.subplots()
    ax.step(lm.wave, lm.flux, label='Observed line')
    ax.scatter(x_in, lm.fit_output.data, color='tab:red', alpha=0.2, label='Input points')
    ax.plot(wave_resample, sum(flux_resample.values()), label='Gaussian fit')
    ax.plot(wave_resample, cont_resample, label='Linear fitting Scipy', linestyle='--')

    ax.scatter(lm.wave[idcsLine_2], lm.flux[idcsLine_2], label='Line points')
    ax.scatter(lm.wave[idcsBlueCont], lm.flux[idcsBlueCont], label='Blue continuum')
    ax.scatter(lm.wave[idcsRedCont], lm.flux[idcsRedCont], label='Red continuum')

    # Plot individual components
    for curve_label, curve_flux in flux_resample.items():
        ax.plot(wave_resample, curve_flux, label=f'Component {curve_label}', linestyle='--')

    # ax.scatter(continuaWave, continuaFlux, label='Continuum regions')
    # ax.plot(lineWave, lineContinuumFit, label='Observed line', linestyle=':')
    # ax.plot(resampleWaveLine, gaussianCurve, label='Gaussian fit', linestyle=':')
    ax.legend()
    ax.update({'xlabel': 'Flux', 'ylabel': 'Wavelength', 'title': 'Gaussian fitting'})
    plt.show()
