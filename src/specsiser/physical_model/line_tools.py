import numpy as np
import pandas as pd
import astropy.units as au
from lmfit.models import GaussianModel, LinearModel, PolynomialModel
from lmfit import Parameters
from astropy.modeling.polynomial import Polynomial1D
from matplotlib.widgets import SpanSelector
from specutils.manipulation import noise_region_uncertainty
from specutils.fitting import find_lines_threshold, find_lines_derivative, fit_generic_continuum
from numpy import ndarray
from specutils import Spectrum1D, SpectralRegion
from matplotlib import pyplot as plt, rcParams
from scipy import stats, optimize, integrate


STANDARD_PLOT = {'figure.figsize': (20, 14),
                 'axes.titlesize': 14,
                 'axes.labelsize': 14,
                 'legend.fontsize': 12,
                 'xtick.labelsize': 12,
                 'ytick.labelsize': 12}
STANDARD_AXES = {'xlabel': r'Wavelength $(\AA)$', 'ylabel': r'Flux $(erg\,cm^{-2} s^{-1} \AA^{-1})$'}

LINEAR_ATTRIBUTES = ['slope', 'intercept']
GAUSSIAN_ATTRIBUTES = ['amplitude', 'center', 'sigma', 'fwhm', 'height']
PARAMETER_ATTRIBUTES = ['name', 'value', 'vary', 'min', 'max', 'expr', 'brute_step']

WAVE_UNITS_DEFAULT, FLUX_UNITS_DEFAULT = au.AA, au.erg/au.s/au.cm**2/au.AA


def leave_axes(event):
    print('leave_axes', event.inaxes)
    event.inaxes.patch.set_facecolor('white')
    event.canvas.draw()


def define_lmfit_param(param_object, param_label, value, user_conf={}, param_attr=PARAMETER_ATTRIBUTES):

    if param_label not in user_conf:
        param_object.add(param_label, value=value)

    else:
        param_conf = dict.fromkeys(PARAMETER_ATTRIBUTES)
        param_conf['name'], param_conf['vary'] = param_label, True

        param_conf.update(user_conf[param_label])
        param_object.add(**param_conf)

    return


def gauss_func(ind_params, a, mu, sigma):

    """
    Gaussian function

    This function returns the gaussian curve as the user speciefies and array of x values, the continuum level and
    the parameters of the gaussian curve

    :param ind_params: 2D array (x, z) where x is the array of abscissa values and z is the continuum level
    :param float a: Amplitude of the gaussian
    :param float mu: Center value of the gaussian
    :param float sigma: Sigma of the gaussian
    :return: Gaussian curve y array of values
    :rtype: np.ndarray
    """

    x, z = ind_params
    return a * np.exp(-((x - mu) * (x - mu)) / (2 * (sigma * sigma))) + z


def generate_object_mask(lines_DF, wavelength, line_labels):

    """

    Algorithm to combine line and features mask

    :param lines_DF:
    :param wavelength:
    :param line_labels:
    :return:
    """

    # TODO This will not work for a redshifted lines log
    idcs_lineMasks = lines_DF.index.isin(line_labels)
    idcs_spectrumMasks = ~lines_DF.index.isin(line_labels)

    # Matrix mask for integrating the emission lines
    n_lineMasks = idcs_lineMasks.sum()
    boolean_matrix = np.zeros((n_lineMasks, wavelength.size), dtype=bool)

    # Total mask for valid regions in the spectrum
    n_objMasks = idcs_spectrumMasks.sum()
    int_mask = np.ones(wavelength.size, dtype=bool)
    object_mask = np.ones(wavelength.size, dtype=bool)

    # Loop through the emission lines
    wmin, wmax = lines_DF['w3'].loc[idcs_lineMasks].values, lines_DF['w4'].loc[idcs_lineMasks].values
    idxMin, idxMax = np.searchsorted(wavelength, [wmin, wmax])
    for i in range(n_lineMasks):
        idx_currentMask = (wavelength >= wavelength[idxMin[i]]) & (wavelength <= wavelength[idxMax[i]])
        boolean_matrix[i, :] = idx_currentMask
        int_mask = int_mask & ~idx_currentMask

    # Loop through the object masks
    wmin, wmax = lines_DF['w3'].loc[idcs_spectrumMasks].values, lines_DF['w4'].loc[idcs_spectrumMasks].values
    idxMin, idxMax = np.searchsorted(wavelength, [wmin, wmax])
    for i in range(n_objMasks):
        idx_currentMask = (wavelength >= wavelength[idxMin[i]]) & (wavelength <= wavelength[idxMax[i]])
        int_mask = int_mask & ~idx_currentMask
        object_mask = object_mask & ~idx_currentMask

    return boolean_matrix


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


LINEMEASURER_PARAMS = ['pixelWidth',
                      'peakWave',
                      'peakInt',
                      'lineIntgFlux',
                      'lineIntgErr',
                      'lineGaussFlux',
                      'lineGaussErr',
                      'n_continuum',
                      'm_continuum',
                      'std_continuum',
                      'fit_function',
                      'p1',
                      'p1_Err']


class LineMeasurer:

    """Class to to measure emission line fluxes and fit them as gaussian curves"""
    # TODO add mechanic for absorption features

    wave: ndarray = None
    flux: ndarray = None
    redshift: float = None
    pixelWidth: float = None
    peakWave: float = None
    peakInt: float = None
    lineIntgFlux: float = None
    lineIntgErr: float = None
    lineWaves: ndarray = np.array([np.nan] * 6)
    eqw: float = None
    eqwErr: float = None
    lineGaussFlux: float = None
    lineGaussErr: float = None
    n_continuum: float = None
    m_continuum: float = None
    std_continuum: float = None
    fit_function: object = None
    fit_params: dict = None
    p1: ndarray = None
    p1_Err: ndarray = None
    paramsConversion = {'lineIntgFlux': 'intg_flux',
                        'lineIntgErr': 'intg_err',
                        'lineGaussFlux': 'gauss_flux',
                        'lineGaussErr': 'gauss_err',
                        'm_continuum': 'm_continuum',
                        'n_continuum': 'n_continuum',
                        'std_continuum': 'std_continuum',
                        'eqw': 'eqw',
                        'eqwErr': 'eqw_err'}

    def __init__(self, input_wave=None, input_flux=None, redshift=None):

        if redshift is not None:
            self.wave = input_wave / (1 + redshift)
            self.redshift = redshift
        else:
            self.wave = input_wave

        self.flux = input_flux

        return

    def define_masks(self, masks_array):

        self.lineWaves = masks_array

        area_indcs = np.searchsorted(self.wave, masks_array)

        idcsLines = ((self.wave[area_indcs[2]] <= self.wave[None]) & (
            self.wave[None] <= self.wave[area_indcs[3]])).squeeze()

        idcsContinua = (
            ((self.wave[area_indcs[0]] <= self.wave[None]) & (self.wave[None] <= self.wave[area_indcs[1]])) |
            ((self.wave[area_indcs[4]] <= self.wave[None]) & (self.wave[None] <= self.wave[area_indcs[5]]))).squeeze()

        return idcsLines, idcsContinua

    def line_properties(self, idcs_line, idcs_continua, bootstrap_size=100):

        # Get regions data
        lineWave, lineFlux = self.wave[idcs_line], self.flux[idcs_line]
        continuaWave, continuaFlux = self.wave[idcs_continua], self.flux[idcs_continua]

        # Linear continuum linear fit
        self.m_continuum, self.n_continuum, r_value, p_value, std_err = stats.linregress(continuaWave, continuaFlux)
        continuaFit = continuaWave * self.m_continuum + self.n_continuum
        lineContinuumFit = lineWave * self.m_continuum + self.n_continuum

        # Line Characteristics
        peakIdx = np.argmax(lineFlux)
        self.peakWave, self.peakInt = lineWave[peakIdx], lineFlux[peakIdx]
        self.pixelWidth = np.diff(lineWave).mean()
        self.std_continuum = np.std(continuaFlux - continuaFit)

        # Monte Carlo to measure line flux and uncertainty
        normalNoise = np.random.normal(0.0, self.std_continuum, (bootstrap_size, lineWave.size))
        lineFluxMatrix = lineFlux + normalNoise
        areasArray = (lineFluxMatrix.sum(axis=1) - lineContinuumFit.sum()) * self.pixelWidth
        self.lineIntgFlux, self.lineIntgErr = areasArray.mean(), areasArray.std()

        # Equivalent width computation
        lineContinuumMatrix = lineContinuumFit + normalNoise
        eqwMatrix = areasArray / lineContinuumMatrix.mean(axis=1)
        self.eqw, self.eqwErr = eqwMatrix.mean(), eqwMatrix.std()

        return

    def line_fitting(self, idcs_line, idcs_continua, bootstrap_size=1000, verbose=False):

        # Get regions data
        lineWave, lineFlux = self.wave[idcs_line], self.flux[idcs_line]
        continuaWave, continuaFlux = self.wave[idcs_continua], self.flux[idcs_continua]

        # Linear continuum linear fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(continuaWave, continuaFlux)
        continuaFit = continuaWave * slope + intercept
        lineContinuumFit = lineWave * slope + intercept

        # Line Characteristics
        peakIdx = np.argmax(lineFlux)
        self.peakWave, self.peakInt = lineWave[peakIdx], lineFlux[peakIdx]
        self.pixelWidth = np.diff(lineWave).mean()
        self.std_continuum = np.std(continuaFlux - continuaFit)

        # Monte Carlo to fit gaussian curve
        normalNoise = np.random.normal(0.0, self.std_continuum, (bootstrap_size, lineWave.size))
        lineFluxMatrix = lineFlux + normalNoise
        p0_array = np.array([self.peakInt, self.peakWave, 1])

        # Bounds system
        paramBounds = ((0, self.peakWave-5, 0),
                       (self.peakInt*2, self.peakWave+5, 5))

        # paramBounds = ((0, self.peakInt*2),
        #                (self.peakWave-5, self.peakWave+5),
        #                (0, 5))

        # Run the fitting
        try:
            # TODO Add logic for all possible combinations
            # TODO Add logic for very small lines
            p1_matrix = np.empty((bootstrap_size, 3))
            for i in np.arange(bootstrap_size):
                p1_matrix[i], pcov = optimize.curve_fit(gauss_func,
                                                        (lineWave, lineContinuumFit),
                                                        lineFluxMatrix[i],
                                                        p0=p0_array,
                                                        # ftol=0.5,
                                                        # xtol=0.5,
                                                        # bounds=paramBounds,
                                                        maxfev=1200)

            self.p1, self.p1_Err = p1_matrix.mean(axis=0), p1_matrix.std(axis=0)
            lineArea = np.sqrt(2 * np.pi * p1_matrix[:, 2] * p1_matrix[:, 2]) * p1_matrix[:, 0]
            self.lineGaussFlux, self.lineGaussErr = lineArea.mean(), lineArea.std()

        except:
            self.p1, self.p1_Err = np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])
            self.lineGaussFlux, self.lineGaussErr = np.nan, np.nan

        return

    def composite_fitting(self, lineLabel, idcs_line, idcs_continua, continuum_check=True,
                         wide_comp=None, user_conf={}):

        """

        :param lineLabel:
        :param idcs_line:
        :param idcs_continua:
        :param continuum_check:
        :param wide_components:
        :param params_conf: Input dictionary with specific fit configuration {name , value, vary, min, max, expr,
        brute_step}. The model parameters ['slope', 'intercept', 'amplitude', 'center', 'sigma', 'fwhm', 'height'].
        The configuration of each parameter (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
        :return:
        """

        # Function storing parameter values
        self.fit_params = Parameters()

        # Linear continuum component
        if continuum_check:

            continuumModel = LinearModel(prefix='cont_')

            define_lmfit_param(self.fit_params, 'cont_slope', self.m_continuum, user_conf)
            define_lmfit_param(self.fit_params, 'cont_intercept', self.n_continuum, user_conf)

            self.fit_function = continuumModel

        # Narrow gaussian components
        narrow_components = lineLabel.split('-')
        for comp in narrow_components:

            narrowModel = GaussianModel(prefix=comp+'_')

            define_lmfit_param(self.fit_params, f'{comp}_amplitude', self.peakInt, user_conf)
            define_lmfit_param(self.fit_params, f'{comp}_center', self.peakWave, user_conf)
            define_lmfit_param(self.fit_params, f'{comp}_sigma', 1, user_conf)

            self.fit_function = narrowModel if self.fit_function is None else self.fit_function + narrowModel

        # Wide gaussian component
        if wide_comp is not None:

            wideModel = GaussianModel(prefix=wide_comp+'_')

            define_lmfit_param(self.fit_params, f'{wide_comp}_amplitude', None, user_conf)
            define_lmfit_param(self.fit_params, f'{wide_comp}_center', None, user_conf)
            define_lmfit_param(self.fit_params, f'{wide_comp}_sigma', None, user_conf)

            self.fit_function += wideModel

        # Combine the line and continuum region in the fitting in cases the continuum is included
        idcs_fit = idcs_line + idcs_continua if continuum_check else idcs_line

        # Perform the fitting
        fitOutput = self.fit_function.fit(self.flux[idcs_fit], self.fit_params, x=self.wave[idcs_fit])

        # Store the results
        for comp in narrow_components:

            for idx, param in enumerate(['_amplitude', '_center', '_sigma']):
                ref = comp + param
                self.p1[idx], self.p1_Err[idx] = fitOutput.params[ref].value, fitOutput.params[ref].stderr

            p1_array = np.random.normal(self.p1, self.p1_Err, size=(100000, 3))
            lineArea = p1_array[:, 0] * np.sqrt(2 * np.pi * np.power(p1_array[:, 2], 2))
            self.lineGaussFlux, self.lineGaussErr = lineArea.mean(), lineArea.std()

        return fitOutput

    def continuum_remover(self, noiseRegionLims, intLineThreshold=((4, 4), (1.5, 1.5)), degree=(3, 7)):

        assert self.wave[0] < noiseRegionLims[0] and noiseRegionLims[1] < self.wave[-1]

        # Identify high flux regions
        idcs_noiseRegion = (noiseRegionLims[0] <= self.wave) & (self.wave <= noiseRegionLims[1])
        noise_mean, noise_std = self.flux[idcs_noiseRegion].mean(), self.flux[idcs_noiseRegion].std()

        # Perform several continuum fits to improve the line detection
        input_wave, input_flux = self.wave, self.flux
        for i in range(len(intLineThreshold)):

            #Mask line regions
            emisLimit = intLineThreshold[i][0] * (noise_mean + noise_std)
            absoLimit = (noise_mean + noise_std) / intLineThreshold[i][1]
            idcsLineMask = np.where((input_flux >= absoLimit) & (input_flux <= emisLimit))
            wave_masked, flux_masked = input_wave[idcsLineMask], input_flux[idcsLineMask]

            # Perform continuum fits iteratively
            poly3Mod = PolynomialModel(prefix=f'poly_{degree[i]}', degree=degree[i])
            poly3Params = poly3Mod.guess(flux_masked, x=wave_masked)
            poly3Out = poly3Mod.fit(flux_masked, poly3Params, x=wave_masked)

            input_flux = input_flux - poly3Out.eval(x=self.wave) + noise_mean

        # Plot the fittings
        # fig, ax = plt.subplots(figsize=(12, 8))
        # ax.step(self.wave, self.flux, label='Observed spectrum')
        # ax.step(wave_masked, flux_masked, label='Masked spectrum', linestyle=':')
        # ax.step(wave_masked, poly3Out.best_fit, label='Fit continuum')
        # ax.plot(input_wave, input_flux, label='Normalized continuum')
        # ax.legend()
        # plt.show()

        # Fitting using astropy
        # wave_masked, flux_masked = self.wave[idcsLineMask], self.flux[idcsLineMask]
        # spectrum_masked = Spectrum1D(flux_masked * FLUX_UNITS_DEFAULT, wave_masked * WAVE_UNITS_DEFAULT)
        # g1_fit = fit_generic_continuum(spectrum_masked, model=Polynomial1D(order))
        # continuum_fit = g1_fit(self.wave * u_spec[0])
        # spectrum_noContinuum = Spectrum1D(self.flux * u_spec[1] - continuum_fit, self.wave * u_spec[0])

        return input_flux - noise_mean

    def line_finder(self, input_flux, noiseWaveLim, intLineThreshold=3, verbose=False):

        assert noiseWaveLim[0] > self.wave[0] or noiseWaveLim[1] < self.wave[-1]

        # Establish noise values
        idcs_noiseRegion = (noiseWaveLim[0] <= self.wave) & (self.wave <= noiseWaveLim[1])
        noise_region = SpectralRegion(noiseWaveLim[0] * WAVE_UNITS_DEFAULT, noiseWaveLim[1] * WAVE_UNITS_DEFAULT)
        flux_threshold = intLineThreshold * input_flux[idcs_noiseRegion].std()

        input_spectrum = Spectrum1D(input_flux * FLUX_UNITS_DEFAULT, self.wave * WAVE_UNITS_DEFAULT)
        input_spectrum = noise_region_uncertainty(input_spectrum, noise_region)
        linesTable = find_lines_derivative(input_spectrum, flux_threshold)

        if verbose:
            print(linesTable)

        return linesTable

    def match_lines(self, obsLineTable, theoLineDF, lineType='emission', tol=5, blendedLineList=[],
                    include_undetected=False):

        # Query the lines from the astropy finder tables # TODO Expand technique for absorption lines
        idcsLineType = obsLineTable['line_type'] == lineType
        idcsLinePeak = np.array(obsLineTable[idcsLineType]['line_center_index'])
        waveObs = self.wave[idcsLinePeak]

        # Theoretical wave values
        waveTheory = theoLineDF.wavelength.values

        # Match the lines with the theoretical emission
        tolerance = np.diff(self.wave).mean() * tol
        theoLineDF['observation'] = 'not detected'
        unidentifiedLine = dict.fromkeys(theoLineDF.columns.values, np.nan)

        for i in np.arange(waveObs.size):

            idx_array = np.where(np.isclose(a=waveTheory, b=waveObs[i], atol=tolerance))

            if len(idx_array[0]) == 0:
                unknownLineLabel = 'xy_{:.0f}A'.format(np.round(waveObs[i]))

                if (unknownLineLabel not in theoLineDF.index) and include_undetected: # Scheme to avoid repeated lines
                    newRow = unidentifiedLine.copy()
                    # newRow.update({'wavelength': waveObs[i], 'w3': self.wave[idx_min], 'w4': self.wave[idx_max],
                    #                'observation': 'not identified'})
                    newRow.update({'wavelength': waveObs[i], 'w3': waveObs[i]-5, 'w4': waveObs[i]+5,
                                   'observation': 'not identified'})
                    theoLineDF.loc[unknownLineLabel] = newRow

            else:
                row_index = theoLineDF.index[theoLineDF.wavelength == waveTheory[idx_array[0][0]]]
                theoLineDF.loc[row_index, 'observation'] = 'detected'
                theoLineLabel = row_index[0]

                minSeparation = 4 if theoLineLabel in blendedLineList else 2
                idx_min = compute_lineWidth(idcsLinePeak[i], self.flux, delta_i=-1, min_delta=minSeparation)
                idx_max = compute_lineWidth(idcsLinePeak[i], self.flux, delta_i=1, min_delta=minSeparation)

                theoLineDF.loc[row_index, 'w3'] = self.wave[idx_min]
                theoLineDF.loc[row_index, 'w4'] = self.wave[idx_max]

        # Sort by wavelength
        theoLineDF.sort_values('wavelength', inplace=True)

        return theoLineDF

    def results_to_database(self, lineLabel, linesDF):

        # Loop through the parameters
        for param in self.paramsConversion:
            linesDF.loc[lineLabel, self.paramsConversion[param]] = self.__getattribute__(param)

        # Gaussian fit parameters
        for idx, val in enumerate(('amp', 'mu', 'sigma')):
            linesDF.loc[lineLabel, val] = self.p1[idx]
            linesDF.loc[lineLabel, val + '_err'] = self.p1_Err[idx]

        return

    def load_lineslog(self, file_address):
        return pd.read_csv(file_address, delim_whitespace=True, header=0, index_col=0)

    def save_lineslog(self, linesDF, file_address):

        with open(file_address, 'wb') as output_file:
            string_DF = linesDF.to_string()
            output_file.write(string_DF.encode('UTF-8'))

        return

    def plot_spectrum_components(self, continuumFlux=None, obsLinesTable=None, matchedLinesDF=None, noise_region=None,
                                 plotConf={}, axConf={}):

        # Plot Configuration
        defaultConf = STANDARD_PLOT.copy()
        defaultConf.update(plotConf)
        rcParams.update(defaultConf)
        fig, ax = plt.subplots()

        # Plot the spectrum
        ax.step(self.wave, self.flux, label='Observed spectrum')

        # Plot the continuum if available
        if continuumFlux is not None:
            ax.plot(self.wave, continuumFlux, label='Continuum')

        # Plot astropy detected lines if available
        if obsLinesTable is not None:
            idcs_emission = obsLinesTable['line_type'] == 'emission'
            idcs_linePeaks = np.array(obsLinesTable[idcs_emission]['line_center_index'])
            ax.scatter(self.wave[idcs_linePeaks], self.flux[idcs_linePeaks], label='Detected lines', facecolors='none',
                       edgecolors='tab:purple')

        if matchedLinesDF is not None:
            idcs_foundLines = (matchedLinesDF.observation.isin(('detected', 'not identified'))) &\
                              (matchedLinesDF.wavelength >= self.wave[0]) &\
                              (matchedLinesDF.wavelength <= self.wave[-1])
            lineLatexLabel, lineWave = matchedLinesDF.loc[idcs_foundLines].latexLabel.values, matchedLinesDF.loc[
                idcs_foundLines].wavelength.values
            w3, w4 = matchedLinesDF.loc[idcs_foundLines].w3.values, matchedLinesDF.loc[idcs_foundLines].w4.values
            observation = matchedLinesDF.loc[idcs_foundLines].observation.values

            for i in np.arange(lineLatexLabel.size):
                if observation[i] == 'detected':
                    color_area = 'tab:red' if observation[i] == 'not identified' else 'tab:green'
                    ax.axvspan(w3[i], w4[i], alpha=0.25, color=color_area)
                    ax.text(lineWave[i], 0, lineLatexLabel[i], rotation=270)

            # for i in np.arange(lineLatexLabel.size):
            #     color_area = 'tab:red' if observation[i] == 'not identified' else 'tab:green'
            #     ax.axvspan(w3[i], w4[i], alpha=0.25, color=color_area)
            #     ax.text(lineWave[i], 0, lineLatexLabel[i], rotation=270)

        if noise_region is not None:
            ax.axvspan(noise_region[0], noise_region[1], alpha=0.15, color='tab:cyan', label='Noise region')

        ax.update({**STANDARD_AXES, **axConf})
        ax.legend()
        plt.tight_layout()
        plt.show()

        return

    def plot_fit_components(self, lmfit_output=None, fig_conf={}, ax_conf={}):

        # Plot Configuration
        defaultConf = STANDARD_PLOT.copy()
        defaultConf.update(fig_conf)
        rcParams.update(defaultConf)
        fig, ax = plt.subplots()

        # Compute line regions
        idcsLine = (self.lineWaves[0] - 5 <= self.wave) & (self.wave <= self.lineWaves[5] + 5)
        idcsContLeft = (self.lineWaves[0] <= self.wave) & (self.wave <= self.lineWaves[1])
        idcsContRight = (self.lineWaves[4] <= self.wave) & (self.wave <= self.lineWaves[5])

        # Plot line spectrum
        ax.step(self.wave[idcsLine], self.flux[idcsLine], label='Line spectrum')

        # Plot line regions
        if self.lineWaves is not None:

            leftWave, leftFlux = self.wave[idcsContLeft], self.flux[idcsContLeft]
            rightWave, rightFlux = self.wave[idcsContRight], self.flux[idcsContRight]

            ax.fill_between(leftWave, 0, leftFlux, facecolor='tab:green', step="pre", alpha=0.4)
            ax.fill_between(rightWave, 0, rightFlux, facecolor='tab:green', step="pre", alpha=0.4)

        # Print lmfit results
        if lmfit_output is not None:
            x_fit, y_fit = lmfit_output.userkws['x'], lmfit_output.data
            wave_resample = np.linspace(x_fit[0], x_fit[-1], 500)
            flux_resample = lmfit_output.eval_components(x=wave_resample)

            ax.scatter(x_fit, y_fit, color='tab:red', label='Input data', alpha=0.4)
            ax.plot(wave_resample, sum(flux_resample.values()), label='LMFIT output', )

            # Plot individual components
            for comp_label, comp_flux in flux_resample.items():
                ax.plot(wave_resample, comp_flux, label=f'Component {comp_label}', linestyle='--')

        defaultConf = STANDARD_AXES.copy()
        defaultConf.update(ax_conf)
        ax.update(defaultConf)
        ax.legend()
        plt.show()

        return


class LineMesurerGUI(LineMeasurer):

    def __init__(self, input_wave=None, input_flux=None, linesDF_address=None):

        # Emission model inheritance
        LineMeasurer.__init__(self, input_wave=input_wave, input_flux=input_flux)

        self.linesLogAddress = linesDF_address

        # Object lines DF
        self.linesDF = None
        if linesDF_address:
            self.linesDF = self.load_lineslog(linesDF_address)

        return

    def database_to_attr(self):

        # Conversion parameters
        for name_attr, name_df in self.paramsConversion.items():
            value_df = self.linesDF.loc[self.lineLabel, name_df]
            self.__setattr__(name_attr, value_df)

        # Gaussian fit parameters
        self.p1, self.p1_Err = np.array([np.nan] * 3), np.array([np.nan] * 3)
        for idx, val in enumerate(('amp', 'mu', 'sigma')):
            self.p1[idx] = self.linesDF.loc[self.lineLabel, val]
            self.p1_Err[idx] = self.linesDF.loc[self.lineLabel, val + '_err']

        return

    def plot_detected_lines(self, matchedLinesDF, plotConf={}, ncols=10, nrows=None):

        # Plot data
        indcsLines = matchedLinesDF.observation.isin(['detected']) # TODO remove this bit
        lineLabels = matchedLinesDF.loc[indcsLines].index.values

        if nrows is None:
            nrows = int(np.ceil(lineLabels.size/ncols))

        # Compute plot grid size
        if 'figure.figsize' not in plotConf:
            nrows = int(np.ceil(lineLabels.size / ncols))
            plotConf['figure.figsize'] = (nrows * 4, 14)

        # Plot format
        defaultConf = STANDARD_PLOT.copy()
        defaultConf.update(plotConf)
        rcParams.update(defaultConf)

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
        axesList = ax.flatten()
        dict_spanSelec = {}

        # Generate plot
        for i in np.arange(lineLabels.size):
            self.lineWaves = matchedLinesDF.loc[lineLabels[i], 'w1':'w6'].values
            self.plot_line_region_i(axesList[i], lineLabels[i], matchedLinesDF)
            dict_spanSelec[f'spanner_{i}'] = SpanSelector(axesList[i], self.on_select, 'horizontal', useblit=True,
                              rectprops=dict(alpha=0.5, facecolor='tab:blue'))

        bpe = fig.canvas.mpl_connect('button_press_event', self.on_click)
        aee = fig.canvas.mpl_connect('axes_enter_event', self.on_enter_axes)
        plt.tight_layout()
        plt.show()

        return

    def plot_line_region_i(self, ax, lineLabel, linesDF, limitPeak=5):

        # Plot line region:
        lineWave = linesDF.loc[lineLabel, 'wavelength']

        # Decide type of plot
        non_nan = (~pd.isnull(self.lineWaves)).sum()

        # Incomplete selections
        if non_nan < 6:  #  selections
            print('CHICO')
            idcsLinePeak = (lineWave - limitPeak <= self.wave) & (self.wave <= lineWave + limitPeak)
            idcsLineArea = (lineWave - limitPeak * 2 <= self.wave) & (lineWave - limitPeak * 2 <= self.lineWaves[3])
            wavePeak, fluxPeak = self.wave[idcsLinePeak], self.flux[idcsLinePeak]
            waveLine, fluxLine = self.wave[idcsLineArea], self.flux[idcsLineArea]
            idxPeakFlux = np.argmax(fluxPeak)

            ax.step(waveLine, fluxLine)

            if non_nan == 2:
                idx1, idx2 = np.searchsorted(self.wave, self.lineWaves[0:2])
                ax.fill_between(self.wave[idx1:idx2], 0.0, self.flux[idx1:idx2], facecolor = 'tab:green',
                                step='mid', alpha=0.5)

            if non_nan == 4:
                idx1, idx2, idx3, idx4 = np.searchsorted(self.wave, self.lineWaves[0:4])
                ax.fill_between(self.wave[idx1:idx2], 0.0, self.flux[idx1:idx2], facecolor = 'tab:green',
                                step='mid', alpha=0.5)

        # Complete selections
        else:
            print(lineLabel, self.lineWaves)

            idcsContLeft = (self.lineWaves[0] <= self.wave) & (self.wave <= self.lineWaves[1])
            idcsContRight = (self.lineWaves[4] <= self.wave) & (self.wave <= self.lineWaves[5])
            idcsLinePeak = (lineWave - limitPeak <= self.wave) & (self.wave <= lineWave + limitPeak)
            idcsLineArea = (self.lineWaves[2] <= self.wave) & (self.wave <= self.lineWaves[3])

            waveCentral, fluxCentral = self.wave[idcsLineArea], self.flux[idcsLineArea]
            wavePeak, fluxPeak = self.wave[idcsLinePeak], self.flux[idcsLinePeak]

            idcsLinePlot = (self.lineWaves[0] - 5 <= self.wave) & (self.wave <= self.lineWaves[5] + 5)
            waveLine, fluxLine = self.wave[idcsLinePlot], self.flux[idcsLinePlot]
            ax.step(waveLine, fluxLine)
            print('HOW BIG', waveLine.size, '/', self.wave.size)

            ax.fill_between(waveCentral, 0, fluxCentral, step="pre", alpha=0.4)
            ax.fill_between(self.wave[idcsContLeft], 0, self.flux[idcsContLeft], facecolor='tab:orange', step="pre", alpha=0.2)
            ax.fill_between(self.wave[idcsContRight], 0, self.flux[idcsContRight], facecolor='tab:orange', step="pre", alpha=0.2)
            idxPeakFlux = np.argmax(fluxPeak)

            # Gaussian curve plot
            p1 = linesDF.loc[lineLabel, 'amp':'sigma'].values
            m, n = linesDF.loc[lineLabel, 'm_continuum'], linesDF.loc[lineLabel, 'n_continuum']
            if (p1[0] is not np.nan) and (p1[0] is not None):
                wave_array = np.linspace(waveLine[0], waveLine[-1], 1000)
                cont_array = m * wave_array + n
                flux_array = gauss_func((wave_array, cont_array), p1[0], p1[1], p1[2])
                ax.plot(wave_array, cont_array, ':', color='tab:orange')
                ax.plot(wave_array, flux_array, ':', color='tab:red')

        # Plot format
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.update({'title': f'{linesDF.loc[lineLabel, "latexLabel"]}'})
        ax.set_yscale('log')
        ax.set_ylim(ymin=np.min(fluxLine) / 5, ymax=fluxPeak[idxPeakFlux] * 1.25)

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
                if (Wlow > self.lineWaves[1] and Whig < self.lineWaves[4]):
                    self.lineWaves[2] = Wlow
                    self.lineWaves[3] = Whig

                # Caso que se corrija el continuum izquierdo
                elif (Wlow < self.lineWaves[2] and Whig < self.lineWaves[2]):
                    self.lineWaves[0] = Wlow
                    self.lineWaves[1] = Whig

                # Caso que se corrija el continuum derecho
                elif (Wlow > self.lineWaves[3] and Whig > self.lineWaves[3]):
                    self.lineWaves[4] = Wlow
                    self.lineWaves[5] = Whig

                # Case we want to select the complete region
                elif (Wlow < self.lineWaves[0] and Whig > self.lineWaves[5]):

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

                # Declare regions data
                idcsLinePeak, idcsContinua = self.define_masks(self.lineWaves)

                # Identify line regions
                self.line_properties(idcsLinePeak, idcsContinua, bootstrap_size=250)

                # Perform gaussian fitting
                self.line_fitting(idcsLinePeak, idcsContinua, bootstrap_size=250)

                # Store results in database
                self.results_to_database(self.lineLabel, self.linesDF)

                # Save results to text file
                self.save_lineslog(self.linesDF, self.linesLogAddress)

            # Else delete previous measurent data (except self.lineWaves):
            else:
                for param in LINEMEASURER_PARAMS:
                    self.__setattr__(param, None)

            # Redraw the line measurement
            self.in_ax.clear()

            self.plot_line_region_i(self.in_ax, self.lineLabel, self.linesDF)

            self.in_fig.canvas.draw()

        return

    def on_enter_axes(self, event):
        # print('enter_axes', event.inaxes, type(event.inaxes))
        # print(type(event.inaxes))
        self.in_fig = event.canvas.figure
        self.in_ax = event.inaxes
        idx_line = self.linesDF.latexLabel == self.in_ax.get_title()
        self.lineLabel = self.linesDF.loc[idx_line].index.values[0]
        self.lineWaves = self.linesDF.loc[idx_line, 'w1':'w6'].values[0]

        self.database_to_attr()
        print(self.lineLabel, self.lineWaves)

        #event.inaxes.patch.set_edgecolor('red')
        event.canvas.draw()

    def on_click(self, event):
        if event.dblclick:
            print(f'{event.button}, {event.x}, {event.y}, {event.xdata}, {event.ydata}')
        else:

            print(f'Wave: {event.xdata}')


if __name__ == '__main__':

    # Generate fake data
    wave = np.linspace(4950, 5050)
    m, n, noise = 0.0, 2.0, np.random.normal(0, 0.05, wave.size)
    flux_cont = (m * wave + n) + noise
    ampTrue, muTrue, sigmaTrue = 10, 5007, 2.3
    flux_gauss = gauss_func((wave, flux_cont), ampTrue, muTrue, sigmaTrue)
    wave_regions = np.array([4960, 4980, 4996, 5015, 5030, 5045])
    areaTrue = np.sqrt(2 * np.pi * sigmaTrue ** 2) * ampTrue

    lm = LineMeasurer()

    # Load observed spectrum
    lm.load_spectrum(wave, flux_gauss)

    # Declare regions data
    idcsLines, idcsContinua = lm.define_masks(wave_regions)

    # Identify line regions
    lm.line_properties(idcsLines, idcsContinua, bootstrap_size=1000)

    # Fit gaussian profile
    lm.line_fitting(idcsLines, idcsContinua, bootstrap_size=1000)

    # Comparing flux integration techniques
    lineWave, lineFlux = lm.wave[idcsLines], lm.flux[idcsLines]
    continuaWave, continuaFlux = lm.wave[idcsContinua], lm.flux[idcsContinua]
    lineContinuumFit = lineWave * lm.m_continuum + lm.n_continuum
    areaSimps = integrate.simps(lineFlux, lineWave) - integrate.simps(lineContinuumFit, lineWave)
    areaTrapz = integrate.trapz(lineFlux, lineWave) - integrate.trapz(lineContinuumFit, lineWave)
    areaIntgPixel = (lm.flux[idcsLines].sum() - lineContinuumFit.sum()) * lm.pixelWidth
    resampleWaveLine = np.linspace(lineWave[0] - 10, lineWave[-1] + 10, 100)
    resampleFluxCont = resampleWaveLine * lm.m_continuum + lm.n_continuum
    gaussianCurve = gauss_func((resampleWaveLine, resampleFluxCont), *lm.p1)
    areaGauss = (gaussianCurve.sum() - resampleFluxCont.sum()) * np.diff(resampleWaveLine).mean()

    # Print the results
    print(f'True area : {areaTrue}')
    print(f'Simpsons rule: {areaSimps}')
    print(f'Trapezoid rule: {areaTrapz}')
    print(f'Pixel intgr: {areaIntgPixel}')
    print(f'Pixel intgr MC: {lm.lineIntgFlux} +/- {lm.lineIntgErr}')
    print(f'Gauss intgr: {areaGauss}')
    print(f'Gauss intgr MC: {lm.lineGaussFlux} +/- {lm.lineGaussErr}')
    print(f'True amp = {ampTrue}, mu = {muTrue}, sigma = {sigmaTrue}')
    print(f'Fit amp = {lm.p1[0]:2f} +/- {lm.p1_Err[0]:2f}, mu = {lm.p1[1]:2f} +/- {lm.p1_Err[1]:2f}, '
          f'sigma = {lm.p1[2]:2f} +/- {lm.p1_Err[2]:2f}')
    print(f'Eqw MC {lm.eqw} +/- {lm.eqwErr}')

    fig, ax = plt.subplots()
    ax.plot(wave, flux_gauss, label='Observed line')
    ax.scatter(continuaWave, continuaFlux, label='Continuum regions')
    ax.plot(lineWave, lineContinuumFit, label='Observed line', linestyle=':')
    ax.plot(resampleWaveLine, gaussianCurve, label='Gaussian fit', linestyle=':')
    ax.legend()
    ax.update({'xlabel': 'Flux', 'ylabel': 'Wavelength', 'title': 'Gaussian fitting'})
    plt.show()
