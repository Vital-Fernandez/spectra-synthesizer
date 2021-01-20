import numpy as np
import pandas as pd
import astropy.units as au
from numpy import ndarray
from pandas import DataFrame
from pathlib import Path
from lmfit.models import GaussianModel, LinearModel, PolynomialModel, height_expr
from lmfit import Parameters, fit_report, Model
from astropy.modeling.polynomial import Polynomial1D
from matplotlib.widgets import SpanSelector
from specutils.manipulation import noise_region_uncertainty
from specutils.fitting import find_lines_threshold, find_lines_derivative, fit_generic_continuum
from specutils import Spectrum1D, SpectralRegion
from matplotlib import pyplot as plt, rcParams, spines
from scipy import stats, optimize, integrate
from data_reading import label_decomposition
from pathlib import Path
from src.specsiser.data_printing import PdfPrinter, label_decomposition

STANDARD_PLOT = {'figure.figsize': (14, 7),
                 'axes.titlesize': 14,
                 'axes.labelsize': 14,
                 'legend.fontsize': 12,
                 'xtick.labelsize': 12,
                 'ytick.labelsize': 12}
STANDARD_AXES = {'xlabel': r'Wavelength $(\AA)$', 'ylabel': r'Flux $(erg\,cm^{-2} s^{-1} \AA^{-1})$'}

LINEAR_ATTRIBUTES = ['slope', 'intercept']
GAUSSIAN_ATTRIBUTES = ['amplitude', 'center', 'sigma', 'fwhm', 'height']
PARAMETER_ATTRIBUTES = ['name', 'value', 'vary', 'min', 'max', 'expr', 'brute_step']
PARAMETER_DEFAULT = dict(name=None, value=None, vary=True, min=-np.inf, max=np.inf, expr=None)
HEIGHT_FORMULA = f'0.3989423 * component_amplitude / component_sigma'

DATABASE_PATH = Path(__file__, '../../').resolve()/'literature_data'/'lines_data.xlsx'

WAVE_UNITS_DEFAULT, FLUX_UNITS_DEFAULT = au.AA, au.erg / au.s / au.cm ** 2 / au.AA

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

PARAMS_CONVERSION = {'lineIntgFlux': 'intg_flux',
                     'lineIntgErr': 'intg_err',
                     'cont': 'cont',
                     'm_continuum': 'm_continuum',
                     'n_continuum': 'n_continuum',
                     'std_continuum': 'std_continuum',
                     'lineGaussFlux': 'gauss_flux',
                     'lineGaussErr': 'gauss_err',
                     'eqw': 'eqw',
                     'eqwErr': 'eqw_err'}


VAL_LIST = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
SYB_LIST = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]

FLUX_TEX_TABLE_HEADERS = [r'$Transition$', '$EW(\AA)$', '$F(\lambda)$', '$I(\lambda)$']
FLUX_TXT_TABLE_HEADERS = [r'$Transition$', 'EW', 'EW_error', 'F(lambda)', 'F(lambda)_error', 'I(lambda)', 'I(lambda)_error']

SQRT2PI = np.sqrt(2*np.pi)


def gauss_area(sigma_true, amp_true):
    # return np.sqrt(2 * np.pi * sigma_true ** 2) * amp_true
    return amp_true * SQRT2PI * sigma_true


def linear_model(x, slope, intercept):
    """a line"""
    return slope * x + intercept


def gaussian_model(x, amplitude, center, sigma):
    """1-d gaussian curve : gaussian(x, amp, cen, wid)"""
    return amplitude * np.exp(-0.5 * (((x-center)/sigma) * ((x-center)/sigma)))


def iraf_snr(input_y):
    avg = np.mean(input_y)
    rms = np.sqrt(np.mean(np.power(input_y - avg, 2)))
    return avg / rms


def leave_axes(event):
    event.inaxes.patch.set_facecolor('white')
    event.canvas.draw()


def define_lmfit_param(param_object, param_label, value=None, user_conf={}):
    param_conf = PARAMETER_DEFAULT.copy()
    param_conf['name'], param_conf['value'] = param_label, value

    if '_amplitude' in param_object:  # TODO this could be and issue for absorptions
        param_conf['min'] = 0.0

    if param_label in user_conf:
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


def int_to_roman(num):
    i, roman_num = 0, ''
    while num > 0:
        for _ in range(num // VAL_LIST[i]):
            roman_num += SYB_LIST[i]
            num -= VAL_LIST[i]
        i += 1
    return roman_num


def lineslogFile_to_DF(lineslog_address):
    '''
    This function includes serveral techniques to import an excel or text file lines log as a dataframe
    :param lineslog_address:
    :return:
    '''

    # Text files
    try:
        lineslogDF = pd.read_csv(lineslog_address, delim_whitespace=True, header=0, index_col=0)
    except:
        try:
            lineslogDF = pd.read_excel(lineslog_address, sheet_name=0, header=0, index_col=0)
        except:
            print(f'- ERROR: Could not open lines log at: {lineslog_address}')

    return lineslogDF


def redshift_calculation(obs_array, emis_array, unit='wavelength', verbose=False):

    if unit == 'wavelength':
        z_array = obs_array/emis_array - 1
    elif unit == 'frequency':
        z_array = emis_array/obs_array - 1
    else:
        print(f'- ERROR: Units {unit} for redshift calculation no understood')

    if verbose:
        print(f'Redshift per line: {z_array}')
        print(f'Mean redshift: {z_array.mean()} {z_array.std()}')

    return z_array.mean(), z_array.std()


def wavelength_to_vel(delta_lambda, lambda_wave, c=299792.458):
    return c * (delta_lambda/lambda_wave)


def save_lineslog(linesDF, file_address):

    with open(file_address, 'wb') as output_file:
        string_DF = linesDF.to_string()
        output_file.write(string_DF.encode('UTF-8'))

    return


class EmissionFitting:

    """Class to to measure emission line fluxes and fit them as gaussian curves"""
    # TODO Add logic for very small lines

    _wave, _flux, _errFlux = None, None, None
    _peakWave, _peakInt = None, None
    _pixelWidth = None
    _lineIntgFlux, _lineIntgErr = None, None
    _lineLabel, _lineWaves = '', np.array([np.nan] * 6)
    _eqw, _eqwErr = None, None
    _lineGaussFlux, _lineGaussErr = None, None
    _cont, _std_continuum = None, None
    _m_continuum, _n_continuum = None, None
    _p1, _p1_Err = None, None
    _v_r, _v_r_Err = None, None
    _sigma_vel, _sigma_vel_Err = None, None
    _fit_params, _fit_output = {}, None
    _blended_check, _mixtureComponents = False, None
    _snr_line, _snr_cont = None, None

    _AMP_PAR = dict(value=None, min=0, max=np.inf, vary=True, expr=None)
    _MU_PAR = dict(value=None, min=-np.inf, max=np.inf, vary=True, expr=None)
    _SIG_PAR = dict(value=None, min=0, max=np.inf, vary=True, expr=None)
    _AREA_PAR = dict(value=None, min=0, max=np.inf, vary=True, expr=None)
    _SLOPE_PAR = dict(value=None, min=-np.inf, max=np.inf, vary=True, expr=None)
    _INTER_PAR = dict(value=None, min=-np.inf, max=np.inf, vary=True, expr=None)

    def __init__(self):

        self.wave, self.flux, self.errFlux = self._wave, self._flux, self._errFlux
        self.peakWave, self.peakFlux = self._peakWave, self._peakInt
        self.pixelWidth = self._pixelWidth
        self.lineIntgFlux, self.lineIntgErr = self._lineIntgFlux, self._lineIntgErr
        self.lineLabel, self.lineWaves = self._lineLabel, self._lineWaves
        self.eqw, self.eqwErr = self._eqw, self._eqwErr
        self.lineGaussFlux, self.lineGaussErr = self._lineGaussFlux, self._lineGaussErr
        self.cont, self.std_continuum = self._cont, self._std_continuum
        self.m_continuum, self.n_continuum = self._m_continuum, self._n_continuum
        self.p1, self.p1_Err = self._p1, self._p1_Err
        self.v_r, self.v_r_Err = self._v_r, self._v_r_Err
        self.sigma_vel, self.sigma_vel_Err = self._sigma_vel, self._sigma_vel_Err
        self.fit_params, self.fit_output = self._fit_params, self._fit_output
        self.blended_check, self.mixtureComponents = self._blended_check, self._mixtureComponents
        self.snr_line, self.snr_cont = self._snr_line, self._snr_cont

        return

    def define_masks(self, masks_array, merge_continua=True):

        # TODO For a matrix these indeces have to be used as. Is this an issue?
        # lineWave, lineFlux = wave[idcsLines[:, i]], fluxNorm[idcsLines[:, i]]
        # continuaWave, continuaFlux = wave[idcsContinua[:, i]], fluxNorm[idcsContinua[:, i]]

        # Make sure it is a matrix
        masks_array = np.array(masks_array, ndmin=2)

        # Find indeces for six points in spectrum
        idcsW = np.searchsorted(self.wave, masks_array)

        # Emission region
        idcsLineRegion = ((self.wave[idcsW[:, 2]] <= self.wave[:, None]) & (self.wave[:, None] <= self.wave[idcsW[:, 3]])).squeeze()

        # Return left and right continua merged in one array
        if merge_continua:
            idcsContRegion = (((self.wave[idcsW[:, 0]] <= self.wave[:, None]) &
                              (self.wave[:, None] <= self.wave[idcsW[:, 1]])) |
                              ((self.wave[idcsW[:, 4]] <= self.wave[:, None]) & (
                               self.wave[:, None] <= self.wave[idcsW[:, 5]]))).squeeze()
            return idcsLineRegion, idcsContRegion

        # Return left and right continua in separated arrays
        else:
            idcsContLeft = ((self.wave[idcsW[:, 0]] <= self.wave[:, None]) & (self.wave[:, None] <= self.wave[idcsW[:, 1]])).squeeze()
            idcsContRight = ((self.wave[idcsW[:, 4]] <= self.wave[:, None]) & (self.wave[:, None] <= self.wave[idcsW[:, 5]])).squeeze()

            return idcsLineRegion, idcsContLeft, idcsContRight

    def continuum_remover(self, noiseRegionLims, intLineThreshold=((4, 4), (1.5, 1.5)), degree=(3, 7)):

        assert self.wave[0] < noiseRegionLims[0] and noiseRegionLims[1] < self.wave[-1]

        # Identify high flux regions
        idcs_noiseRegion = (noiseRegionLims[0] <= self.wave) & (self.wave <= noiseRegionLims[1])
        noise_mean, noise_std = self.flux[idcs_noiseRegion].mean(), self.flux[idcs_noiseRegion].std()

        # Perform several continuum fits to improve the line detection
        input_wave, input_flux = self.wave, self.flux
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

    def line_properties(self, idcs_line, idcs_continua, bootstrap_size=500):

        # Get regions data
        emisWave, emisFlux = self.wave[idcs_line], self.flux[idcs_line]
        contWave, contFlux = self.wave[idcs_continua], self.flux[idcs_continua]

        # Linear continuum linear fit
        self.m_continuum, self.n_continuum, r_value, p_value, std_err = stats.linregress(contWave, contFlux)
        continuaFit = contWave * self.m_continuum + self.n_continuum
        lineLinearCont = emisWave * self.m_continuum + self.n_continuum

        # Line Characteristics
        peakIdx = np.argmax(emisFlux)
        self.peakWave, self.peakFlux = emisWave[peakIdx], emisFlux[peakIdx]
        self.pixelWidth = np.diff(emisWave).mean()
        self.std_continuum = np.std(contFlux - continuaFit)
        self.cont = self.peakWave * self.m_continuum + self.n_continuum
        self.snr_line, self.snr_cont = iraf_snr(emisFlux), iraf_snr(contFlux)

        # Monte Carlo to measure line flux and uncertainty
        normalNoise = np.random.normal(0.0, self.std_continuum, (bootstrap_size, emisWave.size))
        lineFluxMatrix = emisFlux + normalNoise
        areasArray = (lineFluxMatrix.sum(axis=1) - lineLinearCont.sum()) * self.pixelWidth
        self.lineIntgFlux, self.lineIntgErr = areasArray.mean(), areasArray.std()

        # Equivalent width computation
        lineContinuumMatrix = lineLinearCont + normalNoise
        eqwMatrix = areasArray / lineContinuumMatrix.mean(axis=1)
        self.eqw, self.eqwErr = eqwMatrix.mean(), eqwMatrix.std()

        return

    def line_fit(self, algorithm, lineLabel, idcs_line, idcs_continua, iter_n=500, user_conf={}):

        # Check if line is in a blended group
        lineRef = lineLabel
        if '_b' in lineLabel:
            if lineLabel in user_conf:
                self.blended_check = True
                lineRef = user_conf[lineLabel]

        # Define x and y values according to line regions
        idcsFit = idcs_line + idcs_continua
        x_array, y_array = self.wave[idcsFit], self.flux[idcsFit]

        # Define fiting weights according to the error
        if self.errFlux is None:
            weights_array = np.full(idcsFit.sum(), fill_value=1.0/self.std_continuum)
        else: # TODO better to give an option introduce the error you want
            weights_array = 1.0/np.sqrt(np.abs(self.errFlux[idcsFit]))

        # Run fit
        if algorithm == 'mc':
            self.gauss_mcfit(idcs_line, idcs_continua, iter_n)

        if algorithm == 'lmfit':
            self.gauss_lmfit(lineRef, x_array, y_array, weights_array, user_conf)

        return

    def gauss_lmfit(self, line_label, x, y, weights, user_conf={}):

        """
        :param line_label:
        :param idcs_line:
        :param idcs_continua:
        :param continuum_check:
        :param narrow_comp:
        :param user_conf: Input dictionary with specific fit configuration {name , value, vary, min, max, expr,
        brute_step}. The model parameters ['slope', 'intercept', 'amplitude', 'center', 'sigma', 'fwhm', 'height'].
        The configuration of each parameter (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
        :return:
        """

        # Confirm the number of gaussian components
        self.mixtureComponents = np.array(line_label.split('-'), ndmin=1)
        n_comps = self.mixtureComponents.size
        ion_arr, theoWave_arr, latexLabel_arr = label_decomposition(self.mixtureComponents, combined_dict=user_conf)

        # Define initial wavelength for group
        ref_wave = np.array([self.peakWave], ndmin=1)

        # For blended lines replace the first line reference waves by the peak one
        if self.blended_check:
            ref_wave = theoWave_arr
            ref_wave[0] = self.peakWave

        # Define fitting params for each component
        fit_model = Model(linear_model)
        for idx_comp, comp in enumerate(self.mixtureComponents):

            # Linear
            if idx_comp == 0:
                fit_model.prefix = f'{comp}_cont_' # For a blended line the continuum conf is defined by first line
                self.define_param(fit_model, f'{comp}_cont_slope', self.m_continuum, self._SLOPE_PAR, user_conf)
                self.define_param(fit_model, f'{comp}_cont_intercept', self.n_continuum, self._INTER_PAR, user_conf)

            # Gaussian
            fit_model += Model(gaussian_model, prefix=f'{comp}_')
            self.define_param(fit_model, f'{comp}_amplitude', self.peakFlux-self.cont, self._AMP_PAR, user_conf)
            self.define_param(fit_model, f'{comp}_center', ref_wave[idx_comp], self._MU_PAR, user_conf)
            self.define_param(fit_model, f'{comp}_sigma', 1.0, self._SIG_PAR, user_conf)
            self.define_param(fit_model, f'{comp}_area', comp, self._AREA_PAR, user_conf)

        # Fit the line
        self.fit_params = fit_model.make_params()
        self.fit_output = fit_model.fit(y, self.fit_params, x=x, weights=weights)

        # Generate containers for the results
        eqw_g, eqwErr_g = np.empty(n_comps), np.empty(n_comps)
        self.p1, self.p1_Err = np.empty((3, n_comps)), np.empty((3, n_comps))
        self.v_r, self.v_r_Err = np.empty(n_comps), np.empty(n_comps)
        self.sigma_vel, self.sigma_vel_Err = np.empty(n_comps), np.empty(n_comps)
        self.lineGaussFlux, self.lineGaussErr = np.empty(n_comps), np.empty(n_comps)

        # Store lmfit measurements
        for i, line in enumerate(self.mixtureComponents):

            # Gaussian parameters
            for j, param in enumerate(['_amplitude', '_center', '_sigma']):
                param_fit = self.fit_output.params[line + param]
                self.p1[j, i], self.p1_Err[j, i] = param_fit.value, param_fit.stderr

            # Gaussian area
            lineArea = self.fit_output.params[f'{line}_area']
            self.lineGaussFlux[i], self.lineGaussErr[i] = lineArea.value, lineArea.stderr

            # Equivalent with gaussian flux for blended components
            if self.blended_check:
                eqw_g[i], eqwErr_g[i] = self.lineGaussFlux[i]/self.cont, self.lineGaussErr[i]/self.cont

            # Kinematics
            self.v_r[i] = wavelength_to_vel(self.p1[1, i] - ref_wave[i], ref_wave[i])
            self.v_r_Err[i] = np.abs(wavelength_to_vel(self.p1_Err[1, i], ref_wave[i]))
            self.sigma_vel[i] = wavelength_to_vel(self.p1[2, i], ref_wave[i])
            self.sigma_vel_Err[i] = wavelength_to_vel(self.p1_Err[2, i], ref_wave[i])

        if self.blended_check:
            self.eqw, self.eqwErr = eqw_g, eqwErr_g
        else:
            self.eqw, self.eqwErr = np.array(self.eqw, ndmin=1), np.array(self.eqwErr, ndmin=1)

        return

    def define_param(self, model_obj, param_ref, param_value, default_conf={}, user_conf={}):

        # Overwrite default by the one provided by the user
        if param_ref in user_conf:
            param_conf = {**default_conf, **user_conf[param_ref]}
        else:
            param_conf = default_conf.copy()

        # Set initial value estimation from spectrum if not provided by the user
        if param_ref not in user_conf:
            param_conf['value'] = param_value

        else:

            # Special case inequalities: H1_6563A_w1_sigma = '>1.5*H1_6563A_sigma'
            if param_conf['expr'] is not None:
                if ('<' in param_conf['expr']) or ('>' in param_conf['expr']):

                    # Create additional parameter
                    ineq_name = f'{param_ref}_ineq'
                    ineq_operation = '*' # TODO add remaining operations

                    # Split number and ref
                    ineq_expr = param_conf['expr'].replace('<','').replace('>','')
                    ineq_items = ineq_expr.split(ineq_operation)
                    ineq_linkedParam = ineq_items[0] if not ineq_items[0].isdigit() else ineq_items[1]
                    ineq_lim = float(ineq_items[0]) if ineq_items[0].isdigit() else float(ineq_items[1])

                    # Stablish the inequality configuration:
                    ineq_conf = {}
                    if '>' in param_conf['expr']:
                        ineq_conf['value'] = ineq_lim * 1.2
                        ineq_conf['min'] = ineq_lim
                    else:
                        ineq_conf['value'] = ineq_lim * 0.8
                        ineq_conf['max'] = ineq_lim

                    # Define new param
                    model_obj.set_param_hint(name=ineq_name, **ineq_conf)

                    # Prepare definition of param:
                    new_expresion = f'{ineq_name}{ineq_operation}{ineq_linkedParam}'
                    param_conf = dict(expr=new_expresion)

            # Special case inquealities
            elif 'kinematics' in param_conf:
                ref_line = param_conf['kinematics']
                gauss_param = param_ref[param_ref.rfind('_') + 1:]
                assert ref_line in self.linesDF, f'- ERROR kinematics line {ref_line} has not been measured for {param_ref}'
                param_conf['value'] = self.linesDF.loc[ref_line, gauss_param]
                param_conf['vary'] = False

            # Case default value is not provided
            else:
                if param_conf['value'] is None:
                    param_conf['value'] = param_value

        # Special case for the area parameter
        if '_area' in param_ref:
            if (param_conf['expr'] is None) and (param_conf['value'] == param_value):
                param_conf['value'] = None
                param_conf['expr'] = f'{param_value}_amplitude*2.5066282746*{param_value}_sigma'

        # Assign the parameter configuration to the model
        model_obj.set_param_hint(param_ref, **param_conf)

        return

    def gauss_mcfit(self, idcs_line, bootstrap_size=1000):

        # Get regions data
        lineWave, lineFlux = self.wave[idcs_line], self.flux[idcs_line]

        # Linear continuum linear fit
        lineContFit = lineWave * self.m_continuum + self.n_continuum

        # Initial gaussian fit values
        p0_array = np.array([self.peakFlux, self.peakWave, 1])

        # Monte Carlo to fit gaussian curve
        normalNoise = np.random.normal(0.0, self.std_continuum, (bootstrap_size, lineWave.size))
        lineFluxMatrix = lineFlux + normalNoise

        # Run the fitting
        try:
            p1_matrix = np.empty((bootstrap_size, 3))

            for i in np.arange(bootstrap_size):
                p1_matrix[i], pcov = optimize.curve_fit(gauss_func,
                                                        (lineWave, lineContFit),
                                                        lineFluxMatrix[i],
                                                        p0=p0_array,
                                                        # ftol=0.5,
                                                        # xtol=0.5,
                                                        # bounds=paramBounds,
                                                        maxfev=1200)

            p1, p1_Err = p1_matrix.mean(axis=0), p1_matrix.std(axis=0)

            lineArea = np.sqrt(2 * np.pi * p1_matrix[:, 2] * p1_matrix[:, 2]) * p1_matrix[:, 0]
            y_gauss, y_gaussErr = lineArea.mean(), lineArea.std()

        except:
            p1, p1_Err = np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])
            y_gauss, y_gaussErr = np.nan, np.nan

        self.p1, self.p1_Err, self.lineGaussFlux, self.lineGaussErr = p1, p1_Err, y_gauss, y_gaussErr

        return

    def reset_measuerement(self):

        # TODO this reset should recall the __init__
        self.peakWave, self.peakFlux = self._peakWave, self._peakInt
        self.pixelWidth = self.pixelWidth
        self.lineIntgFlux, self.lineIntgErr = self._lineIntgFlux, self._lineIntgErr
        self.lineLabel, self._lineWaves = self._lineLabel, self._lineWaves
        self.eqw, self.eqwErr = self._eqw, self._eqwErr
        self.lineGaussFlux, self.lineGaussErr = self._lineGaussFlux, self._lineGaussErr
        self.cont, self.std_continuum = self._cont, self._std_continuum
        self.m_continuum, self.n_continuum = self._m_continuum, self._n_continuum
        self.p1, self.p1_Err = self._p1, self._p1_Err
        self.v_r, self.v_r_Err = self._v_r, self._v_r_Err
        self.sigma_vel, self.sigma_vel_Err = self._sigma_vel, self._sigma_vel_Err
        self.fit_params, self.fit_output = self._fit_params, self._fit_output
        self.blended_check, self.mixtureComponents = self._blended_check, self._mixtureComponents

        return


class LineMesurer(EmissionFitting):

    _linesDF = None
    _redshift, _normFlux = 0, 1
    _wave_units = 'lambda'

    def __init__(self, input_wave=None, input_flux=None, input_err=None, linesDF_address=None, redshift=None,
                 normFlux=None, crop_waves=None, wave_units='lambda'):

        # Emission model inheritance
        EmissionFitting.__init__(self)

        # Start cropping the input spectrum if necessary
        if crop_waves is not None:
            idcs_cropping = (input_wave >= crop_waves[0]) & (input_wave <= crop_waves[1])
            input_wave = input_wave[idcs_cropping]
            input_flux = input_flux[idcs_cropping]
            if input_err is not None:
                input_err = input_err[idcs_cropping]

        # Import object spectrum
        self.redshift = redshift if redshift is not None else self._redshift
        self.normFlux = normFlux if normFlux is not None else self._normFlux
        self.wave_units = wave_units # TODO Add flexibility for wave changes

        # Apply the redshift correction
        if (input_wave is not None) and (input_flux is not None):
            self.wave = input_wave / (1 + self.redshift)
            self.flux = input_flux * (1 + self.redshift) / self.normFlux
            if input_err is not None:
                self.errFlux = input_err * (1 + self.redshift) / self.normFlux

        # Generate empty dataframe to store measurement use cwd as default storing folder
        if linesDF_address is None:
            self.linesLogAddress = Path.cwd()
            _linesDb = lineslogFile_to_DF(DATABASE_PATH)
            self.linesDF = DataFrame(columns=_linesDb.columns)

        # Otherwise use the one from the user
        else:
            self.linesLogAddress = linesDF_address
            if Path(self.linesLogAddress).is_file():
                self.linesDF = lineslogFile_to_DF(linesDF_address)
            else:
                print(f'-- WARNING: linesLog not found at {self.linesLogAddress}')

        return

    def print_results(self, label=None, show_fit_report=True, show_plot=False):

        # Case no line as input: Show the current measurement
        if label is None:
            if self.lineLabel is not None:
                output_ref = (f'Input line: {self.lineLabel}\n'
                              f'- Line regions: {self.lineWaves}\n'
                              f'- Spectrum: normalization flux: {self.normFlux}; redshift {self.redshift}\n'
                              f'- Peak: wavelength {self.peakWave:.2f}; peak intensity {self.peakFlux:.2f}\n'
                              f'- Continuum: slope {self.m_continuum:.2f}; intercept {self.n_continuum:.2f}\n')

                if self.mixtureComponents.size == 1:
                    output_ref += f'- Intg Eqw: {self.eqw[0]:.2f} +/- {self.eqwErr[0]:.2f}\n'

                output_ref += f'- Intg flux: {self.lineIntgFlux:.3f} +/- {self.lineIntgErr:.3f}\n'

                for i, lineRef in enumerate(self.mixtureComponents):
                    output_ref += (f'- {lineRef} gaussian fitting:\n'
                                   f'-- Gauss flux: {self.lineGaussFlux[i]:.3f} +/- {self.lineGaussErr[i]:.3f}\n'
                                   f'-- Height: {self.p1[0][i]:.3f} +/- {self.p1[0][i]:.3f}\n'
                                   f'-- Center: {self.p1[1][i]:.3f} +/- {self.p1[1][i]:.3f}\n'
                                   f'-- Sigma: {self.p1[2][i]:.3f} +/- {self.p1[2][i]:.3f}\n\n')

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
            self.plot_fit_components(self.fit_output) # TODO this function should read from lines log

        return

    def fit_from_wavelengths(self, label, line_wavelengths, fit_conf={}, algorithm='lmfit'):

        # Clear previous measurement
        self.reset_measuerement()

        # Label the current measurement
        self.lineLabel = label
        self.lineWaves = line_wavelengths

        # Establish spectrum line and continua regions
        idcsLineRegion, idcsContRegion = self.define_masks(self.lineWaves)

        # Integrated line properties
        self.line_properties(idcsLineRegion, idcsContRegion, bootstrap_size=1000)

        # Gaussian line fit properties
        self.line_fit(algorithm, self.lineLabel, idcsLineRegion, idcsContRegion, user_conf=fit_conf)

        # Safe the results to the lineslog
        self.results_to_database(self.lineLabel, self.linesDF, fit_conf)

        return

    def match_lines(self, obsLineTable, theoLineDF, lineType='emission', tol=5, blendedLineList=[], detect_check=False,
                    find_line_borders=True):

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

                if find_line_borders:
                    minSeparation = 4 if theoLineLabel in blendedLineList else 2
                    idx_min = compute_lineWidth(idcsLinePeak[i], self.flux, delta_i=-1, min_delta=minSeparation)
                    idx_max = compute_lineWidth(idcsLinePeak[i], self.flux, delta_i=1, min_delta=minSeparation)
                    theoLineDF.loc[row_index, 'w3'] = self.wave[idx_min]
                    theoLineDF.loc[row_index, 'w4'] = self.wave[idx_max]

        # Sort by wavelength
        theoLineDF.sort_values('wavelength', inplace=True)

        return theoLineDF

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

    def results_to_database(self, lineLabel, linesDF, fit_conf, **kwargs):

        # Recover label data
        ion, waveRef, latexLabel = label_decomposition(self.mixtureComponents, combined_dict=fit_conf)
        ion, waveRef, latexLabel = np.array(ion, ndmin=1), np.array(waveRef, ndmin=1), np.array(latexLabel, ndmin=1)

        # Get the components on the list
        if lineLabel in fit_conf:
            blended_label = fit_conf[lineLabel]
            linesDF.loc[lineLabel, 'blended'] = blended_label
        else:
            blended_label = 'None'

        for i, line in enumerate(self.mixtureComponents):

            linesDF.loc[line, 'wavelength'] = waveRef[i]
            linesDF.loc[line, 'ion'] = ion[i]
            linesDF.loc[line, 'pynebCode'] = waveRef[i]
            linesDF.loc[line, 'w1':'w6'] = self.lineWaves

            linesDF.loc[line, 'intg_flux'] = self.__getattribute__('lineIntgFlux') * self.normFlux
            linesDF.loc[line, 'intg_err'] = self.__getattribute__('lineIntgErr') * self.normFlux
            linesDF.loc[line, 'cont'] = self.__getattribute__('cont') * self.normFlux
            linesDF.loc[line, 'std_continuum'] = self.__getattribute__('std_continuum') * self.normFlux
            linesDF.loc[line, 'm_continuum'] = self.__getattribute__('m_continuum') * self.normFlux
            linesDF.loc[line, 'n_continuum'] = self.__getattribute__('n_continuum')* self.normFlux
            linesDF.loc[line, 'eqw'] = self.__getattribute__('eqw')[i]
            linesDF.loc[line, 'eqw_err'] = self.__getattribute__('eqwErr')[i]
            linesDF.loc[line, 'snr_line'] = self.__getattribute__('snr_line')
            linesDF.loc[line, 'snr_cont'] = self.__getattribute__('snr_cont')

            linesDF.loc[line, 'peak_wave'] = self.__getattribute__('peakWave')
            linesDF.loc[line, 'peak_flux'] = self.__getattribute__('peakFlux') * self.normFlux

            linesDF.loc[line, 'blended'] = blended_label
            linesDF.loc[line, 'latexLabel'] = latexLabel[i]

            linesDF.loc[line, 'gauss_flux'] = self.__getattribute__('lineGaussFlux')[i] * self.normFlux
            linesDF.loc[line, 'gauss_err'] = self.__getattribute__('lineGaussErr')[i] * self.normFlux

            linesDF.loc[line, 'observation'] = 'detected'

            if self.p1 is not None:

                linesDF.loc[line, 'amp'] = self.p1[0, i] * self.normFlux
                linesDF.loc[line, 'amp_err'] = self.p1_Err[0, i] * self.normFlux

                linesDF.loc[line, 'mu'] = self.p1[1, i]
                linesDF.loc[line, 'mu_err'] = self.p1_Err[1, i]

                linesDF.loc[line, 'sigma'] = self.p1[2, i]
                linesDF.loc[line, 'sigma_err'] = self.p1_Err[2, i]

                linesDF.loc[line, 'v_r'] = self.v_r[i]
                linesDF.loc[line, 'v_r_err'] = self.v_r_Err[i]

                linesDF.loc[line, 'sigma_vel'] = self.sigma_vel[i]
                linesDF.loc[line, 'sigma_err_vel'] = self.sigma_vel_Err[i]

                if self.blended_check:
                    linesDF.loc[line, 'wavelength'] = waveRef[i]
                    linesDF.loc[line, 'peak_wave'] = self.p1[1, i]
                    linesDF.loc[line, 'peak_wave'] = self.p1[1, i]

                    # Combined line item
                    combined_latex_label = '+'.join(latexLabel)
                    linesDF.loc[lineLabel, 'wavelength'] = self.peakWave
                    linesDF.loc[lineLabel, 'latexLabel'] = combined_latex_label.replace('$+$', '+')
                    linesDF.loc[lineLabel, 'intg_flux'] = self.__getattribute__('lineIntgFlux') * self.normFlux
                    linesDF.loc[lineLabel, 'intg_err'] = self.__getattribute__('lineIntgErr') * self.normFlux

        # Sort by gaussian mu if possible
        linesDF.sort_values('peak_wave', inplace=True)

        return

    def database_to_attr(self):

        # Conversion parameters
        for name_attr, name_df in PARAMS_CONVERSION.items():
            value_df = self.linesDF.loc[self.lineLabel, name_df]
            self.__setattr__(name_attr, value_df)

        # Gaussian fit parameters
        self.p1, self.p1_Err = np.array([np.nan] * 3), np.array([np.nan] * 3)
        for idx, val in enumerate(('amp', 'mu', 'sigma')):
            self.p1[idx] = self.linesDF.loc[self.lineLabel, val]
            self.p1_Err[idx] = self.linesDF.loc[self.lineLabel, val + '_err']

        return

    def save_lineslog(self, linesDF, file_address):

        with open(file_address, 'wb') as output_file:
            string_DF = linesDF.to_string()
            output_file.write(string_DF.encode('UTF-8'))

        return

    def plot_spectrum_components(self, continuumFlux=None, obsLinesTable=None, matchedLinesDF=None, noise_region=None,
                                 log_scale=False, plotConf={}, axConf={}, output_address=None):

        # Plot Configuration
        defaultConf = STANDARD_PLOT.copy()
        defaultConf.update(plotConf)
        rcParams.update(defaultConf)
        fig, ax = plt.subplots()

        # Plot the spectrum
        ax.step(self.wave, self.flux, label='Observed spectrum')

        # Plot the continuum if available
        if continuumFlux is not None:
            ax.step(self.wave, continuumFlux, label='Error Continuum')

        # Plot astropy detected lines if available
        if obsLinesTable is not None:
            idcs_emission = obsLinesTable['line_type'] == 'emission'
            idcs_linePeaks = np.array(obsLinesTable[idcs_emission]['line_center_index'])
            ax.scatter(self.wave[idcs_linePeaks], self.flux[idcs_linePeaks], label='Detected lines', facecolors='none',
                       edgecolors='tab:purple')

        if matchedLinesDF is not None:
            idcs_foundLines = (matchedLinesDF.observation.isin(('detected', 'not identified'))) & \
                              (matchedLinesDF.wavelength >= self.wave[0]) & \
                              (matchedLinesDF.wavelength <= self.wave[-1])
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

            # for i in np.arange(lineLatexLabel.size):
            #     color_area = 'tab:red' if observation[i] == 'not identified' else 'tab:green'
            #     ax.axvspan(w3[i], w4[i], alpha=0.25, color=color_area)
            #     ax.text(lineWave[i], 0, lineLatexLabel[i], rotation=270)

        if noise_region is not None:
            ax.axvspan(noise_region[0], noise_region[1], alpha=0.15, color='tab:cyan', label='Noise region')

        if log_scale:
            ax.set_yscale('log')

        ax.update({**STANDARD_AXES, **axConf})
        ax.legend()

        if output_address is None:
            plt.tight_layout()
            plt.show()
        else:
            plt.savefig(output_address, bbox_inches='tight')

        plt.close(fig)


        return

    def plot_fit_components(self, lmfit_output=None, fig_conf={}, ax_conf={}, output_address=None, logScale=False):

        # Plot Configuration
        defaultConf = STANDARD_PLOT.copy()
        defaultConf.update(fig_conf)
        rcParams.update(defaultConf)
        fig, ax = plt.subplots()

        # Plot line spectrum
        idcs_plot = (self.lineWaves[0] - 5 <= self.wave) & (self.wave <= self.lineWaves[5] + 5)
        ax.step(self.wave[idcs_plot], self.flux[idcs_plot], label='Line spectrum')

        # Print lmfit results
        if lmfit_output is not None:

            x_fit, y_fit = lmfit_output.userkws['x'], lmfit_output.data

            wave_resample = np.linspace(x_fit[0], x_fit[-1], 500)
            flux_resample = lmfit_output.eval_components(x=wave_resample)

            ax.scatter(x_fit, y_fit, color='tab:red', label='Input data', alpha=0.4)
            ax.plot(x_fit, lmfit_output.best_fit, label='LMFIT best fit')

            # Plot individual components
            for comp_label, comp_flux in flux_resample.items():
                ax.plot(wave_resample, comp_flux, label=f'{comp_label}', linestyle='--')
        else:
            # Plot selection regions
            idcsW = np.searchsorted(self.wave, self.lineWaves)
            ax.fill_between(self.wave[idcsW[0]:idcsW[1]], 0, self.flux[idcsW[0]:idcsW[1]], facecolor='tab:orange',
                            step="pre",
                            alpha=0.2)
            ax.fill_between(self.wave[idcsW[2]:idcsW[3]], 0, self.flux[idcsW[2]:idcsW[3]], facecolor='tab:green',
                            step="pre",
                            alpha=0.2)
            ax.fill_between(self.wave[idcsW[4]:idcsW[5]], 0, self.flux[idcsW[4]:idcsW[5]], facecolor='tab:orange',
                            step="pre",
                            alpha=0.2)

        defaultConf = STANDARD_AXES.copy()
        defaultConf.update(ax_conf)

        if self.normFlux != 1.0:
            defaultConf['ylabel'] = defaultConf['ylabel'] + " $\\times{{{0:.2g}}}$".format(self.normFlux)

        ax.update(defaultConf)
        ax.legend()

        if logScale:
            ax.set_yscale('log')

        if output_address is None:
            plt.tight_layout()
            plt.show()
        else:
            plt.savefig(output_address, bbox_inches='tight')
        plt.close(fig)

        return

    def plot_line_grid(self, linesDF, plotConf={}, ncols=10, nrows=None, output_address=None, log_check=True):

        # Line labels to plot
        lineLabels = linesDF.index.values

        # Define plot axes grid size
        if nrows is None:
            nrows = int(np.ceil(lineLabels.size / ncols))
        if 'figure.figsize' not in plotConf:
            nrows = int(np.ceil(lineLabels.size / ncols))
            plotConf['figure.figsize'] = (ncols * 3, nrows * 3)
        n_axes, n_lines = ncols * nrows, lineLabels.size

        # Define figure
        defaultConf = STANDARD_PLOT.copy()
        defaultConf.update(plotConf)
        rcParams.update(defaultConf)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
        axesList = ax.flatten()

        # Get the lines components from the data frame if possible
        if 'blended' in linesDF.columns:
            conf_dict = linesDF.loc[linesDF.blended != 'None', 'blended'].to_dict()
        else:
            conf_dict = {}

        # Decomponse the lines
        ion_array, wavelength_array, latexLabel_array = label_decomposition(lineLabels, combined_dict=conf_dict)

        # Loop through the lines
        for i in np.arange(n_axes):
            if i < n_lines:

                # Label data region
                ax_i = axesList[i]
                lineLabel, latexLabel = lineLabels[i], latexLabel_array[i]
                lineWaves = linesDF.loc[lineLabel, 'w1':'w6'].values
                idcsLine, idcsContBlue, idcsContRed = self.define_masks(lineWaves, merge_continua=False)
                idcs_blue, idcs_red = np.where(idcsContBlue)[0], np.where(idcsContRed)[0]

                # Plot data
                limit_blue = idcs_blue[0]-10 if idcs_blue[0] > 10 else 0
                limit_red = idcs_red[-1]+10 if idcs_red[-1] + 10 < self.wave.size else self.wave.size - 1
                wave_plot, flux_plot = self.wave[limit_blue:limit_red], self.flux[limit_blue:limit_red]
                line_params = linesDF.loc[lineLabel, ['m_continuum', 'n_continuum']].values
                gaus_params = linesDF.loc[lineLabel, ['amp', 'mu', 'sigma']].values
                wavePeak, fluxPeak = linesDF.loc[lineLabel, ['peak_wave', 'peak_flux']].values

                # Check fittings were done
                cont_check, gauss_check = False, False
                if not pd.isnull(line_params).any():
                    cont_check = True
                    wave_resample = np.linspace(wave_plot[0], wave_plot[-1], 500)
                    m_cont, n_cont = line_params/self.normFlux
                    line_resample = linear_model(wave_resample, m_cont, n_cont)
                if cont_check and not pd.isnull(gaus_params).any():
                    gauss_check = True
                    amp, mu, sigma = gaus_params
                    amp = amp/self.normFlux
                    gauss_resample = gaussian_model(wave_resample, amp, mu, sigma) + line_resample

                # Plot data
                ax_i.step(wave_plot, flux_plot)
                ax_i.fill_between(self.wave[idcsContBlue], 0, self.flux[idcsContBlue], facecolor='tab:orange', step="pre", alpha=0.2)
                ax_i.fill_between(self.wave[idcsLine], 0, self.flux[idcsLine], facecolor='tab:blue', step="pre", alpha=0.2)
                ax_i.fill_between(self.wave[idcsContRed], 0, self.flux[idcsContRed], facecolor='tab:orange', step="pre", alpha=0.2)
                if gauss_check:
                    ax_i.plot(wave_resample, gauss_resample, '--', color='tab:purple', linewidth=1.75)
                else:
                    if cont_check:
                        ax_i.plot(wave_resample, line_resample, ':', color='tab:orange', linewidth=1.75)
                    for child in ax_i.get_children():
                        if isinstance(child, spines.Spine):
                            child.set_color('tab:red')

                # Axis format
                ax_i.set_ylim(ymin=np.min(self.flux[idcsLine])/5, ymax=fluxPeak/self.normFlux * 3.0)
                ax_i.yaxis.set_major_locator(plt.NullLocator())
                ax_i.yaxis.set_ticklabels([])
                ax_i.xaxis.set_major_locator(plt.NullLocator())
                ax_i.axes.yaxis.set_visible(False)
                ax_i.set_title(latexLabel)

                if log_check:
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

    def plot_detected_lines(self, lines_df, plotConf={}, ncols=10, nrows=None, output_address=None):

        # Plot data
        lineLabels = lines_df.index.values

        if nrows is None:
            nrows = int(np.ceil(lineLabels.size / ncols))

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
            self.lineWaves = lines_df.loc[lineLabels[i], 'w1':'w6'].values
            self.plot_line_region_i(axesList[i], lineLabels[i], lines_df)
            dict_spanSelec[f'spanner_{i}'] = SpanSelector(axesList[i], self.on_select, 'horizontal', useblit=True,
                                                          rectprops=dict(alpha=0.5, facecolor='tab:blue'))

        bpe = fig.canvas.mpl_connect('button_press_event', self.on_click)
        aee = fig.canvas.mpl_connect('axes_enter_event', self.on_enter_axes)
        plt.gca().axes.yaxis.set_ticklabels([])
        if output_address is None:
            plt.tight_layout()
            plt.show()
        else:
            plt.savefig(output_address, bbox_inches='tight')

        plt.close(fig)

        return

    def plot_line_region_i(self, ax, lineLabel, linesDF, limitPeak=5):

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


            # # Gaussian curve plot
            # p1 = linesDF.loc[lineLabel, 'amp':'sigma'].values
            # m, n = linesDF.loc[lineLabel, 'm_continuum'], linesDF.loc[lineLabel, 'n_continuum']
            # if (p1[0] is not np.nan) and (p1[0] is not None):
            #     wave_array = np.linspace(waveLine[0], waveLine[-1], 1000)
            #     cont_array = m * wave_array + n
            #     flux_array = gauss_func((wave_array, cont_array), p1[0], p1[1], p1[2])
            #     ax.plot(wave_array, cont_array, ':', color='tab:orange')
            #     ax.plot(wave_array, flux_array, ':', color='tab:red')

        # Plot format
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.xaxis.set_major_locator(plt.NullLocator())
        if 'latexLabel' in linesDF.columns:
            title_axes = linesDF.loc[lineLabel, "latexLabel"]
        else:
            title_axes = lineLabel
        ax.update({'title': title_axes})
        # ax.set_yscale('log')
        try:
            idxPeakFlux = np.argmax(fluxPeak)
            ax.set_ylim(ymin=np.min(fluxLine) / 5, ymax=fluxPeak[idxPeakFlux] * 1.25)
        except:
            print('Fale peak')

        ax.yaxis.set_ticklabels([])
        ax.axes.yaxis.set_visible(False)

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
        ax.yaxis.set_ticklabels([])
        ax.axes.yaxis.set_visible(False)

        return

    def table_fluxes(self, lines_df, tex_address, txt_address, pyneb_rc, scaleTable=1000):

        # TODO this could be included in sr.print

        # Measure line fluxes
        pdf = PdfPrinter()
        pdf.create_pdfDoc(tex_address, pdf_type='table')
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

            if (lines_df.loc[lineLabel, 'blended'] != 'None') and ('_m' not in lineLabel):
                flux, fluxErr = flux_gauss, flux_gaussErr
                label_entry = label_entry + '$_{gauss}$'
            else:
                flux, fluxErr = flux_intg, flux_intgErr

            # Correct the flux
            corr = pyneb_rc.getCorrHb(wavelength)
            intensity, intensityErr = flux * corr, fluxErr * corr

            eqw_entry = r'${:0.2f}\,\pm\,{:0.2f}$'.format(eqw, eqwErr)
            flux_entry = r'${:0.2f}\,\pm\,{:0.2f}$'.format(flux, fluxErr)
            intensity_entry = r'${:0.2f}\,\pm\,{:0.2f}$'.format(intensity, intensityErr)

            # Add row of data
            tex_row_i = [label_entry, eqw_entry, flux_entry, intensity_entry]
            txt_row_i = [label_entry, eqw, eqwErr, flux, fluxErr, intensity, intensityErr]

            lastRow_check = True if lineLabel == obsLines[-1] else False
            pdf.addTableRow(tex_row_i, last_row=lastRow_check)
            tableDF.loc[lineLabel] = txt_row_i[1:]

        # Data last rows
        row_Hbetaflux = [r'$H\beta$ $(erg\,cm^{-2} s^{-1} \AA^{-1})$',
                         '',
                         flux_Hbeta,
                         flux_Hbeta * pyneb_rc.getCorr(4861)]

        row_cHbeta = [r'$c(H\beta)$',
                      '',
                      float(pyneb_rc.cHbeta),
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
            pdf.generate_pdf(clean_tex=True)
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

                idcsLinePeak, idcsContinua = self.define_masks(self.lineWaves)

                self.line_properties(idcsLinePeak, idcsContinua, bootstrap_size=1000)

                # self.line_fit('lmfit', self.lineLabel, idcsLinePeak, idcsContinua, continuum_check=True,
                #               user_conf={})
                # # print(fit_report(self.fit_output))
                # # print(self.fit_params)
                # # self.plot_fit_components(self.fit_output)
                #
                # self.results_to_database(self.lineLabel, self.linesDF, {})

                self.save_lineslog(self.linesDF, str(self.linesLogAddress))

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
        else:
            print(f'Wave: {event.xdata}')


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
    emLine = gauss_func((wave, continuum), ampTrue, muTrue, sigmaTrue)
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
    lineContinuumFit = lineWave * lm.m_continuum + lm.n_continuum
    areaSimps = integrate.simps(lineFlux, lineWave) - integrate.simps(lineContinuumFit, lineWave)
    areaTrapz = integrate.trapz(lineFlux, lineWave) - integrate.trapz(lineContinuumFit, lineWave)
    areaIntgPixel = (lm.flux[idcsLines].sum() - lineContinuumFit.sum()) * lm.pixelWidth

    # Print the results
    print(f'True area : {areaTrue}')
    print(f'Simpsons rule: {areaSimps * lm.normFlux}')
    print(f'Trapezoid rule: {areaTrapz * lm.normFlux}')
    print(f'Fit integration: {lm.lineIntgFlux * lm.normFlux} +/- {lm.lineIntgErr * lm.normFlux}')
    print(f'Fit gaussian: {lm.lineGaussFlux[0] * lm.normFlux} +/- {lm.lineGaussErr[0] * lm.normFlux}')

    line_snr = np.mean(lineFlux) / np.sqrt(np.mean(np.power(lineFlux, 2)))
    cont_snr = np.mean(continuaFlux) / np.sqrt(np.mean(np.power(continuaFlux, 2)))
    print(f'Line signal to noise gaussian: {line_snr} {lm.snr_line}')
    print(f'Continuum signal to noise gaussian: {cont_snr} {lm.snr_cont}')

    # Lmfit output
    x_in, y_in = lm.fit_output.userkws['x'], lm.fit_output.data
    wave_resample = np.linspace(x_in[0], x_in[-1], 500)
    flux_resample = lm.fit_output.eval_components(x=wave_resample)
    cont_resample = lm.m_continuum * wave_resample + lm.n_continuum

    fig, ax = plt.subplots()
    ax.step(lm.wave, lm.flux, label='Observed line')
    ax.scatter(x_in, lm.fit_output.data, color='tab:red', alpha=0.2, label='Input points')
    ax.plot(wave_resample, sum(flux_resample.values()), label='Gaussian fit')
    ax.plot(wave_resample, cont_resample, label='Linear fitting Scipy', linestyle='--')

    ax.scatter(lm.wave[idcsLine_2], lm.flux[idcsLine_2], label='Line points')
    ax.scatter(lm.wave[idcsBlueCont], lm.flux[idcsBlueCont], label='Blue continuum')
    ax.scatter(lm.wave[idcsRedCont], lm.flux[idcsRedCont], label='Red continuum')

    # Plot individual components
    for comp_label, comp_flux in flux_resample.items():
        ax.plot(wave_resample, comp_flux, label=f'Component {comp_label}', linestyle='--')

    # ax.scatter(continuaWave, continuaFlux, label='Continuum regions')
    # ax.plot(lineWave, lineContinuumFit, label='Observed line', linestyle=':')
    # ax.plot(resampleWaveLine, gaussianCurve, label='Gaussian fit', linestyle=':')
    ax.legend()
    ax.update({'xlabel': 'Flux', 'ylabel': 'Wavelength', 'title': 'Gaussian fitting'})
    plt.show()
