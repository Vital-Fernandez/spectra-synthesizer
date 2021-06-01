import numpy as np
from lmfit.models import PolynomialModel, Model
from src.specsiser.data_printing import label_decomposition
from scipy import stats, optimize


c_KMpS = 299792.458  # Speed of light in Km/s (https://en.wikipedia.org/wiki/Speed_of_light)


def wavelength_to_vel(delta_lambda, lambda_wave, light_speed=c_KMpS):
    return light_speed * (delta_lambda/lambda_wave)


def iraf_snr(input_y):
    avg = np.mean(input_y)
    rms = np.sqrt(np.mean(np.power(input_y - avg, 2)))
    return avg / rms


def gaussian_model(x, amplitude, center, sigma):
    """1-d gaussian curve : gaussian(x, amp, cen, wid)"""
    return amplitude * np.exp(-0.5 * (((x-center)/sigma) * ((x-center)/sigma)))


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


def linear_model(x, slope, intercept):
    """a line"""
    return slope * x + intercept


def is_digit(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def import_kinematics_from_line(line_label, user_conf, line_df):

    kinemLine = user_conf.get(f'{line_label}_kinem')

    if kinemLine is not None:

        if kinemLine not in line_df.index:
            print(f'-- WARNING: {kinemLine} has not been measured. Its kinematics were not copied to {line_label}')
        else:
            vr_parent = line_df.loc[kinemLine, 'v_r':'v_r_err'].values
            sigma_parent = line_df.loc[kinemLine, 'sigma_vel':'sigma_err_vel'].values
            wave_parent = line_df.loc[kinemLine, 'wavelength']

            ion, wave_child, latexLabelLine = label_decomposition(line_label, scalar_output=True)

            # Convert parent velocity units to child angstrom units
            for param_ext in ('center', 'sigma'):
                param_label = f'{line_label}_{param_ext}'
                if param_label in user_conf: print(f'-- WARNING: {param_label} overwritten by {kinemLine} kinematics')
                if param_ext == 'center':
                    param_value = wave_child * (1 + vr_parent / c_KMpS)
                else:
                    param_value = wave_child * (sigma_parent / c_KMpS)
                user_conf[param_label] = {'value': param_value[0], 'vary': False}
                user_conf[f'{param_label}_err'] = {'value': param_value[1], 'vary': False}

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

    def define_masks(self, wave_arr, flux_arr, masks_array, merge_continua=True):

        # Make sure it is a matrix
        masks_array = np.array(masks_array, ndmin=2)

        # Find indeces for six points in spectrum
        idcsW = np.searchsorted(self.wave, masks_array)

        # Emission region
        idcsLineRegion = ((wave_arr[idcsW[:, 2]] <= wave_arr[:, None]) &
                          (wave_arr[:, None] <= wave_arr[idcsW[:, 3]])).squeeze()

        # Return left and right continua merged in one array
        # TODO add check for wavelengths beyond wavelengh limits
        if merge_continua:

            idcsContRegion = (((wave_arr[idcsW[:, 0]] <= wave_arr[:, None]) &
                              (wave_arr[:, None] <= wave_arr[idcsW[:, 1]])) |
                              ((wave_arr[idcsW[:, 4]] <= wave_arr[:, None]) & (
                               wave_arr[:, None] <= wave_arr[idcsW[:, 5]]))).squeeze()

            return idcsLineRegion, idcsContRegion

        # Return left and right continua in separated arrays
        else:

            idcsContLeft = ((wave_arr[idcsW[:, 0]] <= wave_arr[:, None]) & (wave_arr[:, None] <= wave_arr[idcsW[:, 1]])).squeeze()
            idcsContRight = ((wave_arr[idcsW[:, 4]] <= wave_arr[:, None]) & (wave_arr[:, None] <= wave_arr[idcsW[:, 5]])).squeeze()

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

    def line_properties(self, emisWave, emisFlux, contWave, contFlux, bootstrap_size=500):

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

    def line_fit(self, algorithm, lineLabel, idcs_line, idcs_continua, iter_n=500, user_conf={}, lineDF=[]):

        # Check if line is in a blended group
        lineRef = lineLabel
        if '_b' in lineLabel:
            if lineLabel in user_conf:
                self.blended_check = True
                lineRef = user_conf[lineLabel]

        # Define x and y values according to line regions
        idcsFit = idcs_line + idcs_continua
        x_array, y_array = self.wave[idcsFit], self.flux[idcsFit]

        # Define fiting weights according to the error # TODO better to give an option introduce the error you want
        if self.errFlux is None:
            self.errFlux = np.full(self.flux.size, fill_value=self.std_continuum)
            weights_array = np.full(idcsFit.sum(), fill_value=1.0/self.std_continuum)
        else:
            weights_array = 1.0/np.sqrt(np.abs(self.errFlux[idcsFit]))

        # Check the kinematics import # TODO to change sigma_err_vel to sigma_vel_err
        import_kinematics_from_line(lineLabel, user_conf, lineDF)

        # Run fit
        if algorithm == 'mc':
            self.gauss_mcfit(idcs_line, idcs_continua, iter_n)

        if algorithm == 'lmfit':
            self.gauss_lmfit(lineRef, x_array, y_array, weights_array, user_conf, lineDF)

        return

    def gauss_lmfit(self, line_label, x, y, weights, user_conf={}, lines_df=[]):

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
        # Check if line is in a blended group
        if '_b' in line_label:
            if line_label in user_conf:
                self.blended_check = True
                line_label = user_conf[line_label]

        # Confirm the number of gaussian components
        self.mixtureComponents = np.array(line_label.split('-'), ndmin=1)
        n_comps = self.mixtureComponents.size
        ion_arr, theoWave_arr, latexLabel_arr = label_decomposition(self.mixtureComponents, combined_dict=user_conf)

        # Define initial wavelength for group
        ref_wave = np.array([self.peakWave], ndmin=1)

        # For blended lines replace the first line reference waves by the peak one
        if self.blended_check:
            ref_wave = theoWave_arr

        # # Import data from previous lines
        # if f'{line_label}_kinem' in user_conf:
        #     parent_line = user_conf[f'{line_label}_kinem']
        #     if parent_line in lines_df.index:
        #         ion_parent, wave_parent, latexLabel_parent = label_decomposition(parent_line, scalar_output=True)
        #         for param_ext in ('center', 'sigma'):
        #             param_label = f'{line_label}_{param_ext}'
        #             parent_value = lines_df.loc[parent_line, KIN_LABEL_CONVERSION[param_ext]]
        #             if param_label in user_conf:
        #                 print(f'-- WARNING: {param_label} overwritten by {parent_line} kinematics')
        #             user_conf[param_label] = {'value': theoWave_arr/wave_parent * parent_value, 'vary': False}
        #     else:
        #         print(f'-- WARNING: {parent_line} has not been measured. Its kinematics were not copied to {line_label}')

        # Define fitting params for each component
        fit_model = Model(linear_model)
        for idx_comp, comp in enumerate(self.mixtureComponents):

            # Linear
            if idx_comp == 0:
                fit_model.prefix = f'{comp}_cont_' # For a blended line the continuum conf is defined by first line
                self.define_param(fit_model, comp, 'cont_slope', self.m_continuum, self._SLOPE_PAR, user_conf)
                self.define_param(fit_model, comp, 'cont_intercept', self.n_continuum, self._INTER_PAR, user_conf)

            # Gaussian
            fit_model += Model(gaussian_model, prefix=f'{comp}_')
            self.define_param(fit_model, comp, 'amplitude', self.peakFlux-self.cont, self._AMP_PAR, user_conf)
            self.define_param(fit_model, comp, 'center', ref_wave[idx_comp], self._MU_PAR, user_conf)
            self.define_param(fit_model, comp, 'sigma', 1.0, self._SIG_PAR, user_conf)
            self.define_param(fit_model, comp, 'area', comp, self._AREA_PAR, user_conf)

        # Fit the line
        self.fit_params = fit_model.make_params()
        self.fit_output = fit_model.fit(y, self.fit_params, x=x, weights=weights)

        if not self.fit_output.errorbars:
            print(f'-- WARNING: Line measuring error at {line_label}')

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
            # TODO we need a robest mechanic to express the uncertainty in the N2_6548A and similar lines
            if line != 'N2_6548A':
                self.lineGaussFlux[i], self.lineGaussErr[i] = lineArea.value, lineArea.stderr
            elif (lineArea.stderr == 0) and ('N2_6584A_area' in self.fit_output.params):
                self.lineGaussFlux[i], self.lineGaussErr[i] = lineArea.value, self.fit_output.params['N2_6584A_area'].stderr/2.5066282746
            else:
                self.lineGaussFlux[i], self.lineGaussErr[i] = lineArea.value, lineArea.stderr

            # Equivalent with gaussian flux for blended components
            if self.blended_check:
                eqw_g[i], eqwErr_g[i] = self.lineGaussFlux[i]/self.cont, self.lineGaussErr[i]/self.cont

            # Kinematics
            self.v_r[i] = wavelength_to_vel(self.p1[1, i] - theoWave_arr[i], theoWave_arr[i])
            self.v_r_Err[i] = np.abs(wavelength_to_vel(self.p1_Err[1, i], theoWave_arr[i]))
            self.sigma_vel[i] = wavelength_to_vel(self.p1[2, i], theoWave_arr[i])
            self.sigma_vel_Err[i] = wavelength_to_vel(self.p1_Err[2, i], theoWave_arr[i])
            # if 'H1' in line:
            #     print(f'-- {line}, mu: {self.p1[1, i]}+/-{self.p1_Err[1, i]}, vr: {self.v_r[i]} +/- {self.v_r_Err[i] }')
            #     print(f'-- {line}, sigma_ang: {self.p1[2, i]}, sigma_kms: {self.sigma_vel[i]} +/- {self.sigma_vel_Err[i]}')

            # if (not self.blended_check) and (f'{line_label}_kinem' in user_conf):
            #     parent_line = user_conf[f'{line_label}_kinem']
            #     v_r_err, sigma_vel_err = lines_df.loc[parent_line, 'v_r_err'], lines_df.loc[parent_line, 'sigma_err_vel']
            #     self.v_r_Err[i], self.sigma_vel_Err[i] = v_r_err, sigma_vel_err

        if self.blended_check:
            self.eqw, self.eqwErr = eqw_g, eqwErr_g
        else:
            self.eqw, self.eqwErr = np.array(self.eqw, ndmin=1), np.array(self.eqwErr, ndmin=1)

        return

    def define_param(self, model_obj, line_label, param_label, param_value, default_conf={}, user_conf={}):

        param_ref = f'{line_label}_{param_label}'

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
                    ineq_expr = param_conf['expr'].replace('<', '').replace('>', '')
                    ineq_items = ineq_expr.split(ineq_operation)
                    ineq_linkedParam = ineq_items[0] if not is_digit(ineq_items[0]) else ineq_items[1]
                    ineq_lim = float(ineq_items[0]) if is_digit(ineq_items[0]) else float(ineq_items[1])

                    # Stablish the inequality configuration:
                    ineq_conf = {} # TODO need to check these limits
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

            # Case default value is not provided
            else:
                if param_conf['value'] is None:
                    param_conf['value'] = param_value

        # Special case importing kinematics in blended group
        if self.blended_check and (f'{line_label}_kinem' in user_conf) and (param_label in ('center', 'sigma')):
            parent_line = user_conf[f'{line_label}_kinem']
            ion_parent, theoWave_parent, latexLabel_parent = label_decomposition(parent_line, scalar_output=True)
            ion_child, theoWave_child, latexLabel_child = label_decomposition(line_label, scalar_output=True)
            parent_param_label = f'{parent_line}_{param_label}'
            param_conf = {'expr': f'{theoWave_child/theoWave_parent:0.8f}*{parent_param_label}'}

            if param_ref in user_conf:
                print(f'-- WARNING: {line_label} overwritten by {parent_line} kinematics')

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

