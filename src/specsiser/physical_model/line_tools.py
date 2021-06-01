# import numpy as np
# import pandas as pd
# import astropy.units as au
# from numpy import ndarray
# from pandas import DataFrame
# from pathlib import Path
# from lmfit.models import GaussianModel, LinearModel, PolynomialModel, height_expr
# from lmfit import Parameters, fit_report, Model
# from astropy.modeling.polynomial import Polynomial1D
# from matplotlib.widgets import SpanSelector
# from specutils.manipulation import noise_region_uncertainty
# from specutils.fitting import find_lines_threshold, find_lines_derivative, fit_generic_continuum
# from specutils import Spectrum1D, SpectralRegion
# from matplotlib import pyplot as plt, rcParams, spines, gridspec
# from scipy import stats, optimize, integrate
# from data_reading import label_decomposition
# from pathlib import Path
# from src.specsiser.data_printing import PdfPrinter, label_decomposition
#
# STANDARD_PLOT = {'figure.figsize': (14, 7),
#                  'axes.titlesize': 14,
#                  'axes.labelsize': 14,
#                  'legend.fontsize': 12,
#                  'xtick.labelsize': 12,
#                  'ytick.labelsize': 12}
# STANDARD_AXES = {'xlabel': r'Wavelength $(\AA)$', 'ylabel': r'Flux $(erg\,cm^{-2} s^{-1} \AA^{-1})$'}
#
# LINEAR_ATTRIBUTES = ['slope', 'intercept']
# GAUSSIAN_ATTRIBUTES = ['amplitude', 'center', 'sigma', 'fwhm', 'height']
# PARAMETER_ATTRIBUTES = ['name', 'value', 'vary', 'min', 'max', 'expr', 'brute_step']
# PARAMETER_DEFAULT = dict(name=None, value=None, vary=True, min=-np.inf, max=np.inf, expr=None)
# HEIGHT_FORMULA = f'0.3989423 * component_amplitude / component_sigma'
#
#
# DATABASE_PATH = Path(__file__, '../../').resolve()/'literature_data'/'lines_data.xlsx'
#
# WAVE_UNITS_DEFAULT, FLUX_UNITS_DEFAULT = au.AA, au.erg / au.s / au.cm ** 2 / au.AA
#
# LINEMEASURER_PARAMS = ['pixelWidth',
#                        'peakWave',
#                        'peakInt',
#                        'lineIntgFlux',
#                        'lineIntgErr',
#                        'lineGaussFlux',
#                        'lineGaussErr',
#                        'n_continuum',
#                        'm_continuum',
#                        'std_continuum',
#                        'fit_function',
#                        'p1',
#                        'p1_Err']
#
# PARAMS_CONVERSION = {'lineIntgFlux': 'intg_flux',
#                      'lineIntgErr': 'intg_err',
#                      'cont': 'cont',
#                      'm_continuum': 'm_continuum',
#                      'n_continuum': 'n_continuum',
#                      'std_continuum': 'std_continuum',
#                      'lineGaussFlux': 'gauss_flux',
#                      'lineGaussErr': 'gauss_err',
#                      'eqw': 'eqw',
#                      'eqwErr': 'eqw_err'}
#
#
# VAL_LIST = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
# SYB_LIST = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
#
# FLUX_TEX_TABLE_HEADERS = [r'$Transition$', '$EW(\AA)$', '$F(\lambda)$', '$I(\lambda)$']
# FLUX_TXT_TABLE_HEADERS = [r'$Transition$', 'EW', 'EW_error', 'F(lambda)', 'F(lambda)_error', 'I(lambda)', 'I(lambda)_error']
#
# KIN_TEX_TABLE_HEADERS = [r'$Transition$', r'$Comp$', r'$v_{r}\left(\nicefrac{km}{s}\right)$', r'$\sigma_{int}\left(\nicefrac{km}{s}\right)$', r'Flux $(\nicefrac{erg}{cm^{-2} s^{-1} \AA^{-1})}$']
# KIN_TXT_TABLE_HEADERS = [r'$Transition$', r'$Comp$', 'v_r', 'v_r_error', 'sigma_int', 'sigma_int_error', 'flux', 'flux_error']
#
#
# SQRT2PI = np.sqrt(2*np.pi)
#
# KIN_LABEL_CONVERSION = {'center': 'mu', 'sigma': 'sigma'}
#
# def latex_science_float(f):
#     float_str = "{0:.2g}".format(f)
#     if "e" in float_str:
#         base, exponent = float_str.split("e")
#         return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
#     else:
#         return float_str
#
#
# def gauss_area(sigma_true, amp_true):
#     # return np.sqrt(2 * np.pi * sigma_true ** 2) * amp_true
#     return amp_true * SQRT2PI * sigma_true
#
#
# def linear_model(x, slope, intercept):
#     """a line"""
#     return slope * x + intercept
#
#
# def gaussian_model(x, amplitude, center, sigma):
#     """1-d gaussian curve : gaussian(x, amp, cen, wid)"""
#     return amplitude * np.exp(-0.5 * (((x-center)/sigma) * ((x-center)/sigma)))
#
#
# def iraf_snr(input_y):
#     avg = np.mean(input_y)
#     rms = np.sqrt(np.mean(np.power(input_y - avg, 2)))
#     return avg / rms
#
#
# def leave_axes(event):
#     event.inaxes.patch.set_facecolor('white')
#     event.canvas.draw()
#
#
# def define_lmfit_param(param_object, param_label, value=None, user_conf={}):
#     param_conf = PARAMETER_DEFAULT.copy()
#     param_conf['name'], param_conf['value'] = param_label, value
#
#     if '_amplitude' in param_object:  # TODO this could be and issue for absorptions
#         param_conf['min'] = 0.0
#
#     if param_label in user_conf:
#         param_conf.update(user_conf[param_label])
#
#     param_object.add(**param_conf)
#
#     return
#
#
# def gauss_func(ind_params, a, mu, sigma):
#     """
#     Gaussian function
#
#     This function returns the gaussian curve as the user speciefies and array of x values, the continuum level and
#     the parameters of the gaussian curve
#
#     :param ind_params: 2D array (x, z) where x is the array of abscissa values and z is the continuum level
#     :param float a: Amplitude of the gaussian
#     :param float mu: Center value of the gaussian
#     :param float sigma: Sigma of the gaussian
#     :return: Gaussian curve y array of values
#     :rtype: np.ndarray
#     """
#
#     x, z = ind_params
#     return a * np.exp(-((x - mu) * (x - mu)) / (2 * (sigma * sigma))) + z
#
#
# def generate_object_mask(lines_DF, wavelength, line_labels):
#     """
#
#     Algorithm to combine line and features mask
#
#     :param lines_DF:
#     :param wavelength:
#     :param line_labels:
#     :return:
#     """
#
#     # TODO This will not work for a redshifted lines log
#     idcs_lineMasks = lines_DF.index.isin(line_labels)
#     idcs_spectrumMasks = ~lines_DF.index.isin(line_labels)
#
#     # Matrix mask for integrating the emission lines
#     n_lineMasks = idcs_lineMasks.sum()
#     boolean_matrix = np.zeros((n_lineMasks, wavelength.size), dtype=bool)
#
#     # Total mask for valid regions in the spectrum
#     n_objMasks = idcs_spectrumMasks.sum()
#     int_mask = np.ones(wavelength.size, dtype=bool)
#     object_mask = np.ones(wavelength.size, dtype=bool)
#
#     # Loop through the emission lines
#     wmin, wmax = lines_DF['w3'].loc[idcs_lineMasks].values, lines_DF['w4'].loc[idcs_lineMasks].values
#     idxMin, idxMax = np.searchsorted(wavelength, [wmin, wmax])
#     for i in range(n_lineMasks):
#         idx_currentMask = (wavelength >= wavelength[idxMin[i]]) & (wavelength <= wavelength[idxMax[i]])
#         boolean_matrix[i, :] = idx_currentMask
#         int_mask = int_mask & ~idx_currentMask
#
#     # Loop through the object masks
#     wmin, wmax = lines_DF['w3'].loc[idcs_spectrumMasks].values, lines_DF['w4'].loc[idcs_spectrumMasks].values
#     idxMin, idxMax = np.searchsorted(wavelength, [wmin, wmax])
#     for i in range(n_objMasks):
#         idx_currentMask = (wavelength >= wavelength[idxMin[i]]) & (wavelength <= wavelength[idxMax[i]])
#         int_mask = int_mask & ~idx_currentMask
#         object_mask = object_mask & ~idx_currentMask
#
#     return boolean_matrix
#
#
# def compute_lineWidth(idx_peak, spec_flux, delta_i, min_delta=2):
#     """
#     Algororithm to measure emision line width given its peak location
#     :param idx_peak:
#     :param spec_flux:
#     :param delta_i:
#     :param min_delta:
#     :return:
#     """
#
#     i = idx_peak
#     while (spec_flux[i] > spec_flux[i + delta_i]) or (np.abs(idx_peak - (i + delta_i)) <= min_delta):
#         i += delta_i
#
#     return i
#
#
# def compute_lineWidth_peligroso(idx_peak, spec_flux, delta_i, min_delta=2):
#     """
#
#     Algorithm to measure emision line width given its peak location
#
#     :param idx_peak:
#     :param spec_flux:
#     :param delta_i:
#     :param min_delta:
#     :return:
#     """
#
#     i = idx_peak
#
#     limit_blue = (i + delta_i) > 0
#     limit_red = (i + delta_i) < spec_flux.size
#
#     if limit_blue or limit_red:
#         while (spec_flux[i] > spec_flux[i + delta_i]) or (np.abs(idx_peak - (i + delta_i)) <= min_delta):
#             i += delta_i
#
#             limit_blue = (i + delta_i) > 0
#             limit_red = (i + delta_i) < spec_flux.size
#
#             if limit_blue or limit_red:
#                 break
#
#     return i
#
#
# def int_to_roman(num):
#     i, roman_num = 0, ''
#     while num > 0:
#         for _ in range(num // VAL_LIST[i]):
#             roman_num += SYB_LIST[i]
#             num -= VAL_LIST[i]
#         i += 1
#     return roman_num
#
#
# def lineslogFile_to_DF(lineslog_address):
#     """
#     This function attemps several approaches to import a lines log from a sheet or text file lines as a pandas
#     dataframe
#     :param lineslog_address: String with the location of the input lines log file
#     :return lineslogDF: Dataframe with line labels as index and default column headers (wavelength, w1 to w6)
#     """
#
#     # Text file
#     try:
#         lineslogDF = pd.read_csv(lineslog_address, delim_whitespace=True, header=0, index_col=0)
#     except ValueError:
#
#         # Excel file
#         try:
#             lineslogDF = pd.read_excel(lineslog_address, sheet_name=0, header=0, index_col=0)
#         except ValueError:
#             print(f'- ERROR: Could not open lines log at: {lineslog_address}')
#
#     return lineslogDF
#
#
# def redshift_calculation(obs_array, emis_array, unit='wavelength', verbose=False):
#
#     if unit == 'wavelength':
#         z_array = obs_array/emis_array - 1
#     elif unit == 'frequency':
#         z_array = emis_array/obs_array - 1
#     else:
#         print(f'- ERROR: Units {unit} for redshift calculation no understood')
#
#     if verbose:
#         print(f'Redshift per line: {z_array}')
#         print(f'Mean redshift: {z_array.mean()} {z_array.std()}')
#
#     return z_array.mean(), z_array.std()
#
#
# def wavelength_to_vel(delta_lambda, lambda_wave):
#     return c_KMpS * (delta_lambda/lambda_wave)
#
#
# def save_lineslog(linesDF, file_address):
#
#     with open(file_address, 'wb') as output_file:
#         string_DF = linesDF.to_string()
#         output_file.write(string_DF.encode('UTF-8'))
#
#     return
#
#
# def isDigit(x):
#     try:
#         float(x)
#         return True
#     except ValueError:
#         return False
#
#
# def kinematic_component_labelling(line_latex_label, comp_ref):
#
#     if len(comp_ref) != 2:
#         print(f'-- Warning: Components label for {line_latex_label} is {comp_ref}. Code only prepare for a 2 character description (ex. n1, w2...)')
#
#     number = comp_ref[-1]
#     letter = comp_ref[0]
#
#     if letter in ('n', 'w'):
#         if letter == 'n':
#             comp_label = f'Narrow {number}'
#         if letter == 'w':
#             comp_label = f'Wide {number}'
#     else:
#         comp_label = f'{letter}{number}'
#
#     if '-' in line_latex_label:
#         lineEmisLabel = line_latex_label.replace(f'-{comp_ref}', '')
#     else:
#         lineEmisLabel = line_latex_label
#
#     return comp_label, lineEmisLabel
#
#
# def import_kinematics_from_line(line_label, user_conf, line_df):
#
#     kinemLine = user_conf.get(f'{line_label}_kinem')
#
#     if kinemLine is not None:
#
#         if kinemLine not in line_df.index:
#             print(f'-- WARNING: {kinemLine} has not been measured. Its kinematics were not copied to {line_label}')
#         else:
#             vr_parent = line_df.loc[kinemLine, 'v_r':'v_r_err'].values
#             sigma_parent = line_df.loc[kinemLine, 'sigma_vel':'sigma_err_vel'].values
#             wave_parent = line_df.loc[kinemLine, 'wavelength']
#
#             ion, wave_child, latexLabelLine = label_decomposition(line_label, scalar_output=True)
#
#             # Convert parent velocity units to child angstrom units
#             for param_ext in ('center', 'sigma'):
#                 param_label = f'{line_label}_{param_ext}'
#                 if param_label in user_conf: print(f'-- WARNING: {param_label} overwritten by {kinemLine} kinematics')
#                 if param_ext == 'center':
#                     param_value = wave_child * (1 + vr_parent / c_KMpS)
#                 else:
#                     param_value = wave_child * (sigma_parent / c_KMpS)
#                 user_conf[param_label] = {'value': param_value[0], 'vary': False}
#                 user_conf[f'{param_label}_err'] = {'value': param_value[1], 'vary': False}
#
#     return
#
#
#
# class LineMesurer(EmissionFitting):
#
#     _linesDF = None
#     _redshift, _normFlux = 0, 1
#     _wave_units = 'lambda'
#
#     # TODO do not include lines with '_b' in output lines log
#     def __init__(self, input_wave=None, input_flux=None, input_err=None, linesDF_address=None, redshift=None,
#                  normFlux=None, crop_waves=None, wave_units='lambda'):
#
#         # Emission model inheritance
#         EmissionFitting.__init__(self)
#
#         # Start cropping the input spectrum if necessary
#         if crop_waves is not None:
#             idcs_cropping = (input_wave >= crop_waves[0]) & (input_wave <= crop_waves[1])
#             input_wave = input_wave[idcs_cropping]
#             input_flux = input_flux[idcs_cropping]
#             if input_err is not None:
#                 input_err = input_err[idcs_cropping]
#
#         # Import object spectrum # TODO Add flexibility for wave changes
#         self.wave_units = wave_units
#
#         # Apply the redshift correction
#         self.redshift = redshift if redshift is not None else self._redshift
#         if (input_wave is not None) and (input_flux is not None):
#             self.wave = input_wave / (1 + self.redshift)
#             self.flux = input_flux * (1 + self.redshift)
#             if input_err is not None:
#                 self.errFlux = input_err * (1 + self.redshift)
#
#         # Normalize the spectrum
#         self.normFlux = normFlux if normFlux is not None else self._normFlux
#         self.flux = self.flux / self.normFlux
#         if input_err is not None:
#             self.errFlux = self.errFlux / self.normFlux
#
#         # Generate empty dataframe to store measurement use cwd as default storing folder
#         if linesDF_address is None:
#             self.linesLogAddress = Path.cwd()
#             _linesDb = lineslogFile_to_DF(DATABASE_PATH)
#             self.linesDF = DataFrame(columns=_linesDb.columns)
#
#         # Otherwise use the one from the user
#         else:
#             self.linesLogAddress = linesDF_address
#             if Path(self.linesLogAddress).is_file():
#                 self.linesDF = lineslogFile_to_DF(linesDF_address)
#             else:
#                 print(f'-- WARNING: linesLog not found at {self.linesLogAddress}')
#
#         return
#
#     def print_results(self, label=None, show_fit_report=True, show_plot=False):
#
#         # Case no line as input: Show the current measurement
#         if label is None:
#             if self.lineLabel is not None:
#                 output_ref = (f'Input line: {self.lineLabel}\n'
#                               f'- Line regions: {self.lineWaves}\n'
#                               f'- Spectrum: normalization flux: {self.normFlux}; redshift {self.redshift}\n'
#                               f'- Peak: wavelength {self.peakWave:.2f}; peak intensity {self.peakFlux:.2f}\n'
#                               f'- Continuum: slope {self.m_continuum:.2f}; intercept {self.n_continuum:.2f}\n')
#
#                 if self.mixtureComponents.size == 1:
#                     output_ref += f'- Intg Eqw: {self.eqw[0]:.2f} +/- {self.eqwErr[0]:.2f}\n'
#
#                 output_ref += f'- Intg flux: {self.lineIntgFlux:.3f} +/- {self.lineIntgErr:.3f}\n'
#
#                 for i, lineRef in enumerate(self.mixtureComponents):
#                     output_ref += (f'- {lineRef} gaussian fitting:\n'
#                                    f'-- Gauss flux: {self.lineGaussFlux[i]:.3f} +/- {self.lineGaussErr[i]:.3f}\n'
#                                    f'-- Height: {self.p1[0][i]:.3f} +/- {self.p1[0][i]:.3f}\n'
#                                    f'-- Center: {self.p1[1][i]:.3f} +/- {self.p1[1][i]:.3f}\n'
#                                    f'-- Sigma: {self.p1[2][i]:.3f} +/- {self.p1[2][i]:.3f}\n\n')
#
#             else:
#                 output_ref = f'- No measurement performed\n'
#
#         # Case with line input: search and show that measurement
#         elif self.linesDF is not None:
#             if label in self.linesDF.index:
#                 output_ref = self.linesDF.loc[label].to_string
#             else:
#                 output_ref = f'- WARNING: {label} not found in  lines table\n'
#         else:
#             output_ref = '- WARNING: Measurement lines log not defined\n'
#
#         # Display the print lmfit report if available
#         if show_fit_report:
#             if self.fit_output is not None:
#                 output_ref += f'- LmFit output:\n{fit_report(self.fit_output)}\n'
#             else:
#                 output_ref += f'- LmFit output not available\n'
#
#         # Show the result
#         print(output_ref)
#
#         # Display plot
#         if show_plot:
#             self.plot_fit_components(self.fit_output) # TODO this function should read from lines log
#
#         return
#
#     def fit_from_wavelengths(self, label, line_wavelengths, fit_conf={}, algorithm='lmfit'):
#
#         # Clear previous measurement
#         self.reset_measuerement()
#
#         # Label the current measurement
#         self.lineLabel = label
#         self.lineWaves = line_wavelengths
#
#         # Establish spectrum line and continua regions
#         idcsLineRegion, idcsContRegion = self.define_masks(self.lineWaves)
#
#         # Integrated line properties
#         self.line_properties(idcsLineRegion, idcsContRegion, bootstrap_size=1000)
#
#         # Gaussian line fit properties
#         self.line_fit(algorithm, self.lineLabel, idcsLineRegion, idcsContRegion, user_conf=fit_conf, lineDF=self.linesDF)
#
#         # Safe the results to the lineslog
#         self.results_to_database(self.lineLabel, self.linesDF, fit_conf)
#
#         return
#
#     def match_lines(self, obsLineTable, theoLineDF, lineType='emission', tol=5, blendedLineList=[], detect_check=False,
#                     find_line_borders='Auto'):
#         #TODO maybe we should remove not detected from output
#
#         # Query the lines from the astropy finder tables # TODO Expand technique for absorption lines
#         idcsLineType = obsLineTable['line_type'] == lineType
#         idcsLinePeak = np.array(obsLineTable[idcsLineType]['line_center_index'])
#         waveObs = self.wave[idcsLinePeak]
#
#         # Theoretical wave values
#         waveTheory = theoLineDF.wavelength.values
#
#         # Match the lines with the theoretical emission
#         tolerance = np.diff(self.wave).mean() * tol
#         theoLineDF['observation'] = 'not detected'
#         unidentifiedLine = dict.fromkeys(theoLineDF.columns.values, np.nan)
#
#         for i in np.arange(waveObs.size):
#
#             idx_array = np.where(np.isclose(a=waveTheory, b=waveObs[i], atol=tolerance))
#
#             if len(idx_array[0]) == 0:
#                 unknownLineLabel = 'xy_{:.0f}A'.format(np.round(waveObs[i]))
#
#                 # Scheme to avoid repeated lines
#                 if (unknownLineLabel not in theoLineDF.index) and detect_check:
#                     newRow = unidentifiedLine.copy()
#                     newRow.update({'wavelength': waveObs[i], 'w3': waveObs[i] - 5, 'w4': waveObs[i] + 5,
#                                    'observation': 'not identified'})
#                     theoLineDF.loc[unknownLineLabel] = newRow
#
#             else:
#                 row_index = theoLineDF.index[theoLineDF.wavelength == waveTheory[idx_array[0][0]]]
#                 theoLineDF.loc[row_index, 'observation'] = 'detected'
#                 theoLineLabel = row_index[0]
#
#                 # TODO lines like Halpha+[NII] this does not work, we should add exclusion
#                 if find_line_borders == True:
#                     minSeparation = 4 if theoLineLabel in blendedLineList else 2
#                     idx_min = compute_lineWidth(idcsLinePeak[i], self.flux, delta_i=-1, min_delta=minSeparation)
#                     idx_max = compute_lineWidth(idcsLinePeak[i], self.flux, delta_i=1, min_delta=minSeparation)
#                     theoLineDF.loc[row_index, 'w3'] = self.wave[idx_min]
#                     theoLineDF.loc[row_index, 'w4'] = self.wave[idx_max]
#                 else:
#                     if find_line_borders == 'Auto':
#                         if '_b' not in theoLineLabel:
#                             minSeparation = 4 if theoLineLabel in blendedLineList else 2
#                             idx_min = compute_lineWidth(idcsLinePeak[i], self.flux, delta_i=-1, min_delta=minSeparation)
#                             idx_max = compute_lineWidth(idcsLinePeak[i], self.flux, delta_i=1, min_delta=minSeparation)
#                             theoLineDF.loc[row_index, 'w3'] = self.wave[idx_min]
#                             theoLineDF.loc[row_index, 'w4'] = self.wave[idx_max]
#
#         # Sort by wavelength
#         theoLineDF.sort_values('wavelength', inplace=True)
#
#         return theoLineDF
#
#     def line_finder(self, input_flux, noiseWaveLim, intLineThreshold=3, verbose=False):
#
#         assert noiseWaveLim[0] > self.wave[0] or noiseWaveLim[1] < self.wave[-1]
#
#         # Establish noise values
#         idcs_noiseRegion = (noiseWaveLim[0] <= self.wave) & (self.wave <= noiseWaveLim[1])
#         noise_region = SpectralRegion(noiseWaveLim[0] * WAVE_UNITS_DEFAULT, noiseWaveLim[1] * WAVE_UNITS_DEFAULT)
#         flux_threshold = intLineThreshold * input_flux[idcs_noiseRegion].std()
#
#         input_spectrum = Spectrum1D(input_flux * FLUX_UNITS_DEFAULT, self.wave * WAVE_UNITS_DEFAULT)
#         input_spectrum = noise_region_uncertainty(input_spectrum, noise_region)
#         linesTable = find_lines_derivative(input_spectrum, flux_threshold)
#
#         if verbose:
#             print(linesTable)
#
#         return linesTable
#
#     def results_to_database(self, lineLabel, linesDF, fit_conf, **kwargs):
#
#         # Recover label data
#         ion, waveRef, latexLabel = label_decomposition(self.mixtureComponents, combined_dict=fit_conf)
#         ion, waveRef, latexLabel = np.array(ion, ndmin=1), np.array(waveRef, ndmin=1), np.array(latexLabel, ndmin=1)
#
#         # Get the components on the list
#         if lineLabel in fit_conf:
#             blended_label = fit_conf[lineLabel]
#             linesDF.loc[lineLabel, 'blended'] = blended_label
#         else:
#             blended_label = 'None'
#
#         for i, line in enumerate(self.mixtureComponents):
#
#             print(line, blended_label)
#             linesDF.loc[line, 'wavelength'] = waveRef[i]
#             linesDF.loc[line, 'ion'] = ion[i]
#             linesDF.loc[line, 'pynebCode'] = waveRef[i]
#             linesDF.loc[line, 'w1':'w6'] = self.lineWaves
#
#             linesDF.loc[line, 'intg_flux'] = self.__getattribute__('lineIntgFlux') * self.normFlux
#             linesDF.loc[line, 'intg_err'] = self.__getattribute__('lineIntgErr') * self.normFlux
#             linesDF.loc[line, 'cont'] = self.__getattribute__('cont') * self.normFlux
#             linesDF.loc[line, 'std_continuum'] = self.__getattribute__('std_continuum') * self.normFlux
#             linesDF.loc[line, 'm_continuum'] = self.__getattribute__('m_continuum') * self.normFlux
#             linesDF.loc[line, 'n_continuum'] = self.__getattribute__('n_continuum')* self.normFlux
#             linesDF.loc[line, 'eqw'] = self.__getattribute__('eqw')[i]
#             linesDF.loc[line, 'eqw_err'] = self.__getattribute__('eqwErr')[i]
#             linesDF.loc[line, 'snr_line'] = self.__getattribute__('snr_line')
#             linesDF.loc[line, 'snr_cont'] = self.__getattribute__('snr_cont')
#
#             linesDF.loc[line, 'peak_wave'] = self.__getattribute__('peakWave')
#             linesDF.loc[line, 'peak_flux'] = self.__getattribute__('peakFlux') * self.normFlux
#
#             linesDF.loc[line, 'blended'] = blended_label
#             linesDF.loc[line, 'latexLabel'] = latexLabel[i]
#
#             linesDF.loc[line, 'gauss_flux'] = self.__getattribute__('lineGaussFlux')[i] * self.normFlux
#             linesDF.loc[line, 'gauss_err'] = self.__getattribute__('lineGaussErr')[i] * self.normFlux
#
#             linesDF.loc[line, 'observation'] = 'detected'
#
#             if self.p1 is not None:
#
#                 linesDF.loc[line, 'amp'] = self.p1[0, i] * self.normFlux
#                 linesDF.loc[line, 'amp_err'] = self.p1_Err[0, i] * self.normFlux
#
#                 linesDF.loc[line, 'mu'] = self.p1[1, i]
#                 linesDF.loc[line, 'mu_err'] = self.p1_Err[1, i]
#
#                 linesDF.loc[line, 'sigma'] = self.p1[2, i]
#                 linesDF.loc[line, 'sigma_err'] = self.p1_Err[2, i]
#
#                 linesDF.loc[line, 'v_r'] = self.v_r[i]
#                 linesDF.loc[line, 'v_r_err'] = self.v_r_Err[i]
#
#                 linesDF.loc[line, 'sigma_vel'] = self.sigma_vel[i]
#                 linesDF.loc[line, 'sigma_err_vel'] = self.sigma_vel_Err[i]
#
#                 # if self.blended_check:
#                 #     linesDF.loc[line, 'wavelength'] = waveRef[i]
#                 #     linesDF.loc[line, 'peak_wave'] = self.p1[1, i]
#                 #     linesDF.loc[line, 'peak_wave'] = self.p1[1, i]
#                 #
#                 #     # Combined line item
#                 #     combined_latex_label = '+'.join(latexLabel)
#                 #     linesDF.loc[lineLabel, 'wavelength'] = self.peakWave
#                 #     linesDF.loc[lineLabel, 'latexLabel'] = combined_latex_label.replace('$+$', '+')
#                 #     linesDF.loc[lineLabel, 'intg_flux'] = self.__getattribute__('lineIntgFlux') * self.normFlux
#                 #     linesDF.loc[lineLabel, 'intg_err'] = self.__getattribute__('lineIntgErr') * self.normFlux
#
#         # Remove blended line from log (Only store the deblended components)
#         if self.blended_check:
#             linesDF.drop(index=lineLabel, inplace=True)
#
#         # Sort by gaussian mu if possible
#         linesDF.sort_values('peak_wave', inplace=True)
#
#         return
#
#     def database_to_attr(self):
#
#         # Conversion parameters
#         for name_attr, name_df in PARAMS_CONVERSION.items():
#             value_df = self.linesDF.loc[self.lineLabel, name_df]
#             self.__setattr__(name_attr, value_df)
#
#         # Gaussian fit parameters
#         self.p1, self.p1_Err = np.array([np.nan] * 3), np.array([np.nan] * 3)
#         for idx, val in enumerate(('amp', 'mu', 'sigma')):
#             self.p1[idx] = self.linesDF.loc[self.lineLabel, val]
#             self.p1_Err[idx] = self.linesDF.loc[self.lineLabel, val + '_err']
#
#         return
#
#     def save_lineslog(self, linesDF, file_address):
#
#         with open(file_address, 'wb') as output_file:
#             string_DF = linesDF.to_string()
#             output_file.write(string_DF.encode('UTF-8'))
#
#         return
#
#     def plot_spectrum(self, continuumFlux=None, obsLinesTable=None, matchedLinesDF=None, noise_region=None,
#                       log_scale=False, plotConf={}, axConf={}, specLabel='Observed spectrum', output_address=None):
#
#         # Plot Configuration
#         defaultConf = STANDARD_PLOT.copy()
#         defaultConf.update(plotConf)
#         rcParams.update(defaultConf)
#         fig, ax = plt.subplots()
#
#         # Plot the spectrum
#         ax.step(self.wave, self.flux, label=specLabel)
#
#         # Plot the continuum if available
#         if continuumFlux is not None:
#             ax.step(self.wave, continuumFlux, label='Error Continuum', linestyle=':')
#
#         # Plot astropy detected lines if available
#         if obsLinesTable is not None:
#             idcs_emission = obsLinesTable['line_type'] == 'emission'
#             idcs_linePeaks = np.array(obsLinesTable[idcs_emission]['line_center_index'])
#             ax.scatter(self.wave[idcs_linePeaks], self.flux[idcs_linePeaks], label='Detected lines', facecolors='none',
#                        edgecolors='tab:purple')
#
#         if matchedLinesDF is not None:
#             idcs_foundLines = (matchedLinesDF.observation.isin(('detected', 'not identified'))) & \
#                               (matchedLinesDF.wavelength >= self.wave[0]) & \
#                               (matchedLinesDF.wavelength <= self.wave[-1])
#             if 'latexLabel' in matchedLinesDF:
#                 lineLatexLabel = matchedLinesDF.loc[idcs_foundLines].latexLabel.values
#             else:
#                 lineLatexLabel = matchedLinesDF.loc[idcs_foundLines].index.values
#             lineWave = matchedLinesDF.loc[idcs_foundLines].wavelength.values
#             w3, w4 = matchedLinesDF.loc[idcs_foundLines].w3.values, matchedLinesDF.loc[idcs_foundLines].w4.values
#             observation = matchedLinesDF.loc[idcs_foundLines].observation.values
#
#             for i in np.arange(lineLatexLabel.size):
#                 if observation[i] == 'detected':
#                     color_area = 'tab:red' if observation[i] == 'not identified' else 'tab:green'
#                     ax.axvspan(w3[i], w4[i], alpha=0.25, color=color_area)
#                     ax.text(lineWave[i], 0, lineLatexLabel[i], rotation=270)
#
#             # for i in np.arange(lineLatexLabel.size):
#             #     color_area = 'tab:red' if observation[i] == 'not identified' else 'tab:green'
#             #     ax.axvspan(w3[i], w4[i], alpha=0.25, color=color_area)
#             #     ax.text(lineWave[i], 0, lineLatexLabel[i], rotation=270)
#
#         if noise_region is not None:
#             ax.axvspan(noise_region[0], noise_region[1], alpha=0.15, color='tab:cyan', label='Noise region')
#
#         if log_scale:
#             ax.set_yscale('log')
#
#         if self.normFlux != 1:
#             if 'ylabel' not in axConf:
#                 y_label = STANDARD_AXES['ylabel']
#                 axConf['ylabel'] = y_label.replace('Flux', r'$Flux\,/\,{}$'.format(latex_science_float(self.normFlux)))
#
#         ax.update({**STANDARD_AXES, **axConf})
#         ax.legend()
#
#         if output_address is None:
#             plt.tight_layout()
#             plt.show()
#         else:
#             plt.savefig(output_address, bbox_inches='tight')
#
#         plt.close(fig)
#
#
#         return
#
#     def plot_fit_components(self, lmfit_output=None, fig_conf={}, ax_conf={}, output_address=None, logScale=False):
#
#         # Plot Configuration
#         defaultConf = STANDARD_PLOT.copy()
#         defaultConf.update(fig_conf)
#         rcParams.update(defaultConf)
#
#         if lmfit_output is None:
#             fig, ax = plt.subplots()
#             ax = [ax]
#         else:
#             # fig, ax = plt.subplots(nrows=2)
#             gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
#             ax = [plt.subplot(gs[0]), plt.subplot(gs[1])]
#
#         # Plot line spectrum
#         idcs_plot = (self.lineWaves[0] - 5 <= self.wave) & (self.wave <= self.lineWaves[5] + 5)
#         ax[0].step(self.wave[idcs_plot], self.flux[idcs_plot], label='Line spectrum')
#
#         # Print lmfit results
#         if lmfit_output is not None:
#
#             # Determine line Label:
#             lineLabel = 'None'
#             for comp in lmfit_output.var_names:
#                 if '_cont' in comp:
#                     lineLabel = comp[0:comp.find('_cont')]
#                 break
#
#             x_in, y_in = lmfit_output.userkws['x'], lmfit_output.data
#
#             wave_resample = np.linspace(x_in[0], x_in[-1], 500)
#             flux_resample = lmfit_output.eval_components(x=wave_resample)
#
#             ax[0].scatter(x_in, y_in, color='tab:red', label='Input data', alpha=0.4)
#             ax[0].plot(x_in, lmfit_output.best_fit, label='LMFIT best fit')
#
#             # Plot individual components
#             contLabel = f'{lineLabel}_cont_'
#             cont_flux = flux_resample.get(contLabel, 0.0)
#             for comp_label, comp_flux in flux_resample.items():
#                 comp_flux = comp_flux + cont_flux if comp_label != contLabel else comp_flux
#                 ax[0].plot(wave_resample, comp_flux, label=f'{comp_label}', linestyle='--')
#
#             # Plot the residuals:
#             residual = (y_in - lmfit_output.best_fit)/lmfit_output.best_fit
#             ax[1].step(x_in, residual)
#             # ax[1].plot(x_fit, 1/lmfit_output.weights)
#
#             err_norm = np.sqrt(self.errFlux[idcs_plot])/self.cont
#             ax[1].fill_between(self.wave[idcs_plot], -err_norm, err_norm, facecolor='tab:red', alpha=0.5,
#                                label=r'$\sigma_{Error} / \overline{F(linear)}$')
#
#             y_low, y_high = -self.std_continuum/self.cont, self.std_continuum/self.cont
#             ax[1].fill_between(x_in, y_low, y_high, facecolor='tab:orange', alpha=0.5,
#                                label = r'$\sigma_{Continuum} / \overline{F(linear)}$')
#
#             ax[1].set_xlim(ax[0].get_xlim())
#             ax[1].set_ylim(2*residual.min(), 2*residual.max())
#             ax[1].legend(loc='center left', framealpha=1)
#             ax[1].set_ylabel(r'$\frac{F_{obs}}{F_{fit}} - 1$')
#             ax[1].set_xlabel(r'Wavelength $(\AA)$')
#
#         # Plot selection regions
#         idcsW = np.searchsorted(self.wave, self.lineWaves)
#         ax[0].fill_between(self.wave[idcsW[0]:idcsW[1]], 0, self.flux[idcsW[0]:idcsW[1]], facecolor='tab:orange',
#                         step="pre", alpha=0.2)
#         ax[0].fill_between(self.wave[idcsW[2]:idcsW[3]], 0, self.flux[idcsW[2]:idcsW[3]], facecolor='tab:green',
#                         step="pre", alpha=0.2)
#         ax[0].fill_between(self.wave[idcsW[4]:idcsW[5]], 0, self.flux[idcsW[4]:idcsW[5]], facecolor='tab:orange',
#                         step="pre", alpha=0.2)
#
#         defaultConf = STANDARD_AXES.copy()
#         defaultConf.update(ax_conf)
#
#         if self.normFlux != 1.0:
#             defaultConf['ylabel'] = defaultConf['ylabel'] + " $\\times{{{0:.2g}}}$".format(self.normFlux)
#
#         ax[0].update(defaultConf)
#         ax[0].legend()
#
#         if logScale:
#             ax[0].set_yscale('log')
#
#         if output_address is None:
#             plt.tight_layout()
#             plt.show()
#         else:
#             plt.savefig(output_address, bbox_inches='tight')
#
#         return
#
#     def plot_fit_components_backUp(self, lmfit_output=None, fig_conf={}, ax_conf={}, output_address=None, logScale=False):
#
#         # Plot Configuration
#         defaultConf = STANDARD_PLOT.copy()
#         defaultConf.update(fig_conf)
#         rcParams.update(defaultConf)
#         fig, ax = plt.subplots()
#
#         # Plot line spectrum
#         idcs_plot = (self.lineWaves[0] - 5 <= self.wave) & (self.wave <= self.lineWaves[5] + 5)
#         ax.step(self.wave[idcs_plot], self.flux[idcs_plot], label='Line spectrum')
#
#         # Print lmfit results
#         if lmfit_output is not None:
#
#             # Determine line Label:
#             lineLabel = 'None'
#             for comp in lmfit_output.var_names:
#                 if '_cont' in comp:
#                     lineLabel = comp[0:comp.find('_cont')]
#                 break
#
#             x_fit, y_fit = lmfit_output.userkws['x'], lmfit_output.data
#
#             wave_resample = np.linspace(x_fit[0], x_fit[-1], 500)
#             flux_resample = lmfit_output.eval_components(x=wave_resample)
#
#             ax.scatter(x_fit, y_fit, color='tab:red', label='Input data', alpha=0.4)
#             ax.plot(x_fit, lmfit_output.best_fit, label='LMFIT best fit')
#
#             # Plot individual components
#             contLabel = f'{lineLabel}_cont_'
#             cont_flux = flux_resample.get(contLabel, 0.0)
#             for comp_label, comp_flux in flux_resample.items():
#                 comp_flux = comp_flux + cont_flux if comp_label != contLabel else comp_flux
#                 ax.plot(wave_resample, comp_flux, label=f'{comp_label}', linestyle='--')
#
#         # Plot selection regions
#         idcsW = np.searchsorted(self.wave, self.lineWaves)
#         ax.fill_between(self.wave[idcsW[0]:idcsW[1]], 0, self.flux[idcsW[0]:idcsW[1]], facecolor='tab:orange',
#                         step="pre", alpha=0.2)
#         ax.fill_between(self.wave[idcsW[2]:idcsW[3]], 0, self.flux[idcsW[2]:idcsW[3]], facecolor='tab:green',
#                         step="pre", alpha=0.2)
#         ax.fill_between(self.wave[idcsW[4]:idcsW[5]], 0, self.flux[idcsW[4]:idcsW[5]], facecolor='tab:orange',
#                         step="pre", alpha=0.2)
#
#         defaultConf = STANDARD_AXES.copy()
#         defaultConf.update(ax_conf)
#
#         if self.normFlux != 1.0:
#             defaultConf['ylabel'] = defaultConf['ylabel'] + " $\\times{{{0:.2g}}}$".format(self.normFlux)
#
#         ax.update(defaultConf)
#         ax.legend()
#
#         if logScale:
#             ax.set_yscale('log')
#
#         if output_address is None:
#             plt.tight_layout()
#             plt.show()
#         else:
#             plt.savefig(output_address, bbox_inches='tight')
#         plt.close(fig)
#
#         return
#
#     def plot_line_grid(self, linesDF, plotConf={}, ncols=10, nrows=None, output_address=None, log_check=True):
#
#         # Line labels to plot
#         lineLabels = linesDF.index.values
#
#         # Define plot axes grid size
#         if nrows is None:
#             nrows = int(np.ceil(lineLabels.size / ncols))
#         if 'figure.figsize' not in plotConf:
#             nrows = int(np.ceil(lineLabels.size / ncols))
#             plotConf['figure.figsize'] = (ncols * 3, nrows * 3)
#         n_axes, n_lines = ncols * nrows, lineLabels.size
#
#         # Define figure
#         defaultConf = STANDARD_PLOT.copy()
#         defaultConf.update(plotConf)
#         rcParams.update(defaultConf)
#         fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
#         axesList = ax.flatten()
#
#         # Get the lines components from the data frame if possible
#         if 'blended' in linesDF.columns:
#             conf_dict = linesDF.loc[linesDF.blended != 'None', 'blended'].to_dict()
#         else:
#             conf_dict = {}
#
#         # Decomponse the lines
#         ion_array, wavelength_array, latexLabel_array = label_decomposition(lineLabels, combined_dict=conf_dict)
#
#         # Loop through the lines
#         for i in np.arange(n_axes):
#             if i < n_lines:
#
#                 # Label data region
#                 ax_i = axesList[i]
#                 lineLabel, latexLabel = lineLabels[i], latexLabel_array[i]
#                 lineWaves = linesDF.loc[lineLabel, 'w1':'w6'].values
#                 idcsLine, idcsContBlue, idcsContRed = self.define_masks(lineWaves, merge_continua=False)
#                 idcs_blue, idcs_red = np.where(idcsContBlue)[0], np.where(idcsContRed)[0]
#
#                 # Plot data
#                 limit_blue = idcs_blue[0]-10 if idcs_blue[0] > 10 else 0
#                 limit_red = idcs_red[-1]+10 if idcs_red[-1] + 10 < self.wave.size else self.wave.size - 1
#                 wave_plot, flux_plot = self.wave[limit_blue:limit_red], self.flux[limit_blue:limit_red]
#                 line_params = linesDF.loc[lineLabel, ['m_continuum', 'n_continuum']].values
#                 gaus_params = linesDF.loc[lineLabel, ['amp', 'mu', 'sigma']].values
#                 # wavePeak, fluxPeak = linesDF.loc[lineLabel, ['peak_wave', 'peak_flux']].values
#                 idx_peak = np.argmax(self.flux[idcsLine])
#                 wavePeak, fluxPeak = self.wave[idcsLine][idx_peak], self.flux[idcsLine][idx_peak]
#
#                 # Check fittings were done
#                 cont_check, gauss_check = False, False
#                 if not pd.isnull(line_params).any():
#                     cont_check = True
#                     wave_resample = np.linspace(wave_plot[0], wave_plot[-1], 500)
#                     m_cont, n_cont = line_params/self.normFlux
#                     line_resample = linear_model(wave_resample, m_cont, n_cont)
#                 if cont_check and not pd.isnull(gaus_params).any():
#                     gauss_check = True
#                     amp, mu, sigma = gaus_params
#                     amp = amp/self.normFlux
#                     gauss_resample = gaussian_model(wave_resample, amp, mu, sigma) + line_resample
#
#                 # Plot data
#                 ax_i.step(wave_plot, flux_plot)
#                 ax_i.fill_between(self.wave[idcsContBlue], 0, self.flux[idcsContBlue], facecolor='tab:orange', step="pre", alpha=0.2)
#                 ax_i.fill_between(self.wave[idcsLine], 0, self.flux[idcsLine], facecolor='tab:blue', step="pre", alpha=0.2)
#                 ax_i.fill_between(self.wave[idcsContRed], 0, self.flux[idcsContRed], facecolor='tab:orange', step="pre", alpha=0.2)
#                 if gauss_check:
#                     ax_i.plot(wave_resample, gauss_resample, '--', color='tab:purple', linewidth=1.75)
#                 else:
#                     if cont_check:
#                         ax_i.plot(wave_resample, line_resample, ':', color='tab:orange', linewidth=1.75)
#                     for child in ax_i.get_children():
#                         if isinstance(child, spines.Spine):
#                             child.set_color('tab:red')
#
#                 # Axis format
#                 ax_i.set_ylim(ymin=np.min(self.flux[idcsLine])/5, ymax=fluxPeak * 3.0)
#                 ax_i.yaxis.set_major_locator(plt.NullLocator())
#                 ax_i.yaxis.set_ticklabels([])
#                 ax_i.xaxis.set_major_locator(plt.NullLocator())
#                 ax_i.axes.yaxis.set_visible(False)
#                 ax_i.set_title(latexLabel)
#
#                 if log_check:
#                     ax_i.set_yscale('log')
#
#             # Clear not filled axes
#             else:
#                 fig.delaxes(axesList[i])
#
#         if output_address is None:
#             plt.tight_layout()
#             plt.show()
#         else:
#             plt.savefig(output_address, bbox_inches='tight')
#
#         plt.close(fig)
#
#         return
#
#     def plot_detected_lines(self, lines_df, plotConf={}, ncols=10, nrows=None, output_address=None):
#
#         # Plot data
#         lineLabels = lines_df.index.values
#
#         if nrows is None:
#             nrows = int(np.ceil(lineLabels.size / ncols))
#
#         # Compute plot grid size
#         if 'figure.figsize' not in plotConf:
#             nrows = int(np.ceil(lineLabels.size / ncols))
#             plotConf['figure.figsize'] = (nrows * 4, 14)
#
#         # Plot format
#         defaultConf = STANDARD_PLOT.copy()
#         defaultConf.update(plotConf)
#         rcParams.update(defaultConf)
#
#         fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
#         axesList = ax.flatten()
#         dict_spanSelec = {}
#
#         # Generate plot
#         for i in np.arange(lineLabels.size):
#             self.lineWaves = lines_df.loc[lineLabels[i], 'w1':'w6'].values
#             self.plot_line_region_i(axesList[i], lineLabels[i], lines_df)
#             dict_spanSelec[f'spanner_{i}'] = SpanSelector(axesList[i], self.on_select, 'horizontal', useblit=True,
#                                                           rectprops=dict(alpha=0.5, facecolor='tab:blue'))
#
#         bpe = fig.canvas.mpl_connect('button_press_event', self.on_click)
#         aee = fig.canvas.mpl_connect('axes_enter_event', self.on_enter_axes)
#         plt.gca().axes.yaxis.set_ticklabels([])
#         if output_address is None:
#             plt.tight_layout()
#             plt.show()
#         else:
#             plt.savefig(output_address, bbox_inches='tight')
#
#         plt.close(fig)
#
#         return
#
#     def plot_line_region_i(self, ax, lineLabel, linesDF, limitPeak=5):
#
#         # Plot line region:
#         lineWave = linesDF.loc[lineLabel, 'wavelength']
#
#         # Decide type of plot
#         non_nan = (~pd.isnull(self.lineWaves)).sum()
#
#         # Incomplete selections
#         if non_nan < 6:  # selections
#
#             idcsLinePeak = (lineWave - limitPeak <= self.wave) & (self.wave <= lineWave + limitPeak)
#             idcsLineArea = (lineWave - limitPeak * 2 <= self.wave) & (lineWave - limitPeak * 2 <= self.lineWaves[3])
#             wavePeak, fluxPeak = self.wave[idcsLinePeak], self.flux[idcsLinePeak]
#             waveLine, fluxLine = self.wave[idcsLineArea], self.flux[idcsLineArea]
#             idxPeakFlux = np.argmax(fluxPeak)
#
#             ax.step(waveLine, fluxLine)
#
#             if non_nan == 2:
#                 idx1, idx2 = np.searchsorted(self.wave, self.lineWaves[0:2])
#                 ax.fill_between(self.wave[idx1:idx2], 0.0, self.flux[idx1:idx2], facecolor='tab:green',
#                                 step='mid', alpha=0.5)
#
#             if non_nan == 4:
#                 idx1, idx2, idx3, idx4 = np.searchsorted(self.wave, self.lineWaves[0:4])
#                 ax.fill_between(self.wave[idx1:idx2], 0.0, self.flux[idx1:idx2], facecolor='tab:green',
#                                 step='mid', alpha=0.5)
#                 ax.fill_between(self.wave[idx3:idx4], 0.0, self.flux[idx3:idx4], facecolor='tab:green',
#                                 step='mid', alpha=0.5)
#
#
#         # Complete selections
#         else:
#
#             # Proceed to measurment
#
#             idcsContLeft = (self.lineWaves[0] <= self.wave) & (self.wave <= self.lineWaves[1])
#             idcsContRight = (self.lineWaves[4] <= self.wave) & (self.wave <= self.lineWaves[5])
#             idcsLinePeak = (lineWave - limitPeak <= self.wave) & (self.wave <= lineWave + limitPeak)
#             idcsLineArea = (self.lineWaves[2] <= self.wave) & (self.wave <= self.lineWaves[3])
#
#             waveCentral, fluxCentral = self.wave[idcsLineArea], self.flux[idcsLineArea]
#             wavePeak, fluxPeak = self.wave[idcsLinePeak], self.flux[idcsLinePeak]
#
#             idcsLinePlot = (self.lineWaves[0] - 5 <= self.wave) & (self.wave <= self.lineWaves[5] + 5)
#             waveLine, fluxLine = self.wave[idcsLinePlot], self.flux[idcsLinePlot]
#             ax.step(waveLine, fluxLine)
#
#             ax.fill_between(waveCentral, 0, fluxCentral, step="pre", alpha=0.4)
#             ax.fill_between(self.wave[idcsContLeft], 0, self.flux[idcsContLeft], facecolor='tab:orange', step="pre", alpha=0.2)
#             ax.fill_between(self.wave[idcsContRight], 0, self.flux[idcsContRight], facecolor='tab:orange', step="pre", alpha=0.2)
#
#
#             # # Gaussian curve plot
#             # p1 = linesDF.loc[lineLabel, 'amp':'sigma'].values
#             # m, n = linesDF.loc[lineLabel, 'm_continuum'], linesDF.loc[lineLabel, 'n_continuum']
#             # if (p1[0] is not np.nan) and (p1[0] is not None):
#             #     wave_array = np.linspace(waveLine[0], waveLine[-1], 1000)
#             #     cont_array = m * wave_array + n
#             #     flux_array = gauss_func((wave_array, cont_array), p1[0], p1[1], p1[2])
#             #     ax.plot(wave_array, cont_array, ':', color='tab:orange')
#             #     ax.plot(wave_array, flux_array, ':', color='tab:red')
#
#         # Plot format
#         ax.yaxis.set_major_locator(plt.NullLocator())
#         ax.xaxis.set_major_locator(plt.NullLocator())
#         if 'latexLabel' in linesDF.columns:
#             title_axes = linesDF.loc[lineLabel, "latexLabel"]
#         else:
#             title_axes = lineLabel
#         ax.update({'title': title_axes})
#         # ax.set_yscale('log')
#         try:
#             idxPeakFlux = np.argmax(fluxPeak)
#             ax.set_ylim(ymin=np.min(fluxLine) / 5, ymax=fluxPeak[idxPeakFlux] * 1.25)
#         except:
#             print('Fale peak')
#
#         ax.yaxis.set_ticklabels([])
#         ax.axes.yaxis.set_visible(False)
#
#         return
#
#     def plot_line_region_i_fit(self, ax, lineLabel, linesDF, limitPeak=5):
#
#         # Plot line region:
#         lineWave = linesDF.loc[lineLabel, 'wavelength']
#
#         # Decide type of plot
#         non_nan = (~pd.isnull(self.lineWaves)).sum()
#
#         # Incomplete selections
#         if non_nan < 6:  # selections
#
#             idcsLinePeak = (lineWave - limitPeak <= self.wave) & (self.wave <= lineWave + limitPeak)
#             idcsLineArea = (lineWave - limitPeak * 2 <= self.wave) & (lineWave - limitPeak * 2 <= self.lineWaves[3])
#             wavePeak, fluxPeak = self.wave[idcsLinePeak], self.flux[idcsLinePeak]
#             waveLine, fluxLine = self.wave[idcsLineArea], self.flux[idcsLineArea]
#             idxPeakFlux = np.argmax(fluxPeak)
#
#             ax.step(waveLine, fluxLine)
#
#             if non_nan == 2:
#                 idx1, idx2 = np.searchsorted(self.wave, self.lineWaves[0:2])
#                 ax.fill_between(self.wave[idx1:idx2], 0.0, self.flux[idx1:idx2], facecolor='tab:green',
#                                 step='mid', alpha=0.5)
#
#             if non_nan == 4:
#                 idx1, idx2, idx3, idx4 = np.searchsorted(self.wave, self.lineWaves[0:4])
#                 ax.fill_between(self.wave[idx1:idx2], 0.0, self.flux[idx1:idx2], facecolor='tab:green',
#                                 step='mid', alpha=0.5)
#                 ax.fill_between(self.wave[idx3:idx4], 0.0, self.flux[idx3:idx4], facecolor='tab:green',
#                                 step='mid', alpha=0.5)
#
#
#         # Complete selections
#         else:
#
#             # Proceed to measurment
#             idcsContLeft = (self.lineWaves[0] <= self.wave) & (self.wave <= self.lineWaves[1])
#             idcsContRight = (self.lineWaves[4] <= self.wave) & (self.wave <= self.lineWaves[5])
#             idcsLinePeak = (lineWave - limitPeak <= self.wave) & (self.wave <= lineWave + limitPeak)
#             idcsLineArea = (self.lineWaves[2] <= self.wave) & (self.wave <= self.lineWaves[3])
#
#             waveCentral, fluxCentral = self.wave[idcsLineArea], self.flux[idcsLineArea]
#             wavePeak, fluxPeak = self.wave[idcsLinePeak], self.flux[idcsLinePeak]
#
#             idcsLinePlot = (self.lineWaves[0] - 5 <= self.wave) & (self.wave <= self.lineWaves[5] + 5)
#             waveLine, fluxLine = self.wave[idcsLinePlot], self.flux[idcsLinePlot]
#             ax.step(waveLine, fluxLine)
#
#             ax.fill_between(waveCentral, 0, fluxCentral, step="pre", alpha=0.4)
#             ax.fill_between(self.wave[idcsContLeft], 0, self.flux[idcsContLeft], facecolor='tab:orange', step="pre", alpha=0.2)
#             ax.fill_between(self.wave[idcsContRight], 0, self.flux[idcsContRight], facecolor='tab:orange', step="pre", alpha=0.2)
#             idxPeakFlux = np.argmax(fluxPeak)
#
#             # Gaussian curve plot
#             p1 = linesDF.loc[lineLabel, 'amp':'sigma'].values
#             m, n = linesDF.loc[lineLabel, 'm_continuum'], linesDF.loc[lineLabel, 'n_continuum']
#             if (p1[0] is not np.nan) and (p1[0] is not None):
#                 wave_array = np.linspace(waveLine[0], waveLine[-1], 1000)
#                 cont_array = m * wave_array + n
#                 flux_array = gauss_func((wave_array, cont_array), p1[0], p1[1], p1[2])
#                 ax.plot(wave_array, cont_array, ':', color='tab:orange')
#                 ax.plot(wave_array, flux_array, ':', color='tab:red')
#
#         # Plot format
#         ax.yaxis.set_major_locator(plt.NullLocator())
#         ax.xaxis.set_major_locator(plt.NullLocator())
#         ax.update({'title': f'{linesDF.loc[lineLabel, "latexLabel"]}'})
#         ax.set_yscale('log')
#         ax.set_ylim(ymin=np.min(fluxLine) / 5, ymax=fluxPeak[idxPeakFlux] * 1.25)
#         ax.yaxis.set_ticklabels([])
#         ax.axes.yaxis.set_visible(False)
#
#         return
#
#     def table_fluxes(self, lines_df, table_address, pyneb_rc=None, scaleTable=1000):
#
#         # TODO this could be included in sr.print
#         tex_address = f'{table_address}'
#         txt_address = f'{table_address}.txt'
#
#         # Measure line fluxes
#         pdf = PdfPrinter()
#         pdf.create_pdfDoc(pdf_type='table')
#         pdf.pdf_insert_table(FLUX_TEX_TABLE_HEADERS)
#
#         # Dataframe as container as a txt file
#         tableDF = pd.DataFrame(columns=FLUX_TXT_TABLE_HEADERS[1:])
#
#         # Normalization line
#         if 'H1_4861A' in lines_df.index:
#             flux_Hbeta = lines_df.loc['H1_4861A', 'intg_flux']
#         else:
#             flux_Hbeta = scaleTable
#
#         obsLines = lines_df.index.values
#         for lineLabel in obsLines:
#
#             label_entry = lines_df.loc[lineLabel, 'latexLabel']
#             wavelength = lines_df.loc[lineLabel, 'wavelength']
#             eqw, eqwErr = lines_df.loc[lineLabel, 'eqw'], lines_df.loc[lineLabel, 'eqw_err']
#
#             flux_intg = lines_df.loc[lineLabel, 'intg_flux'] / flux_Hbeta * scaleTable
#             flux_intgErr = lines_df.loc[lineLabel, 'intg_err'] / flux_Hbeta * scaleTable
#             flux_gauss = lines_df.loc[lineLabel, 'gauss_flux'] / flux_Hbeta * scaleTable
#             flux_gaussErr = lines_df.loc[lineLabel, 'gauss_err'] / flux_Hbeta * scaleTable
#
#             if (lines_df.loc[lineLabel, 'blended'] != 'None') and ('_m' not in lineLabel):
#                 flux, fluxErr = flux_gauss, flux_gaussErr
#                 label_entry = label_entry + '$_{gauss}$'
#             else:
#                 flux, fluxErr = flux_intg, flux_intgErr
#
#             # Correct the flux
#             if pyneb_rc is not None:
#                 corr = pyneb_rc.getCorrHb(wavelength)
#                 intensity, intensityErr = flux * corr, fluxErr * corr
#                 intensity_entry = r'${:0.2f}\,\pm\,{:0.2f}$'.format(intensity, intensityErr)
#             else:
#                 intensity, intensityErr = '-', '-'
#                 intensity_entry = '-'
#
#             eqw_entry = r'${:0.2f}\,\pm\,{:0.2f}$'.format(eqw, eqwErr)
#             flux_entry = r'${:0.2f}\,\pm\,{:0.2f}$'.format(flux, fluxErr)
#
#             # Add row of data
#             tex_row_i = [label_entry, eqw_entry, flux_entry, intensity_entry]
#             txt_row_i = [label_entry, eqw, eqwErr, flux, fluxErr, intensity, intensityErr]
#
#             lastRow_check = True if lineLabel == obsLines[-1] else False
#             pdf.addTableRow(tex_row_i, last_row=lastRow_check)
#             tableDF.loc[lineLabel] = txt_row_i[1:]
#
#         if pyneb_rc is not None:
#
#             # Data last rows
#             row_Hbetaflux = [r'$H\beta$ $(erg\,cm^{-2} s^{-1} \AA^{-1})$',
#                              '',
#                              flux_Hbeta,
#                              flux_Hbeta * pyneb_rc.getCorr(4861)]
#
#             row_cHbeta = [r'$c(H\beta)$',
#                           '',
#                           float(pyneb_rc.cHbeta),
#                           '']
#         else:
#             # Data last rows
#             row_Hbetaflux = [r'$H\beta$ $(erg\,cm^{-2} s^{-1} \AA^{-1})$',
#                              '',
#                              flux_Hbeta,
#                              '-']
#
#             row_cHbeta = [r'$c(H\beta)$',
#                           '',
#                           '-',
#                           '']
#
#         pdf.addTableRow(row_Hbetaflux, last_row=False)
#         pdf.addTableRow(row_cHbeta, last_row=False)
#         tableDF.loc[row_Hbetaflux[0]] = row_Hbetaflux[1:] + [''] * 3
#         tableDF.loc[row_cHbeta[0]] = row_cHbeta[1:] + [''] * 3
#
#         # Format last rows
#         pdf.table.add_hline()
#         pdf.table.add_hline()
#
#         # Save the pdf table
#         try:
#             pdf.generate_pdf(table_address, clean_tex=True)
#         except:
#             print('-- PDF compilation failure')
#
#         # Save the txt table
#         with open(txt_address, 'wb') as output_file:
#             string_DF = tableDF.to_string()
#             string_DF = string_DF.replace('$', '')
#             output_file.write(string_DF.encode('UTF-8'))
#
#         return
#
#     def table_kinematics(self, lines_df, table_address, flux_normalization=1.0):
#
#         # TODO this could be included in sr.print
#         tex_address = f'{table_address}'
#         txt_address = f'{table_address}.txt'
#
#         # Measure line fluxes
#         pdf = PdfPrinter()
#         pdf.create_pdfDoc(pdf_type='table')
#         pdf.pdf_insert_table(KIN_TEX_TABLE_HEADERS)
#
#         # Dataframe as container as a txt file
#         tableDF = pd.DataFrame(columns=KIN_TXT_TABLE_HEADERS[1:])
#
#         obsLines = lines_df.index.values
#         for lineLabel in obsLines:
#
#             if not lineLabel.endswith('_b'):
#                 label_entry = lines_df.loc[lineLabel, 'latexLabel']
#
#                 # Establish component:
#                 blended_check = (lines_df.loc[lineLabel, 'blended'] != 'None') and ('_m' not in lineLabel)
#                 if blended_check:
#                     blended_group = lines_df.loc[lineLabel, 'blended']
#                     comp = 'n1' if lineLabel.count('_') == 1 else lineLabel[lineLabel.rfind('_')+1:]
#                 else:
#                     comp = 'n1'
#                 comp_label, lineEmisLabel = kinematic_component_labelling(label_entry, comp)
#
#                 wavelength = lines_df.loc[lineLabel, 'wavelength']
#                 v_r, v_r_err =  lines_df.loc[lineLabel, 'v_r':'v_r_err']
#                 sigma_vel, sigma_err_vel = lines_df.loc[lineLabel, 'sigma_vel':'sigma_err_vel']
#
#                 flux_intg = lines_df.loc[lineLabel, 'intg_flux']
#                 flux_intgErr = lines_df.loc[lineLabel, 'intg_err']
#                 flux_gauss = lines_df.loc[lineLabel, 'gauss_flux']
#                 flux_gaussErr = lines_df.loc[lineLabel, 'gauss_err']
#
#                 # Format the entries
#                 vr_entry = r'${:0.1f}\,\pm\,{:0.1f}$'.format(v_r, v_r_err)
#                 sigma_entry = r'${:0.1f}\,\pm\,{:0.1f}$'.format(sigma_vel, sigma_err_vel)
#
#                 if blended_check:
#                     flux, fluxErr = flux_gauss, flux_gaussErr
#                     label_entry = lineEmisLabel
#                 else:
#                     flux, fluxErr = flux_intg, flux_intgErr
#
#                 # Correct the flux
#                 flux_entry = r'${:0.2f}\,\pm\,{:0.2f}$'.format(flux, fluxErr)
#
#                 # Add row of data
#                 tex_row_i = [label_entry, comp_label, vr_entry, sigma_entry, flux_entry]
#                 txt_row_i = [lineLabel, comp_label.replace(' ', '_'), v_r, v_r_err, sigma_vel, sigma_err_vel, flux, fluxErr]
#
#                 lastRow_check = True if lineLabel == obsLines[-1] else False
#                 pdf.addTableRow(tex_row_i, last_row=lastRow_check)
#                 tableDF.loc[lineLabel] = txt_row_i[1:]
#
#         pdf.table.add_hline()
#
#         # Save the pdf table
#         try:
#             pdf.generate_pdf(tex_address)
#         except:
#             print('-- PDF compilation failure')
#
#         # Save the txt table
#         with open(txt_address, 'wb') as output_file:
#             string_DF = tableDF.to_string()
#             string_DF = string_DF.replace('$', '')
#             output_file.write(string_DF.encode('UTF-8'))
#
#         return
#
#     def on_select(self, Wlow, Whig):
#
#         # Check we are not just clicking on the plot
#         if Wlow != Whig:
#
#             # Count number of empty entries to determine next step
#             non_nans = (~pd.isnull(self.lineWaves)).sum()
#
#             # Case selecting 1/3 region
#             if non_nans == 0:
#                 self.lineWaves[0] = Wlow
#                 self.lineWaves[1] = Whig
#
#             # Case selecting 2/3 region
#             elif non_nans == 2:
#                 self.lineWaves[2] = Wlow
#                 self.lineWaves[3] = Whig
#                 self.lineWaves = np.sort(self.lineWaves)
#
#             # Case selecting 3/3 region
#             elif non_nans == 4:
#                 self.lineWaves[4] = Wlow
#                 self.lineWaves[5] = Whig
#                 self.lineWaves = np.sort(self.lineWaves)
#
#             elif non_nans == 6:
#                 self.lineWaves = np.sort(self.lineWaves)
#
#                 # Caso que se corrija la region de la linea
#                 if Wlow > self.lineWaves[1] and Whig < self.lineWaves[4]:
#                     self.lineWaves[2] = Wlow
#                     self.lineWaves[3] = Whig
#
#                 # Caso que se corrija el continuum izquierdo
#                 elif Wlow < self.lineWaves[2] and Whig < self.lineWaves[2]:
#                     self.lineWaves[0] = Wlow
#                     self.lineWaves[1] = Whig
#
#                 # Caso que se corrija el continuum derecho
#                 elif Wlow > self.lineWaves[3] and Whig > self.lineWaves[3]:
#                     self.lineWaves[4] = Wlow
#                     self.lineWaves[5] = Whig
#
#                 # Case we want to select the complete region
#                 elif Wlow < self.lineWaves[0] and Whig > self.lineWaves[5]:
#
#                     # # Remove line from dataframe and save it
#                     # self.remove_lines_df(self.current_df, self.Current_Label)
#                     #
#                     # # Save lines log df
#                     # self.save_lineslog_dataframe(self.current_df, self.lineslog_df_address)
#
#                     # Clear the selections
#                     self.lineWaves = np.array([np.nan] * 6)
#
#                 else:
#                     print('- WARNING: Unsucessful line selection:')
#                     print(f'-- {self.lineLabel}: w_low: {Wlow}, w_high: {Whig}')
#
#             # Check number of measurements after selection
#             non_nans = (~pd.isnull(self.lineWaves)).sum()
#
#             # Proceed to re-measurement if possible:
#             if non_nans == 6:
#
#                 self.linesDF.loc[self.lineLabel, 'w1':'w6'] = self.lineWaves
#
#                 idcsLinePeak, idcsContinua = self.define_masks(self.lineWaves)
#
#                 self.line_properties(idcsLinePeak, idcsContinua, bootstrap_size=1000)
#
#                 # self.line_fit('lmfit', self.lineLabel, idcsLinePeak, idcsContinua, continuum_check=True,
#                 #               user_conf={})
#                 # # print(fit_report(self.fit_output))
#                 # # print(self.fit_params)
#                 # # self.plot_fit_components(self.fit_output)
#                 #
#                 # self.results_to_database(self.lineLabel, self.linesDF, {})
#
#                 self.save_lineslog(self.linesDF, str(self.linesLogAddress))
#
#             # Else delete previous measurent data (except self.lineWaves):
#             else:
#                 for param in LINEMEASURER_PARAMS:
#                     self.__setattr__(param, None)
#
#             # Redraw the line measurement
#             self.in_ax.clear()
#
#             self.plot_line_region_i(self.in_ax, self.lineLabel, self.linesDF)
#
#             self.in_fig.canvas.draw()
#
#         return
#
#     def on_enter_axes(self, event):
#
#         self.in_fig = event.canvas.figure
#         self.in_ax = event.inaxes
#         idx_line = self.linesDF.latexLabel == self.in_ax.get_title()
#         self.lineLabel = self.linesDF.loc[idx_line].index.values[0]
#         self.lineWaves = self.linesDF.loc[idx_line, 'w1':'w6'].values[0]
#
#         self.database_to_attr()
#
#         # event.inaxes.patch.set_edgecolor('red')
#         event.canvas.draw()
#
#     def on_click(self, event):
#         if event.dblclick:
#             print(f'{event.button}, {event.x}, {event.y}, {event.xdata}, {event.ydata}')
#         else:
#             print(f'Wave: {event.xdata}')
#
#
# if __name__ == '__main__':
#
#     # Fake data
#     pixels_n = 200
#     noise_mag = 1.5
#     m, n = 0.005, 4.0
#     ampTrue, muTrue, sigmaTrue = 20, 5007, 2.3
#     areaTrue = np.sqrt(2 * np.pi * sigmaTrue ** 2) * ampTrue
#     linelabel, wave_regions = 'O3_5007A', np.array([4960, 4980, 4996, 5015, 5030, 5045])
#
#     red_path = '/Users/Dania/Documents/Proyectos/J0838_cubo/gemini_data/red'
#
#     # Spectrum generation
#     wave = np.linspace(4950, 5050, num=200)
#     continuum = (m * wave + n)
#     noise = np.random.normal(0, noise_mag, pixels_n)
#     emLine = gauss_func((wave, continuum), ampTrue, muTrue, sigmaTrue)
#     flux = noise + emLine
#
#     # Call funcions
#     lm = LineMesurer(wave, flux, normFlux=10)
#
#     # Perform fit
#     lm.fit_from_wavelengths(linelabel, wave_regions)
#     lm.print_results(show_fit_report=True, show_plot=True)
#
#     # Comparing flux integration techniques
#     idcsLines, idcsContinua = lm.define_masks(wave_regions)
#     idcsLine_2, idcsBlueCont, idcsRedCont = lm.define_masks(wave_regions, merge_continua=False)
#     lineWave, lineFlux = lm.wave[idcsLines], lm.flux[idcsLines]
#     continuaWave, continuaFlux = lm.wave[idcsContinua], lm.flux[idcsContinua]
#     lineContinuumFit = lineWave * lm.m_continuum + lm.n_continuum
#     areaSimps = integrate.simps(lineFlux, lineWave) - integrate.simps(lineContinuumFit, lineWave)
#     areaTrapz = integrate.trapz(lineFlux, lineWave) - integrate.trapz(lineContinuumFit, lineWave)
#     areaIntgPixel = (lm.flux[idcsLines].sum() - lineContinuumFit.sum()) * lm.pixelWidth
#
#     # Print the results
#     print(f'True area : {areaTrue}')
#     print(f'Simpsons rule: {areaSimps * lm.normFlux}')
#     print(f'Trapezoid rule: {areaTrapz * lm.normFlux}')
#     print(f'Fit integration: {lm.lineIntgFlux * lm.normFlux} +/- {lm.lineIntgErr * lm.normFlux}')
#     print(f'Fit gaussian: {lm.lineGaussFlux[0] * lm.normFlux} +/- {lm.lineGaussErr[0] * lm.normFlux}')
#
#     line_snr = np.mean(lineFlux) / np.sqrt(np.mean(np.power(lineFlux, 2)))
#     cont_snr = np.mean(continuaFlux) / np.sqrt(np.mean(np.power(continuaFlux, 2)))
#     print(f'Line signal to noise gaussian: {line_snr} {lm.snr_line}')
#     print(f'Continuum signal to noise gaussian: {cont_snr} {lm.snr_cont}')
#
#     # Lmfit output
#     x_in, y_in = lm.fit_output.userkws['x'], lm.fit_output.data
#     wave_resample = np.linspace(x_in[0], x_in[-1], 500)
#     flux_resample = lm.fit_output.eval_components(x=wave_resample)
#     cont_resample = lm.m_continuum * wave_resample + lm.n_continuum
#
#     fig, ax = plt.subplots()
#     ax.step(lm.wave, lm.flux, label='Observed line')
#     ax.scatter(x_in, lm.fit_output.data, color='tab:red', alpha=0.2, label='Input points')
#     ax.plot(wave_resample, sum(flux_resample.values()), label='Gaussian fit')
#     ax.plot(wave_resample, cont_resample, label='Linear fitting Scipy', linestyle='--')
#
#     ax.scatter(lm.wave[idcsLine_2], lm.flux[idcsLine_2], label='Line points')
#     ax.scatter(lm.wave[idcsBlueCont], lm.flux[idcsBlueCont], label='Blue continuum')
#     ax.scatter(lm.wave[idcsRedCont], lm.flux[idcsRedCont], label='Red continuum')
#
#     # Plot individual components
#     for curve_label, curve_flux in flux_resample.items():
#         ax.plot(wave_resample, curve_flux, label=f'Component {curve_label}', linestyle='--')
#
#     # ax.scatter(continuaWave, continuaFlux, label='Continuum regions')
#     # ax.plot(lineWave, lineContinuumFit, label='Observed line', linestyle=':')
#     # ax.plot(resampleWaveLine, gaussianCurve, label='Gaussian fit', linestyle=':')
#     ax.legend()
#     ax.update({'xlabel': 'Flux', 'ylabel': 'Wavelength', 'title': 'Gaussian fitting'})
#     plt.show()
