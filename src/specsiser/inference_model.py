import pymc3
import theano
import theano.tensor as tt
import numpy as np
import pickle
from pathlib import Path
from data_reading import parseConfDict
from data_printing import MCOutputDisplay
from physical_model.gasEmission_functions import storeValueInTensor
from physical_model.chemical_model import TOIII_from_TSIII_relation, TSIII_from_TOIII_relation, TOII_from_TOIII_relation
from physical_model.gasEmission_functions import assignFluxEq2Label, gridInterpolatorFunction, EmissionFluxModel
from physical_model.photo_ionization_model import ModelGridWrapper
from astropy.io import fits

# Disable compute_test_value in theano zeros tensor
theano.config.compute_test_value = "ignore"

log10_factor = 1.0 / np.log(10)

FITS_INPUTS_EXTENSION = {'line_list': '20A', 'line_fluxes': 'E', 'line_err': 'E'}
FITS_OUTPUTS_EXTENSION = {'parameter_list': '20A',
                          'mean': 'E',
                          'std': 'E',
                          'median': 'E',
                          'p16th': 'E',
                          'p84th': 'E',
                          'true': 'E'}


def displaySimulationData(model):
    # Check test_values are finite
    print('\n-- Test points:')
    model_var = model.test_point
    for var in model_var:
        displayText = '{} = {}'.format(var, model_var[var])
        print(displayText)

    # Checks log probability of random variables
    print('\n-- Log probability variable:')
    print(model.check_test_point())

    return


def fits_db(fits_address, model_db, ext_name='', header=None):

    line_labels = model_db['inputs']['line_list']
    params_traces = model_db['outputs']

    sec_label = 'synthetic_fluxes' if ext_name == '' else f'{ext_name}_synthetic_fluxes'

    # ---------------------------------- Input data

    # Data
    list_columns = []
    for data_label, data_format in FITS_INPUTS_EXTENSION.items():
        data_array = model_db['inputs'][data_label]
        data_col = fits.Column(name=data_label, format=data_format, array=data_array)
        list_columns.append(data_col)

    # Header
    hdr_dict = {}
    for i_line, lineLabel in enumerate(line_labels):
        hdr_dict[f'hierarch {lineLabel}'] = model_db['inputs']['line_fluxes'][i_line]
        hdr_dict[f'hierarch {lineLabel}_err'] = model_db['inputs']['line_err'][i_line]

    # Inputs extension
    cols = fits.ColDefs(list_columns)
    sec_label = 'inputs' if ext_name == '' else f'{ext_name}_inputs'
    hdu_inputs = fits.BinTableHDU.from_columns(cols, name=sec_label, header=fits.Header(hdr_dict))

    # ---------------------------------- Output data
    params_list = model_db['inputs']['parameter_list']
    param_matrix = np.array([params_traces[param] for param in params_list])
    param_col = fits.Column(name='parameters_list', format=FITS_OUTPUTS_EXTENSION['parameter_list'], array=params_list)
    param_val = fits.Column(name='parameters_fit', format='E', array=param_matrix.mean(axis=1))
    param_err = fits.Column(name='parameters_err', format='E', array=param_matrix.std(axis=1))
    list_columns = [param_col, param_val, param_err]

    # Header
    hdr_dict = {}
    for i, param in enumerate(params_list):
        param_trace = params_traces[param]
        hdr_dict[f'hierarch {param}'] = np.mean(param_trace)
        hdr_dict[f'hierarch {param}_err'] = np.std(param_trace)

    for lineLabel in line_labels:
        param_trace = params_traces[lineLabel]
        hdr_dict[f'hierarch {lineLabel}'] = np.mean(param_trace)
        hdr_dict[f'hierarch {lineLabel}_err'] = np.std(param_trace)

    # # Data
    # param_array = np.array(list(params_traces.keys()))
    # paramMatrix = np.array([params_traces[param] for param in param_array])
    #
    # list_columns.append(fits.Column(name='parameter', format='20A', array=param_array))
    # list_columns.append(fits.Column(name='mean', format='E', array=np.mean(paramMatrix, axis=0)))
    # list_columns.append(fits.Column(name='std', format='E', array=np.std(paramMatrix, axis=0)))
    # list_columns.append(fits.Column(name='median', format='E', array=np.median(paramMatrix, axis=0)))
    # list_columns.append(fits.Column(name='p16th', format='E', array=np.percentile(paramMatrix, 16, axis=0)))
    # list_columns.append(fits.Column(name='p84th', format='E', array=np.percentile(paramMatrix, 84, axis=0)))

    cols = fits.ColDefs(list_columns)
    sec_label = 'outputs' if ext_name == '' else f'{ext_name}_outputs'
    hdu_outputs = fits.BinTableHDU.from_columns(cols, name=sec_label, header=fits.Header(hdr_dict))

    # ---------------------------------- traces data
    list_columns = []
    for param, trace_array in params_traces.items():
        col_trace = fits.Column(name=param, format='E', array=params_traces[param])
        list_columns.append(col_trace)

    cols = fits.ColDefs(list_columns)
    sec_label = 'traces' if ext_name == '' else f'{ext_name}_traces'
    hdu_traces = fits.BinTableHDU.from_columns(cols, name=sec_label)

    # ---------------------------------- Save fits files
    hdu_list = [hdu_inputs, hdu_outputs, hdu_traces]

    if fits_address.is_file():
        for hdu in hdu_list:
            try:
                fits.update(fits_address, data=hdu.data, header=hdu.header, extname=hdu.name, verify=True)
            except KeyError:
                fits.append(fits_address, data=hdu.data, header=hdu.header, extname=hdu.name)
    else:
        hdul = fits.HDUList([fits.PrimaryHDU()] + hdu_list)
        hdul.writeto(fits_address, overwrite=True, output_verify='fix')

    return


class SpectraSynthesizer(MCOutputDisplay, ModelGridWrapper):

    def __init__(self):

        ModelGridWrapper.__init__(self)
        MCOutputDisplay.__init__(self)

        self.paramDict = {}
        self.priorDict = {}
        self.inferenModel = None

        self.lineLabels = None
        self.lineIons = None
        self.emissionFluxes = None
        self.emissionErr = None
        self.lineFlambda = None
        self.emtt = None

        self.obsIons = None

        self.lowTemp_check = None
        self.highTemp_check = None
        self.idcs_highTemp_ions = None

        self.ftauCoef = None
        self.emisGridInterpFun = None

        self.ionizationModels_Check = False

        self.total_regions = 1
        self.fit_results = None

    def define_region(self, objLinesDF, ion_model=None, extinction_model=None, chemistry_model=None, minErr=0.02,
                      normLine='H1_4861A'):

        # Lines data
        normLine = normLine if ion_model is None else ion_model.normLine
        idcs_lines = (objLinesDF.index != normLine)
        self.lineLabels = objLinesDF.loc[idcs_lines].index.values
        self.lineIons = objLinesDF.loc[idcs_lines].ion.values
        self.emissionFluxes = objLinesDF.loc[idcs_lines].intg_flux.values
        self.emissionErr = objLinesDF.loc[idcs_lines].intg_err.values

        if extinction_model is not None:
            self.lineFlambda = extinction_model.gasExtincParams(wave=objLinesDF.loc[idcs_lines].obsWave.values)

        # Emissivity data
        if ion_model is not None:
            emisGridDict = ion_model.emisGridDict
            self.ftauCoef = ion_model.ftau_coeffs
            self.emisGridInterpFun = gridInterpolatorFunction(emisGridDict, ion_model.tempRange, ion_model.denRange)

        if chemistry_model is not None:
            self.emtt = EmissionFluxModel(self.lineLabels, self.lineIons)
            self.obsIons = chemistry_model.obsAtoms
            self.idcs_highTemp_ions = chemistry_model.indcsHighTemp[idcs_lines]

        # Establish minimum error on lines
        if minErr is not None:
            err_fraction = self.emissionErr / self.emissionFluxes
            idcs_smallErr = err_fraction < minErr
            self.emissionErr[idcs_smallErr] = minErr * self.emissionFluxes[idcs_smallErr]

        return

    def simulation_configuration(self, model_parameters, prior_conf_dict, n_regions=0, grid_interpolator=None,
                                 photo_ionization_grid=False,
                                 verbose=True, T_low_diag='S3_6312A', T_high_diag='O3_4363A'):

        # Priors configuration # TODO this one should be detected automatically
        for param in model_parameters:
            priorConf = prior_conf_dict[param + '_prior']
            self.priorDict[param] = priorConf
        self.priorDict['logParams_list'] = prior_conf_dict['logParams_list']

        # Interpolator object
        if grid_interpolator is not None:
            self.gridInterp = grid_interpolator

        # Load photoIonization models
        if photo_ionization_grid:
            self.ionizationModels_Check = True
            self.HII_Teff_models(self.lineLabels, self.emissionFluxes, self.emissionErr)
        else:
            self.idx_analysis_lines = np.zeros(self.lineLabels.size)

        self.lowTemp_check = True if T_low_diag in self.lineLabels else False
        self.highTemp_check = True if T_high_diag in self.lineLabels else False

        if verbose:
            print(f'\n- Input lines ({self.lineLabels.size})')
            for i in range(self.lineLabels.size):
                print(f'-- {self.lineLabels[i]} '
                      f'({self.lineIons[i]})'
                      f'flux = {self.emissionFluxes[i]:.4f} +/- {self.emissionErr[i]:.4f} '
                      f'|| err/flux = {100 * self.emissionErr[i] / self.emissionFluxes[i]:.2f} %')

            # TODO Only display those which we are actually using
            # Display prior configuration
            print(f'\n- Priors configuration ({len(model_parameters)} parameters):')
            for param in model_parameters:
                print(f'-- {param} {self.priorDict[param][0]} dist : '
                      f'mu = {self.priorDict[param][1]}, '
                      f'std = {self.priorDict[param][2]},'
                      f' normConst = {self.priorDict[param][3]},'  # TODO This will need to increase for parametrisaton
                      f' n_regions = {n_regions}')

        return

    def set_prior(self, param, abund=False, name_param=None):

        # Read distribution configuration
        dist_name = self.priorDict[param][0]
        dist_loc, dist_scale = self.priorDict[param][1], self.priorDict[param][2]
        dist_norm, dist_reLoc = self.priorDict[param][3], self.priorDict[param][4]

        # Load the corresponding probability distribution
        probDist = getattr(pymc3, dist_name)

        if abund:
            priorFunc = probDist(name_param, dist_loc, dist_scale) * dist_norm + dist_reLoc

        elif probDist.__name__ in ['HalfCauchy']:  # These distributions only have one parameter
            priorFunc = probDist(param, dist_loc, shape=self.total_regions) * dist_norm + dist_reLoc

        elif probDist.__name__ == 'Uniform':
            # priorFunc = probDist(param, lower=dist_loc, upper=dist_scale) * dist_norm + dist_reLoc
            if param == 'logOH':
                priorFunc = pymc3.Bound(pymc3.Normal, lower=7.1, upper=9.1)('logOH', mu=8.0, sigma=1.0)
            if param == 'logU':
                priorFunc = pymc3.Bound(pymc3.Normal, lower=-4.0, upper=-1.5)('logU', mu=-2.75, sigma=1.5)
            if param == 'logNO':
                priorFunc = pymc3.Bound(pymc3.Normal, lower=-2.0, upper=0.0)('logNO', mu=-1.0, sigma=0.5)
        else:
            priorFunc = probDist(param, dist_norm, dist_scale, shape=self.total_regions) * dist_norm + dist_reLoc

        self.paramDict[param] = priorFunc

        return

    def inference_photoionization(self, OH, cHbeta, OH_err=0.10):

        # Define observable input
        inputGridFlux = np.log10(self.grid_emissionFluxes)
        inputGridErr = np.log10(1 + self.grid_emissionFluxErrs / self.grid_emissionFluxes)
        linesTensorLabels = np.array([f'{self.grid_LineLabels[i]}_Op' for i in range(self.grid_LineLabels.size)])

        for i, lineLabel in enumerate(self.grid_LineLabels):
            print(lineLabel, self.grid_emissionFluxes[i], self.grid_emissionFluxErrs[i], self.idx_analysis_lines[i])

        # Define the counters for loops
        linesRangeArray = np.arange(self.lineLabels.size)

        with pymc3.Model() as self.inferenModel:

            # Priors
            self.set_prior('Teff')
            self.set_prior('logU')
            # OH_err = pymc3.Normal('OH_err', mu=0, sigma=0.10)
            # OH_err_prior = np.random.normal('O_abund_err', mu=0, sigma=OH_err)
            # OH_prior = pymc3.Normal('OH', mu=OH, sigma=OH_err)
            # OH_err_prior = pymc3.Normal('OH_err', mu=OH, sigma=OH_err)
            OH_err_prior = np.random.normal(loc=OH, scale=OH_err)

            # Interpolation coord
            grid_coord = tt.stack([[self.paramDict['logU']], [self.paramDict['Teff']], [OH_err_prior]], axis=-1)

            for i in linesRangeArray:

                if self.idx_analysis_lines[i]:
                    # Declare line properties
                    lineLabel = self.lineLabels[i]
                    lineFlambda = self.lineFlambda[i]

                    # Line Flux
                    lineInt = self.gridInterp[lineLabel](grid_coord)[0][0]

                    # Line Intensity
                    lineFlux = lineInt - cHbeta * lineFlambda

                    # Inference
                    pymc3.Deterministic(linesTensorLabels[i], lineFlux)
                    print(lineLabel, inputGridFlux[i], inputGridErr[i])
                    pymc3.Normal(lineLabel, mu=lineFlux, sd=inputGridErr[i], observed=inputGridFlux[i])

            # Display simulation data
            displaySimulationData(self.inferenModel)

        return

    def photoionization_sampling(self, parameter_list):

        # Define observable input
        inputGridFlux = np.log10(self.emissionFluxes.astype(float))
        inputGridErr = self.emissionErr.astype(float) / self.emissionFluxes.astype(float) * log10_factor
        linesTensorLabels = np.array([f'{self.lineLabels[i]}_Op' for i in range(self.lineLabels.size)])

        # Define the counters for loops
        linesRangeArray = np.arange(self.lineLabels.size)

        with pymc3.Model() as self.inferenModel:

            # Priors
            for param in parameter_list:
                self.set_prior(param)

            # Interpolation coord
            grid_coord = tt.stack([[self.paramDict['logOH']], [self.paramDict['logU']], [self.paramDict['logNO']]],
                                  axis=-1)

            for i in linesRangeArray:
                # Declare line properties
                lineLabel = self.lineLabels[i]
                # lineFlambda = self.lineFlambda[i]

                # Line Flux
                lineInt = self.gridInterp[lineLabel](grid_coord)[0][0]

                # Line Intensity
                # lineFlux = lineInt - cHbeta * lineFlambda
                lineFlux = lineInt
                pymc3.Deterministic(linesTensorLabels[i], lineFlux)

                # Inference
                pymc3.Normal(lineLabel, mu=lineFlux, sd=inputGridErr[i], observed=inputGridFlux[i])

            # Display simulation data
            displaySimulationData(self.inferenModel)

        return

    def inference_model(self, fit_T_low=True, fit_T_high=True):

        # Container to store the synthetic line fluxes
        self.paramDict = {}  # FIXME do I need this one for loop inferences

        # for i, line_i in enumerate(obj1_model.lineLabels):
        #     print(f'{i} {line_i} {obj1_model.lineIons[i]}: {obj1_model.emissionFluxes[i]}, {obj1_model.lineFlambda[i]}')

        # Define observable input
        fluxTensor = tt.zeros(self.lineLabels.size)
        inputFlux = np.log10(self.emissionFluxes)
        inputFluxErr = np.log10(1 + self.emissionErr / self.emissionFluxes)

        if self.ionizationModels_Check:
            inputGridFlux = np.log10(self.grid_emissionFluxes)
            inputGridErr = np.log10(1 + self.grid_emissionFluxErrs / self.grid_emissionFluxes)
            linesTensorLabels = np.array([f'{self.grid_LineLabels[i]}_Op' for i in range(self.grid_LineLabels.size)])

        # Define the counters for loops
        linesRangeArray = np.arange(self.lineLabels.size)

        # Assign variable values
        self.paramDict['H1'] = 0.0

        with pymc3.Model() as self.inferenModel:

            # Declare model parameters priors
            self.set_prior('n_e')
            self.set_prior('cHbeta')

            # Establish model temperature structure
            self.temperature_selection(fit_T_low, fit_T_high)

            # Define grid interpolation variables
            emisCoord_low = tt.stack([[self.paramDict['T_low'][0]], [self.paramDict['n_e'][0]]], axis=-1)
            emisCoord_high = tt.stack([[self.paramDict['T_high'][0]], [self.paramDict['n_e'][0]]], axis=-1)

            # Establish model composition
            for ion in self.obsIons:
                if ion != 'H1':
                    self.set_prior(ion, abund=True, name_param=ion)

            if self.ionizationModels_Check:
                self.set_prior('Teff')
                self.set_prior('logU')

                O2_abund = tt.power(10, self.paramDict['O2'] - 12)
                O3_abund = tt.power(10, self.paramDict['O3'] - 12)
                OH = tt.log10(O2_abund + O3_abund) + 12

                grid_coord = tt.stack([[self.paramDict['logU']], [self.paramDict['Teff']], [OH]], axis=-1)

            # Loop through the lines to compute the synthetic fluxes
            for i in linesRangeArray:

                # Declare line properties
                lineLabel = self.lineLabels[i]
                lineIon = self.lineIons[i]
                lineFlambda = self.lineFlambda[i]

                # Compute emisivity for the corresponding ion temperature
                T_calc = emisCoord_high if self.idcs_highTemp_ions[i] else emisCoord_low
                line_emis = self.emisGridInterpFun[lineLabel](T_calc)

                # Declare fluorescence correction
                lineftau = 0.0

                # Compute line flux
                lineFlux_i = self.emtt.compute_flux(lineLabel,
                                                    line_emis[0][0],
                                                    self.paramDict['cHbeta'],
                                                    lineFlambda,
                                                    self.paramDict[lineIon],
                                                    lineftau,
                                                    O3=self.paramDict['O3'],
                                                    T_high=self.paramDict['T_high'])

                if self.idx_analysis_lines[i]:
                    # Line Flux
                    lineInt = self.gridInterp[lineLabel](grid_coord)

                    # Line Intensity
                    lineFlux = lineInt - self.paramDict['cHbeta'] * lineFlambda

                    # Inference
                    pymc3.Deterministic(linesTensorLabels[i], lineFlux)
                    Y_grid = pymc3.Normal(lineLabel, mu=lineFlux, sd=inputGridErr[i], observed=inputGridFlux[i])

                # Assign the new value in the tensor
                fluxTensor = storeValueInTensor(i, lineFlux_i[0], fluxTensor)

            # Store computed fluxes
            pymc3.Deterministic('calcFluxes_Op', fluxTensor)

            # Likelihood gas components
            Y_emision = pymc3.Normal('Y_emision', mu=fluxTensor, sd=inputFluxErr, observed=inputFlux)

            # Display simulation data
            displaySimulationData(self.inferenModel)

        # self.inferenModel.profile(self.inferenModel.logpt).summary()
        # self.inferenModel.profile(pymc3.gradient(self.inferenModel.logpt, self.inferenModel.vars)).summary()

        return

    def temperature_selection(self, fit_T_low=True, fit_T_high=True):

        if self.lowTemp_check and fit_T_low:
            self.set_prior('T_low')

            if self.highTemp_check:
                self.set_prior('T_high')
            else:
                self.paramDict['T_high'] = TOIII_from_TSIII_relation(self.paramDict['T_low'])

        else:
            if self.highTemp_check and fit_T_high:
                self.set_prior('T_high')

            self.paramDict['T_low'] = TOII_from_TOIII_relation(self.paramDict['T_high'], self.paramDict['n_e'])

        return

    def run_sampler(self, iterations, tuning, nchains=2, njobs=2):

        # ---------------------------- Launch model
        print('\n- Launching sampler')
        trace = pymc3.sample(iterations, tune=tuning, chains=nchains, cores=njobs, model=self.inferenModel)

        #  ---------------------------- Treat traces and store outputs
        model_params = []
        output_dict = {}
        traces_ref = np.array(trace.varnames)
        for param in traces_ref:

            # Exclude pymc3 variables
            if ('_log__' not in param) and ('interval' not in param):

                trace_array = trace[param]

                # Restore prior parametrisation
                if param in self.priorDict:

                    reparam0, reparam1 = self.priorDict[param][3], self.priorDict[param][4]
                    if param not in self.priorDict['logParams_list']:
                        trace_array = trace_array * reparam0 + reparam1
                    else:
                        trace_array = np.power(10, trace_array * reparam0 + reparam1)

                    model_params.append(param)
                    output_dict[param] = trace_array
                    trace.add_values({param: trace_array}, overwrite=True)

                # Line traces
                elif param.endswith('_Op'):

                    # Convert to natural scale
                    trace_array = np.power(10, trace_array)

                    if param == 'calcFluxes_Op':  # Flux matrix case
                        for i in range(trace_array.shape[0]):
                            output_dict[self.lineLabels[i]] = trace_array[:, i]
                        trace.add_values({'calcFluxes_Op': trace_array}, overwrite=True)
                    else:  # Individual line
                        ref_line = param[:-3]  # Not saving '_Op' extension
                        output_dict[ref_line] = trace_array
                        trace.add_values({param: trace_array}, overwrite=True)

                # None physical
                else:
                    model_params.append(param)
                    output_dict[param] = trace_array

        # ---------------------------- Save inputs
        inputs = {'line_list': self.lineLabels,
                  'line_fluxes': self.emissionFluxes,
                  'line_err': self.emissionErr,
                  'parameter_list': model_params}

        # ---------------------------- Store fit
        self.fit_results = {'model': self.inferenModel, 'trace': trace, 'inputs': inputs, 'outputs': output_dict}

        return

    def save_fit(self, output_address, ext_name='', output_format='pickle', user_header=None):

        if output_format == 'cfg':

            # Input data
            input_data = self.fit_results['inputs']
            sec_label = 'inputs' if ext_name == '' else f'{ext_name}_inputs'
            sec_dict = {}
            for i, lineLabel in enumerate(input_data['line_list']):
                lineFlux, lineErr = input_data['line_fluxes'][i], input_data['line_err'][i]
                sec_dict[lineLabel] = np.array([lineFlux, lineErr])
            sec_dict['parameter_list'] = input_data['parameter_list']
            parseConfDict(str(output_address), sec_dict, section_name=sec_label, clear_section=True)

            # Output data
            sec_label = 'outputs' if ext_name == '' else f'{ext_name}_outputs'
            sec_dict = {}
            for param in self.fit_results['inputs']['parameter_list']:
                param_trace = self.fit_results['outputs'][param]
                sec_dict[param] = np.array([np.mean(param_trace), np.std(param_trace)])
            parseConfDict(str(output_address), sec_dict, section_name=sec_label, clear_section=True)

            # Synthetic fluxes
            sec_label = 'synthetic_fluxes' if ext_name == '' else f'{ext_name}_synthetic_fluxes'
            sec_dict = {}
            for lineLabel in self.fit_results['inputs']['line_list']:
                line_trace = self.fit_results['outputs'][lineLabel]
                sec_dict[lineLabel] = np.array([np.mean(line_trace), np.std(line_trace)])
            parseConfDict(str(output_address), sec_dict, section_name=sec_label, clear_section=True)

        if output_format == 'pickle':
            with open(output_address, 'wb') as db_pickle:
                pickle.dump(self.fit_results, db_pickle)

        if output_format == 'fits':
            fits_db(output_address, model_db=self.fit_results, ext_name=ext_name, header=user_header)

