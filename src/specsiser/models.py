import pymc3
import numpy as np
import theano
from theano import tensor as tt, function
from .components.chemical_model import TOIII_from_TSIII_relation, TSIII_from_TOIII_relation, TOII_from_TOIII_relation

# Disable compute_test_value in theano zeros tensor
theano.config.compute_test_value = "ignore"

log10_factor = 1.0 / np.log(10)

def displaySimulationData(model, prior_dict):

    # Check test_values are finite
    print('\n-- Test points:')
    model_var = model.test_point
    for var in model_var:
        # displayText = '{} = {} (mu, std, norm)'.format(var, model_var[var])
        displayText = f'{var} = {model_var[var]}'
        prior_conf = '' if var not in prior_dict else f' ({prior_dict[var][0]}: mu = {prior_dict[var][1]}, std = {prior_dict[var][2]},' \
                                                    f' mu_norm = {prior_dict[var][3]}, std_norm = {prior_dict[var][4]})'

        print(displayText + prior_conf)

    # Checks log probability of random variables
    print('\n-- Log probability variable:')
    print(model.check_test_point())

    return


def storeValueInTensor(idx, value, tensor1D):
    return tt.inc_subtensor(tensor1D[idx], value)


class EmissionFluxModel:

    def __init__(self, label_list, ion_list):

        # Dictionary storing the functions corresponding to each emission line
        self.emFluxEqDict = {}

        # Dictionary storing the parameter list corresponding to each emission line for theano functions
        self.emFluxParamDict = {}

        # Loop through all the observed lines and assign a flux equation
        self.declare_emission_flux_functions(label_list, ion_list)

        # Define dictionary with flux functions with flexible number of arguments
        self.assign_flux_eqtt()

        return

    def declare_emission_flux_functions(self, label_list, ion_list):

        # TODO this could read the attribute fun directly
        # Dictionary storing emission flux equation database (log scale)
        emFluxDb_log = {'H1': self.ion_H1r_flux_log,
                        'He1': self.ion_He1r_flux_log,
                        'He2': self.ion_He2r_flux_log,
                        'metals': self.metals_flux_log,
                        'O2_7319A_b': self.ion_O2_7319A_b_flux_log}

        for i, lineLabel in enumerate(label_list):

            if ion_list[i] in ('H1', 'He1', 'He2'):
                self.emFluxEqDict[lineLabel] = emFluxDb_log[ion_list[i]]
                self.emFluxParamDict[lineLabel] = ['emis_ratio', 'cHbeta', 'flambda', 'abund', 'ftau']

            elif lineLabel == 'O2_7319A_b':
                self.emFluxEqDict[lineLabel] = emFluxDb_log['O2_7319A_b']
                self.emFluxParamDict[lineLabel] = ['emis_ratio', 'cHbeta', 'flambda', 'abund', 'ftau', 'O3', 'T_high']

            else:
                self.emFluxEqDict[lineLabel] = emFluxDb_log['metals']
                self.emFluxParamDict[lineLabel] = ['emis_ratio', 'cHbeta', 'flambda', 'abund', 'ftau']

        return

    def assign_lambda_function(self, linefunction, input_dict, label):

        # Single lines
        if label != 'O2_7319A_b':
            input_dict[label] = lambda emis_ratio, cHbeta, flambda, abund, ftau, kwargs: \
                linefunction(emis_ratio, cHbeta, flambda, abund, ftau)

        # Blended lines # FIXME currently only working for O2_7319A_b the only blended line
        else:
            input_dict[label] = lambda emis_ratio, cHbeta, flambda, abund, ftau, kwargs: \
                linefunction(emis_ratio, cHbeta, flambda, abund, ftau, kwargs['O3'], kwargs['T_high'])

        return

    def assign_flux_eqtt(self):

        for label, func in self.emFluxEqDict.items():
            self.assign_lambda_function(func, self.emFluxEqDict, label)

        return

    def compute_flux(self, lineLabel, emis_ratio, cHbeta, flambda, abund, ftau, **kwargs):
        return self.emFluxEqDict[lineLabel](emis_ratio, cHbeta, flambda, abund, ftau, kwargs)

    def ion_H1r_flux_log(self, emis_ratio, cHbeta, flambda, abund, ftau):
        return emis_ratio - flambda * cHbeta

    def ion_He1r_flux_log(self, emis_ratio, cHbeta, flambda, abund, ftau):
        return abund + emis_ratio - cHbeta * flambda

    def ion_He2r_flux_log(self, emis_ratio, cHbeta, flambda, abund, ftau):
        return abund + emis_ratio - cHbeta * flambda

    def metals_flux_log(self, emis_ratio, cHbeta, flambda, abund, ftau):
        return abund + emis_ratio - flambda * cHbeta - 12

    def ion_O2_7319A_b_flux_log(self, emis_ratio, cHbeta, flambda, abund, ftau, O3, T_high):
        col_ext = tt.power(10, abund + emis_ratio - flambda * cHbeta - 12)
        recomb = tt.power(10, O3 + 0.9712758 + tt.log10(tt.power(T_high/10000.0, 0.44)) - flambda * cHbeta - 12)
        return tt.log10(col_ext + recomb)


class EmissionTensors(EmissionFluxModel):

    def __init__(self, label_list, ion_list):

        self.emtt = None

        # Inherit Emission flux model
        print('\n- Compiling theano flux equations')
        EmissionFluxModel.__init__(self,  label_list, ion_list)

        # Loop through all the observed lines and assign a flux equation
        self.declare_emission_flux_functions(label_list, ion_list)

        # Compile the theano functions for all the input emission lines
        for label, func in self.emFluxEqDict.items():
            func_params = tt.dscalars(self.emFluxParamDict[label])
            self.emFluxEqDict[label] = function(inputs=func_params,
                                                outputs=func(*func_params),
                                                on_unused_input='ignore')

        # Assign function dictionary with flexible arguments
        self.assign_flux_eqtt()
        print('-- Completed\n')

        return


class PhotoIonizationModels(EmissionTensors):

    def __init__(self):

        # EmissionTensors.__init__()

        self.lineLabels = None
        self.lineIons = None
        self.emissionFluxes = None
        self.emissionErr = None
        self.lineFlambda = None
        self.obsIons = None

        self.prior_vars = {}
        self.priorDict = {}
        self.inferenModel = None

    def set_prior(self, param, abund_type=False, name_param=None):

        # Read distribution configuration
        dist_name = self.priorDict[param][0]
        dist_loc, dist_scale = self.priorDict[param][1], self.priorDict[param][2]
        dist_norm, dist_reLoc = self.priorDict[param][3], self.priorDict[param][4]

        # Load the corresponding probability distribution
        probDist = getattr(pymc3, dist_name)

        if abund_type:
            priorFunc = probDist(name_param, dist_loc, dist_scale) * dist_norm + dist_reLoc

        elif probDist.__name__ in ['HalfCauchy']:  # These distributions only have one parameter
            priorFunc = probDist(param, dist_loc, shape=self.total_regions) * dist_norm + dist_reLoc

        elif probDist.__name__ == 'Uniform':

            if param == 'logOH':
                priorFunc = pymc3.Bound(pymc3.Normal, lower=7.1, upper=9.1)('logOH', mu=8.0, sigma=1.0, testval=8.1)
            if param == 'logU':
                priorFunc = pymc3.Bound(pymc3.Normal, lower=-4.0, upper=-1.5)('logU', mu=-2.75, sigma=1.5, testval=-2.75)
            if param == 'logNO':
                priorFunc = pymc3.Bound(pymc3.Normal, lower=-2.0, upper=0.0)('logNO', mu=-1.0, sigma=0.5, testval=-1.0)

        else:
            priorFunc = probDist(param, dist_norm, dist_scale, shape=self.total_regions) * dist_norm + dist_reLoc

        self.prior_vars[param] = priorFunc

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
            grid_coord = tt.stack([[self.prior_vars['logU']], [self.prior_vars['Teff']], [OH_err_prior]], axis=-1)

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
            displaySimulationData(self.inferenModel, self.priorDict)

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
            grid_coord = tt.stack([[self.prior_vars['logOH']], [self.prior_vars['logU']], [self.prior_vars['logNO']]],
                                  axis=-1)

            for i in linesRangeArray:

                # Declare line properties
                lineLabel = self.lineLabels[i]
                lineFlambda = self.lineFlambda[i]
                lineInt = self.gridInterp[lineLabel](grid_coord)[0]

                # Line Intensity
                lineFlux = lineInt #- self.prior_vars['cHbeta'] * lineFlambda
                pymc3.Deterministic(linesTensorLabels[i], lineFlux)

                # Inference
                pymc3.Normal(lineLabel, mu=lineFlux, sd=inputGridErr[i], observed=inputGridFlux[i])

            # Display simulation data
            displaySimulationData(self.inferenModel, self.priorDict)

        return

    def inference_model(self, fit_T_low=True, fit_T_high=True):

        # Container to store the synthetic line fluxes
        self.prior_vars = {}  # FIXME do I need this one for loop inferences

        # Define observable input
        fluxTensor = tt.zeros(self.lineLabels.size)
        inputFlux = np.log10(self.emissionFluxes)
        inputFluxErr = np.log10(1 + self.emissionErr / self.emissionFluxes)

        if self.grid_check:
            inputGridFlux = np.log10(self.grid_emissionFluxes)
            inputGridErr = np.log10(1 + self.grid_emissionFluxErrs / self.grid_emissionFluxes)
            linesTensorLabels = np.array([f'{self.grid_LineLabels[i]}_Op' for i in range(self.grid_LineLabels.size)])

        # Define the counters for loops
        linesRangeArray = np.arange(self.lineLabels.size)

        # Assign variable values
        self.prior_vars['H1'] = 0.0

        with pymc3.Model() as self.inferenModel:

            # Declare model parameters priors
            self.set_prior('n_e')
            self.set_prior('cHbeta')

            # Establish model temperature structure
            self.temperature_selection(fit_T_low, fit_T_high)

            # Define grid interpolation variables
            emisCoord_low = tt.stack([[self.prior_vars['T_low'][0]], [self.prior_vars['n_e'][0]]], axis=-1)
            emisCoord_high = tt.stack([[self.prior_vars['T_high'][0]], [self.prior_vars['n_e'][0]]], axis=-1)

            # Establish model composition
            for ion in self.obsIons:
                if ion != 'H1':
                    self.set_prior(ion, abund_type=True, name_param=ion)

            if self.grid_check:
                self.set_prior('Teff')
                self.set_prior('logU')

                O2_abund = tt.power(10, self.prior_vars['O2'] - 12)
                O3_abund = tt.power(10, self.prior_vars['O3'] - 12)
                OH = tt.log10(O2_abund + O3_abund) + 12

                grid_coord = tt.stack([[self.prior_vars['logU']], [self.prior_vars['Teff']], [OH]], axis=-1)

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
                                                    self.prior_vars['cHbeta'],
                                                    lineFlambda,
                                                    self.prior_vars[lineIon],
                                                    lineftau,
                                                    O3=self.prior_vars['O3'],
                                                    T_high=self.prior_vars['T_high'])

                if self.idx_analysis_lines[i]:
                    # Line Flux
                    lineInt = self.gridInterp[lineLabel](grid_coord)

                    # Line Intensity
                    lineFlux = lineInt - self.prior_vars['cHbeta'] * lineFlambda

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
            displaySimulationData(self.inferenModel, self.priorDict)

        # self.inferenModel.profile(self.inferenModel.logpt).summary()
        # self.inferenModel.profile(pymc3.gradient(self.inferenModel.logpt, self.inferenModel.vars)).summary()

        return

    def temperature_selection(self, fit_T_low=True, fit_T_high=True):

        if self.lowTemp_check and fit_T_low:
            self.set_prior('T_low')

            if self.highTemp_check:
                self.set_prior('T_high')
            else:
                self.prior_vars['T_high'] = TOIII_from_TSIII_relation(self.prior_vars['T_low'])

        else:
            if self.highTemp_check and fit_T_high:
                self.set_prior('T_high')

            self.prior_vars['T_low'] = TOII_from_TOIII_relation(self.prior_vars['T_high'], self.prior_vars['n_e'])

        return

    def run_sampler(self, iterations, tuning, nchains=2, njobs=2, init='auto'):

        # ---------------------------- Launch model
        print('\n- Launching sampler')
        trace = pymc3.sample(iterations, tune=tuning, chains=nchains, cores=njobs, model=self.inferenModel, init=init,
                             progressbar=True)

        #  ---------------------------- Treat traces and store outputs
        model_params = []
        output_dict = {}
        traces_ref = np.array(trace.varnames)
        for param in traces_ref:

            # Exclude pymc3 variables
            if ('_log__' not in param) and ('interval' not in param):

                trace_array = np.squeeze(trace[param]) # TODO why is this squeeze necesary...

                # Restore prior parametrisation
                if param in self.priorDict:

                    reparam0, reparam1 = self.priorDict[param][3], self.priorDict[param][4]
                    if 'logParams_list' in self.priorDict:
                        if param not in self.priorDict['logParams_list']:
                            trace_array = trace_array * reparam0 + reparam1
                        else:
                            trace_array = np.power(10, trace_array * reparam0 + reparam1)
                    else:
                        trace_array = trace_array * reparam0 + reparam1

                    model_params.append(param)
                    output_dict[param] = trace_array
                    trace.add_values({param: trace_array}, overwrite=True)

                # Line traces
                elif param.endswith('_Op'):

                    # Convert to natural scale
                    trace_array = np.power(10, trace_array)

                    if param == 'calcFluxes_Op':  # Flux matrix case
                        for i in range(trace_array.shape[1]):
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

