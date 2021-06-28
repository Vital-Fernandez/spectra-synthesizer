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

# Disable compute_test_value in theano zeros tensor
theano.config.compute_test_value = "ignore"


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

    def define_region(self, objLinesDF, ion_model=None, extinction_model=None, chemistry_model=None, minErr=0.02,
                      normLine='H1_4861A'):

        # Lines data
        normLine = normLine if ion_model is None else ion_model.normLine
        idcs_lines = (objLinesDF.index != normLine)
        self.lineLabels = objLinesDF.loc[idcs_lines].index.values
        self.lineIons = objLinesDF.loc[idcs_lines].ion.values
        self.emissionFluxes = objLinesDF.loc[idcs_lines].obsFlux.values
        self.emissionErr = objLinesDF.loc[idcs_lines].obsFluxErr.values

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

    def simulation_configuration(self, model_parameters, prior_conf_dict, n_regions=0, photo_ionization_grid=False,
                                 verbose=True, T_low_diag='S3_6312A', T_high_diag='O3_4363A'):

        # Priors configuration # TODO this one should be detected automatically
        for param in model_parameters:
            priorConf = prior_conf_dict[param + '_prior']
            self.priorDict[param] = priorConf
        self.priorDict['logParams_list'] = prior_conf_dict['logParams_list']

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
                      f'|| err/flux = {100 * self.emissionErr[i]/self.emissionFluxes[i]:.2f} %')

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
            priorFunc = probDist(param, lower=dist_loc, upper=dist_scale) * dist_norm + dist_reLoc

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

    def inference_backUp(self, include_reddening=True, include_Thigh_prior=True):

        # Container to store the synthetic line fluxes
        self.paramDict = {}  # FIXME do I need this one for loop inferences

        # Define observable input
        fluxTensor = tt.zeros(self.lineLabels.size)
        inputFlux = np.log10(self.emissionFluxes)
        inputFluxErr = np.log10(1 + self.emissionErr / self.emissionFluxes)

        # Define the counters for loops
        linesRangeArray = np.arange(self.lineLabels.size)

        # Assign variable values
        self.paramDict['H1'] = 0.0

        with pymc3.Model() as self.inferenModel:

            # Declare model parameters priors
            self.set_prior('n_e')
            self.set_prior('T_low')
            self.set_prior('cHbeta')

            # Establish model temperature structure
            if include_Thigh_prior:
                self.set_prior('T_high')
            else:
                self.paramDict['T_high'] = TOIII_from_TSIII_relation(self.paramDict['T_low'])
            emisCoord_low = tt.stack([[self.paramDict['T_low'][0]], [self.paramDict['n_e'][0]]], axis=-1)
            emisCoord_high = tt.stack([[self.paramDict['T_high'][0]], [self.paramDict['n_e'][0]]], axis=-1)

            # Establish model composition
            for ion in self.obsIons:
                if ion != 'H1':
                    self.set_prior(ion, abund=True, name_param=ion)

            # Loop through the lines to compute the synthetic fluxes
            for i in linesRangeArray:

                # Declare line properties
                lineLabel = self.lineLabels[i]
                lineIon = self.lineIons[i]
                lineFlambda = self.lineFlambda[i]

                # Compute emisivity for the corresponding ion temperature
                T_calc = emisCoord_high if self.highTemp_check[i] else emisCoord_low
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

                # Y_emision_i = pymc3.Normal(lineLabel, mu=lineFlux_i, sd=self.logFluxErr[i], observed=self.logFlux[i])

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

    def run_sampler(self, db_location, iterations, tuning, nchains=2, njobs=2, fits_store=False):

        # Confirm output folder
        db_location = Path(db_location)
        if db_location.parent.exists() is False:
            print('- WARNING: Output simulation folder does not exist')
            exit()

        # Launch model
        print('\n- Launching sampler')
        trace = pymc3.sample(iterations, tune=tuning, chains=nchains, cores=njobs, model=self.inferenModel)

        # Adapt the database to the prior configuration
        model_param = np.array(trace.varnames)

        for idx in range(model_param.size):

            param = model_param[idx]

            # Clean the extension to get the parametrisation
            ref_name = param
            for region in range(self.total_regions):
                ext_region = f'_{region}'
                ref_name = ref_name.replace(ext_region, '')

            # Apply the parametrisation
            if ref_name in self.priorDict:
                reparam0, reparam1 = self.priorDict[ref_name][3], self.priorDict[ref_name][4]

                if param not in self.priorDict['logParams_list']:
                    trace.add_values({param: trace[param] * reparam0 + reparam1}, overwrite=True)
                else:
                    trace.add_values({param: np.power(10, trace[param] * reparam0 + reparam1)}, overwrite=True)

            # Fluxes parametrisation
            if param == 'calcFluxes_Op':
                trace.add_values({param: np.power(10, trace[param])}, overwrite=True)

        # Convert line fluxes to natural scale
        linesTensorLabels = np.array([f'{self.lineLabels[i]}_Op' for i in range(self.lineLabels.size)])
        for i, lineTensorLabel in enumerate(linesTensorLabels):
            if np.any(lineTensorLabel in model_param):
                trace.add_values({lineTensorLabel: np.power(10, trace[lineTensorLabel])}, overwrite=True)

        # Save the database
        with open(db_location, 'wb') as trace_pickle:
            pickle.dump({'model': self.inferenModel, 'trace': trace}, trace_pickle)

        # Output configuration file
        configFileAddress = db_location.with_suffix('.txt')

        # Safe input data
        input_params = {'lineLabels_list': self.lineLabels, 'inputFlux_array': self.emissionFluxes,
                        'inputErr_array': self.emissionErr}
        parseConfDict(str(configFileAddress), input_params, section_name='Input_data')

        # Safe output variables data
        output_params = {}
        for parameter in trace.varnames:
            if ('_log__' not in parameter) and ('interval' not in parameter) and (parameter != 'calcFluxes_Op'):
                if '_Op' not in parameter:
                    trace_i = trace[parameter]
                    output_params[parameter] = [trace_i.mean(axis=0), trace_i.std(axis=0)]
                else:
                    trace_i = trace[parameter]
                    output_params[parameter] = [trace_i.mean(axis=0), trace_i.std(axis=0)]
        parseConfDict(str(configFileAddress), output_params, section_name='Fitting_results')

        # Output fluxes
        if 'calcFluxes_Op' in trace:
            trace_i = trace['calcFluxes_Op']
            output_params = {'outputFlux_array': trace_i.mean(axis=0), 'outputErr_array': trace_i.std(axis=0)}
            parseConfDict(str(configFileAddress), output_params, section_name='Simulation_fluxes')


