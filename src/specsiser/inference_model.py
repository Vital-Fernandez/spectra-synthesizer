import pymc3
import theano
import theano.tensor as tt
import numpy as np
import pickle
from pathlib import Path
from data_reading import parseConfDict
from data_printing import MCOutputDisplay
from physical_model.gasEmission_functions import storeValueInTensor
from physical_model.chemical_model import TOIII_TSIII_relation
from physical_model.gasEmission_functions import assignFluxEq2Label, gridInterpolatorFunction, EmissionFluxModel


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


class SpectraSynthesizer(MCOutputDisplay):

    def __init__(self):

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
        self.highTemp_check = None

        self.ftauCoef = None
        self.emisGridInterpFun = None

        self.total_regions = 1

    def define_region(self, objLinesDF, ion_model, extinction_model, chemistry_model, minErr=0.02):

        # Lines data
        self.lineLabels = objLinesDF.index.values
        self.lineIons = objLinesDF.ion.values
        self.emissionFluxes = objLinesDF.obsFlux.values
        self.emissionErr = objLinesDF.obsFluxErr.values
        print(objLinesDF.obsWave.values)
        self.lineFlambda = extinction_model.gasExtincParams(wave=objLinesDF.obsWave.values)
        self.emtt = EmissionFluxModel(self.lineLabels, self.lineIons)

        self.obsIons = chemistry_model.obsAtoms
        self.highTemp_check = chemistry_model.indcsHighTemp

        # Emissivity data
        emisGridDict = ion_model.emisGridDict
        self.ftauCoef = ion_model.ftau_coeffs
        self.emisGridInterpFun = gridInterpolatorFunction(emisGridDict, ion_model.tempRange, ion_model.denRange)

        # Establish minimum error on lines
        if minErr is not None:
            err_fraction = self.emissionErr / self.emissionFluxes
            idcs_smallErr = err_fraction < minErr
            self.emissionErr[idcs_smallErr] = minErr * self.emissionFluxes[idcs_smallErr]

        return

    def simulation_configuration(self, model_parameters, prior_conf_dict, n_regions=0, verbose=True):

        # Priors configuration
        for param in model_parameters:
            priorConf = prior_conf_dict[param + '_prior']
            self.priorDict[param] = priorConf
        self.priorDict['logParams_list'] = prior_conf_dict['logParams_list']

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

        # Load the corresponding probability distribution
        probDist = getattr(pymc3, self.priorDict[param][0])

        if abund:
            priorFunc = probDist(name_param, self.priorDict[param][1], self.priorDict[param][2]) \
                        * self.priorDict[param][3] + self.priorDict[param][4]

        elif probDist.__name__ in ['HalfCauchy']:  # These distributions only have one parameter
            priorFunc = probDist(param, self.priorDict[param][1], shape=self.total_regions) \
                        * self.priorDict[param][3] + self.priorDict[param][4]

        else:
            priorFunc = probDist(param, self.priorDict[param][1], self.priorDict[param][2], shape=self.total_regions) \
                        * self.priorDict[param][3] + self.priorDict[param][4]

        self.paramDict[param] = priorFunc

        return

    def inference_model(self, include_reddening=True, include_Thigh_prior=True):

        # Container to store the synthetic line fluxes
        self.paramDict = {}  # FIXME do I need this one for loop inferences

        # Define observable input
        fluxTensor = tt.zeros(self.lineLabels.size)
        inputFlux = np.log10(self.emissionFluxes)
        inputFluxErr = np.log10(1 + self.emissionErr / self.emissionFluxes)

        # Define the counters for loops
        linesRangeArray = np.arange(self.lineLabels.size)

        # Assign variable values
        self.paramDict['H1r'] = 0.0

        with pymc3.Model() as self.inferenModel:

            # Declare model parameters priors
            self.set_prior('n_e')
            self.set_prior('T_low')
            self.set_prior('cHbeta')

            # Establish model temperature structure
            if include_Thigh_prior:
                self.set_prior('T_high')
            else:
                self.paramDict['T_high'] = TOIII_TSIII_relation(self.paramDict['T_low'])
            emisCoord_low = tt.stack([[self.paramDict['T_low'][0]], [self.paramDict['n_e'][0]]], axis=-1)
            emisCoord_high = tt.stack([[self.paramDict['T_high'][0]], [self.paramDict['n_e'][0]]], axis=-1)

            # Establish model composition
            for ion in self.obsIons:
                if ion != 'H1r':
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

    def run_sampler(self, db_location, iterations, tuning, nchains=2, njobs=2):

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
                trace_i = trace[parameter]
                output_params[parameter] = [trace_i.mean(axis=0), trace_i.std(axis=0)]
        parseConfDict(str(configFileAddress), output_params, section_name='Fitting_results')

        # Output fluxes
        trace_i = trace['calcFluxes_Op']
        output_params = {'outputFlux_array': trace_i.mean(axis=0), 'outputErr_array': trace_i.std(axis=0)}
        parseConfDict(str(configFileAddress), output_params, section_name='Simulation_fluxes')


    # for param in param_list:
    #     for n_region in range(self.total_regions):
    #         param_key = f'{param}_{n_region}'
    #         if n_region == 0:
    #             self.__setattr__(param, self.region_data[param_key])
    #         else:
    #             self.__setattr__(param, np.append(self.__getattribute__(param), self.region_data[param_key]))
    #
    # for param in dictionary_list:
    #     for n_region in range(self.total_regions):
    #         param_key = f'{param}_{n_region}'
    #         if n_region == 0:
    #             self.__setattr__(param, self.region_data[param_key])
    #         else:
    #             current_dict = self.__getattribute__(param)
    #             current_dict.update(self.region_data[param_key])
    #             self.__setattr__(param, current_dict)