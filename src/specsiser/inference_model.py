import pymc3
import theano
import theano.tensor as tt
import numpy as np
import pickle
from data_reading import parseConfDict
from data_printing import MCOutputDisplay
from physical_model.gasEmission_functions import storeValueInTensor
from physical_model.chemical_model import TOIII_TSIII_relation
from physical_model.gasEmission_functions import calcEmFluxes_Eq, calcEmFluxes_Grid, EmissionTensors,\
    assignFluxEq2Label, gridInterpolatorFunction, EmissionFluxModel

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

        self.modelParameters = None
        self.priorDict = {}  # TODO This one must include the different coordinates
        self.paramDict = {}

        self.emissionFluxes = None
        self.emissionErr = None

        self.obsIons = None  # TODO Assign later
        self.emissionCheck = False

        self.linesRangeArray = None  # TODO Assign later

        self.total_regions = 0
        self.region_data = {}
        self.region_vector = np.array([], dtype=int)

    def define_region(self, objLinesDF, ion_model, extinction_model, chemistry_model, n_region=0, minErr=None,
                      verbose=True):

        ext_region = f'_{n_region}'
        n_lines = objLinesDF.index.size
        self.region_data[f'lineLabels_{n_region}'] = objLinesDF.index.values
        self.region_data[f'lineIons_{n_region}'] = objLinesDF.ion.values
        self.region_data[f'emissionFluxes_{n_region}'] = objLinesDF.obsFlux.values
        self.region_data[f'emissionErr_{n_region}'] = objLinesDF.obsFluxErr.values

        self.region_data[f'lineFlambda_{n_region}'] = extinction_model.gasExtincParams(wave=objLinesDF.obsWave.values)

        self.region_data[f'emisCoef_{n_region}'] = ion_model.emisCoeffs
        self.region_data[f'emisEq_{n_region}'] = ion_model.ionEmisEq_fit
        self.region_data[f'ftauCoef_{n_region}'] = ion_model.ftau_coeffs

        # TODO establish a loop to store the properties
        self.region_data[f'emisGridDict_{n_region}'] = ion_model.emisGridDict

        self.region_data[f'indcsLabelLines_{n_region}'] = chemistry_model.indcsLabelLines
        self.region_data[f'indcsIonLines_{n_region}'] = chemistry_model.indcsIonLines
        self.region_data[f'highTemp_check_{n_region}'] = chemistry_model.indcsHighTemp

        self.region_data[f'obsIons_{n_region}'] = chemistry_model.obsAtoms

        self.total_regions += 1
        self.region_vector = np.append(self.region_vector, np.ones(n_lines, dtype=int) * n_region)

        # TODO this should update for different regions
        # Compile exoplanet interpolator functions so they can be used wit numpy
        self.emisGridInterpFun = gridInterpolatorFunction(ion_model.emisGridDict, ion_model.tempRange, ion_model.denRange)

        return

    def set_simulation_variables(self, minErr=None, verbose=True):

        param_list = ['lineLabels', 'lineIons', 'emissionFluxes', 'emissionErr', 'lineFlambda', 'lineFlambda',
                      'highTemp_check', 'obsIons']

        dictionary_list = ['emisCoef', 'emisEq', 'ftauCoef', 'emisEq', 'emisGridDict']

        combine_dict_list = ['indcsLabelLines', 'indcsIonLines']

        self.emissionCheck = True

        for param in param_list:
            for n_region in range(self.total_regions):
                param_key = f'{param}_{n_region}'
                if n_region == 0:
                    self.__setattr__(param, self.region_data[param_key])
                else:
                    self.__setattr__(param, np.append(self.__getattribute__(param), self.region_data[param_key]))

        for param in dictionary_list:
            for n_region in range(self.total_regions):
                param_key = f'{param}_{n_region}'
                if n_region == 0:
                    self.__setattr__(param, self.region_data[param_key])
                else:
                    current_dict = self.__getattribute__(param)
                    current_dict.update(self.region_data[param_key])
                    self.__setattr__(param, current_dict)

        self.linesRangeArray = np.arange(self.lineLabels.size)
        self.eqLabelArray = assignFluxEq2Label(self.lineLabels, self.lineIons)

        # Vector with local obsserved abundances
        self.abundObjArray = np.array([]).astype(str)
        for n_region in range(self.total_regions):
            ext_region = f'_{n_region}'
            region_Ions = self.region_data[f'obsIons_{n_region}'].astype(str)
            self.abundObjArray = np.append(self.abundObjArray, np.core.defchararray.add(region_Ions, ext_region))

        # Vector with array of local ions
        self.lineLocalIon = np.empty(self.region_vector.size, dtype='U8')
        for i in range(self.region_vector.size):
            idx_region = int(self.region_vector[i])
            ext_region = f'_{idx_region}'
            self.lineLocalIon[i] = self.lineIons[i] + ext_region

        # Determine the line indeces
        # This dictionary of lines has a boolean indexing of the lines belonging to it
        self.indcsLabelLines = {}  # TODO in here we are just overwritting
        for line in np.unique(self.lineLabels):
            self.indcsLabelLines[line] = (self.lineLabels == line)

        # Determine the lines belonging to observed ions
        # This dictionary of ions has a boolean indexing of the lines belonging to it
        self.indcsIonLines = {}  # TODO in here should I use local?
        for ion in np.unique(self.lineIons):
            self.indcsIonLines[ion] = (self.lineIons == ion)

        # Dictionary with simulation abundances
        self.abund_dict = {}
        for i in range(self.total_regions):
            hydrogen_key = 'H1r_' + str(i)
            self.abund_dict[hydrogen_key] = 1.0

        # Establish minimum error on lines: # TODO Should this operation be at the point we import the fluxes?
        if minErr is not None:
            err_fraction = self.emissionErr / self.emissionFluxes
            idcs_smallErr = err_fraction < minErr
            self.emissionErr[idcs_smallErr] = minErr * self.emissionFluxes[idcs_smallErr]


        # Load flux tensors
        self.emtt = EmissionFluxModel(self.lineLabels, self.lineIons)

        if verbose:
            print(f'\n- Input lines ({self.lineLabels.size})')
            for i in range(self.lineLabels.size):
                lineReference = f'-- {self.lineLabels[i]} ({self.lineIons[i]}) '
                lineFlux = ' flux = {:.4f} +/- {:.4f} || err % = {:.4f}'.format(self.emissionFluxes[i],
                                                                                self.emissionErr[i],
                                                                                self.emissionErr[i] /
                                                                                self.emissionFluxes[i])
                print(lineReference + lineFlux)

        return


    def simulation_configuration(self, model_parameters, prior_conf_dict, n_regions=0, verbose=True):

        # Combine regions data
        self.set_simulation_variables(minErr=0.02)

        # Simulation prios configuration # TODO we are not using these ones
        self.modelParameters = model_parameters

        if verbose:
            print(f'\n- Priors configuration ({len(model_parameters)} parameters):')

        for param in self.modelParameters:
            priorConf = prior_conf_dict[param + '_prior']
            self.priorDict[param] = np.append(priorConf, n_regions)  # TODO update this for a flexible input of regions

            # Display prior configuration
            if verbose:
                print(f'-- {param} {self.priorDict[param][0]} dist : mu = {self.priorDict[param][1]}, '
                      f'std = {self.priorDict[param][2]}, normConst = {self.priorDict[param][3]},'  # TODO This will need to increase for parametrisaton
                      f' n_regions = {n_regions}')

        # Add parameters sampled in log scale
        self.priorDict['logParams_list'] = prior_conf_dict['logParams_list']

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

        return priorFunc

    def set_prior_backUp(self, param, abund=False, name_param=None):

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

        return priorFunc

    def inference_emisEq_model(self, include_reddening=True, include_Thigh_prior=True):

        # Container to store the synthetic line fluxes
        fluxTensor = tt.zeros(self.lineLabels.size)

        with pymc3.Model() as self.inferenModel:

            # Gas priors
            n_e = self.set_prior('n_e')
            T_low = self.set_prior('T_low')
            cHbeta = self.set_prior('cHbeta')  # TODO add the a mechanism to preload a reddening
            T_high = self.set_prior('T_high') if include_Thigh_prior else TOIII_TSIII_relation(T_low)

            # Abundance priors
            for idx in range(self.obsIons.size):
                ion = self.obsIons[idx]
                if ion != 'H1r':  # TODO check right place to exclude the hydrogen atoms
                    self.abund_dict[self.abundObjArray[idx]] = self.set_prior(self.obsIons[idx], abund=True,
                                                                              name_param=self.abundObjArray[idx])

            # Specific transition priors
            tau = self.set_prior('tau') if 'He1r' in self.obsIons else 0.0

            # Loop through the lines and compute the synthetic fluxes
            for i in self.linesRangeArray:
                i_region = self.region_vector[i]
                lineLabel = self.lineLabels[i]
                lineIonRef = self.lineIons[i]
                lineIon = self.lineLocalIon[i]  # TODO warning this is local
                lineFlambda = self.lineFlambda[i]
                fluxEq = self.emtt.emFlux_ttMethods[self.eqLabelArray[i]]
                emisCoef = self.emisCoef[lineLabel]
                emisEq = self.emisEq[lineLabel]

                lineFlux_i = calcEmFluxes_Eq(T_low[i_region], T_high[i_region], n_e[i_region], cHbeta[i_region],
                                             tau[i_region],
                                             self.abund_dict,
                                             i, lineLabel, lineIon, lineFlambda,
                                             fluxEq=fluxEq,
                                             ftau_coeffs=self.ftauCoef,
                                             emisCoeffs=emisCoef,
                                             emis_func=emisEq,
                                             indcsLabelLines=self.indcsLabelLines,
                                             He1r_check=self.indcsIonLines['He1r'],
                                             HighTemp_check=self.highTemp_check,
                                             region_ext=f'_{i_region}')

                # Assign the new value in the tensor
                fluxTensor = storeValueInTensor(i, lineFlux_i, fluxTensor)

            # Store computed fluxes
            pymc3.Deterministic('calcFluxes_Op', fluxTensor)

            # Likelihood gas components
            Y_emision = pymc3.Normal('Y_emision', mu=fluxTensor, sd=self.emissionErr, observed=self.emissionFluxes)

            # Display simulation data
            displaySimulationData(self.inferenModel)

        return

    def inference_emisGrid_model(self, include_reddening=True, include_Thigh_prior=True):

        # Container to store the synthetic line fluxes
        fluxTensor = tt.zeros(self.lineLabels.size)

        self.logFlux = np.log10(self.emissionFluxes)
        self.logFluxErr = np.log10(1 + self.emissionErr / self.emissionFluxes)
        self.paramDict = {}  # FIXME do I need this one for loop inferences

        with pymc3.Model() as self.inferenModel:

            # Declare model parameters priors
            self.paramDict['n_e'] = self.set_prior('n_e')
            self.paramDict['T_low'] = self.set_prior('T_low')
            self.paramDict['cHbeta'] = self.set_prior('cHbeta')

            # Establish model temperature structure
            if include_Thigh_prior:
                self.paramDict['T_high'] = self.set_prior('T_high')
            else:
                self.paramDict['T_high'] = TOIII_TSIII_relation(self.paramDict['T_low'])
            emisCoord_low = tt.stack([[self.paramDict['T_low'][0]], [self.paramDict['n_e'][0]]], axis=-1)
            emisCoord_high = tt.stack([[self.paramDict['T_high'][0]], [self.paramDict['n_e'][0]]], axis=-1)

            # Establish model composition
            for ion in self.obsIons:
                if ion != 'H1r':
                    self.paramDict[ion] = self.set_prior(ion, abund=True, name_param=ion)
                else:
                    self.paramDict[ion] = 0.0

            # Loop through the lines to compute the synthetic fluxes
            for i in self.linesRangeArray:

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


                # Assign the new value in the tensor
                fluxTensor = storeValueInTensor(i, lineFlux_i[0], fluxTensor)
                # tt.inc_subtensor(fluxTensor[i], lineFlux_i)
                #                       x             y

            # Store computed fluxes
            pymc3.Deterministic('calcFluxes_Op', fluxTensor)

            # Likelihood gas components
            Y_emision = pymc3.Normal('Y_emision', mu=fluxTensor, sd=self.logFluxErr, observed=self.logFlux)

            # Display simulation data
            displaySimulationData(self.inferenModel)

        return

    def inference_emisGrid_model_backUp(self, include_reddening=True, include_Thigh_prior=True):

        # Container to store the synthetic line fluxes
        fluxTensor = tt.zeros(self.lineLabels.size)

        self.logFlux = np.log10(self.emissionFluxes)
        self.logFluxErr = np.log10(1 + self.emissionErr / self.emissionFluxes)

        with pymc3.Model() as self.inferenModel:

            # Gas priors
            n_e = self.set_prior('n_e')
            T_low = self.set_prior('T_low')
            cHbeta = self.set_prior('cHbeta')  # TODO add the a mechanism to preload a reddening
            T_high = self.set_prior('T_high') if include_Thigh_prior else TOIII_TSIII_relation(T_low)

            # TODO very inefficient for multiple regions?
            emisCoord_low = tt.stack([[T_low[0]], [n_e[0]]], axis=-1)
            emisCoord_high = tt.stack([[T_high[0]], [n_e[0]]], axis=-1)

            # Abundance priors
            for idx in range(self.obsIons.size):
                ion = self.obsIons[idx]
                if ion != 'H1r':  # TODO check right place to exclude the hydrogen atoms
                    self.abund_dict[self.abundObjArray[idx]] = self.set_prior(self.obsIons[idx], abund=True,
                                                                              name_param=self.abundObjArray[idx])

            # Specific transition priors
            if 'He1r' not in self.indcsIonLines:
                self.indcsIonLines['He1r'] = np.zeros(self.region_vector.size,
                                                      dtype=bool)  # TODO nasty trick will need to remake
            tau = self.set_prior('tau') if 'He1r' in self.obsIons else np.zeros(self.region_vector.size)

            if 'O2_7319A_b' not in self.indcsLabelLines:
                self.indcsLabelLines['O2_7319A_b'] = np.zeros(self.region_vector.size, dtype=bool)

            # Loop through the lines and compute the synthetic fluxes
            for i in self.linesRangeArray:
                i_region = self.region_vector[i]
                lineLabel = self.lineLabels[i]
                lineIonRef = self.lineIons[i]
                lineIon = self.lineLocalIon[i]  # TODO warning this is local
                lineFlambda = self.lineFlambda[i]
                fluxEq = self.emtt.emFlux_ttMethods[self.eqLabelArray[i]]
                emisInter = self.emisGridDict[lineLabel]

                lineFlux_i = calcEmFluxes_Grid(T_low[i_region], T_high[i_region], n_e[i_region], emisCoord_low,
                                               emisCoord_high,
                                               cHbeta[i_region], tau[i_region], self.abund_dict,
                                               i, lineLabel, lineIon, lineFlambda,
                                               emisInter, fluxEq, self.ftauCoef,
                                               self.indcsLabelLines, self.indcsIonLines['He1r'],
                                               self.highTemp_check,
                                               region_ext=f'_{i_region}')

                # Assign the new value in the tensor
                fluxTensor = storeValueInTensor(i, lineFlux_i, fluxTensor)

            # Store computed fluxes
            pymc3.Deterministic('calcFluxes_Op', fluxTensor)

            # Likelihood gas components
            Y_emision = pymc3.Normal('Y_emision', mu=fluxTensor, sd=self.logFluxErr, observed=self.logFlux)

            # Display simulation data
            displaySimulationData(self.inferenModel)

        return

    def inference_IFUmodel_emission_backUp(self, include_reddening=True, include_Thigh_prior=True):

        # Container to store the synthetic line fluxes
        fluxTensor = tt.zeros(self.lineLabels.size)

        with pymc3.Model() as self.inferenModel:

            # Gas priors
            n_e = self.set_prior('n_e')
            T_low = self.set_prior('T_low')
            cHbeta = self.set_prior('cHbeta')  # TODO add the a mechanism to preload a reddening
            T_high = self.set_prior('T_high') if include_Thigh_prior else TOIII_TSIII_relation(T_low)

            # Abundance priors
            for idx in range(self.obsIons.size):
                ion = self.obsIons[idx]
                if ion != 'H1r':  # TODO check right place to exclude the hydrogen atoms
                    self.abund_dict[self.abundObjArray[idx]] = self.set_prior(self.obsIons[idx], abund=True,
                                                                              name_param=self.abundObjArray[idx])

            # Specific transition priors
            tau = self.set_prior('tau') if 'He1r' in self.obsIons else 0.0

            # Loop through the lines and compute the synthetic fluxes
            for i in self.linesRangeArray:
                i_region = self.region_vector[i]
                lineLabel = self.lineLabels[i]
                lineIonRef = self.lineIons[i]
                lineIon = self.lineLocalIon[i]  # TODO warning this is local
                lineFlambda = self.lineFlambda[i]
                fluxEq = self.emtt.emFlux_ttMethods[self.eqLabelArray[i]]
                emisCoef = self.emisCoef[lineLabel]
                emisEq = self.emisEq[lineLabel]

                lineFlux_i = calcEmFluxes_IFU(T_low[i_region], T_high[i_region], n_e[i_region], cHbeta[i_region],
                                              tau[i_region],
                                              self.abund_dict,
                                              i, lineLabel, lineIon, lineFlambda,
                                              fluxEq=fluxEq,
                                              ftau_coeffs=self.ftauCoef,
                                              emisCoeffs=emisCoef,
                                              emis_func=emisEq,
                                              indcsLabelLines=self.indcsLabelLines,
                                              He1r_check=self.indcsIonLines['He1r'],
                                              HighTemp_check=self.highTemp_check,
                                              idx_region=i_region)

                # Assign the new value in the tensor
                fluxTensor = storeValueInTensor(i, lineFlux_i, fluxTensor)

            # Store computed fluxes
            pymc3.Deterministic('calcFluxes_Op', fluxTensor)

            # Likelihood gas components
            Y_emision = pymc3.Normal('Y_emision', mu=fluxTensor, sd=self.emissionErr, observed=self.emissionFluxes)

            # Display simulation data
            displaySimulationData(self.inferenModel)

        return

    def run_sampler(self, db_location, iterations, tuning, nchains=2, njobs=2):

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

    def load_sampler_results(self, db_location, conf_file=None, n_regions=1):

        # Load the .db file
        with open(db_location, 'rb') as trace_restored:
            db = pickle.load(trace_restored)

        # Restore parameters data
        inferenModel, trace = db['model'], db['trace']

        # Save mean and std from parameters into the object log save the database and store the data
        store_params = {}
        for parameter in trace.varnames:
            if ('_log__' not in parameter) and ('interval' not in parameter):
                trace_i = trace[parameter]
                store_params[parameter] = [trace_i.mean(axis=0), trace_i.std(axis=0)]

        # Save results summary to configuration file
        if conf_file is not None:
            parseConfDict(conf_file, store_params, section_name='Fitting_results')

        return trace



   # def declare_model_data(self, objLinesDF, ion_model, extinction_model, chemistry_model, n_region, minErr=None,
    #                        verbose=True):
    #
    #     self.emissionCheck = True
    #
    #     self.lineLabels = objLinesDF.index.values
    #     self.lineIons = objLinesDF.ion.values
    #     self.lineFlambda = extinction_model.gasExtincParams(wave=objLinesDF.obsWave.values)
    #     self.linesRangeArray = np.arange(self.lineLabels.size)
    #
    #     self.emissionFluxes = objLinesDF.obsFlux.values
    #     self.emissionErr = objLinesDF.obsFluxErr.values
    #
    #     self.emisCoef = ion_model.emisCoeffs
    #     self.emisEq = ion_model.ionEmisEq_fit
    #     self.emisGridInterp = ion_model.emisGridInterp
    #
    #     self.emtt = EmissionFluxModel(self.lineLabels, self.lineIons) #EmissionTensors()
    #     self.eqLabelArray = assignFluxEq2Label(self.lineLabels, self.lineIons)
    #     self.ftauCoef = ion_model.ftau_coeffs
    #
    #     self.indcsLabelLines = chemistry_model.indcsLabelLines
    #     self.indcsIonLines = chemistry_model.indcsIonLines
    #     self.highTemp_check = chemistry_model.indcsHighTemp
    #     self.obsIons = chemistry_model.obsAtoms
    #
    #     # Establish minimum error on lines:
    #     # TODO Should this operation be at the point we import the fluxes?
    #     if minErr is not None:
    #         err_fraction = self.emissionErr / self.emissionFluxes
    #         idcs_smallErr = err_fraction < minErr
    #         self.emissionFluxes = minErr * self.obsLineFluxes[idcs_smallErr]
    #
    #     if verbose:
    #         print(f'\n- Input lines ({self.lineLabels.size})')
    #         for i in range(self.lineLabels.size):
    #             lineReference = f'-- {self.lineLabels[i]} ({self.lineIons[i]}) '
    #             lineFlux = ' flux = {:.3f} +/- {:.3f} || err % = {:.3f}'.format(self.emissionFluxes[i],
    #                                                                             self.emissionErr[i],
    #                                                                             self.emissionErr[i] /
    #                                                                             self.emissionFluxes[i])
    #             print(lineReference + lineFlux)
    #
    #             # warnLine = '{}'.format('|| WARNING obsLineErr = {:.4f}'.format(lineErr[i]) if lineErr[i] != lineFitErr[i] else '')
    #             # displayText = '{} flux = {:.4f} +/- {:.4f} || err % = {:.5f} {}'.format(lineLabels[i], lineFluxes[i],
    #             #                                                                         lineFitErr[i],
    #             #                                                                         lineFitErr[i] / lineFluxes[i],
    #             #                                                                         warnLine)
    #             # print(displayText)
    #
    #     return