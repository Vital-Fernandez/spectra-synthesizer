import pymc3
import theano
import theano.tensor as tt
import numpy as np
import pickle
from data_reading import parseConfDict
from data_printing import MCOutputDisplay
from physical_model.gasEmission_functions import storeValueInTensor
from physical_model.chemical_model import TOIII_TSIII_relation
from physical_model.gasEmission_functions import calcEmFluxes, calcEmFluxes_IFU, EmissionTensors, assignFluxEq2Label

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

        self.emissionFluxes = None
        self.emissionErr = None

        self.obsIons = None  # TODO Assign later
        self.emissionCheck = False

        self.linesRangeArray = None  # TODO Assign later

        self.total_regions = 0
        self.region_data = {}
        self.region_vector = np.array([], dtype=int)

        self.pymc3Dist = {'Normal': pymc3.Normal, 'Lognormal': pymc3.Lognormal}

    def define_region(self, objLinesDF, ion_model, extinction_model, chemistry_model, n_region, minErr=None, verbose=True):

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
        self.region_data[f'emisGridInterp_{n_region}'] = ion_model.emisGridInterp

        self.region_data[f'indcsLabelLines_{n_region}'] = chemistry_model.indcsLabelLines
        self.region_data[f'indcsIonLines_{n_region}'] = chemistry_model.indcsIonLines
        self.region_data[f'highTemp_check_{n_region}'] = chemistry_model.indcsHighTemp

        self.region_data[f'obsIons_{n_region}'] = chemistry_model.obsAtoms

        self.total_regions += 1
        self.region_vector = np.append(self.region_vector, np.ones(n_lines, dtype=int) * n_region)

    def set_simulation_variables(self, minErr = None, verbose = True):

        param_list = ['lineLabels', 'lineIons', 'emissionFluxes', 'emissionErr', 'lineFlambda', 'lineFlambda',
                      'highTemp_check', 'obsIons']

        dictionary_list = ['emisCoef', 'emisEq', 'ftauCoef', 'emisEq', 'emisGridInterp']

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
        self.indcsLabelLines = {} # TODO in here we are just overwritting
        for line in np.unique(self.lineLabels):
            self.indcsLabelLines[line] = (self.lineLabels == line)

        # Determine the lines belonging to observed ions
        # This dictionary of ions has a boolean indexing of the lines belonging to it
        self.indcsIonLines = {} # TODO in here should I use local?
        for ion in np.unique(self.lineIons):
            self.indcsIonLines[ion] = (self.lineIons == ion)

        # Load flux tensors
        self.emtt = EmissionTensors()

        # Dictionary with simulation abundances
        self.abund_dict = {}
        for i in range(self.total_regions):
            hydrogen_key = 'H1r_' + str(i)
            self.abund_dict[hydrogen_key] = 1.0

        for idx in range(len(self.lineLabels)):
            print(self.lineLabels[idx], self.lineIons[idx], self.lineLocalIon[idx])

        # Establish minimum error on lines: # TODO Should this operation be at the point we import the fluxes?
        if minErr is not None:
            err_fraction = self.emissionErr / self.emissionFluxes
            idcs_smallErr = err_fraction < minErr
            self.emissionFluxes = minErr * self.obsLineFluxes[idcs_smallErr]

        if verbose:
            print(f'\n- Input lines ({self.lineLabels.size})')
            for i in range(self.lineLabels.size):
                lineReference = f'-- {self.lineLabels[i]} ({self.lineIons[i]}) '
                lineFlux = ' flux = {:.3f} +/- {:.3f} || err % = {:.3f}'.format(self.emissionFluxes[i], self.emissionErr[i],
                                                                                self.emissionErr[i] / self.emissionFluxes[i])
                print(lineReference + lineFlux)

        return

    def declare_model_data(self, objLinesDF, ion_model, extinction_model, chemistry_model, n_region, minErr=None, verbose=True):

        self.emissionCheck = True

        self.lineLabels = objLinesDF.index.values
        self.lineIons = objLinesDF.ion.values
        self.lineFlambda = extinction_model.gasExtincParams(wave=objLinesDF.obsWave.values)
        self.linesRangeArray = np.arange(self.lineLabels.size)

        self.emissionFluxes = objLinesDF.obsFlux.values
        self.emissionErr = objLinesDF.obsFluxErr.values

        self.emisCoef = ion_model.emisCoeffs
        self.emisEq = ion_model.ionEmisEq_fit
        self.emisGridInterp = ion_model.emisGridInterp

        self.emtt = EmissionTensors()
        self.eqLabelArray = assignFluxEq2Label(self.lineLabels, self.lineIons)
        self.ftauCoef = ion_model.ftau_coeffs

        self.indcsLabelLines = chemistry_model.indcsLabelLines
        self.indcsIonLines = chemistry_model.indcsIonLines
        self.highTemp_check = chemistry_model.indcsHighTemp
        self.obsIons = chemistry_model.obsAtoms

        # Establish minimum error on lines:
        # TODO Should this operation be at the point we import the fluxes?
        if minErr is not None:
            err_fraction = self.emissionErr / self.emissionFluxes
            idcs_smallErr = err_fraction < minErr
            self.emissionFluxes = minErr * self.obsLineFluxes[idcs_smallErr]

        if verbose:
            print(f'\n- Input lines ({self.lineLabels.size})')
            for i in range(self.lineLabels.size):
                lineReference = f'-- {self.lineLabels[i]} ({self.lineIons[i]}) '
                lineFlux = ' flux = {:.3f} +/- {:.3f} || err % = {:.3f}'.format(self.emissionFluxes[i],
                                                                                self.emissionErr[i],
                                                                                self.emissionErr[i] /
                                                                                self.emissionFluxes[i])
                print(lineReference + lineFlux)

                # warnLine = '{}'.format('|| WARNING obsLineErr = {:.4f}'.format(lineErr[i]) if lineErr[i] != lineFitErr[i] else '')
                # displayText = '{} flux = {:.4f} +/- {:.4f} || err % = {:.5f} {}'.format(lineLabels[i], lineFluxes[i],
                #                                                                         lineFitErr[i],
                #                                                                         lineFitErr[i] / lineFluxes[i],
                #                                                                         warnLine)
                # print(displayText)

        return

    def simulation_configuration(self, model_parameters, prior_conf_dict, n_regions=1, verbose=True):

        # Combine regions data
        self.set_simulation_variables()

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
                      f'std = {self.priorDict[param][2]}, normConst = {self.priorDict[param][3]},' # TODO This will need to increase for parametrisaton
                      f' n_regions = {n_regions}')

        return

    def set_prior(self, param, abund = False, name_param=None):
        probDist = getattr(pymc3, self.priorDict[param][0])
        if abund:
            return probDist(name_param, self.priorDict[param][1], self.priorDict[param][2]) * self.priorDict[param][3]
        else:
            return probDist(param, self.priorDict[param][1], self.priorDict[param][2], shape=self.total_regions) * self.priorDict[param][3]

    def inference_model_emission(self, include_reddening=True, include_Thigh_prior=True):

        # Container to store the synthetic line fluxes
        fluxTensor = tt.zeros(self.lineLabels.size)

        with pymc3.Model() as self.inferenModel:

            # Gas priors
            n_e = self.set_prior('n_e')
            T_low = self.set_prior('T_low')
            cHbeta = self.set_prior('cHbeta')  # TODO add the a mechanism to preload a reddening
            T_high = self.set_prior('T_high') if include_Thigh_prior else TOIII_TSIII_relation(T_low)

            # Abundance priors
            abund_dict = {'H1r': 1.0}
            for ion in self.obsIons:
                if ion != 'H1r':  # TODO check right place to exclude the hydrogen atoms
                    abund_dict[ion] = self.set_prior(ion)

            # Specific transition priors
            tau = self.set_prior('tau') if 'He1r' in self.obsIons else 0.0

            # Loop through the lines and compute the synthetic fluxes
            for i in self.linesRangeArray:
                lineLabel = self.lineLabels[i]
                lineIon = self.lineIons[i]
                lineFlambda = self.lineFlambda[i]
                fluxEq = self.emtt.emFlux_ttMethods[self.eqLabelArray[i]]
                emisCoef = self.emisCoef[lineLabel]
                emisEq = self.emisEq[lineLabel]

                lineFlux_i = calcEmFluxes(T_low, T_high, n_e, cHbeta, tau, abund_dict,
                                          i, lineLabel, lineIon, lineFlambda,
                                          fluxEq=fluxEq,
                                          ftau_coeffs=self.ftauCoef,
                                          emisCoeffs=emisCoef,
                                          emis_func=emisEq,
                                          indcsLabelLines=self.indcsLabelLines,
                                          He1r_check=self.indcsIonLines['He1r'],
                                          HighTemp_check=self.highTemp_check)

                # Assign the new value in the tensor
                fluxTensor = storeValueInTensor(i, lineFlux_i, fluxTensor)

            # Store computed fluxes
            pymc3.Deterministic('calcFluxes_Op', fluxTensor)

            # Likelihood gas components
            Y_emision = pymc3.Normal('Y_emision', mu=fluxTensor, sd=self.emissionErr, observed=self.emissionFluxes)

            # Display simulation data
            displaySimulationData(self.inferenModel)

        return

    def inference_IFUmodel_emission(self, include_reddening=True, include_Thigh_prior=True):

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
                    self.abund_dict[self.abundObjArray[idx]] = self.set_prior(self.obsIons[idx], abund=True, name_param=self.abundObjArray[idx])

            # Specific transition priors
            tau = self.set_prior('tau') if 'He1r' in self.obsIons else 0.0

            # Loop through the lines and compute the synthetic fluxes
            for i in self.linesRangeArray:
                i_region = self.region_vector[i]
                lineLabel = self.lineLabels[i]
                lineIonRef = self.lineIons[i]
                lineIon = self.lineLocalIon[i] # TODO warning this is local
                lineFlambda = self.lineFlambda[i]
                fluxEq = self.emtt.emFlux_ttMethods[self.eqLabelArray[i]]
                emisCoef = self.emisCoef[lineLabel]
                emisEq = self.emisGridInterp[lineLabel]

                lineFlux_i = calcEmFluxes_IFU(T_low[i_region], T_high[i_region], n_e[i_region], cHbeta[i_region], tau[i_region],
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
                    self.abund_dict[self.abundObjArray[idx]] = self.set_prior(self.obsIons[idx], abund=True, name_param=self.abundObjArray[idx])

            # Specific transition priors
            tau = self.set_prior('tau') if 'He1r' in self.obsIons else 0.0

            # Loop through the lines and compute the synthetic fluxes
            for i in self.linesRangeArray:
                i_region = self.region_vector[i]
                lineLabel = self.lineLabels[i]
                lineIonRef = self.lineIons[i]
                lineIon = self.lineLocalIon[i] # TODO warning this is local
                lineFlambda = self.lineFlambda[i]
                fluxEq = self.emtt.emFlux_ttMethods[self.eqLabelArray[i]]
                emisCoef = self.emisCoef[lineLabel]
                emisEq = self.emisEq[lineLabel]

                lineFlux_i = calcEmFluxes_IFU(T_low[i_region], T_high[i_region], n_e[i_region], cHbeta[i_region], tau[i_region],
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

        #Adapt the database to the prior configuration
        print('HI')
        model_param = np.array(trace.varnames)
        prior_param = self.priorDict.keys()

        for idx in range(model_param.size):

            param = model_param[idx]

            # Clean the extension to get the parametrisation
            ref_name = param
            for region in range(self.total_regions):
                ext_region = f'_{region}'
                ref_name = ref_name.replace(ext_region, '')

            if ref_name in prior_param:
                reparam0 = self.priorDict[ref_name][3]
                print('--',idx, param, ref_name, reparam0, '\n')
                trace.add_values({param:trace[param] * reparam0}, overwrite=True)

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
                trace_norm = self.priorDict[parameter][3] if parameter in self.priorDict else 1.0
                trace_i = trace_norm * trace[parameter]
                store_params[parameter] = [trace_i.mean(axis=0), trace_i.std(axis=0)]

        # Save results summary to configuration file
        if conf_file is not None:
            parseConfDict(conf_file, store_params, section_name='Fitting_results')

        return trace
