import pymc3
import theano.tensor as tt
from data_reading import save_MC_fitting, load_MC_fitting, parseObjData
from physical_model.gasEmission_functions import storeValueInTensor
from physical_model.chemical_model import TOIII_TSIII_relation
from physical_model.gasEmission_functions import calcEmFluxes, EmissionTensors, assignFluxEq2Label


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


class SpectraSynthesizer:

    def __init__(self):

        self.modelParameters = None
        self.priorDict = {}

        self.emissionFluxes = None
        self.emissionErr = None

        self.obsIons = None
        self.emissionCheck = False

        self.linesRangeArray = None

        self.pymc3Dist = {'Normal': pymc3.Normal, 'Lognormal': pymc3.Lognormal}

    def declare_model_data(self, objLinesDF, ion_model, extinction_model, chemistry_model, minErr = None, verbose = True):

        self.emissionCheck = True

        self.lineLabels = objLinesDF.index.values
        self.lineIons = objLinesDF.ion.values
        self.lineFlambda = extinction_model.gasExtincParams(wave=objLinesDF.obsWave.values)

        self.emissionFluxes = objLinesDF.obsFlux.values
        self.emissionErr = objLinesDF.obsFluxErr.values

        self.emisCoef = ion_model.emisCoeffs
        self.emisEq = ion_model.ionEmisEq_fit

        self.emtt = EmissionTensors()
        self.eqLabelArray = assignFluxEq2Label(self.lineLabels, self.lineIons)
        self.ftauCoef = ion_model.ftau_coeffs

        self.indcsLabelLines = chemistry_model.indcsLabelLines
        self.indcsIonLines = chemistry_model.indcsIonLines['He1r']
        self.highTemp_check = chemistry_model.indcsHighTemp

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
                lineFlux = ' flux = {:.3f} +/- {:.3f} || err % = {:.3f}'.format(self.emissionFluxes[i], self.emissionErr[i],
                                                                                self.emissionErr[i] / self.emissionFluxes[i])
                print(lineReference + lineFlux)



                # warnLine = '{}'.format('|| WARNING obsLineErr = {:.4f}'.format(lineErr[i]) if lineErr[i] != lineFitErr[i] else '')
                # displayText = '{} flux = {:.4f} +/- {:.4f} || err % = {:.5f} {}'.format(lineLabels[i], lineFluxes[i],
                #                                                                         lineFitErr[i],
                #                                                                         lineFitErr[i] / lineFluxes[i],
                #                                                                         warnLine)
                # print(displayText)

        return

    def priors_configuration(self, model_parameters, prior_conf_dict, verbose=True):

        self.modelParameters = model_parameters

        if verbose:
            print(f'\n- Priors configuration ({len(model_parameters)} parameters):')

        for param in self.modelParameters:
            priorConf = prior_conf_dict[param + '_prior']
            self.priorDict[param] = priorConf

            # Display prior configuration
            if verbose:
                print(f'-- {param} {priorConf[0]} dist : mu = {priorConf[1]}, std = {priorConf[2]}, normConst = {priorConf[3]}')

        return

    def set_prior(self, param):
        return self.pymc3Dist[self.priorDict[0]](param, self.priorDict[1], self.priorDict[2]) * self.priorDict[3]

    def inference_model_emission(self, flux, equations, include_reddening=True, include_Thigh_prior=True):

        # Container to store the synthetic line fluxes
        fluxTensor = tt.zeros(self.lineLabels.size)

        with pymc3.Model() as model:

            # Gas priors
            n_e = self.set_prior('n_e')
            T_low = self.set_prior('T_low')
            cHbeta = self.set_prior('cHbeta')
            T_high = self.set_prior('T_high') if include_Thigh_prior else TOIII_TSIII_relation(T_low)

            # Abundance priors
            abund_dict = {'H1r': 1.0}
            for ion in self.obsIons:
                abund_dict[ion] = self.set_prior(ion)

            # Specific transition priors
            tau = self.set_prior('tau') if self.elementCheck['Her1'] else 0.0

            # Loop through the lines and compute the synthetic fluxes
            for i in self.linesRangeArray:
                lineLabel = self.lineLabels[i]
                lineIon = self.lineIons[i]
                lineFlambda = self.lineFlambda[i]
                fluxEq = self.emtt.emFluxTensors[self.eqLabelArray[i]]
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
                                                HighTemp_check=self.highTemp_check.indcsHighTemp)

                # Assign the new value in the tensor
                fluxTensor = storeValueInTensor(i, lineFlux_i, fluxTensor)

            # Store computed fluxes
            pymc3.Deterministic('calcFluxes_Op', fluxTensor)

            # Likelihood gas components
            Y_emision = pymc3.Normal('Y_emision', mu=fluxTensor, sd=self.emissionErr, observed=self.emissionFluxes)

            # Display simulation data
            displaySimulationData(model)

        return model

    def run_sampler(self, simulation_mame, db_address, iterations, tuning, normContants):

        # Select the model
        model = self.emission_model()

        # Launch model
        print('\n- Launching sampler')
        trace = pymc3.sample(iterations, tune=tuning, nchains=2, njobs=1, model=model)

        # Save the database
        save_MC_fitting(db_address, trace, model)

        # Save mean and std from parameters into the object log
        store_params = {}
        for parameter in trace.varnames:
            if ('_log__' not in parameter) and ('interval' not in parameter):
                trace_norm = normContants[parameter] if parameter in normContants else 1.0
                trace_i = trace_norm * trace[parameter]
                store_params[parameter] = [trace_i.mean(), trace_i.std()]
        parseObjData(self.configFile, self.objName + '_results', store_params)
