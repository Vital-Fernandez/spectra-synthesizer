import os
import pymc3
import numpy as np
import pickle
import src.specsiser as ss
import theano.tensor as tt
from inference_model import displaySimulationData
from physical_model.gasEmission_functions import storeValueInTensor

# Use the default user folder to store the results
user_folder = os.path.join(os.path.expanduser('~'), '')
output_db = f'{user_folder}/interPolationTesting_fitting_db'

# Dictionary storing synthetic data
objParams = {}

# Declare artificial data for the emission line region
objParams['true_values'] = {'flux_hbeta': 5e-14,
                            'n_e': 150.0,
                            'T_low': 10000.0,
                            'T_high': ss.TOIII_TSIII_relation(10000.0),
                            'tau': 0.60,
                            'cHbeta': 0.08,
                            'H1r': 0.0,
                            'He1r': 0.070,
                            'He2r': 0.00088,
                            'O2': 7.80,
                            'O3': 8.05,
                            'N2': 5.84,
                            'S2': 5.48,
                            'S3': 6.36,
                            'Ar3': 5.72,
                            'Ar4': 5.06,
                            'err_lines': 0.02}

# Declare lines to simulate
objParams['input_lines'] = ['H1_4341A', 'H1_6563A',
                            'O3_4363A', 'O3_4959A', 'O3_5007A', 'N2_6548A', 'N2_6584A',
                            'S2_6716A', 'S2_6731A', 'S3_6312A', 'S3_9069A', 'S3_9531A']

# We use the default simulation configuration for the remaining settings
objParams.update(ss._default_cfg)

# We use the default lines database to generate the synthetic emission lines log for this simulation
linesLogPath = os.path.join(ss._literatureDataFolder, ss._default_cfg['lines_data_file'])

# Prepare dataframe with the observed lines labeled
objLinesDF = ss.import_emission_line_data(linesLogPath, include_lines=objParams['input_lines'])

# Declare extinction properties
objRed = ss.ExtinctionModel(Rv=objParams['R_v'],
                            red_curve=objParams['reddenig_curve'],
                            data_folder=objParams['external_data_folder'])

# Compute the flambda value for the input emission lines
lineFlambdas = objRed.gasExtincParams(wave=objLinesDF.obsWave.values)
lineWaves = objLinesDF.pynebCode.values

# Establish atomic data references
ftau_file_path = os.path.join(ss._literatureDataFolder, objParams['ftau_file'])
objIons = ss.IonEmissivity(ftau_file_path=ftau_file_path, tempGrid=objParams['temp_grid'],
                           denGrid=objParams['den_grid'])

# Define the dictionary with the pyneb ion objects
ionDict = objIons.get_ions_dict(np.unique(objLinesDF.ion.values))

# Compute the emissivity surfaces for the observed emission lines # TODO this database is not necesary since we duplicate the logs
objIons.computeEmissivityGrid(objLinesDF, ionDict, linesDb=ss._linesDb)

# Declare the paradigm for the chemical composition measurement
objChem = ss.DirectMethod()

# Tag the emission features for the chemical model implementation
objChem.label_ion_features(linesDF=objLinesDF, highTempIons=objParams['high_temp_ions_list'])

# Declare a dictionary with the synthetic abundances
abundDict = {}
for ion in objChem.obsAtoms:
    abundDict[ion] = objParams['true_values'][ion]

# We generate an object with the tensor emission functions
emtt = ss.EmissionTensors()

# Array with the equation labels
eqLabelArray = ss.assignFluxEq2Label(objLinesDF.index.values, objLinesDF.ion.values)

# Compute the emission line fluxes
lineFluxes = np.empty(lineWaves.size)
lineErrFlux = np.empty(lineWaves.size)
lineIons = objLinesDF.ion.values
lineLabels = objLinesDF.index.values
obsIons = np.unique(lineIons)

for i in np.arange(len(objLinesDF)):
    lineLabel = objLinesDF.iloc[i].name
    lineIon = objLinesDF.iloc[i].ion
    lineFlambda = lineFlambdas[i]
    lineWave = lineWaves[i]
    lineEmisInterp = objIons.emisGridInterp[lineLabel]
    line_ftau = None

    fluxEq = emtt.emFluxTensors[eqLabelArray[i]]
    # emisCoef = objIons.emisCoeffs[lineLabel]
    # emisEq = objIons.ionEmisEq_fit[lineLabel]

    Tlow, Thigh, ne, cHbeta, tau = objParams['true_values']['T_low'], objParams['true_values']['T_high'], \
                                   objParams['true_values']['n_e'], objParams['true_values']['cHbeta'], \
                                   objParams['true_values']['tau']

    Te_calc = Thigh if objChem.indcsHighTemp[i] else Tlow
    emisCoord = np.stack(([Te_calc], [ne]), axis=-1)

    lineEmis_true = np.log10(ionDict[lineIon].getEmissivity(Te_calc, ne, wave=lineWave) / ionDict['H1r'].getEmissivity(Te_calc, ne, wave=4861))
    lineEmis_interp = lineEmisInterp.evaluate(emisCoord).eval()

    # self.emisGridDict[line_label]
    lineFluxes[i] = fluxEq(lineEmis_true, cHbeta, lineFlambda, abundDict[lineIon], line_ftau, continuum=0.0)
    lineErrFlux[i] = lineFluxes[i] * objParams['lines_minimum_error']
    # lineFluxes[i] = ss.calcEmFluxes(Tlow, Thigh, ne, cHbeta, tau, abundDict,
    #                                 i, lineLabel, lineIon, lineFlambda,
    #                                 fluxEq=fluxEq, ftau_coeffs=objIons.ftau_coeffs, emisCoeffs=emisCoef, emis_func=emisEq,
    #                                 indcsLabelLines=objChem.indcsLabelLines, He1r_check=objChem.indcsIonLines['He1r'],
    #                                 HighTemp_check=objChem.indcsHighTemp)

    print(lineLabel, lineIon, lineFlambda, lineWaves[i], Te_calc, ne, lineEmis_true, lineEmis_interp[0][0], lineFluxes[i])


# Calculation with pymc3
fluxTensor = tt.zeros(lineWaves.size)
rangeLines = np.arange(lineWaves.size)
HighTemp_check = objChem.indcsHighTemp
interpDict = objIons.emisGridInterp

with pymc3.Model() as inferenModel:

    # Prior
    ne = 100.0 * pymc3.Lognormal('n_e', mu=0, sigma=1, shape=(1,))
    T_low = pymc3.Normal('T_low', mu=15000, sigma=2500, shape=(1,))
    T_high = pymc3.Normal('T_high', mu=15000, sigma=2500, shape=(1,))
    cHbeta_prior = pymc3.Lognormal('cHbeta', mu=0, sigma=1)

    emisCoord_low = tt.stack([T_low, ne], axis=-1)
    emisCoord_high = tt.stack([T_high, ne], axis=-1)

    # Abundance priors
    abund_dict = {'H1r': 1.0}
    for ion in obsIons:
        if ion != 'H1r':  # TODO check right place to exclude the hydrogen atoms
            abund_dict[ion] = pymc3.Normal(ion, mu=5, sigma=5)

    # Specific transition priors
    tau = 0.0

    for i in rangeLines:

        lineLabel = lineLabels[i]
        lineIon = lineIons[i]
        line_abund = abund_dict[lineIon]
        line_flambda = lineFlambdas[i]
        line_ftau = None

        # Appropriate data for the ion
        Te_calc = emisCoord_high if HighTemp_check[i] else emisCoord_low

        # Line Emissivity
        line_emis = interpDict[lineLabel].evaluate(Te_calc)
        fluxEq = emtt.emFlux_ttMethods[eqLabelArray[i]]

        #Compute flux
        lineFlux_i = fluxEq(line_emis[0][0], cHbeta_prior, line_flambda, line_abund, line_ftau, continuum=0.0)

        # Assign the new value in the tensor
        fluxTensor = storeValueInTensor(i, lineFlux_i, fluxTensor)

    # Store computed fluxes
    pymc3.Deterministic('calcFluxes_Op', fluxTensor)

    # Likelihood gas components
    Y_emision = pymc3.Normal('Y_emision', mu=fluxTensor, sd=lineErrFlux, observed=lineFluxes)

    # Display simulation data
    displaySimulationData(inferenModel)

    # Run the model
    trace = pymc3.sample(5000, tune=2000, chains=2, cores=1)

# Save the database
with open(output_db, 'wb') as trace_pickle:
    pickle.dump({'model': inferenModel, 'trace': trace}, trace_pickle)

# Load the .db file
with open(output_db, 'rb') as trace_restored:
    db = pickle.load(trace_restored)

# Restore parameters data
inferenModel, trace = db['model'], db['trace']
traces_dict = ss.load_MC_fitting(output_db)

# Declare sampler
obj1_model = ss.SpectraSynthesizer()

modelParams = list(traces_dict.keys())
modelParams.remove('calcFluxes_Op')

# Traces and Posteriors
print('-- Model parameters posterior diagram')
figure_file = f'{user_folder}interPolationTesting_ParamsPosteriors.txt'
obj1_model.tracesPosteriorPlot(modelParams, traces_dict, true_values=objParams)
obj1_model.savefig(figure_file, resolution=200)

# # Traces and Posteriors
# print('-- Model parameters corner diagram')
# figure_file = f'{user_folder}interPolationTesting_corner'
# obj1_model.corner_plot(modelParams, traces_dict, true_values=objParams)
# obj1_model.savefig(figure_file, resolution=200)

# print('-- Model parameters Traces-Posterior diagram')
# figure_file = f'{user_folder}interPolationTesting_paramsTracesPost'
# obj1_model.tracesPriorPostComp(modelParams, traces_dict, true_values=objParams)
# obj1_model.savefig(figure_file, resolution=200)