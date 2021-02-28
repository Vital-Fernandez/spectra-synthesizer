import numpy as np
import src.specsiser as sr
from pathlib import Path
import theano.tensor as tt
import pymc3
import matplotlib.pyplot as plt
import arviz as az
from src.specsiser.inference_model import displaySimulationData


# Data folder
user_folder = Path('D:/AstroModels')

# Compute the ionization parameter from the grids
gridLineDict, gridAxDict = sr.load_ionization_grid(log_scale=True)
gridInterp = sr.gridInterpolatorFunction(gridLineDict, gridAxDict['logU'],
                                                       gridAxDict['Teff'],
                                                       gridAxDict['OH'],
                                                       interp_type='cube')

for key, value in gridAxDict.items():
    print(key, value)

# -------------------------------------- Test 1 --------------------------------------------------

# # Generate synthetic object
# # temp_true, logU_true, OH_true = 53500.0, -1.70, 7.4
# temp_true, logU_true, OH_true = 53500.0, -1.82, 7.65
# coord_true = np.stack(([logU_true], [temp_true], [OH_true]), axis=-1)
#
# lineFluxes = np.zeros(len(gridInterp.keys()))
# for i, item in enumerate(gridInterp.items()):
#     lineLabel, lineInterpolator = item
#     lineFluxes[i] = np.power(10, lineInterpolator(coord_true).eval()[0][0])
# lineErr = lineFluxes * 0.05
#
# lineLabels = np.array(list(gridInterp.keys()))
# lineRange = np.arange(lineLabels.size)
# inputFlux = np.log10(lineFluxes)
# inputFluxErr = np.log10(1 + lineErr / lineFluxes)
#
# for i, lineLabel in enumerate(lineLabels):
#     print(f' - {lineLabel}: {lineFluxes[i]:0.3f} +/- {lineErr[i]:0.3f} => {inputFlux[i]:0.3f} +/- {inputFluxErr[i]:0.3f}')
#
# # Inference model
# with pymc3.Model() as model:
#
#     # Priors
#     OH = OH_true
#     Teff = pymc3.Uniform('Teff', lower=30000.0, upper=90000.0)
#     logU = pymc3.Uniform('logU', lower=-4, upper=-1.5)
#
#     # Interpolation coord
#     grid_coord = tt.stack([[logU], [Teff], [OH]], axis=-1)
#
#     # Loop throught
#     for i in lineRange:
#
#         # Line intensity
#         lineInt = gridInterp[lineLabels[i]](grid_coord)
#
#         # Inference
#         Y_emision = pymc3.Normal(lineLabels[i], mu=lineInt, sd=inputFluxErr[i], observed=inputFlux[i])
#
#     displaySimulationData(model)
#
#     trace = pymc3.sample(5000, tune=2000, chains=2, cores=1, model=model)
#
# print(trace)
# print(pymc3.summary(trace))
# az.plot_trace(trace)
# plt.show()
# az.plot_posterior(trace)
# plt.show()

# -------------------------------------- Test 2 --------------------------------------------------

# Load synthetic observation
linesLogAddress = user_folder / f'GridEmiss_region1of1_linesLog.txt'
simulationData_file = user_folder / f'GridEmiss_region1of1_config.txt'

# Load simulation parameters
objParams = sr.loadConfData(simulationData_file, group_variables=False)

# Load emission lines
merged_lines = {'O2_3726A_m': 'O2_3726A-O2_3729A', 'O2_7319A_m': 'O2_7319A-O2_7330A'}
objLinesDF = sr.import_emission_line_data(linesLogAddress, include_lines=objParams['inference_model_configuration']['input_lines'], exclude_lines=['S3_9069A'])

# Establish lines used in photo-ionization model
lineLabels = objLinesDF.index.values
lines_Grid = np.array(list(gridLineDict.keys()))
idx_analysis_lines = np.in1d(lineLabels, lines_Grid)
lineFluxes = objLinesDF.obsFlux.values
lineErr = objLinesDF.obsFluxErr.values
lineWaves = objLinesDF.wavelength.values

# Extinction parameters
objRed = sr.ExtinctionModel(Rv=objParams['simulation_properties']['R_v'],
                            red_curve=objParams['simulation_properties']['reddenig_curve'],
                            data_folder=objParams['data_location']['external_data_folder'])
lineFlambdas = objRed.gasExtincParams(wave=lineWaves)

lineRange = np.arange(lineLabels.size)
inputFlux = np.log10(lineFluxes)
inputFluxErr = np.log10(1 + lineErr / lineFluxes)

O2_abund = np.power(10, objParams['true_values']['O2'] - 12)
O3_abund = np.power(10, objParams['true_values']['O3'] - 12)
OH_true = np.log10(O2_abund + O3_abund) + 12
cHbeta = objParams['true_values']['cHbeta']

for i, lineLabel in enumerate(lineLabels):
    print(f' - {lineLabel}: {lineFluxes[i]:0.3f} +/- {lineErr[i]:0.3f} => {inputFlux[i]:0.3f} +/- {inputFluxErr[i]:0.3f}')

# Inference model
with pymc3.Model() as model:

    # Priors
    OH = OH_true
    Teff = pymc3.Uniform('Teff', lower=30000.0, upper=90000.0)
    logU = pymc3.Uniform('logU', lower=-4, upper=-1.5)

    # Interpolation coord
    grid_coord = tt.stack([[logU], [Teff], [OH]], axis=-1)

    # Loop throught
    for i in lineRange:

        if idx_analysis_lines[i]:

            # Line Flux
            lineInt = gridInterp[lineLabels[i]](grid_coord)

            # Line Intensity
            lineFlux = lineInt - cHbeta * lineFlambdas[i]

            # Inference
            Y_emision = pymc3.Normal(lineLabels[i], mu=lineInt, sd=inputFluxErr[i], observed=inputFlux[i])

    displaySimulationData(model)

    trace = pymc3.sample(5000, tune=2000, chains=2, cores=1, model=model)

print(trace)
print(pymc3.summary(trace))
az.plot_trace(trace)
plt.show()
az.plot_posterior(trace)
plt.show()
