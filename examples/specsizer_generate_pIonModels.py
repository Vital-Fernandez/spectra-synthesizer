import os
import numpy as np
import pandas as pd
import src.specsiser as sr
from src.specsiser.physical_model.photo_ionization_model import ModelGridWrapper
from physical_model.gasEmission_functions import gridInterpolatorFunction

# Use the default user folder to store the results
user_folder = 'D:\\AstroModels\\'

# We use the default simulation configuration as a base
objParams = sr._default_cfg.copy()

# # Set the paramter values
# objParams['true_values'] = {'OH': 7.4,
#                             'Teff': 45678.0,
#                             'logU': -2.15,
#                             'cHbeta': 0.5}

# Set the paramter values
objParams['true_values'] = {'OH': 7.4,
                            'Teff': 60000.0,
                            'logU': -2.00,
                            'cHbeta': 0.5}

# Load the ionization grids
gridLineDict, gridAxDict = sr.load_ionization_grid(log_scale=True)
lineInterpolator_dict = gridInterpolatorFunction(gridLineDict,
                                                 gridAxDict['logU'],
                                                 gridAxDict['Teff'],
                                                 gridAxDict['OH'],
                                                 interp_type='cube')

# Declare lines to simulate
lineLabels = np.array(['O2_3726A_m', 'He1_4471A', 'He2_4686A', 'O3_5007A', 'He1_5876A', 'S2_6716A_m'])
objParams['input_lines'] = lineLabels

# We use the default lines database to generate the synthetic emission lines log for this simulation
ion_array, wavelength_array, latexLabel_array = sr.label_decomposition(objParams['input_lines'])

# Define a pandas dataframe to contain the lines data
linesLogHeaders = ['wavelength', 'obsFlux', 'obsFluxErr', 'ion', 'blended', 'latexLabel']
objLinesDF = pd.DataFrame(index=lineLabels, columns=linesLogHeaders)
objLinesDF['ion'] = ion_array
objLinesDF['wavelength'] = wavelength_array
objLinesDF['latexLabel'] = latexLabel_array

# Declare extinction properties
objRed = sr.ExtinctionModel(Rv=objParams['simulation_properties']['R_v'],
                            red_curve=objParams['simulation_properties']['reddenig_curve'],
                            data_folder=objParams['data_location']['external_data_folder'])

# Compute the reddening curve for the input emission lines
lineFlambdas = objRed.gasExtincParams(wave=objLinesDF.wavelength.values)

# Generate the line fluxes
lineLogFluxes = np.zeros(lineLabels.size)

coord_true = np.stack(([objParams['true_values']['logU']],
                       [objParams['true_values']['Teff']],
                       [objParams['true_values']['OH']]), axis=-1)

for i, lineLabel in enumerate(lineLabels):
    lineInt = lineInterpolator_dict[lineLabel](coord_true).eval()[0][0]
    print(i, lineLabel, np.power(10, lineInt), lineFlambdas[i])
    lineLogFluxes[i] = lineInt - objParams['true_values']['cHbeta'] * lineFlambdas[i]
lineFluxes = np.power(10, lineLogFluxes)

# Convert to a natural scale
objLinesDF['obsFlux'] = lineFluxes
objLinesDF['obsFluxErr'] = lineFluxes * objParams['simulation_properties']['lines_minimum_error']
objLinesDF.sort_values(by=['wavelength'], ascending=True, inplace=True)

# We proceed to safe the synthetic spectrum as if it were a real observation
print(f'- Saving synthetic observation at: {user_folder}')
synthLinesLogPath = f'{user_folder}Teff_LogU_epmGrids_linesLog.txt'
print('-- Storing computed line fluxes in:\n--', synthLinesLogPath)
with open(synthLinesLogPath, 'w') as f:
    f.write(objLinesDF.to_string(index=True, index_names=False))

# Finally we safe a configuration file to fit the spectra afterwards
synthConfigPath = f'{user_folder}Teff_LogU_epmGrids_config.txt'
sr.safeConfData(synthConfigPath, objParams)
