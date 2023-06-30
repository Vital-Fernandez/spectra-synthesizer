import numpy as np
import pandas as pd
import src.specsiser as sr

from pathlib import Path
from src.specsiser.components.chemical_model import TOIII_from_TSIII_relation

# Use the default user folder to store the results
user_folder = Path.home()

# We use the default simulation configuration as a base
objParams = sr._default_cfg.copy()

# Set the paramter values
objParams['true_values'] = {'n_e': 150.0,
                            'T_low': 12500.0,
                            'T_high': TOIII_from_TSIII_relation(12500.0),
                            'tau': 0.60 + 0.15,
                            'cHbeta': 0.08 + 0.02,
                            'H1': 0.0,
                            'He1': np.log10(0.070 + 0.005),
                            'He2': np.log10(0.00088 + 0.0002),
                            'O2': 7.805 + 0.15,
                            'O3': 8.055 + 0.15,
                            'N2': 5.845 + 0.15,
                            'S2': 5.485 + 0.15,
                            'S3': 6.94 + 0.15,
                            'Ne3': 7.065 + 0.15,
                            'Fe3': 5.055 + 0.15,
                            'Ar3': 5.725 + 0.15,
                            'Ar4': 5.065 + 0.15}

# Declare lines to simulate
input_lines = ['H1_4341A', 'H1_4861A', 'H1_6563A',
                'He1_4026A', 'He1_4471A', 'He1_5876A', 'He1_6678A', 'He1_7065A', 'He2_4686A',
                'O2_3726A_m', 'O2_7319A_m', 'O3_4363A', 'O3_4959A', 'O3_5007A',
                'Ne3_3968A',
                'N2_6548A', 'N2_6584A',
                'S2_6716A', 'S2_6731A', 'S3_6312A', 'S3_9069A', 'S3_9531A',
                'Ar3_7136A', 'Ar4_4740A',
                'Fe3_4658A']

merged_lines = {'O2_3726A_m': 'O2_3726A-O2_3729A',
                'O2_7319A_m': 'O2_7319A-O2_7330A'}

ion_array, wavelength_array, latexLabel_array = label_decomposition(input_lines, blended_dict=merged_lines)

# Define a pandas dataframe to contain the lines data
linesLogHeaders = ['wavelength', 'intg_flux', 'intg_err', 'ion', 'blended_label']
log = pd.DataFrame(index=input_lines, columns=linesLogHeaders)
log = log.assign(wavelength=wavelength_array, ion=ion_array, blended_label='None')
for line, components in merged_lines.items(): log.loc[line, 'blended_label'] = components
log.sort_values(by=['wavelength'], ascending=True, inplace=True)

# Compute the reddening curve for the input emission lines
f_lambda_array = sr.flambda_calc(wave=log.wavelength.values,
                                 R_V=objParams['simulation_properties']['R_v'],
                                 red_curve=objParams['simulation_properties']['reddenig_curve'])

# Interpolator functions for the emissivity grids
emis_grid_interp = sr.emissivity_grid_calc(lines_array=log.index.values,
                                           comp_dict=merged_lines)

# We generate an object with the tensor emission functions
emtt = sr.EmissionTensors(log.index.values, log.ion.values)

# Compute the emission line fluxess
T_High_ions = objParams['simulation_properties']['high_temp_ions_list']

# Loop throught the lines and compute their flux from the model
params_dict = objParams['true_values']
lineLogFluxes = np.empty(log.index.size)
for i in np.arange(len(log)):

    # Declare line properties
    lineLabel = log.iloc[i].name
    lineIon = log.iloc[i].ion
    lineFlambda = f_lambda_array[i]

    # Declare grid interpolation parameters
    emisCoord_low = np.stack(([params_dict['T_low']], [params_dict['n_e']]), axis=-1)
    emisCoord_high = np.stack(([params_dict['T_high']], [params_dict['n_e']]), axis=-1)

    # Compute emisivity for the corresponding ion temperature
    T_calc = emisCoord_high if lineIon in T_High_ions else emisCoord_low
    line_emis = emis_grid_interp[lineLabel](T_calc).eval()

    # Compute line flux
    lineLogFluxes[i] = emtt.compute_flux(lineLabel,
                                         line_emis[0][0],
                                         params_dict['cHbeta'],
                                         lineFlambda,
                                         params_dict[lineIon],
                                         ftau=0.0,
                                         O3=params_dict['O3'],
                                         T_high=params_dict['T_high'])

# Convert to a natural scale
lineFluxes = np.power(10, lineLogFluxes)
log['intg_flux'] = lineFluxes
log['intg_err'] = lineFluxes * objParams['simulation_properties']['lines_minimum_error']

# We proceed to safe the synthetic spectrum as if it were a real observation
synthLinesLogPath = f'{user_folder}/obj_linesLog.txt'
print('-- Storing computed line fluxes in:\n--', synthLinesLogPath)
with open(synthLinesLogPath, 'w') as f:
    f.write(log.to_string(index=True, index_names=False))

# Save the parameters in the natural scale
for param in objParams['priors_configuration']['logParams_list']:
    param_true_ref = param + '_true'
    objParams['true_values'][param] = np.power(10, objParams['true_values'][param])

# Finally we safe a configuration file to fit the spectra afterwards
synthConfigPath = f'{user_folder}/obj_fitting_config.txt'
objParams['inference_model_configuration']['input_lines_list'] = log.index.values
sr.safeConfData(synthConfigPath, objParams)


