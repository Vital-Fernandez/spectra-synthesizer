import numpy as np
import src.specsiser as sr
import lime
from pathlib import Path

from fastprogress import fastprogress
fastprogress.printing = lambda: True

# user_folder = os.path.join(os.path.expanduser('~'), 'Documents/Tests_specSyzer/')
user_folder = Path.home()
synthConfigPath = f'{user_folder}/obj_fitting_config.txt'
synthLinesLogPath = f'{user_folder}/obj_linesLog.txt'
output_db = user_folder/f'obj_output_db'

# Load simulation parameters
objParams = lime.load_cfg(synthConfigPath)

# Load emission lines
input_lines = objParams['inference_model_configuration']['input_lines_list']
merged_lines = {'O2_3726A_m': 'O2_3726A-O2_3729A', 'O2_7319A_m': 'O2_7319A-O2_7330A'}
log = lime.load_lines_log(synthLinesLogPath)

normLine = 'H1_4861A'
idcs_lines = (log.index != normLine)
lineLabels = log.loc[idcs_lines].index
lineWaves = log.loc[idcs_lines, 'wavelength'].values
lineIons = log.loc[idcs_lines, 'ion'].values
lineFluxes = log.loc[idcs_lines, 'intg_flux'].values
lineErr = log.loc[idcs_lines, 'intg_err'].values

# Reddening curve
flambda = sr.flambda_calc(lineWaves,
                          R_V=objParams['simulation_properties']['R_v'],
                          red_curve=objParams['simulation_properties']['reddenig_curve'])

# Interpolator functions for the emissivity grids
emis_grid_interp = sr.emissivity_grid_calc(lines_array=log.index.values,
                                           comp_dict=merged_lines)

# Declare sampler
obj1_model = sr.SpectraSynthesizer(emis_grid_interp)

# Declare region physical model
obj1_model.define_region(lineLabels, lineFluxes, lineErr, flambda, merged_lines)

# Declare sampling properties
obj1_model.simulation_configuration(prior_conf_dict=objParams['priors_configuration'],
                                    highTempIons=objParams['simulation_properties']['high_temp_ions_list'],)

# Declare simulation inference model
obj1_model.inference_model()

# Run the simulation
obj1_model.run_sampler(2000, 2000, nchains=4, njobs=4)
obj1_model.save_fit(output_db)

# Load the results
fit_pickle = sr.load_fit_results(output_db)
inLines, inParameters = fit_pickle['inputs']['lines_list'], fit_pickle['inputs']['parameter_list']
inFlux, inErr = fit_pickle['inputs']['line_fluxes'].astype(float), fit_pickle['inputs']['line_err'].astype(float)
traces_dict = fit_pickle['outputs']

# Print the results
print('-- Model parameters table')
figure_file = user_folder/f'obj_fitted_fluxes'
sr.table_fluxes(figure_file, inLines, inFlux, inErr, traces_dict, merged_lines)

# Print the results
print('-- Fitted fluxes table')
figure_file = user_folder/f'obj_MeanOutputs'
sr.table_params(figure_file, inParameters, traces_dict, true_values=objParams['true_values'])

print('-- Model parameters posterior diagram')
figure_file = user_folder/f'obj_traces_plot.png'
sr.plot_traces(figure_file, inParameters, traces_dict, true_values=objParams['true_values'])

print('-- Line flux posteriors')
figure_file = user_folder/f'obj_fluxes_grid.png'
sr.plot_flux_grid(figure_file, inLines, inFlux, inErr, traces_dict)

print('-- Model parameters corner diagram')
figure_file = user_folder/f'obj_corner.png'
sr.plot_corner(figure_file, inParameters, traces_dict, true_values=objParams['true_values'])



