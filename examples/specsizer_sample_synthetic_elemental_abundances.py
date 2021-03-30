import numpy as np
from pathlib import Path
import src.specsiser as sr
from pathlib import Path

# Search for the data in the default user folder
n_objs = 1

# user_folder = os.path.join(os.path.expanduser('~'), 'Documents/Tests_specSyzer/')
user_folder = Path.home()/'Astro-data/Models/'
output_db = user_folder/f'GridEmiss_regions{n_objs}_db'

# Declare sampler
obj1_model = sr.SpectraSynthesizer()

# Loop through the number of regions
idx_obj = 0

# State the objects to study
linesLogAddress = user_folder/f'GridEmiss_region{idx_obj+1}of{n_objs}_linesLog.txt'
simulationData_file = user_folder/f'GridEmiss_region{idx_obj+1}of{n_objs}_config.txt'

# Load simulation parameters
objParams = sr.loadConfData(simulationData_file, group_variables=False)

# Load emission lines
merged_lines = {'O2_3726A_m': 'O2_3726A-O2_3729A', 'O2_7319A_m': 'O2_7319A-O2_7330A'}

# Plot the results
fit_results = sr.load_MC_fitting(output_db)

# Compute abundances
objChem = sr.DirectMethod()
table_file = user_folder/f'GridEmiss_region{n_objs}_ElementalAbund'
objChem.abundances_from_db(output_db, save_results_address=table_file)

# # Print the results
# print('-- Model parameters table')
# figure_file = user_folder/f'GridEmiss_region{n_objs}_MeanOutputs'
# obj1_model.table_mean_outputs(figure_file, fit_results, true_values=objParams['true_values'])
#
# print('-- Flux values table')
# figure_file = user_folder/f'GridEmiss_region{n_objs}_FluxComparison'
# obj1_model.table_line_fluxes(figure_file, fit_results, combined_dict=merged_lines)
#
# print('-- Model parameters posterior diagram')
# figure_file = user_folder/f'GridEmiss_region{n_objs}_ParamsPosteriors.png'
# obj1_model.tracesPosteriorPlot(figure_file, fit_results, true_values=objParams['true_values'])
#
# print('-- Line flux posteriors')
# figure_file = user_folder/f'GridEmiss_region{n_objs}_lineFluxPosteriors.png'
# obj1_model.fluxes_distribution(figure_file, fit_results, combined_dict=merged_lines)
#
# print('-- Model parameters corner diagram')
# figure_file = user_folder/f'GridEmiss_region{n_objs}_cornerPlot.png'
# obj1_model.corner_plot(figure_file, fit_results, true_values=objParams['true_values'])

