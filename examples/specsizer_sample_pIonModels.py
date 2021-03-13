import os
import numpy as np
from pathlib import Path
import src.specsiser as sr

# user_folder = os.path.join(os.path.expanduser('~'), 'Documents/Tests_specSyzer/')
user_folder = Path('D:/AstroModels')
output_db = user_folder/f'Teff_LogU_epmGrids_db'

# Declare sampler
obj1_model = sr.SpectraSynthesizer()

# State the objects to study
linesLogAddress = user_folder/f'Teff_LogU_epmGrids_linesLog.txt'
simulationData_file = user_folder/f'Teff_LogU_epmGrids_config.txt'

# Load simulation parameters
objParams = sr.loadConfData(simulationData_file, group_variables=False)

# Load emission lines
inputLines = objParams['inference_model_configuration']['input_lines']
objLinesDF = sr.import_emission_line_data(linesLogAddress, include_lines=inputLines)

# Declare simulation physical properties
objRed = sr.ExtinctionModel(Rv=objParams['simulation_properties']['R_v'],
                            red_curve=objParams['simulation_properties']['reddenig_curve'],
                            data_folder=objParams['data_location']['external_data_folder'])

# Declare region physical model
obj1_model.define_region(objLinesDF, extinction_model=objRed)

# Declare sampling properties
obj1_model.simulation_configuration(objParams['inference_model_configuration']['parameter_list'],
                                    prior_conf_dict=objParams['priors_configuration'],
                                    photo_ionization_grid=True)

# Declare simulation inference model
obj1_model.inference_photoionization(OH=objParams['true_values']['OH']-0.05, cHbeta=objParams['true_values']['cHbeta'])

# grid_coord = np.stack(([objParams['true_values']['logU']],
#                        [objParams['true_values']['Teff']],
#                        [objParams['true_values']['OH']]), axis=-1)
#
# print('Hola')
# linesRangeArray = np.arange(obj1_model.lineLabels.size)
# for i in linesRangeArray:
#     if obj1_model.idx_analysis_lines[i]:
#         lineLabel = obj1_model.lineLabels[i]
#         loglineInt = obj1_model.gridInterp[lineLabel](grid_coord).eval()[0][0]
#         lineInt = np.power(10, loglineInt)
#         print(lineLabel, lineInt)
#         print(loglineInt, np.log10(obj1_model.grid_emissionFluxes[i]))


# Run the simulation
obj1_model.run_sampler(output_db, 5000, 2000, njobs=1)

# Plot the results
fit_results = sr.load_MC_fitting(output_db)

# Print the results
print('-- Model parameters table')
figure_file = user_folder/f'Teff_LogU_epmGrids_MeanOutputs'
obj1_model.table_mean_outputs(figure_file, fit_results, true_values=objParams['true_values'])

print('-- Model parameters posterior diagram')
figure_file = user_folder/f'Teff_LogU_epmGrids_ParamsPosteriors.png'
obj1_model.tracesPosteriorPlot(figure_file, fit_results, true_values=objParams['true_values'])

print('-- Model parameters corner diagram')
figure_file = user_folder/f'Teff_LogU_epmGrids_cornerPlot.png'
obj1_model.corner_plot(figure_file, fit_results, true_values=objParams['true_values'])

print('-- Model emission flux posteriors')
figure_file = user_folder/f'Teff_LogU_epmGrids_EmFluxPosteriors.png'
obj1_model.fluxes_photoIonization_distribution(figure_file, fit_results, combined_dict={'O2_3726A_m': 'O2_3726A-O2_3729A',
                                                                                        'S2_6716A_m': 'S2_6716A-S2_6731A'})
