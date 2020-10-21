import os
import numpy as np
from pathlib import Path
import src.specsiser as sr

# Search for the data in the default user folder
n_objs = 1

# user_folder = os.path.join(os.path.expanduser('~'), 'Documents/Tests_specSyzer/')
user_folder = Path('D:/AstroModels')
output_db = user_folder/f'GridEmiss_regions{n_objs}_db'

# Declare sampler
obj1_model = sr.SpectraSynthesizer()

# Simulation lines
excludeLines = np.array(['H1_4341A', 'O3_4363A', 'Ar4_4740A', 'O3_4959A', 'O3_5007A', 'S3_6312A', 'N2_6548A', 'H1_6563A',
                        'He1_5876A', 'He1_6678A', 'He1_7065A', 'He2_4686A', 'N2_6584A', 'S2_6716A', 'S2_6731A',
                         'Ar3_7136A', 'O2_7319A_b', 'S3_9069A', 'S3_9531A'])

# Loop through the number of regions
for idx_obj in range(n_objs):

    # State the objects to study
    linesLogAddress = user_folder/f'GridEmiss_region{idx_obj+1}of{n_objs}_linesLog.txt'
    simulationData_file = user_folder/f'GridEmiss_region{idx_obj+1}of{n_objs}_config.txt'

    # Load simulation parameters
    objParams = sr.loadConfData(simulationData_file, group_variables=False)

    # Load emission lines
    objLinesDF = sr.import_emission_line_data(linesLogAddress, include_lines=excludeLines)

    # Declare simulation physical properties
    objRed = sr.ExtinctionModel(Rv=objParams['simulation_properties']['R_v'],
                                red_curve=objParams['simulation_properties']['reddenig_curve'],
                                data_folder=objParams['data_location']['external_data_folder'])

    objIons = sr.IonEmissivity(tempGrid=objParams['simulation_properties']['temp_grid'],
                               denGrid=objParams['simulation_properties']['den_grid'])

    # Generate interpolator from the emissivity grids
    ionDict = objIons.get_ions_dict(np.unique(objLinesDF.ion.values))
    objIons.computeEmissivityGrids(objLinesDF, ionDict, linesDb=sr._linesDb,
                                   combined_dict={'O2_7319A_b': 'O2_7319A-O2_7330A'})

    # Declare chemical model
    objChem = sr.DirectMethod(linesDF=objLinesDF, highTempIons=objParams['simulation_properties']['high_temp_ions_list'])

    # Declare region physical model
    obj1_model.define_region(objLinesDF, objIons, objRed, objChem)

# Declare sampling properties
obj1_model.simulation_configuration(objParams['inference_model_configuration']['parameter_list'], prior_conf_dict=objParams['priors_configuration'], n_regions=n_objs)

# Declare simulation inference model
obj1_model.inference_model()

# Run the simulation
obj1_model.run_sampler(output_db, 5000, 2000, njobs=1)

# Plot the results
fit_results = sr.load_MC_fitting(output_db)

# Print the results
print('-- Model parameters table')
figure_file = user_folder/f'GridEmiss_region{n_objs}_MeanOutputs'
obj1_model.table_mean_outputs(figure_file, fit_results, true_values=objParams['true_values'])

print('-- Flux values table')
figure_file = user_folder/f'GridEmiss_region{n_objs}_FluxComparison'
obj1_model.table_line_fluxes(figure_file, fit_results, combined_dict={'O2_7319A_b': 'O2_7319A-O2_7330A'})

print('-- Model parameters posterior diagram')
figure_file = user_folder/f'GridEmiss_region{n_objs}_ParamsPosteriors.png'
obj1_model.tracesPosteriorPlot(figure_file, fit_results, true_values=objParams['true_values'])

print('-- Line flux posteriors')
figure_file = user_folder/f'GridEmiss_region{n_objs}_lineFluxPosteriors.png'
obj1_model.fluxes_distribution(figure_file, fit_results, combined_dict={'O2_7319A_b': 'O2_7319A-O2_7330A'})

# print('-- Model parameters corner diagram')
# figure_file = simFolder/f'{objName}_corner'
# obj1_model.corner_plot(objParams['parameter_list'], traces_dict)
# obj1_model.savefig(figure_file, resolution=200)

