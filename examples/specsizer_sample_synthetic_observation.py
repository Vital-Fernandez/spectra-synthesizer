import numpy as np
import src.specsiser as sr
from pathlib import Path

from fastprogress import fastprogress
fastprogress.printing = lambda: True

# Search for the data in the default user folder
n_objs = 1

# user_folder = os.path.join(os.path.expanduser('~'), 'Documents/Tests_specSyzer/')
user_folder = Path.home()
output_db = user_folder/f'GridEmiss_regions{n_objs}_db'

# Declare sampler
obj1_model = sr.SpectraSynthesizer()

# Loop through the number of regions
for idx_obj in range(n_objs):

    # State the objects to study
    linesLogAddress = user_folder/f'GridEmiss_region{idx_obj+1}of{n_objs}_linesLog.txt'
    simulationData_file = user_folder/f'GridEmiss_region{idx_obj+1}of{n_objs}_config.txt'

    # Load simulation parameters
    objParams = sr.loadConfData(simulationData_file, group_variables=False)

    # Load emission lines
    merged_lines = {'O2_3726A_m': 'O2_3726A-O2_3729A', 'O2_7319A_m': 'O2_7319A-O2_7330A'}
    default_lines = objParams['inference_model_configuration']['input_lines']
    objLinesDF = sr.import_emission_line_data(linesLogAddress, include_lines=default_lines)

    normLine = 'H1_4861A'
    idcs_lines = (objLinesDF.index != normLine)
    lineLabels = objLinesDF.loc[idcs_lines].index
    lineIons = objLinesDF.loc[idcs_lines, 'ion'].values
    lineFluxes = objLinesDF.loc[idcs_lines, 'intg_flux'].values
    lineErr = objLinesDF.loc[idcs_lines, 'intg_err'].values

    # Declare simulation physical properties
    objRed = sr.ExtinctionModel(Rv=objParams['simulation_properties']['R_v'],
                                red_curve=objParams['simulation_properties']['reddenig_curve'],
                                data_folder=objParams['data_location']['external_data_folder'])

    objIons = sr.IonEmissivity(tempGrid=objParams['simulation_properties']['temp_grid'],
                               denGrid=objParams['simulation_properties']['den_grid'])

    # Generate interpolator from the emissivity grids
    ionDict = objIons.get_ions_dict(np.unique(lineIons))
    objIons.computeEmissivityGrids(lineLabels, ionDict, combined_dict=merged_lines)

    # Declare chemical model
    objChem = sr.DirectMethod(lineLabels, highTempIons=objParams['simulation_properties']['high_temp_ions_list'])

    # Declare region physical model
    obj1_model.define_region(lineLabels, lineFluxes, lineErr, objIons, objRed, objChem)

# Declare sampling properties
obj1_model.simulation_configuration(objParams['inference_model_configuration']['parameter_list'],
                                    prior_conf_dict=objParams['priors_configuration'],
                                    photo_ionization_grid=False,
                                    n_regions=n_objs)

# Declare simulation inference model
obj1_model.inference_model()

# Run the simulation
obj1_model.run_sampler(2000, 2000, nchains=1, njobs=1)
obj1_model.save_fit(output_db)

# Load the results
fit_pickle = sr.load_MC_fitting(output_db)
inLines, inParameters = fit_pickle['inputs']['line_list'], fit_pickle['inputs']['parameter_list']
inFlux, inErr = fit_pickle['inputs']['line_fluxes'].astype(float), fit_pickle['inputs']['line_err'].astype(float)
traces_dict = fit_pickle['outputs']


# Print the results
# TODO make plots independent of obj1_model
print('-- Model parameters table')
figure_file = user_folder/f'GridEmiss_region{n_objs}_MeanOutputs'
obj1_model.table_mean_outputs(figure_file, inParameters, traces_dict, true_values=objParams['true_values'])

print('-- Flux values table')
figure_file = user_folder/f'GridEmiss_region{n_objs}_FluxComparison'
obj1_model.table_line_fluxes(figure_file, inLines, inFlux, inErr, traces_dict)

print('-- Model parameters posterior diagram')
figure_file = user_folder/f'GridEmiss_region{n_objs}_ParamsPosteriors.png'
obj1_model.tracesPosteriorPlot(figure_file, inParameters, traces_dict, true_values=objParams['true_values'])

print('-- Line flux posteriors')
figure_file = user_folder/f'GridEmiss_region{n_objs}_lineFluxPosteriors.png'
obj1_model.fluxes_distribution(figure_file, inLines, inFlux, inErr, traces_dict)

print('-- Model parameters corner diagram')
figure_file = user_folder/f'GridEmiss_region{n_objs}_cornerPlot.png'
obj1_model.corner_plot(figure_file, inParameters, traces_dict, true_values=objParams['true_values'])


