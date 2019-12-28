import os
import numpy as np
import src.specsyzer as ss

# Search for the data in the default user folder
n_objs = 10
user_folder = os.path.join(os.path.expanduser('~'), '')
output_db = f'{user_folder}/{n_objs}IFU_fitting_db'

# Declare sampler
obj1_model = ss.SpectraSynthesizer()

# Loop through the number of regions
for idx_obj in range(n_objs):

    # State the objects to study
    linesLogAddress = f'{user_folder}{n_objs}IFU_region{idx_obj}_linesLog.txt'
    simulationData_file = f'{user_folder}{n_objs}IFU_region{idx_obj}_config.txt'

    # Load simulation parameters
    objParams = ss.loadConfData(simulationData_file)

    # Load emission lines
    objLinesDF = ss.import_emission_line_data(linesLogAddress, input_lines='all')

    # Declare simulation physical properties
    objRed = ss.ExtinctionModel(Rv=objParams['R_v'],
                                red_curve=objParams['reddenig_curve'],
                                data_folder=objParams['external_data_folder'])

    objIons = ss.IonEmissivity(ftau_file_path=objParams['ftau_file'],
                               tempGrid=objParams['temp_grid'],
                               denGrid=objParams['den_grid'])

    # Load coefficients for emissivity fittings:
    objIons.load_emis_coeffs(objLinesDF.index.values, objParams)

    objChem = ss.DirectMethod(linesDF=objLinesDF,
                              highTempIons=objParams['high_temp_ions_list'])

    # Declare region physical model
    obj1_model.define_region(objLinesDF, objIons, objRed, objChem, n_region=idx_obj)

# Declare sampling properties
obj1_model.simulation_configuration(objParams['parameter_list'], prior_conf_dict=objParams, n_regions=n_objs)

# Declare simulation inference model
obj1_model.inference_IFUmodel_emission()

# Run the simulation
obj1_model.run_sampler(output_db, 5000, 2000)
=======
#     # Load emission lines
#     objLinesDF = ss.import_emission_line_data(linesLogAddress, input_lines='all')
#
#     # Declare simulation physical properties
#     objRed = ss.ExtinctionModel(Rv=objParams['R_v'],
#                                 red_curve=objParams['reddenig_curve'],
#                                 data_folder=objParams['external_data_folder'])
#
#     objIons = ss.IonEmissivity(ftau_file_path=objParams['ftau_file'],
#                                tempGrid=objParams['temp_grid'],
#                                denGrid=objParams['den_grid'])
#
#     # Load coefficients for emissivity fittings:
#     objIons.load_emis_coeffs(objLinesDF.index.values, objParams)
#
#     objChem = ss.DirectMethod(linesDF=objLinesDF,
#                               highTempIons=objParams['high_temp_ions_list'])
#
#     # Declare region physical model
#     obj1_model.define_region(objLinesDF, objIons, objRed, objChem, n_region=idx_obj)
#
# Declare sampling properties
obj1_model.simulation_configuration(objParams['parameter_list'], prior_conf_dict=objParams, n_regions=n_objs)

# Declare simulation inference model
obj1_model.inference_IFUmodel_emission()

# Run the simulation
obj1_model.run_sampler(output_db, 5000, 2000, njobs=1)
simulation_outputfile = f'{user_folder}{n_objs}IFU_results.txt'
obj1_model.load_sampler_results(output_db, simulation_outputfile, n_regions=n_objs)

# Plot the results
traces_dict = ss.load_MC_fitting(output_db, normConstants=objParams)

# Table mean values
true_values = {k.replace('_true', ''): v for k, v in objParams.items() if '_true' in k}

# print('-- Model parameters table')
# obj1_model.table_mean_outputs(user_folder+'obj1_meanOutput', traces_dict, objParams)

# Traces and Posteriors
# print('-- Model parameters posterior diagram')
# figure_file = f'{user_folder}{n_objs}IFU_ParamsTracesPosterios.txt'
# obj1_model.tracesPosteriorPlot(objParams['parameter_list'], traces_dict, idx_obj, true_values)
# obj1_model.savefig(figure_file, resolution=200)

# Traces and Posteriors
# print('-- Model parameters corner diagram')
# figure_file = f'{user_folder}{n_objs}IFU_corner'
# obj1_model.corner_plot(objParams['parameter_list'], traces_dict, idx_obj, true_values)
# obj1_model.savefig(figure_file, resolution=200)

# print('-- Model parameters corner diagram')
# figure_file = f'{user_folder}{n_objs}IFU_corner'
# obj1_model.corner_plot(['T_low', 'T_high', 'n_e', 'O2_0', 'O3_0', 'S2_0', 'S3_0'], traces_dict, idx_obj, true_values)
# obj1_model.savefig(figure_file, resolution=200)

print('-- Model parameters corner diagram')
figure_file = f'E:/Dropbox/Astrophysics/Seminars/PyConES_2019/{n_objs}priorPostComp'
obj1_model.tracesPriorPostComp(objParams['parameter_list'], traces_dict, idx_obj, true_values)
obj1_model.savefig(figure_file, resolution=200)

