import os
import src.specsyzer as ss

# Search for the data in the default user folder
user_folder = os.path.join(os.path.expanduser('~'), '')
n_objs = 3
output_db = f'{user_folder}/IFU_{n_objs}regions_fitting_db'

# Declare sampler
obj1_model = ss.SpectraSynthesizer()

# Loop through the number of regions
for n_obj in range(n_objs):

    # State the objects to study
    linesLogAddress = f'{user_folder}IFU_region{n_obj}_linesLog.txt'
    simulationData_file = f'{user_folder}IFU_region{n_obj}_config.txt'

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
    obj1_model.declare_model_region(objLinesDF, objIons, objRed, objChem, n_region = n_obj)

# Join IFU data
obj1_model.declare_IFU_data()

# Declare Region priors
obj1_model.priors_configuration(objParams['parameter_list'], prior_conf_dict=objParams, n_regions=n_objs)

# Declare simulation inference model
obj1_model.inference_IFUmodel_emission()

# Run the simulation
obj1_model.run_sampler(output_db, 5000, 2000, njobs=1)
simulation_outputfile = f'{user_folder}IFU_region{n_objs}_results.txt'
obj1_model.load_sampler_results(output_db, simulation_outputfile, n_regions=n_objs)


# # Plot the results
# traces_dict = ss.load_MC_fitting(output_db, normConstants=objParams)
#
# # Table mean values
# true_values = {k.replace('_true', ''): v for k, v in objParams.items() if '_true' in k}
#
# # print('-- Model parameters table')
# # obj1_model.table_mean_outputs(user_folder+'obj1_meanOutput', traces_dict, objParams)
#
# # Traces and Posteriors
# print('-- Model parameters posterior diagram')
# obj1_model.tracesPosteriorPlot(objParams['parameter_list'], traces_dict, true_values)
# obj1_model.savefig(user_folder+'obj1_ParamsTracesPosterios', resolution=200)
