import os
import src.specsyzer as ss

# Search for the data in the default user folder
user_folder = os.path.join(os.path.expanduser('~'), '')

# State the objects to study
objName = 'syntheticObject'
linesLogAddress = f'{user_folder}{objName}_linesLog.txt'
simulationData_file = f'{user_folder}{objName}_config.txt'

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

# Declare sampler properties
obj1_model = ss.SpectraSynthesizer()
obj1_model.declare_model_data(objLinesDF, objIons, objRed, objChem)

# Set parameters priors
obj1_model.priors_configuration(objParams['parameter_list'], prior_conf_dict=objParams)

# Declare simulation inference model
obj1_model.inference_model_emission()

# Run the simulation
print('PERO ESTO VA BIEN?')

# Plot the results


# # Compute reddening curve # TODO also export Rv and curve here
# objLinesDF['lineFlambda'] = ss.gasExtincParams(objLinesDF['lineWaves'].values)
#
# # Variables to make the iterations simpler # TODO this is more like a sampler rather than gas
# ss.gasSamplerVariables(objLinesDF['lineIons'].values, ss.config['high_temp_ions'],
#                        lineLabels=objLinesDF.index.values, lineFlambda=objLinesDF['lineFlambda'].values)
#
# # Declare the sampler
# objHMC_sampler = ss.SpectraSynthesizer()
#
#
# # Declaring configuration design
# fit_conf = dict(obs_data=objParams,
#                 ssp_data=None,
#                 output_folder=user_folder,
#                 spectra_components=['emission'],
#                 input_lines=objParams['input_lines'],
#                 prefit_ssp=False,
#                 normalized_by_Hbeta=True)
#
# # Prepare fit data
# objHMC_sampler.prepareSimulation(**fit_conf)
#
# # Run the simulation
# simulationName = '{}_emissionHMC'.format(objList[0])
# objHMC_sampler.fitSpectra(model_name=simulationName, iterations=6000, tuning=3000,
#                  include_reddening=objParams['redening_check'], include_Thigh_prior=objParams['Thigh_check'])

