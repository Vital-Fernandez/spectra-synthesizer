import os
import numpy as np
import src.specsiser as ss

# Search for the data in the default user folder
n_objs = 1

# user_folder = os.path.join(os.path.expanduser('~'), 'Documents/Tests_specSyzer/')
user_folder = 'D:\\AstroModels\\'
fileStructure = f'{user_folder}/GridEmiss_region'
output_db = f'{fileStructure}s{n_objs}_db'

# Declare sampler
obj1_model = ss.SpectraSynthesizer()

excludeLines = np.array(['H1_4341A', 'O3_4363A', 'Ar4_4740A', 'O3_4959A', 'O3_5007A', 'S3_6312A', 'N2_6548A', 'H1_6563A',
                        'He1_5876A', 'He1_6678A', 'He1_7065A', 'He2_4686A', 'N2_6584A', 'S2_6716A', 'S2_6731A',
                         'Ar3_7136A', 'O2_7319A', 'O2_7330A', 'S3_9069A', 'S3_9531A'])

# Loop through the number of regions
for idx_obj in range(n_objs):

    # State the objects to study
    linesLogAddress = f'{fileStructure}{idx_obj+1}of{n_objs}_linesLog.txt'
    simulationData_file = f'{fileStructure}{idx_obj+1}of{n_objs}_config.txt'

    # Load simulation parameters
    objParams = ss.loadConfData(simulationData_file)

    # Load emission lines
    objLinesDF = ss.import_emission_line_data(linesLogAddress, include_lines=excludeLines)

    # Declare simulation physical properties
    objRed = ss.ExtinctionModel(Rv=objParams['R_v'],
                                red_curve=objParams['reddenig_curve'],
                                data_folder=objParams['external_data_folder'])

    objIons = ss.IonEmissivity(tempGrid=objParams['temp_grid'],
                               denGrid=objParams['den_grid'])

    # Generate interpolator from the emissivity grids
    ionDict = objIons.get_ions_dict(np.unique(objLinesDF.ion.values))
    objIons.computeEmissivityGrids(objLinesDF, ionDict, linesDb=ss._linesDb)

    # Declare chemical model
    objChem = ss.DirectMethod(linesDF=objLinesDF, highTempIons=objParams['high_temp_ions_list'])

    # Declare region physical model
    obj1_model.define_region(objLinesDF, objIons, objRed, objChem, n_region=idx_obj)

# Declare sampling properties
obj1_model.simulation_configuration(objParams['parameter_list'], prior_conf_dict=objParams, n_regions=n_objs)

# Declare simulation inference model
obj1_model.inference_model()

# Run the simulation
obj1_model.run_sampler(output_db, 5000, 2000, njobs=1)

simulation_outputfile = f'{fileStructure}s{n_objs}_results.txt'
obj1_model.load_sampler_results(output_db, simulation_outputfile, n_regions=n_objs)

# Plot the results
traces_dict = ss.load_MC_fitting(output_db, normConstants=objParams)

# Table mean values
true_values = {k.replace('_true', ''): v for k, v in objParams.items() if '_true' in k}

print('-- Model parameters table')
figure_file = f'{fileStructure}s{n_objs}_MeanOutputs'
obj1_model.table_mean_outputs(figure_file, traces_dict, true_values)

print('-- Flux values table')
figure_file = f'{fileStructure}s{n_objs}_EmissionFluxComparison'
obj1_model.table_line_fluxes(figure_file, traces_dict, obj1_model.lineLabels, obj1_model.emissionFluxes, obj1_model.emissionErr)

print('-- Model parameters posterior diagram')
figure_file = f'{fileStructure}s{n_objs}_ParamsPosteriors.txt'
obj1_model.tracesPosteriorPlot(objParams['parameter_list'], traces_dict, true_values=true_values)
obj1_model.savefig(figure_file, resolution=200)

print('-- Model parameters corner diagram')
figure_file = f'{fileStructure}s{n_objs}_corner'
obj1_model.corner_plot(objParams['parameter_list'], traces_dict, idx_obj, true_values)
obj1_model.savefig(figure_file, resolution=200)

print('-- Line flux posteriors')
figure_file = f'{fileStructure}s{n_objs}_lineFluxPosteriors.txt'
obj1_model.fluxes_distribution(obj1_model.lineLabels, obj1_model.lineIons, 'calcFluxes_Op', traces_dict, obj1_model.emissionFluxes, obj1_model.emissionErr, objLinesDF.latexLabel.values)
obj1_model.savefig(figure_file, resolution=200)

# print('-- Model parameters Traces-Posterior diagram')
# figure_file = f'{fileStructure}s{n_objs}_paramsTracesPost'
# obj1_model.tracesPriorPostComp(objParams['parameter_list'], traces_dict, true_values=true_values)
# obj1_model.savefig(figure_file, resolution=200)
