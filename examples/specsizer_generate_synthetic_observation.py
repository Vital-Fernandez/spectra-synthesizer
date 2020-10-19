import os
import numpy as np
import src.specsiser as sr

# Use the default user folder to store the results
# user_folder = os.path.join(os.path.expanduser('~'), 'Documents/Tests_specSyzer/')
user_folder = 'D:\\AstroModels\\'

# Declare whether the observation is a single spectrum or multi-spectra
n_objs = 1

# Loop through the number of objects and generate a spectrum for each one
for n_obj in range(n_objs):

    # We use the default simulation configuration as a base
    objParams = sr._default_cfg.copy()

    # Set the paramter values
    objParams['true_values'] = {'n_e': 100.0 + 50.0 * n_obj,
                                'T_low': 10000.0 * (1.0 + 0.05 * n_obj),
                                'T_high': sr.TOIII_TSIII_relation(10000.0 * (1.0 + 0.05 * n_obj)),
                                'tau': 0.60 + 0.15 * n_obj,
                                'cHbeta': 0.08 + 0.02 * n_obj,
                                'H1r': 0.0,
                                'He1r': np.log10(0.070 + 0.005 * n_obj),
                                'He2r': np.log10(0.00088 + 0.0002 * n_obj),
                                'O2': 7.80 + 0.15 * n_obj,
                                'O3': 8.05 + 0.15 * n_obj,
                                'N2': 5.84 + 0.15 * n_obj,
                                'S2': 5.48 + 0.15 * n_obj,
                                'S3': 6.36 + 0.15 * n_obj,
                                'Ar3': 5.72 + 0.15 * n_obj,
                                'Ar4': 5.06 + 0.15 * n_obj}

    # Declare lines to simulate
    input_lines = ['H1_4341A', 'H1_4861A', 'H1_6563A', 'He1_4471A', 'He1_5876A', 'He1_6678A', 'He1_7065A',
                   'He2_4686A', 'O2_7319A_b', 'O2_7319A', 'O2_7330A', 'O3_4363A', 'O3_4959A', 'O3_5007A', 'N2_6548A',
                   'N2_6584A', 'S2_6716A', 'S2_6731A', 'S3_6312A', 'S3_9069A', 'S3_9531A', 'Ar3_7136A', 'Ar4_4740A']
    objParams['input_lines'] = input_lines

    # We use the default lines database to generate the synthetic emission lines log for this simulation
    linesLogPath = os.path.join(sr._literatureDataFolder, sr._default_cfg['data_location']['lines_data_file'])

    # Prepare dataframe with the observed lines labeled
    objLinesDF = sr.import_emission_line_data(linesLogPath, include_lines=objParams['input_lines'])

    # Declare extinction properties
    objRed = sr.ExtinctionModel(Rv=objParams['simulation_properties']['R_v'],
                                red_curve=objParams['simulation_properties']['reddenig_curve'],
                                data_folder=objParams['data_location']['external_data_folder'])

    # Compute the reddening curve for the input emission lines
    lineFlambdas = objRed.gasExtincParams(wave=objLinesDF.obsWave.values)

    # Establish atomic data references
    objIons = sr.IonEmissivity(tempGrid=objParams['simulation_properties']['temp_grid'],
                               denGrid=objParams['simulation_properties']['den_grid'])
    ftau_file_path = os.path.join(sr._literatureDataFolder, objParams['data_location']['ftau_file'])
    objIons.compute_ftau_grids(ftau_file_path)

    # Define the dictionary with the pyneb ion objects
    ionDict = objIons.get_ions_dict(np.unique(objLinesDF.ion.values))

    # Compute the emissivity surfaces for the observed emission lines
    objIons.computeEmissivityGrids(objLinesDF, ionDict, linesDb=objLinesDF)

    # Declare the paradigm for the chemical composition measurement
    objChem = sr.DirectMethod()

    # Tag the emission features for the chemical model implementation
    objChem.label_ion_features(linesDF=objLinesDF, highTempIons=objParams['simulation_properties']['high_temp_ions_list'])

    # We generate an object with the tensor emission functions
    emtt = sr.EmissionTensors(objLinesDF.index.values, objLinesDF.ion.values)

    # Compile exoplanet interpolator functions so they can be used wit numpy
    emisGridInterpFun = sr.gridInterpolatorFunction(objIons.emisGridDict, objIons.tempRange, objIons.denRange)

    # Compute the emission line fluxess
    lineLogFluxes = np.empty(len(objLinesDF))
    params_dict = objParams['true_values']
    He1_fluorescence_check = objChem.indcsIonLines['He1r']

    for i in np.arange(len(objLinesDF)):

        # Declare line properties
        lineLabel = objLinesDF.iloc[i].name
        lineIon = objLinesDF.iloc[i].ion
        lineFlambda = lineFlambdas[i]

        # Declare grid interpolation parameters
        emisCoord_low = np.stack(([params_dict['T_low']], [params_dict['n_e']]), axis=-1)
        emisCoord_high = np.stack(([params_dict['T_high']], [params_dict['n_e']]), axis=-1)

        # Compute emisivity for the corresponding ion temperature
        T_calc = emisCoord_high if objChem.indcsHighTemp[i] else emisCoord_low
        line_emis = emisGridInterpFun[lineLabel](T_calc).eval()

        # Declare fluorescence correction
        lineftau = 0.0

        # Compute line flux
        lineLogFluxes[i] = emtt.compute_flux(lineLabel,
                                             line_emis[0][0],
                                             params_dict['cHbeta'],
                                             lineFlambda,
                                             params_dict[lineIon],
                                             lineftau,
                                             O3=params_dict['O3'],
                                             T_high=params_dict['T_high'])

    # Convert to a natural scale
    lineFluxes = np.power(10, lineLogFluxes)

    objLinesDF.insert(loc=1, column='obsFlux', value=lineFluxes)
    objLinesDF.insert(loc=2, column='obsFluxErr', value=lineFluxes * objParams['simulation_properties']['lines_minimum_error'])

    # We proceed to safe the synthetic spectrum as if it were a real observation
    print(f'- Saving synthetic observation at: {user_folder}')
    synthLinesLogPath = f'{user_folder}GridEmiss_region{n_obj+1}of{n_objs}_linesLog.txt'
    print('-- Storing computed line fluxes in:\n--', synthLinesLogPath)
    with open(synthLinesLogPath, 'w') as f:
        f.write(objLinesDF.to_string(index=True, index_names=False))

    # Save the parameters in the natural scale
    for param in objParams['priors_configuration']['logParams_list']:
        param_true_ref = param + '_true'
        objParams['true_values'][param] = np.power(10, objParams['true_values'][param])
    # objLinesDF['intg_flux'] = lineFluxes
    # objLinesDF['intg_err'] = lineFluxes * objParams['lines_minimum_error']

    # Finally we safe a configuration file to fit the spectra afterwards
    synthConfigPath = f'{user_folder}GridEmiss_region{n_obj+1}of{n_objs}_config.txt'
    sr.safeConfData(synthConfigPath, objParams)
