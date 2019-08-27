import os
import numpy as np
import src.specsyzer as ss

# Dictionary storing synthetic data
objParams = {}

# Use the default user folder to store the results
user_folder = os.path.join(os.path.expanduser('~'), '')

# Declare artificial data for the emission line region #TODO this notation is in conflict with the abundances
objParams['true_values'] = {'flux_hbeta': 5e-14,
                            'n_e': 125.0,
                            'T_low': 12500.0,
                            'T_high': ss.TOIII_TSIII_relation(12500.0),
                            'tau': 0.875,
                            'cHbeta': 0.125,
                            'H1r': 0.0,
                            'He1r': 0.0869,
                            'He2r': 0.00088,
                            'O2': 7.80,
                            'O3': 8.05,
                            'N2': 5.84,
                            'S2': 5.48,
                            'S3': 6.36,
                            'Ar3': 5.72,
                            'Ar4': 5.06,
                            'err_lines': 0.02}

# We use the default simulation configuration for the remaining settings
objParams.update(ss._default_cfg)

# Declare lines to simulate
objParams['input_lines'] = ['H1_4341A', 'H1_6563A', 'He1_3889A', 'He1_4471A', 'He1_5876A', 'He1_6678A', 'He1_7065A',
                            'He2_4686A', 'O2_7319A_b', 'O3_4363A', 'O3_4959A', 'O3_5007A', 'N2_6548A', 'N2_6584A',
                            'S2_6716A', 'S2_6731A', 'S3_6312A', 'S3_9069A', 'S3_9531A', 'Ar3_7136A', 'Ar4_4740A']

# We use the default lines database to generate the synthetic emission lines log for this simulation
linesLogPath = os.path.join(ss._literatureDataFolder, ss._default_cfg['lines_data_file'])

# Prepare dataframe with the observed lines labeled
objLinesDF = ss.import_emission_line_data(linesLogPath, input_lines=objParams['input_lines'])

# Declare extinction properties
objRed = ss.ExtinctionModel(Rv=objParams['R_v'],
                            red_curve=objParams['reddenig_curve'],
                            data_folder=objParams['external_data_folder'])

# Compute the flambda value for the input emission lines
lineFlambdas = objRed.gasExtincParams(wave=objLinesDF.obsWave.values)

# Establish atomic data references
ftau_file_path = os.path.join(ss._literatureDataFolder, objParams['ftau_file'])
objIons = ss.IonEmissivity(ftau_file_path=ftau_file_path, tempGrid=objParams['temp_grid'], denGrid=objParams['den_grid'])

# Define the dictionary with the pyneb ion objects
ionDict = objIons.get_ions_dict(np.unique(objLinesDF.ion.values))

# Compute the emissivity surfaces for the observed emission lines # TODO this database is not necesary since we duplicate the logs
objIons.computeEmissivityGrid(objLinesDF, ionDict, linesDb=ss._linesDb)

# Fit emissivity grids to a surface
objIons.fitEmissivityPlane(objLinesDF)

# Declare the paradigm for the chemical composition measurement
objChem = ss.DirectMethod()

# Tag the emission features for the chemical model implementation
objChem.label_ion_features(linesDF=objLinesDF, highTempIons=objParams['high_temp_ions_list'])

# We generate an object with the tensor emission functions
emtt = ss.EmissionTensors()

# Array with the equation labels
eqLabelArray = ss.assignFluxEq2Label(objLinesDF.index.values, objLinesDF.ion.values)

# Declare a dictionary with the synthetic abundances
abundDict = {}
for ion in objChem.obsAtoms:
    abundDict[ion] = objParams['true_values'][ion]

# Compute the emission line fluxes
lineFluxes = np.empty(len(objLinesDF))
for i in np.arange(len(objLinesDF)):

    lineLabel = objLinesDF.iloc[i].name
    lineIon = objLinesDF.iloc[i].ion
    lineFlambda = lineFlambdas[i]
    fluxEq = emtt.emFluxTensors[eqLabelArray[i]]
    emisCoef = objIons.emisCoeffs[lineLabel]
    emisEq = objIons.ionEmisEq_fit[lineLabel]

    Tlow, Thigh, ne, cHbeta, tau = objParams['true_values']['T_low'], objParams['true_values']['T_high'], \
                                   objParams['true_values']['n_e'], objParams['true_values']['cHbeta'], \
                                   objParams['true_values']['tau']

    lineFluxes[i] = ss.calcEmFluxes(Tlow, Thigh, ne, cHbeta, tau, abundDict,
                                    i, lineLabel, lineIon, lineFlambda,
                                    fluxEq=fluxEq, ftau_coeffs=objIons.ftau_coeffs, emisCoeffs=emisCoef, emis_func=emisEq,
                                    indcsLabelLines=objChem.indcsLabelLines, He1r_check=objChem.indcsIonLines['He1r'],
                                    HighTemp_check=objChem.indcsHighTemp)

    print('{} {} {}'.format(lineLabel, lineIon, lineFluxes[i]))

# We set the error as a 2% for all the lines and we add the data to the lines log
objLinesDF.insert(loc=1, column='obsFlux', value=lineFluxes)
objLinesDF.insert(loc=2, column='obsFluxErr', value=lineFluxes * objParams['lines_minimum_error'])

# We proceed to safe the synthetic spectrum as if it were a real observation
synthLinesLogPath = f'{user_folder}syntheticObject_linesLog.txt'
print('saving in', synthLinesLogPath)
with open(synthLinesLogPath, 'w') as f:
    f.write(objLinesDF.to_string(index=True, index_names=False))

# Finally we safe a configuration file which specsyser can use to sample this spectrum

# Include the emissivity coefficient fitted
objParams['emisCoeffs'] = objIons.emisCoeffs

# Include opacite parametrisation file
objParams['ftau_file'] = os.path.join(ss._literatureDataFolder, objParams['ftau_file'])

synthConfigPath = f'{user_folder}syntheticObject_config.txt'
ss.safeConfData(synthConfigPath, objParams)
