import os
import numpy as np
import configparser
import pandas as pd
import pickle
from errno import ENOENT
from scipy.interpolate import interp1d
from distutils.util import strtobool
from collections import Sequence

__all__ = ['loadConfData', 'safeConfData', 'import_emission_line_data', 'save_MC_fitting', 'load_MC_fitting',
           'parseConfDict', 'parseConfList']


CONFIGPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.ini')
STRINGCONFKEYS = ['sampler', 'reddenig_curve', 'norm_line_label', 'norm_line_pynebCode']


# Function to import configuration data
def parseObjData(file_address, sectionName, objData):

    parser = configparser.SafeConfigParser()
    parser.optionxform = str
    if os.path.isfile(file_address):
        parser.read(file_address)

    if not parser.has_section(sectionName):
        parser.add_section(sectionName)

    for key in objData.keys():
        value = objData[key]
        if value is not None:
            if isinstance(value, list) or isinstance(value, np.ndarray):
                value = ','.join(str(x) for x in value)
            else:
                value = str(value)
        else:
            value = ''

        parser.set(sectionName, key, value)

    with open(file_address, 'w') as f:
        parser.write(f)

    return


# Function to import configparser
def importConfigFile(config_path):

    # Check if file exists
    if os.path.isfile(config_path):
        cfg = configparser.ConfigParser()
        cfg.optionxform = str
        cfg.read(config_path)
    else:
        exit(f'--WARNING: Configuration file {config_path} was not found. Exiting program')

    return cfg


# Function to delete files
def silent_remove(filename_list):
    for filename in filename_list:
        try:
            os.remove(filename)
        except OSError as e:  # this would be "except OSError, e:" before Python 2.6
            if e.errno != ENOENT:  # errno.ENOENT = no such file or directory
                raise  # re-raise exception if a different error occurred


# Function to map variables to strings
def formatConfEntry(entry_value, float_format=None, nan_format='nan'):

    # Check None entry
    if entry_value is not None:

        # Check string entry
        if isinstance(entry_value, str):
            formatted_value = entry_value

        else:

            # Case of an array
            scalarVariable = True
            if isinstance(entry_value, (Sequence, np.ndarray)):

                # Confirm is not a single value array
                if len(entry_value) == 1:
                    entry_value = entry_value[0]

                # Case of an array
                else:
                    scalarVariable = False
                    formatted_value = ','.join([str(item) for item in entry_value])

            if scalarVariable:

                # Case single float
                if np.isnan(entry_value):
                    formatted_value = nan_format

                else:
                    formatted_value = str(entry_value)

    else:
        formatted_value = 'None'

    return formatted_value


# Function to check for nan entries
def check_missing_flux_values(flux):
    # Evaluate the nan array
    nan_idcs = np.isnan(flux)
    nan_count = np.sum(nan_idcs)

    # Directly save if not nan
    if nan_count > 0:
        print('--WARNING: missing flux entries')

    return


# Function to import SpecSyzer configuration file #TODO repeated
def loadConfData(filepath, objName=None):

    # Open the file
    cfg = importConfigFile(filepath)

    # Read conversion settings # TODO exclude this metadata from file
    string_parameter = cfg['conf_entries']['string_conf'].split(',')

    # Loop through configuration file sections and merge into a dictionary
    confDict = {}
    for i in range(1, len(cfg.sections())):
        section = cfg.sections()[i]
        for option in cfg.options(section):
            confDict[option] = cfg[section][option]

    # TODO add mechanic to read the many objects are results
    # TODO make confDict[key] a variable
    # Convert the entries to the right format # TODO Add security warnings for wrong data
    for key in confDict.keys():

        value = confDict[key]

        # Empty variable
        if value == '':
            confDict[key] = None

        # None variable
        elif (value == 'None') or (value is None):
            confDict[key] = None

        # Arrays (The last boolean overrides the parameters # TODO unstable in case of one item lists
        elif ',' in value:

            # Specia cases conversion
            if key == 'input_lines':
                if value == 'all':
                    confDict[key] = 'all'
                else:
                    confDict[key] = np.array(confDict[key].split(','))

            elif '_prior' in key:
                value = value.split(',')
                confDict[key] = np.array([float(value[i]) if i > 0 else value[i] for i in range(len(value))], dtype=object) # TODO this one should read as an array

            # List of strings
            elif '_list' in key:
                    confDict[key] = np.array(confDict[key].split(','))

            # Objects arrays
            else:
                newArray = []
                textArrays = value.split(',')
                for item in textArrays:
                    convertValue = float(item) if item != 'None' else np.nan
                    newArray.append(convertValue)
                confDict[key] = np.array(newArray)

        # Boolean
        elif '_check' in key:
            confDict[key] = strtobool(confDict[key]) == 1

        # Remaining are either strings or floats
        elif (key not in STRINGCONFKEYS) and ('_folder' not in key) and ('_file' not in key) and ('_list' not in key):
            confDict[key] = float(value)

    return confDict


# Function to save data to configuration file based on a previous dictionary
def safeConfData(fileAddress, parametersDict, conf_style_path=None):

    # Declare the default configuration file and load it
    if conf_style_path is None:
        conf_style_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.ini')

    # Check if file exists
    if os.path.isfile(conf_style_path):
        _default_cfg = configparser.ConfigParser()
        _default_cfg.optionxform = str
        _default_cfg.read(conf_style_path)
    else:
        exit(f'--WARNING: Configuration file {conf_style_path} was not found. Exiting program')

    # Create new configuration object
    output_cfg = configparser.ConfigParser()
    output_cfg.optionxform = str

    # Loop through the default configuration file sections and options using giving preference to the new data
    sections_cfg = _default_cfg.sections()
    for i in range(len(sections_cfg)):

        section = sections_cfg[i]

        options_cfg = _default_cfg.options(sections_cfg[i])

        output_cfg.add_section(section)

        for j in range(len(options_cfg)):

            option = options_cfg[j]

            if options_cfg[j] in parametersDict:
                value = parametersDict[option]
            else:
                value = _default_cfg[section][option]

            value_formatted = formatConfEntry(value)

            # print(f'-- {section} {option} {value} --> {value_formatted}')
            output_cfg.set(section, option, value_formatted)

    # Additional sections not included by default

    # TODO create a generic function to add sections to input file
    # Emissivity coefficients
    if 'emisCoeffs' in parametersDict:

        section = 'emissivity_fitting_coefficients'
        if output_cfg.has_section(section) is False:
            output_cfg.add_section(section)

        for lineLabel in parametersDict['emisCoeffs']:
            value = parametersDict['emisCoeffs'][lineLabel]
            value_formatted = formatConfEntry(value)
            output_cfg.set(section, lineLabel, value_formatted)

    # User input lines
    if 'input_lines' in parametersDict:

        section = 'objData_1'
        if output_cfg.has_section(section) is False:
            output_cfg.add_section(section)

        value_formatted = formatConfEntry(parametersDict['input_lines'])
        output_cfg.set(section, 'input_lines', value_formatted)

    # User true parameter values
    if 'true_values' in parametersDict:
        section = 'objData_1'

        if output_cfg.has_section(section) is False:
            output_cfg.add_section(section)

        for item in parametersDict['true_values']:
            value_formatted = formatConfEntry(parametersDict['true_values'][item])
            output_cfg.set(section, f'{item}_true', value_formatted)

    # Safe the configuration file
    with open(fileAddress, 'w') as f:
        output_cfg.write(f)

    return


# Function to save a parameter key-value item (or list) into the dictionary
def parseConfList(output_file, param_key, param_value, section_name):

    # Check if file exists
    if os.path.isfile(output_file):
        output_cfg = configparser.ConfigParser()
        output_cfg.optionxform = str
        output_cfg.read(output_file)
    else:
        # Create new configuration object
        output_cfg = configparser.ConfigParser()
        output_cfg.optionxform = str

    # Map key values to the expected format and store them
    if isinstance(param_key, (Sequence, np.array)):
        for i in range(len(param_key)):
            value_formatted = formatConfEntry(param_value[i])
            output_cfg.set(section_name, param_key[i], value_formatted)
    else:
        value_formatted = formatConfEntry(param_value)
        output_cfg.set(section_name, param_key, value_formatted)

    # Save the text file data
    with open(output_file, 'w') as f:
        output_cfg.write(f)

    return


# Function to save a parameter dictionary into a cfg dictionary
def parseConfDict(output_file, param_dict, section_name):

    # TODO add logic to erase section previous results

    # Check if file exists
    if os.path.isfile(output_file):
        output_cfg = configparser.ConfigParser()
        output_cfg.optionxform = str
        output_cfg.read(output_file)
    else:
        # Create new configuration object
        output_cfg = configparser.ConfigParser()
        output_cfg.optionxform = str

    # Add new section if it is not there
    if not output_cfg.has_section(section_name):
        output_cfg.add_section(section_name)

    # Map key values to the expected format and store them
    for item in param_dict:
        value_formatted = formatConfEntry(param_dict[item])
        output_cfg.set(section_name, item, value_formatted)

    # Save the text file data
    with open(output_file, 'w') as f:
        output_cfg.write(f)

    return


# Function to resample and trim spectra
def treat_input_spectrum(output_dict, spec_wave, spec_flux, wavelengh_limits=None, resample_inc=None, norm_interval=None):

    # TODO we should remove the nBases requirement by some style which can just read the number of dimensions

    # Store input values
    output_dict['wavelengh_limits'] = wavelengh_limits
    output_dict['resample_inc'] = resample_inc
    output_dict['norm_interval'] = norm_interval

    # Special case using 0, -1 indexing
    if wavelengh_limits is not None:
        if (wavelengh_limits[0] != 0) and (wavelengh_limits[0] != -1):
            inputWaveLimits = wavelengh_limits
        else:
            inputWaveLimits = wavelengh_limits
            if wavelengh_limits[0] == 0:
                inputWaveLimits[0] = int(np.ceil(spec_wave[0]) + 1)
            if wavelengh_limits[-1] == -1:
                inputWaveLimits[-1] = int(np.floor(spec_wave[-1]) - 1)

    # Resampling the spectra
    if resample_inc is not None:
        wave_resam = np.arange(inputWaveLimits[0], inputWaveLimits[-1], resample_inc, dtype=float)

        # Loop throught the fluxes (In the case of the bases it is assumed they may have different wavelength ranges)
        if isinstance(spec_flux, list):

            flux_resam = np.empty((output_dict['nBases'], len(wave_resam)))
            for i in range(output_dict['nBases']):
                flux_resam[i, :] = interp1d(spec_wave[i], spec_flux[i], bounds_error=True)(wave_resam)

        # In case only one dimension
        elif spec_flux.ndim == 1:
            flux_resam = interp1d(spec_wave, spec_flux, bounds_error=True)(wave_resam)

        output_dict['wave_resam'] = wave_resam
        output_dict['flux_resam'] = flux_resam

    else:
        output_dict['wave_resam'] = spec_wave
        output_dict['flux_resam'] = spec_flux

    # Normalizing the spectra
    if norm_interval is not None:

        # Loop throught the fluxes (In the case of the bases it is assumed they may have different wavelength ranges)
        if isinstance(spec_flux, list):

            normFlux_coeff = np.empty(output_dict['nBases'])
            flux_norm = np.empty((output_dict['nBases'], len(wave_resam)))
            for i in range(output_dict['nBases']):
                idx_Wavenorm_min, idx_Wavenorm_max = np.searchsorted(spec_wave[i], norm_interval)
                normFlux_coeff[i] = np.mean(spec_flux[i][idx_Wavenorm_min:idx_Wavenorm_max])
                flux_norm[i] = output_dict['flux_resam'][i] / normFlux_coeff[i]

        elif spec_flux.ndim == 1:
            idx_Wavenorm_min, idx_Wavenorm_max = np.searchsorted(spec_wave, norm_interval)
            normFlux_coeff = np.mean(spec_flux[idx_Wavenorm_min:idx_Wavenorm_max])
            flux_norm = output_dict['flux_resam'] / normFlux_coeff

        output_dict['flux_norm'] = flux_norm
        output_dict['normFlux_coeff'] = normFlux_coeff

    else:

        output_dict['flux_norm'] = output_dict['flux_resam']
        output_dict['normFlux_coeff'] = 1.0

    return


# Function to generate mask according to input emission lines
def generate_object_mask(linesDf, wavelength, linelabels):

    # TODO This will not work for a redshifted lines log
    idcs_lineMasks = linesDf.index.isin(linelabels)
    idcs_spectrumMasks = ~linesDf.index.isin(linelabels)

    # Matrix mask for integring the emission lines
    n_lineMasks = idcs_lineMasks.sum()
    boolean_matrix = np.zeros((n_lineMasks, wavelength.size), dtype=bool)

    # Array with line wavelength resolution which we fill with default value (This is because there are lines beyong the continuum range)
    lineRes = np.ones(n_lineMasks) * (wavelength[1] - wavelength[0])

    # Total mask for valid regions in the spectrum
    n_objMasks = idcs_spectrumMasks.sum()
    int_mask = np.ones(wavelength.size, dtype=bool)
    object_mask = np.ones(wavelength.size, dtype=bool)

    # Loop through the emission lines
    wmin, wmax = linesDf['w3'].loc[idcs_lineMasks].values, linesDf['w4'].loc[idcs_lineMasks].values
    idxMin, idxMax = np.searchsorted(wavelength, [wmin, wmax])
    for i in range(n_lineMasks):
        if not np.isnan(wmin[i]) and not np.isnan(wmax[i]) and (wmax[i] < wavelength[-1]): # We need this for lines beyong continuum range #TODO propose better
            w2, w3 = wavelength[idxMin[i]], wavelength[idxMax[i]]
            idx_currentMask = (wavelength >= w2) & (wavelength <= w3)
            boolean_matrix[i, :] = idx_currentMask
            int_mask = int_mask & ~idx_currentMask
            lineRes[i] = wavelength[idxMax[i]] - wavelength[idxMax[i] - 1]

    # Loop through the object masks
    wmin, wmax = linesDf['w3'].loc[idcs_spectrumMasks].values, linesDf['w4'].loc[idcs_spectrumMasks].values
    idxMin, idxMax = np.searchsorted(wavelength, [wmin, wmax])
    for i in range(n_objMasks):
        if not np.isnan(wmin[i]) and not np.isnan(wmax[i]) and (wmax[i] < wavelength[-1]):
            w2, w3 = wavelength[idxMin[i]], wavelength[idxMax[i]]
            idx_currentMask = (wavelength >= w2) & (wavelength <= w3)
            int_mask = int_mask & ~idx_currentMask
            object_mask = object_mask & ~idx_currentMask

    return int_mask, object_mask


# Function to label the lines provided by the user
def import_emission_line_data(linesLogAddress, linesDb = None, input_lines='all'):

    # Output DF # TODO we need a better mechanic to discremeate
    try:
        outputDF = pd.read_excel(linesLogAddress, sheet_name=0, header=0, index_col=0)

    except:
        outputDF = pd.read_csv(linesLogAddress, delim_whitespace=True, header=0, index_col=0)

    if linesDb is None:
        cfg = importConfigFile(CONFIGPATH)
        dir_path = os.path.dirname(os.path.realpath(__file__))

        # Declare default data folder
        literatureDataFolder = os.path.join(dir_path, cfg['data_location']['external_data_folder'])

        # Load library databases
        linesDatabasePath = os.path.join(literatureDataFolder, cfg['data_location']['lines_data_file'])
        linesDb = pd.read_excel(linesDatabasePath, sheet_name=0, header=0, index_col=0)

    # If wavelengths are provided for the observation we use them, else we use the theoretical values
    if 'obsWave' not in outputDF.columns:
        idx_obs_labels = linesDb.index.isin(outputDF.index)
        outputDF['obsWave'] = linesDb.loc[idx_obs_labels].wavelength

    # Sort the dataframe by wavelength value in case it isn't
    outputDF.sort_values(by=['obsWave'], ascending=True, inplace=True)

    # Get the references for the lines treatment
    idx_obj_lines = linesDb.index.isin(outputDF.index)
    outputDF['ion'] = linesDb.loc[idx_obj_lines].ion.astype(str)
    outputDF['lineType'] = linesDb.loc[idx_obj_lines].lineType.astype(str)
    outputDF['pynebCode'] = linesDb.loc[idx_obj_lines].pynebCode
    outputDF['latexLabel'] = linesDb.loc[idx_obj_lines].latexLabel
    outputDF['blended'] = linesDb.loc[idx_obj_lines].blended

    # Remove non desired lines  TODO add check of possible lines w.r.t spectrum
    # TODO maybe better to exclude than include
    if input_lines is not 'all':
        idx_excludedLines = ~(outputDF.index.isin(input_lines))
        outputDF.drop(index=outputDF.loc[idx_excludedLines].index.values, inplace=True)

    return outputDF


# Function to save data to configuration file section
def import_optical_depth_coeff_table(file_address):

    opticalDepthCoeffs_df = pd.read_csv(file_address, delim_whitespace=True, header=0)

    opticalDepthCoeffs = {}
    for column in opticalDepthCoeffs_df.columns:
        opticalDepthCoeffs[column] = opticalDepthCoeffs_df[column].values

    return opticalDepthCoeffs


# Function to save PYMC3 simulations using pickle
def save_MC_fitting(db_address, trace, model, sampler='pymc3'):

    if sampler is 'pymc3':
        with open(db_address, 'wb') as trace_pickle:
            pickle.dump({'model': model, 'trace': trace}, trace_pickle)

    return


# Function to restore PYMC3 simulations using pickle
def load_MC_fitting(db_address, return_dictionary=True, normConstants=None):

    # Restore the trace
    with open(db_address, 'rb') as trace_restored:
        db = pickle.load(trace_restored)

    # Return Pymc3 db object
    if return_dictionary is False:
        return db

    # Return dictionary with the traces
    else:
        model_reference, trace = db['model'], db['trace']

        traces_dict = {}
        for parameter in trace.varnames:
            if ('_log__' not in parameter) and ('interval' not in parameter):
                prior_key = parameter + '_prior'
                trace_norm = normConstants[prior_key][3] if prior_key in normConstants else 1.0
                trace_i = trace_norm * trace[parameter]
                traces_dict[parameter] = trace_i

        return traces_dict



