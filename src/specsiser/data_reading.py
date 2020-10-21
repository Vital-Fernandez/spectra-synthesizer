import os
import numpy as np
import configparser
import pandas as pd
import pickle
import copy
from errno import ENOENT
from scipy.interpolate import interp1d
from distutils.util import strtobool
from collections import Sequence
from pathlib import Path
from data_printing import int_to_roman, label_decomposition

__all__ = ['loadConfData', 'safeConfData', 'import_emission_line_data', 'save_MC_fitting', 'load_MC_fitting',
           'parseConfDict', 'parseConfList']


CONFIGPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.ini')
STRINGCONFKEYS = ['sampler', 'reddenig_curve', 'norm_line_label', 'norm_line_pynebCode']
GLOBAL_LOCAL_GROUPS = ['_line_fitting']


# Function to check if variable can be converte to float else leave as string
def check_numeric_Value(s):
    try:
        output = float(s)
        return output
    except ValueError:
        return s


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


# Function to map a string to its variable-type
def formatStringEntry(entry_value, key_label, section_label='', float_format=None, nan_format='nan'):

    output_variable = None

    # None variable
    if (entry_value == 'None') or (entry_value is None):
        output_variable = None

    # Dictionary blended lines
    elif '_line_fitting' in section_label:
        output_variable = {}
        keys_and_values = entry_value.split(',')
        for pair in keys_and_values:

            # Conversion for parameter class atributes
            if ':' in pair:
                key, value = pair.split(':')
                if value == 'None':
                    output_variable[key] = None
                elif key in ['value', 'min', 'max']:
                    output_variable[key] = float(value)
                elif key == 'vary':
                    output_variable[key] = strtobool(value) == 1
                else:
                    output_variable[key] = value

            # Conversion for non-parameter class atributes (only str conversion possible)
            else:
                output_variable = check_numeric_Value(entry_value)

    # Arrays (The last boolean overrides the parameters # TODO unstable in case of one item lists
    elif ',' in entry_value:

        # Specia cases conversion
        if key_label == 'input_lines':
            if entry_value == 'all':
                output_variable = 'all'
            else:
                output_variable = np.array(entry_value.split(','))

        elif '_array' in key_label:
            output_variable = np.fromstring(entry_value, dtype=np.float, sep=',')

        elif '_prior' in key_label:
            entry_value = entry_value.split(',')
            output_variable = np.array([float(entry_value[i]) if i > 0 else
                                        entry_value[i] for i in range(len(entry_value))], dtype=object)

        # List of strings
        elif '_list' in key_label:
            output_variable = np.array(entry_value.split(','))

        # Objects arrays
        else:
            newArray = []
            textArrays = entry_value.split(',')
            for item in textArrays:
                convertValue = float(item) if item != 'None' else np.nan
                newArray.append(convertValue)
            output_variable = np.array(newArray)

    # Boolean
    elif '_check' in key_label:
        output_variable = strtobool(entry_value) == 1

    # Standard strings
    elif ('_folder' in key_label) or ('_file' in key_label) or ('_list' in key_label):
        output_variable = entry_value

    # elif (key_label not in STRINGCONFKEYS) and ('_folder' not in key_label) and ('_file' not in key_label) and \
    #         ('_list' not in key_label) and ('_b_components' not in key_label) and section_label not in ['blended_groups', 'merged_groups']:
    #
    #     output_variable = float(entry_value)

    # Check if numeric possible else string
    else:
        output_variable = check_numeric_Value(entry_value)


    return output_variable


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
def loadConfData(filepath, objList=None, group_variables=True):

    # Open the file
    cfg = importConfigFile(filepath)

    if group_variables:

        # Read conversion settings # TODO exclude this metadata from file
        string_parameter = cfg['conf_entries']['string_conf'].split(',')

        # Loop through configuration file sections and merge into a dictionary
        confDict = {}
        for i in range(1, len(cfg.sections())):
            section = cfg.sections()[i]
            for option in cfg.options(section):
                confDict[option] = cfg[section][option]

        # Convert the entries to the right format
        for key, value in confDict.items():
            confDict[key] = formatStringEntry(value, key)

    else:

        # Loop through configuration file sections and merge into a dictionary
        confDict = {}

        # Loop through configuration file sections and merge into a dictionary
        for section in cfg.sections():
            confDict[section] = {}
            for option_key in cfg.options(section):
                option_value = cfg[section][option_key]
                confDict[section][option_key] = formatStringEntry(option_value, option_key, section)

        # Combine sample with obj properties if available
        if objList is not None:
            for key_group in GLOBAL_LOCAL_GROUPS:
                global_group = f'default{key_group}'
                if global_group in confDict:
                    for objname in objList:
                        local_group = f'{objname}{key_group}'
                        dict_global = copy.deepcopy(confDict[global_group])
                        if local_group in confDict:
                            dict_global.update(confDict[local_group])
                        confDict[local_group] = dict_global

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

    # User input lines # TODO this might have to go to another place
    if 'input_lines' in parametersDict:

        section = 'inference_model_configuration'

        if output_cfg.has_section(section) is False:
            output_cfg.add_section(section)

        value_formatted = formatConfEntry(parametersDict['input_lines'])
        output_cfg.set(section, 'input_lines', value_formatted)

    # User true parameter values
    if 'true_values' in parametersDict:

        section = 'true_values'

        if output_cfg.has_section(section) is False:
            output_cfg.add_section(section)

        for item in parametersDict['true_values']:
            value_formatted = formatConfEntry(parametersDict['true_values'][item])
            output_cfg.set(section, f'{item}', value_formatted)

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
def import_emission_line_data(linesLogAddress, linesDb=None, include_lines=None, exclude_lines=None):

    """
    Read emission line fluxes table

    This function imports the line fluxes from a specsiser text file format and returns a pandas dataframe.

    The user can provide an auxiliariary dataframe with the theoretical information on the lines. Otherwise, the code
    default database will be used. Lines not included in the databse will still be imported but without
    the sumplementary information (ion, theoretical wavelength, pyneb reference, latex label...)

    The user can specify the lines to include or exclude (the latter take preference if the same line is included in
    both lists). Otherwise all the lines in the input files will be added be included.

    :param string linesLogAddress: Address of lines log. A succesful import requires a specsiser log format.
    :param pd.DataFrame linesDb: Dataframe with theoretical data on the lines log file. If None, the default database
    will supply the used to supply the missing data
    :param list include_lines: Array of line labels to be imported. If None, all the lines in the file will be considered
    :param list exclude_lines: Array of line labels not to be imported. If None, all the lines in the file will be considered.
    :return: Dataframe with the line data from the specified lines log file compared against the theoretical database.
    :rtype: pd.DataFrame
    """

    # Output DF # TODO we need to replace for a open excel format
    try:
        outputDF = pd.read_excel(linesLogAddress, sheet_name=0, header=0, index_col=0)

    except:
        outputDF = pd.read_csv(linesLogAddress, delim_whitespace=True, header=0, index_col=0)

    # Trim with include lines
    if include_lines is not None:
        idx_includeLines = ~(outputDF.index.isin(include_lines))
        outputDF.drop(index=outputDF.loc[idx_includeLines].index.values, inplace=True)

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
    outputDF['lineType'] = linesDb.loc[idx_obj_lines].lineType.astype(str)
    outputDF['pynebCode'] = linesDb.loc[idx_obj_lines].pynebCode
    outputDF['latexLabel'] = linesDb.loc[idx_obj_lines].latexLabel
    outputDF['blended'] = linesDb.loc[idx_obj_lines].blended

    # Trim with exclude lines
    if exclude_lines is not None:
        idx_excludedLines = outputDF.index.isin(exclude_lines)
        outputDF.drop(index=outputDF.loc[idx_excludedLines].index.values, inplace=True)

    # Correct missing ions from line labels if possible
    idx_no_ion = pd.isnull(outputDF.ion)
    for linelabel in outputDF.loc[idx_no_ion].index:
        ion_array, wave_array, latexLabel_array = label_decomposition([linelabel])
        if ion_array[0] in ('H1', 'He1', 'He2'):
            outputDF.loc[linelabel, 'ion'] = ion_array[0] + 'r'
        else:
            outputDF.loc[linelabel, 'ion'] = ion_array[0]

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
def load_MC_fitting(output_address):

    db_address = Path(output_address)

    # Output dictionary with the results
    fit_results = {}

    # Restore the pymc output file
    with open(db_address, 'rb') as trace_restored:
        db = pickle.load(trace_restored)

    model_reference, trace = db['model'], db['trace']
    fit_results.update(db)

    # params_dict = {}
    # for parameter in trace.varnames:
    #     if ('_log__' not in parameter) and ('interval' not in parameter):
    #         params_dict[parameter] = trace[parameter]
    # fit_results['parameters'] = params_dict

    # Restore the input data file
    configFileAddress = db_address.with_suffix('.txt')
    output_dict = loadConfData(configFileAddress, group_variables=False)
    fit_results['Input_data'] = output_dict['Input_data']
    fit_results['Fitting_results'] = output_dict['Fitting_results']
    fit_results['Simulation_fluxes'] = output_dict['Simulation_fluxes']

    return fit_results



