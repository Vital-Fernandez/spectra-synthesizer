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
from astropy.io import fits
from pylatex import MultiColumn, MultiRow, utils
from lime.io import progress_bar

VAL_LIST = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
SYB_LIST = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]

CONFIGPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'default.cfg')
STRINGCONFKEYS = ['sampler', 'reddenig_curve', 'norm_line_label', 'norm_line_pynebCode']
GLOBAL_LOCAL_GROUPS = ['_line_fitting', '_chemical_model']

FITS_INPUTS_EXTENSION = {'line_list': '20A', 'line_fluxes': 'E', 'line_err': 'E'}
FITS_OUTPUTS_EXTENSION = {'parameter_list': '20A',
                          'mean': 'E',
                          'std': 'E',
                          'median': 'E',
                          'p16th': 'E',
                          'p84th': 'E',
                          'true': 'E'}

def save_log_maps(log_file_address, param_list, output_folder, mask_file_address=None, ext_mask='all',
                    ext_log='_INPUTS', default_spaxel_value=np.nan, output_files_prefix=None, page_hdr={}):

    assert Path(log_file_address).is_file(), f'- ERROR: lines log at {log_file_address} not found'
    assert Path(output_folder).is_dir(), f'- ERROR: Output parameter maps folder {output_folder} not found'

    # Compile the list of voxels to recover the provided masks
    if mask_file_address is not None:

        assert Path(mask_file_address).is_file(), f'- ERROR: mask file at {mask_file_address} not found'

        with fits.open(mask_file_address) as maskHDUs:

            # Get the list of mask extensions
            if ext_mask == 'all':
                if ('PRIMARY' in maskHDUs) and (len(maskHDUs) > 1):
                    mask_list = []
                    for i, HDU in enumerate(maskHDUs):
                        mask_name = HDU.name
                        if mask_name != 'PRIMARY':
                            mask_list.append(mask_name)
                    mask_list = np.array(mask_list)
                else:
                    mask_list = np.array(['PRIMARY'])
            else:
                mask_list = np.array(ext_mask, ndmin=1)

            # Combine all the mask voxels into one
            for i, mask_name in enumerate(mask_list):
                if i == 0:
                    mask_array = maskHDUs[mask_name].data
                    image_shape = mask_array.shape
                else:
                    assert image_shape == maskHDUs[
                        mask_name].data.shape, '- ERROR: Input masks do not have the same dimensions'
                    mask_array += maskHDUs[mask_name].data

            # Convert to boolean
            mask_array = mask_array.astype(bool)

            # List of spaxels in list [(idx_j, idx_i), ...] format
            spaxel_list = np.argwhere(mask_array)

    # No mask file is provided and the user just defines an image size tupple (nY, nX)
    else:
        exit()

    # Generate containers for the data:
    images_dict = {}
    for param in param_list:
        images_dict[f'{param}'] = np.full(image_shape, default_spaxel_value)
        images_dict[f'{param}_err'] = np.full(image_shape, default_spaxel_value)

    # Loop through the spaxels and fill the parameter images
    n_spaxels = spaxel_list.shape[0]
    spaxel_range = np.arange(n_spaxels)

    with fits.open(log_file_address) as logHDUs:

        for i_spaxel in spaxel_range:
            idx_j, idx_i = spaxel_list[i_spaxel]
            spaxel_ref = f'{idx_j}-{idx_i}{ext_log}'

            progress_bar(i_spaxel, n_spaxels, post_text=f'spaxels treated ({n_spaxels})')

            # Confirm log extension exists
            if spaxel_ref in logHDUs:

                # Recover extension data
                log_data, log_header = logHDUs[spaxel_ref].data, logHDUs[spaxel_ref].header

                # Loop through the parameters and the lines:
                for param in param_list:
                    if param in log_header:
                        images_dict[f'{param}'][idx_j, idx_i] = log_header[param]
                        images_dict[f'{param}_err'][idx_j, idx_i] = log_header[f'{param}_err']

    # New line after the rustic progress bar
    print()

    # Save the parameter maps as individual fits files with one line per page
    output_files_prefix = '' if output_files_prefix is None else output_files_prefix
    for param in param_list:

        # Primary header
        paramHDUs = fits.HDUList()
        paramHDUs.append(fits.PrimaryHDU())

        # ImageHDU for the parameter maps
        hdr = fits.Header({'PARAM': param})
        hdr.update(page_hdr)
        data = images_dict[f'{param}']
        paramHDUs.append(fits.ImageHDU(name=param, data=data, header=hdr, ver=1))

        # ImageHDU for the parameter error maps
        hdr = fits.Header({'PARAMERR': param})
        hdr.update(page_hdr)
        data_err = images_dict[f'{param}_err']
        paramHDUs.append(fits.ImageHDU(name=f'{param}_err', data=data_err, header=hdr, ver=1))

        # Write to new file
        output_file = Path(output_folder) / f'{output_files_prefix}{param}.fits'
        paramHDUs.writeto(output_file, overwrite=True, output_verify='fix')

    return


def numberStringFormat(value, cifras = 4):
    if value > 0.001:
        newFormat = f'{value:.{cifras}f}'
    else:
        newFormat = f'{value:.{cifras}e}'

    return newFormat


def printSimulationData(model, priorsDict, lineLabels, lineFluxes, lineErr, lineFitErr):

    print('\n- Simulation configuration')

    # Print input lines and fluxes
    print('\n-- Input lines')
    for i in range(lineLabels.size):
        warnLine = '{}'.format('|| WARNING obsLineErr = {:.4f}'.format(lineErr[i]) if lineErr[i] != lineFitErr[i] else '')
        displayText = '{} flux = {:.4f} +/- {:.4f} || err % = {:.5f} {}'.format(lineLabels[i], lineFluxes[i], lineFitErr[i], lineFitErr[i] / lineFluxes[i], warnLine)
        print(displayText)

    # Present the model data
    print('\n-- Priors design:')
    for prior in priorsDict:
        displayText = '{} : mu = {}, std = {}'.format(prior, priorsDict[prior][0], priorsDict[prior][1])
        print(displayText)

    # Check test_values are finite
    print('\n-- Test points:')
    model_var = model.test_point
    for var in model_var:
        displayText = '{} = {}'.format(var, model_var[var])
        print(displayText)

    # Checks log probability of random variables
    print('\n-- Log probability variable:')
    print(model.check_test_point())

    return


def format_for_table(entry, rounddig=4, rounddig_er=2, scientific_notation=False, nan_format='-'):

    if rounddig_er == None: #TODO declare a universal tool
        rounddig_er = rounddig

    # Check None entry
    if entry != None:

        # Check string entry
        if isinstance(entry, (str, bytes)):
            formatted_entry = entry

        elif isinstance(entry, (MultiColumn, MultiRow, utils.NoEscape)):
            formatted_entry = entry

        # Case of Numerical entry
        else:

            # Case of an array
            scalarVariable = True
            if isinstance(entry, (Sequence, np.ndarray)):

                # Confirm is not a single value array
                if len(entry) == 1:
                    entry = entry[0]
                # Case of an array
                else:
                    scalarVariable = False
                    formatted_entry = '_'.join(entry)  # we just put all together in a "_' joined string

            # Case single scalar
            if scalarVariable:

                # Case with error quantified # TODO add uncertainty protocol for table
                # if isinstance(entry, UFloat):
                #     formatted_entry = round_sig(nominal_values(entry), rounddig,
                #                                 scien_notation=scientific_notation) + r'$\pm$' + round_sig(
                #         std_devs(entry), rounddig_er, scien_notation=scientific_notation)

                # Case single float
                if np.isnan(entry):
                    formatted_entry = nan_format

                # Case single float
                else:
                    formatted_entry = numberStringFormat(entry, rounddig)
    else:
        # None entry is converted to None
        formatted_entry = 'None'

    return formatted_entry


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
    elif 'line_fitting' in section_label:
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
    elif (',' in entry_value) or ('_array' in key_label) or ('_list' in key_label):

        # Specia cases conversion
        if key_label in ['input_lines']:
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
            output_variable = entry_value.split(',')

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
    elif ('_folder' in key_label) or ('_file' in key_label):
        output_variable = entry_value

    # Check if numeric possible else string
    else:

        if '_list' in key_label:
            output_variable = [entry_value]

        elif '_array' in key_label:
            output_variable = np.array([entry_value], ndmin=1)

        else:
            output_variable = check_numeric_Value(entry_value)

    return output_variable


def formatStringOutput(value, key, section_label=None, float_format=None, nan_format='nan'):

    # TODO this one should be the default option
    # TODO add more cases for dicts
    # Check None entry
    if value is not None:

        # Check string entry
        if isinstance(value, str):
            formatted_value = value

        else:

            # Case of an array
            scalarVariable = True
            if isinstance(value, (Sequence, np.ndarray)):

                # Confirm is not a single value array
                if len(value) == 1:
                    value = value[0]

                # Case of an array
                else:
                    scalarVariable = False
                    formatted_value = ','.join([str(item) for item in value])

            if scalarVariable:

                # Case single float
                if isinstance(value, str):
                    formatted_value = value
                else:
                    if np.isnan(value):
                        formatted_value = nan_format
                    else:
                        formatted_value = str(value)

    else:
        formatted_value = 'None'

    return formatted_value


# Function to map variables to strings
def formatConfEntry(entry_value, float_format=None, nan_format='nan'):
    # TODO this one should be replaced by formatStringEntry
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
                print(entry_value)
                if np.isnan(entry_value):
                    formatted_value = nan_format

                else:
                    formatted_value = str(entry_value)

    else:
        formatted_value = 'None'

    return formatted_value


# Function to import SpecSyzer configuration file
def loadConfData(filepath, objList_check=False, group_variables=False):
    # Open the file
    if Path(filepath).is_file():
        cfg = importConfigFile(filepath)
        # TODO keys with array are always converted to numpy array even if just one
    else:
        exit(f'-ERROR Configuration file not found at:\n{filepath}')

    if group_variables:
        #
        # # Read conversion settings # TODO exclude this metadata from file
        # string_parameter = cfg['conf_entries']['string_conf'].split(',')

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

        confDict = {}

        for section in cfg.sections():
            confDict[section] = {}
            for option_key in cfg.options(section):
                option_value = cfg[section][option_key]
                confDict[section][option_key] = formatStringEntry(option_value, option_key, section)

        if objList_check is True:

            assert 'file_information' in confDict, '- No file_information section in configuration file'
            assert 'object_list' in confDict['file_information'], '- No object_list option in configuration file'
            objList = confDict['file_information']['object_list']

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


# Function to save a parameter key-value item (or list) into the dictionary
def safeConfData(output_file, param_dict, section_name=None, clear_section=False):

    """
    This function safes the input dictionary into a configuration file. If no section is provided the input dictionary
    overwrites the data

    """
    # TODO add mechanic for commented conf lines. Currently they are being erased in the load/safe process

    # Creating a new file (overwritting old if existing)
    if section_name == None:

        # Check all entries are dictionaries
        values_list = [*param_dict.values()]
        section_check = all(isinstance(x, dict) for x in values_list)
        assert section_check, f'ERROR: Dictionary for {output_file} cannot be converted to configuration file. Confirm all its values are dictionaries'

        output_cfg = configparser.ConfigParser()
        output_cfg.optionxform = str

        # Loop throught he sections and options to create the files
        for section_name, options_dict in param_dict.items():
            output_cfg.add_section(section_name)
            for option_name, option_value in options_dict.items():
                option_formatted = formatStringOutput(option_value, option_name, section_name)
                output_cfg.set(section_name, option_name, option_formatted)

        # Save to a text format
        with open(output_file, 'w') as f:
            output_cfg.write(f)

    # Updating old file
    else:

        # Confirm file exists
        file_check = os.path.isfile(output_file)

        # Load original cfg
        if file_check:
            output_cfg = configparser.ConfigParser()
            output_cfg.optionxform = str
            output_cfg.read(output_file)
        # Create empty cfg
        else:
            output_cfg = configparser.ConfigParser()
            output_cfg.optionxform = str

        # Clear section upon request
        if clear_section:
            if output_cfg.has_section(section_name):
                output_cfg.remove_section(section_name)

        # Add new section if it is not there
        if not output_cfg.has_section(section_name):
            output_cfg.add_section(section_name)

        # Map key values to the expected format and store them
        for option_name, option_value in param_dict.items():
            option_formatted = formatStringOutput(option_value, option_name, section_name)
            output_cfg.set(section_name, option_name, option_formatted)

        # Save to a text file
        with open(output_file, 'w') as f:
            output_cfg.write(f)

    return


# Function to save a parameter dictionary into a cfg dictionary
def parseConfDict(output_file, param_dict, section_name, clear_section=False):
    # TODO add logic to erase section previous results
    # TODO add logic to create a new file from dictionary of dictionaries

    # Check if file exists
    if os.path.isfile(output_file):
        output_cfg = configparser.ConfigParser()
        output_cfg.optionxform = str
        output_cfg.read(output_file)
    else:
        # Create new configuration object
        output_cfg = configparser.ConfigParser()
        output_cfg.optionxform = str

    # Clear the section upon request
    if clear_section:
        if output_cfg.has_section(section_name):
            output_cfg.remove_section(section_name)

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
def treat_input_spectrum(output_dict, spec_wave, spec_flux, wavelengh_limits=None, resample_inc=None,
                         norm_interval=None):
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
        if not np.isnan(wmin[i]) and not np.isnan(wmax[i]) and (
                wmax[i] < wavelength[-1]):  # We need this for lines beyong continuum range #TODO propose better
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
def import_emission_line_data(linesLogAddress=None, linesDF=None, include_lines=None, exclude_lines=None,
                              remove_neg=True):



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

    # TODO include the gaussian/integr distinction, normalize option, warning option for error in lime logs
    # Output DF # TODO we need to replace for a open excel format
    if linesDF is None:
        try:
            outputDF = pd.read_excel(linesLogAddress, sheet_name=0, header=0, index_col=0)

        except:
            outputDF = pd.read_csv(linesLogAddress, delim_whitespace=True, header=0, index_col=0)

    else:
        outputDF = linesDF

    if remove_neg:
        idx_neg = outputDF.intg_flux < 0.0
        outputDF.drop(index=outputDF.loc[idx_neg].index.values, inplace=True)

    # Trim with include lines
    if include_lines is not None:
        idx_includeLines = ~(outputDF.index.isin(include_lines))
        outputDF.drop(index=outputDF.loc[idx_includeLines].index.values, inplace=True)

    # Trim with exclude lines
    if exclude_lines is not None:
        idx_excludedLines = outputDF.index.isin(exclude_lines)
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
    if sampler == 'pymc3':
        with open(db_address, 'wb') as trace_pickle:
            pickle.dump({'model': model, 'trace': trace}, trace_pickle)

    return


# Function to restore PYMC3 simulations using pickle
def load_fit_results(output_address, ext_name='', output_format='pickle'):
    # TODO make this a class

    db_address = Path(output_address)

    # Configuration file (traces are not restored)
    if output_format == 'cfg':
        fit_results = loadConfData(db_address)

    # Pickle file (Pymc3 is restored alongside formated inputs and outputs)
    if output_format == 'pickle':
        with open(db_address, 'rb') as trace_restored:
            fit_results = pickle.load(trace_restored)

    # Fits file
    if output_format == 'fits':
        fit_results = {}
        with fits.open(db_address) as hdul:
            for i, sec in enumerate(['inputs', 'outputs', 'traces']):
                sec_label = sec if ext_name == '' else f'{ext_name}_{sec}'
                fit_results[sec_label] = [hdul[sec_label].data, hdul[sec_label].header]

    return fit_results


# Function to save the PYMC3 simulation as a fits log
def fits_db(fits_address, model_db, ext_name='', header=None):

    line_labels = model_db['inputs']['line_list']
    params_traces = model_db['outputs']

    sec_label = 'synthetic_fluxes' if ext_name == '' else f'{ext_name}_synthetic_fluxes'

    # ---------------------------------- Input data

    # Data
    list_columns = []
    for data_label, data_format in FITS_INPUTS_EXTENSION.items():
        data_array = model_db['inputs'][data_label]
        data_col = fits.Column(name=data_label, format=data_format, array=data_array)
        list_columns.append(data_col)

    # Header
    hdr_dict = {}
    for i_line, lineLabel in enumerate(line_labels):
        hdr_dict[f'hierarch {lineLabel}'] = model_db['inputs']['line_fluxes'][i_line]
        hdr_dict[f'hierarch {lineLabel}_err'] = model_db['inputs']['line_err'][i_line]

    # User values:
    for key, value in header.items():
        if key not in ['logP_values', 'r_hat']:
            hdr_dict[f'hierarch {key}'] = value

    # Inputs extension
    cols = fits.ColDefs(list_columns)
    sec_label = 'inputs' if ext_name == '' else f'{ext_name}_inputs'
    hdu_inputs = fits.BinTableHDU.from_columns(cols, name=sec_label, header=fits.Header(hdr_dict))

    # ---------------------------------- Output data
    params_list = model_db['inputs']['parameter_list']
    param_matrix = np.array([params_traces[param] for param in params_list])
    param_col = fits.Column(name='parameters_list', format=FITS_OUTPUTS_EXTENSION['parameter_list'], array=params_list)
    param_val = fits.Column(name='parameters_fit', format='E', array=param_matrix.mean(axis=1))
    param_err = fits.Column(name='parameters_err', format='E', array=param_matrix.std(axis=1))
    list_columns = [param_col, param_val, param_err]

    # Header
    hdr_dict = {}
    for i, param in enumerate(params_list):
        param_trace = params_traces[param]
        hdr_dict[f'hierarch {param}'] = np.mean(param_trace)
        hdr_dict[f'hierarch {param}_err'] = np.std(param_trace)

    for lineLabel in line_labels:
        param_trace = params_traces[lineLabel]
        hdr_dict[f'hierarch {lineLabel}'] = np.mean(param_trace)
        hdr_dict[f'hierarch {lineLabel}_err'] = np.std(param_trace)

    # # Data
    # param_array = np.array(list(params_traces.keys()))
    # paramMatrix = np.array([params_traces[param] for param in param_array])
    #
    # list_columns.append(fits.Column(name='parameter', format='20A', array=param_array))
    # list_columns.append(fits.Column(name='mean', format='E', array=np.mean(paramMatrix, axis=0)))
    # list_columns.append(fits.Column(name='std', format='E', array=np.std(paramMatrix, axis=0)))
    # list_columns.append(fits.Column(name='median', format='E', array=np.median(paramMatrix, axis=0)))
    # list_columns.append(fits.Column(name='p16th', format='E', array=np.percentile(paramMatrix, 16, axis=0)))
    # list_columns.append(fits.Column(name='p84th', format='E', array=np.percentile(paramMatrix, 84, axis=0)))

    cols = fits.ColDefs(list_columns)
    sec_label = 'outputs' if ext_name == '' else f'{ext_name}_outputs'
    hdu_outputs = fits.BinTableHDU.from_columns(cols, name=sec_label, header=fits.Header(hdr_dict))

    # ---------------------------------- traces data
    list_columns = []

    # Data
    for param, trace_array in params_traces.items():
        col_trace = fits.Column(name=param, format='E', array=params_traces[param])
        list_columns.append(col_trace)

    cols = fits.ColDefs(list_columns)

    # Header fitting properties
    hdr_dict = {}
    for stats_dict in ['logP_values', 'r_hat']:
        if stats_dict in header:
            for key, value in header[stats_dict].items():
                hdr_dict[f'hierarch {key}_{stats_dict}'] = value

    sec_label = 'traces' if ext_name == '' else f'{ext_name}_traces'
    hdu_traces = fits.BinTableHDU.from_columns(cols, name=sec_label, header=fits.Header(hdr_dict))

    # ---------------------------------- Save fits files
    hdu_list = [hdu_inputs, hdu_outputs, hdu_traces]

    if fits_address.is_file():
        for hdu in hdu_list:
            try:
                fits.update(fits_address, data=hdu.data, header=hdu.header, extname=hdu.name, verify=True)
            except KeyError:
                fits.append(fits_address, data=hdu.data, header=hdu.header, extname=hdu.name)
    else:
        hdul = fits.HDUList([fits.PrimaryHDU()] + hdu_list)
        hdul.writeto(fits_address, overwrite=True, output_verify='fix')

    return


