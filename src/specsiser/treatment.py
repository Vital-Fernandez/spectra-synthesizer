import numpy as np
import pickle
from lime import label_decomposition
from pathlib import Path

from .inout import parseConfDict, fits_db
from .models import EmissionTensors, PhotoIonizationModels, EmissionFluxModel
from .grids import gridInterpolatorFunction, GridWrapper
from .components.atomic_model import IonEmissivity


def emissivity_grid_calc(lines_array, comp_dict, temp_grid_points=(9000, 20000, 251), den_grid_points=(1, 600, 101)):

    print(f'- Computing emissivity grids for {len(lines_array)} lines\n')

    # Compute the atomic data grids
    objIons = IonEmissivity(tempGrid=temp_grid_points, denGrid=den_grid_points)

    ion_array, wave_array, latex_array = label_decomposition(lines_array, comp_dict=comp_dict)

    # Define the dictionary with the pyneb ion objects
    ionDict = objIons.get_ions_dict(ion_array)

    # Compute the emissivity surfaces for the observed emission lines
    objIons.computeEmissivityGrids(lines_array, ionDict, combined_dict=comp_dict)

    # Compile exoplanet interpolator functions so they can be used wit numpy
    emisGridInterpFun = gridInterpolatorFunction(objIons.emisGridDict, objIons.tempRange, objIons.denRange)

    print(f'-- completed')

    return emisGridInterpFun


class SpectraSynthesizer(GridWrapper, PhotoIonizationModels):

    def __init__(self, emisGridInterFun=None, emis_tensors=None, grid_interp=None, ftau_coeff=None, grid_sampling=False):

        GridWrapper.__init__(self)
        PhotoIonizationModels.__init__(self)

        # Security checks
        self.lowTemp_check = None
        self.highTemp_check = None
        self.idcs_highTemp_ions = None
        self.grid_check = grid_sampling

        # Indexes
        self.indcsLabelLines = {}
        self.indcsIonLines = {}
        self.idcs_highTemp_ions = None
        self.ionicAbundCheck = {}

        # Simulation preload data
        self.ftauCoef = ftau_coeff
        self.emisGridInterpFun = emisGridInterFun
        self.emtt = emis_tensors
        self.gridInterp = grid_interp

        # Number of regions
        self.total_regions = 1

        # Output container
        self.fit_results = None

    def define_region(self, line_labels, line_fluxes, line_errs, lineFlambda=None, comp_dict=None, minErr=0.02):

        # Lines data
        ion_array, wave_array, latexLabel_array = label_decomposition(line_labels)
        self.lineLabels = line_labels
        self.lineIons = ion_array
        self.emissionFluxes = line_fluxes
        self.emissionErr = line_errs if line_errs is not None else line_fluxes * minErr
        self.lineFlambda = lineFlambda

        # Establish minimum error on lines
        if minErr is not None:
            err_fraction = self.emissionErr / self.emissionFluxes
            idcs_smallErr = err_fraction < minErr
            self.emissionErr[idcs_smallErr] = minErr * self.emissionFluxes[idcs_smallErr]

        # Parameters separation for direct and grid sampling models
        if not self.grid_check:

            # If not provided compute emissivity interpolation functions
            if self.emisGridInterpFun is None:
                self.emisGridInterpFun = emissivity_grid_calc(line_labels, comp_dict)

            # If not provided compute the emission flux tensors
            if self.emtt is None:
                self.emtt = EmissionFluxModel(self.lineLabels, self.lineIons)

        # else:
        #     self.HII_Teff_models(self.lineLabels, self.emissionFluxes, self.emissionErr)

        return

    def simulation_configuration(self, prior_conf_dict, highTempIons=None, T_low_diag='S3_6312A', T_high_diag='O3_4363A',
                                 verbose=True,):

        # Priors configuration
        for key, value in prior_conf_dict.items():
            if '_prior' in key:
                param = key.split('_prior')[0]
                priorConf = prior_conf_dict[param + '_prior']
                self.priorDict[param] = priorConf

        if 'logParams_list' in prior_conf_dict:
            self.priorDict['logParams_list'] = prior_conf_dict['logParams_list']

        # Load photoIonization models
        if not self.grid_check:

            self.idx_analysis_lines = np.zeros(self.lineLabels.size)

            # High and low temperature distinction
            self.lowTemp_check = any(T_low_diag in lineLabel for lineLabel in self.lineLabels) # TODO Change this for the intercept method
            self.highTemp_check = any(T_high_diag in lineLabel for lineLabel in self.lineLabels)

            # Index the lines
            self.label_ion_features(self.lineLabels, highTempIons)

        # # Interpolator object
        # if grid_interpolator is not None:
        #     self.gridInterp = grid_interpolator

        # self.obsIons = chemistry_model.obsIons
        # self.idcs_highTemp_ions = chemistry_model.indcsHighTemp # TODO this is dangerous repeat out

        if verbose:
            for i in range(self.lineLabels.size):
                print(f'-- {self.lineLabels[i]} '
                      f'({self.lineIons[i]})'
                      f'flux = {self.emissionFluxes[i]:.4f} +/- {self.emissionErr[i]:.4f} '
                      f'|| err/flux = {100 * self.emissionErr[i] / self.emissionFluxes[i]:.2f} %')

        return

    def label_ion_features(self, line_labels, highTempIons=None):

        ion_array, wavelength_array, latexLabel_array = label_decomposition(line_labels)

        # Establish the ions from the available lines
        self.obsIons = np.unique(ion_array)

        # Determine the line indeces
        for line in line_labels:
            self.indcsLabelLines[line] = (line_labels == line)

        # Determine the lines belonging to observed ions
        for ion in self.obsIons:
            self.indcsIonLines[ion] = (ion_array == ion)

        # Establish index of lines which below to high and low ionization zones # TODO increase flexibility for more Te
        if highTempIons is not None:
            self.idcs_highTemp_ions = np.in1d(ion_array, highTempIons)
        else:
            self.idcs_highTemp_ions = np.zeros(line_labels.size, dtype=bool)

        # Establish the ionic abundance logic from the available lines
        for ion in self.obsIons:
            self.ionicAbundCheck[ion] = True if ion in self.obsIons else False

        return

    def save_fit(self, output_address, ext_name='', output_format='pickle', user_header={}):

        output_path = Path(output_address)
        file_stem, file_root = output_path.stem, output_path.suffix

        if output_format == 'cfg':

            # Input data
            input_data = self.fit_results['inputs']
            sec_label = 'inputs' if ext_name == '' else f'{ext_name}_inputs'
            sec_dict = {}
            for i, lineLabel in enumerate(input_data['lines_list']):
                lineFlux, lineErr = input_data['line_fluxes'][i], input_data['line_err'][i]
                sec_dict[lineLabel] = np.array([lineFlux, lineErr])
            sec_dict['parameter_list'] = input_data['parameter_list']
            parseConfDict(str(output_address), sec_dict, section_name=sec_label, clear_section=True)

            # Output data
            sec_label = 'outputs' if ext_name == '' else f'{ext_name}_outputs'
            sec_dict = {}
            for param in self.fit_results['inputs']['parameter_list']:
                param_trace = self.fit_results['outputs'][param]
                sec_dict[param] = np.array([np.mean(param_trace), np.std(param_trace)])
            parseConfDict(str(output_address), sec_dict, section_name=sec_label, clear_section=True)

            # Synthetic fluxes
            sec_label = 'synthetic_fluxes' if ext_name == '' else f'{ext_name}_synthetic_fluxes'
            sec_dict = {}
            for lineLabel in self.fit_results['inputs']['lines_list']:
                line_trace = self.fit_results['outputs'][lineLabel]
                sec_dict[lineLabel] = np.array([np.mean(line_trace), np.std(line_trace)])
            parseConfDict(str(output_address), sec_dict, section_name=sec_label, clear_section=True)

        if output_format == 'pickle':
            with open(output_address, 'wb') as db_pickle:
                pickle.dump(self.fit_results, db_pickle)

        if output_format == 'fits':
            user_header['logP_values'] = dict(self.inferenModel.check_test_point())
            # user_header['r_hat'] = dict(pymc3.summary(self.fit_results['trace'])['r_hat']) # TODO check r_hat values
            fits_db(output_path, model_db=self.fit_results, ext_name=ext_name, header=user_header)

