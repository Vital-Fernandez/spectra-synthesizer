import os
import numpy as np
import pyneb as pn
from inspect import getfullargspec
from scipy.optimize import curve_fit
from data_reading import import_optical_depth_coeff_table


def compute_emissivity_grid(tempGrid, denGrid):
    tempRange = np.linspace(tempGrid[0], tempGrid[1], tempGrid[2])
    denRange = np.linspace(denGrid[0], denGrid[1], denGrid[2])
    X, Y = np.meshgrid(tempRange, denRange)
    tempGridFlatten, denGridFlatten = X.flatten(), Y.flatten()

    return tempGridFlatten, denGridFlatten


class EmissivitySurfaceFitter():

    def __init__(self):

        self.tempGridFlatten = None
        self.denGridFlatten = None
        self.emisGridDict = None
        self.emisCoeffs = None

        # Class with the tensor operations of this class
        # TODO we should read this from the xlsx file
        self.ionEmisEq_fit = {'S2_6716A': self.emisEquation_TeDe,
                              'S2_6731A': self.emisEquation_TeDe,
                              'S3_6312A': self.emisEquation_Te,
                              'S3_9069A': self.emisEquation_Te,
                              'S3_9531A': self.emisEquation_Te,
                              'Ar4_4740A': self.emisEquation_Te,
                              'Ar3_7136A': self.emisEquation_Te,
                              'Ar3_7751A': self.emisEquation_Te,
                              'O3_4363A': self.emisEquation_Te,
                              'O3_4959A': self.emisEquation_Te,
                              'O3_5007A': self.emisEquation_Te,
                              'O2_7319A': self.emisEquation_TeDe,
                              'O2_7330A': self.emisEquation_TeDe,
                              'O2_7319A_b': self.emisEquation_TeDe,
                              'N2_6548A': self.emisEquation_Te,
                              'N2_6584A': self.emisEquation_Te,
                              'H1_4102A': self.emisEquation_HI,
                              'H1_4341A': self.emisEquation_HI,
                              'H1_6563A': self.emisEquation_HI,
                              'He1_3889A': self.emisEquation_HeI_fit,
                              'He1_4026A': self.emisEquation_HeI_fit,
                              'He1_4471A': self.emisEquation_HeI_fit,
                              'He1_5876A': self.emisEquation_HeI_fit,
                              'He1_6678A': self.emisEquation_HeI_fit,
                              'He1_7065A': self.emisEquation_HeI_fit,
                              'He1_10830A': self.emisEquation_HeI_fit,
                              'He2_4686A': self.emisEquation_HeII_fit}

        # Initial coeffient values to help with the fitting
        self.epm2017_emisCoeffs = {'He1_3889A': np.array([0.173, 0.00054, 0.904, 1e-5]),
                                   'He1_4026A': np.array([-0.09, 0.0000063, 4.297, 1e-5]),
                                   'He1_4471A': np.array([-0.1463, 0.0005, 2.0301, 1.5e-5]),
                                   'He1_5876A': np.array([-0.226, 0.0011, 0.745, -5.1e-5]),
                                   'He1_6678A': np.array([-0.2355, 0.0016, 2.612, 0.000146]),
                                   'He1_7065A': np.array([0.368, 0.0017, 4.329, 0.0024]),
                                   'He1_10830A': np.array([0.14, 0.00189, 0.337, -0.00027])}

        return

    def emisEquation_Te(self, xy_space, a, b, c):
        temp_range, den_range = xy_space
        return a + b / (temp_range / 10000.0) + c * np.log10(temp_range / 10000)

    def emisEquation_TeDe(self, xy_space, a, b, c, d, e):
        temp_range, den_range = xy_space
        return a + b / (temp_range / 10000.0) + c * np.log10(temp_range / 10000) + np.log10(1 + e * den_range)

    def emisEquation_HI(self, xy_space, a, b, c):
        temp_range, den_range = xy_space
        return a + b * np.log10(temp_range) + c * np.log10(temp_range) * np.log10(temp_range)

    def emisEquation_HeI_fit(self, xy_space, a, b, c, d):
        temp_range, den_range = xy_space
        return (a + b * den_range) * np.log10(temp_range / 10000.0) - np.log10(c + d * den_range)

    def emisEquation_HeII_fit(self, xy_space, a, b):
        temp_range, den_range = xy_space
        return a + b * np.log(temp_range / 10000)

    def fitEmis(self, func_emis, xy_space, line_emis, p0=None):
        p1, p1_cov = curve_fit(func_emis, xy_space, line_emis, p0)
        return p1, p1_cov

    def fitEmissivityPlane(self, linesDF):

        labels_list = linesDF.index.values

        # Dictionary to store the emissivity surface coeffients
        self.emisCoeffs = {}
        for i in range(labels_list.size):
            lineLabel = labels_list[i]

            # Get equation type to fit the emissivity
            line_func = self.ionEmisEq_fit[lineLabel]
            n_args = len(getfullargspec(
                line_func).args) - 2  # TODO Not working in python 2.7 https://stackoverflow.com/questions/847936/how-can-i-find-the-number-of-arguments-of-a-python-function

            # Compute emissivity functions coefficients
            emis_grid_i = self.emisGridDict[lineLabel]
            p0 = self.epm2017_emisCoeffs[lineLabel] if lineLabel in self.epm2017_emisCoeffs else np.zeros(n_args)
            p1, cov1 = self.fitEmis(line_func, (self.tempGridFlatten, self.denGridFlatten), emis_grid_i, p0=p0)
            self.emisCoeffs[lineLabel] = p1

        return


class IonEmissivity(EmissivitySurfaceFitter):

    def __init__(self, atomic_references=None, ftau_file_path=None, tempGrid=None, denGrid=None):

        self.ftau_coeffs = None

        EmissivitySurfaceFitter.__init__(self)

        # Load user atomic data references # TODO upgrade the right method? # CHECK for pyneb abundances
        if atomic_references is not None:
            pn.atomicData.defaultDict = atomic_references
            pn.atomicData.resetDataFileDict()

        # Import Optical depth function coefficients
        if ftau_file_path is not None:
            self.ftau_coeffs = import_optical_depth_coeff_table(ftau_file_path)

        # Defining temperature and density grids
        if (tempGrid is not None) and (denGrid is not None):
            self.tempGridFlatten, self.denGridFlatten = compute_emissivity_grid(tempGrid, denGrid)

    def get_ions_dict(self, ions_list, atomic_references=pn.atomicData.defaultDict):

        # Check if the atomic dataset is the default one
        if atomic_references == pn.atomicData.defaultDict:
            pn.atomicData.resetDataFileDict()
            pn.atomicData.removeFitsPath()
        else:
            pn.atomicData.includeFitsPath()
            pn.atomicData.setDataFileDict(atomic_references)

        # Generate the dictionary with pyneb ions
        ionDict = pn.getAtomDict(ions_list)

        return ionDict

    def computeEmissivityGrid(self, linesDF, ionDict, grids_folder=None, load_grids=False, norm_Ion='H1r', norm_pynebCode=4861,
                              linesDb=None):

        labels_list = linesDF.index.values
        ions_list = linesDF.ion.values
        pynebCode_list = linesDF.pynebCode.values
        blended_list = linesDF.blended.values

        # Generate a grid with the default reference line
        Hbeta_emis_grid = ionDict[norm_Ion].getEmissivity(self.tempGridFlatten, self.denGridFlatten,
                                                               wave=norm_pynebCode, product=False)

        self.emisGridDict = {}
        for i in range(len(labels_list)):

            # Line emissivity references
            line_label = labels_list[i]

            if (grids_folder is not None) and load_grids:
                emis_grid_i = np.load(grids_folder, line_label)

            # Otherwise generate it (and save it)
            else:

                # Check if it is a blended line:
                if '_b' not in line_label:
                    # TODO I should change wave by label
                    emis_grid_i = ionDict[ions_list[i]].getEmissivity(self.tempGridFlatten, self.denGridFlatten,
                                                                           wave=float(pynebCode_list[i]), product=False)
                else:
                    for component in blended_list[i].split(','):
                        component_wave = float(linesDb.loc[component].pynebCode)
                        emis_grid_i += ionDict[ions_list[i]].getEmissivity(self.tempGridFlatten,
                                                                                self.denGridFlatten,
                                                                                wave=component_wave, product=False)
                if (grids_folder is not None):
                    np.save(grids_folder, emis_grid_i)

            # Save along the number of points
            self.emisGridDict[line_label] = np.log10(emis_grid_i / Hbeta_emis_grid)

        return

