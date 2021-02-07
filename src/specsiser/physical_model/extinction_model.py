import pyneb as pn
import numpy as np
import os
from scipy.interpolate import interp1d


class ExtinctionModel:

    def __init__(self, Rv=None, red_curve=None, data_folder=None):

        self.R_v = Rv
        self.red_curve = red_curve

        # Dictionary with the reddening curves
        self.reddening_curves_calc = {'MM72': self.f_Miller_Mathews1972,
                                      'CCM89': self.X_x_Cardelli1989,
                                      'G03_bar': self.X_x_Gordon2003_bar,
                                      'G03_average': self.X_x_Gordon2003_average,
                                      'G03_supershell': self.X_x_Gordon2003_supershell}

        self.literatureDataFolder = data_folder

    def reddening_correction(self, wave, flux, reddening_curve, cHbeta=None, E_BV=None, R_v=None):

        # By default we perform the calculation using the colour excess
        E_BV = E_BV if E_BV is not None else self.Ebv_from_cHbeta(cHbeta, reddening_curve, R_v)

        # Perform reddening correction
        wavelength_range_Xx = self.reddening_Xx(wave, reddening_curve, R_v)
        flux_range_derred = flux * np.power(10, 0.4 * wavelength_range_Xx * E_BV)

        return flux_range_derred

    def Ebv_from_cHbeta(self, cHbeta, reddening_curve, R_v):

        E_BV = cHbeta * 2.5 / self.reddening_Xx(np.array([self.Hbeta_wavelength]), reddening_curve, R_v)[0]
        return E_BV

    def flambda_from_Xx(self, Xx, reddening_curve, R_v):

        X_Hbeta = self.reddening_Xx(np.array([self.Hbeta_wavelength]), reddening_curve, R_v)[0]

        f_lines = Xx / X_Hbeta - 1

        return f_lines

    def reddening_Xx(self, waves, curve_methodology, R_v):

        self.R_v = R_v
        self.wavelength_rc = waves
        return self.reddening_curves_calc[curve_methodology]()

    def f_Miller_Mathews1972(self):

        if isinstance(self.wavelength_rc, np.ndarray):
            y = 1.0 / (self.wavelength_rc / 10000.0)
            y_beta = 1.0 / (4862.683 / 10000.0)

            ind_low = np.where(y <= 2.29)[0]
            ind_high = np.where(y > 2.29)[0]

            dm_lam_low = 0.74 * y[ind_low] - 0.34 + 0.341 * self.R_v - 1.014
            dm_lam_high = 0.43 * y[ind_high] + 0.37 + 0.341 * self.R_v - 1.014
            dm_beta = 0.74 * y_beta - 0.34 + 0.341 * self.R_v - 1.014

            dm_lam = np.concatenate((dm_lam_low, dm_lam_high))

            f = dm_lam / dm_beta - 1

        else:

            y = 1.0 / (self.wavelength_rc / 10000.0)
            y_beta = 1.0 / (4862.683 / 10000.0)

            if y <= 2.29:
                dm_lam = 0.74 * y - 0.34 + 0.341 * self.R_v - 1.014
            else:
                dm_lam = 0.43 * y + 0.37 + 0.341 * self.R_v - 1.014

            dm_beta = 0.74 * y_beta - 0.34 + 0.341 * self.R_v - 1.014

            f = dm_lam / dm_beta - 1

        return f

    def X_x_Cardelli1989(self):

        x_true = 1.0 / (self.wavelength_rc / 10000.0)
        y = x_true - 1.82

        y_coeffs = np.array(
            [np.ones(len(y)), y, np.power(y, 2), np.power(y, 3), np.power(y, 4), np.power(y, 5), np.power(y, 6),
             np.power(y, 7)])
        a_coeffs = np.array([1, 0.17699, -0.50447, -0.02427, 0.72085, 0.01979, -0.77530, 0.32999])
        b_coeffs = np.array([0, 1.41338, 2.28305, 1.07233, -5.38434, -0.62251, 5.30260, -2.09002])

        a_x = np.dot(a_coeffs, y_coeffs)
        b_x = np.dot(b_coeffs, y_coeffs)

        X_x = a_x + b_x / self.R_v

        return X_x

    def X_x_Gordon2003_bar(self):

        # Default R_V is 3.4
        R_v = self.R_v if self.R_v != None else 3.4  # This is not very nice
        x = 1.0 / (self.wavelength_rc / 10000.0)

        # This file format has 1/um in column 0 and A_x/A_V in column 1
        curve_address = os.path.join(self.literatureDataFolder, 'gordon_2003_SMC_bar.txt')
        file_data = np.loadtxt(curve_address)

        # This file has column
        Xx_interpolator = interp1d(file_data[:, 0], file_data[:, 1])
        X_x = R_v * Xx_interpolator(x)
        return X_x

    def X_x_Gordon2003_average(self):

        # Default R_V is 3.4
        R_v = self.R_v if self.R_v != None else 3.4  # This is not very nice
        x = 1.0 / (self.wavelength_rc / 10000.0)

        # This file format has 1/um in column 0 and A_x/A_V in column 1
        curve_address = os.path.join(self.literatureDataFolder, 'gordon_2003_LMC_average.txt')
        file_data = np.loadtxt(curve_address)

        # This file has column
        Xx_interpolator = interp1d(file_data[:, 0], file_data[:, 1])
        X_x = R_v * Xx_interpolator(x)
        return X_x

    def X_x_Gordon2003_supershell(self):

        # Default R_V is 3.4
        R_v = self.R_v if self.R_v != None else 3.4  # This is not very nice
        x = 1.0 / (self.wavelength_rc / 10000.0)

        # This file format has 1/um in column 0 and A_x/A_V in column 1
        curve_address = os.path.join(self.literatureDataFolder, 'gordon_2003_LMC2_supershell.txt')
        file_data = np.loadtxt(curve_address)

        # This file has column
        Xx_interpolator = interp1d(file_data[:, 0], file_data[:, 1])
        X_x = R_v * Xx_interpolator(x)
        return X_x

    def Epm_ReddeningPoints(self):

        x_true = np.arange(1.0, 2.8, 0.1)  # in microns -1
        X_Angs = 1 / x_true * 1e4

        Xx = np.array(
            [1.36, 1.44, 1.84, 2.04, 2.24, 2.44, 2.66, 2.88, 3.14, 3.36, 3.56, 3.77, 3.96, 4.15, 4.26, 4.40, 4.52,
             4.64])
        f_lambda = np.array(
            [-0.63, -0.61, -0.5, -0.45, -0.39, -0.34, -0.28, -0.22, -0.15, -0.09, -0.03, 0.02, 0.08, 0.13, 0.16, 0.20,
             0.23, 0.26])

        return x_true, X_Angs, Xx, f_lambda

    def gasExtincParams(self, wave, R_v = None, red_curve = None, normWave = 4861.331):

        if R_v is None:
            R_v = self.R_v
        if red_curve is None:
            red_curve = self.red_curve

        self.rcGas = pn.RedCorr(R_V=R_v, law=red_curve)

        HbetaXx = self.rcGas.X(normWave)
        lineXx = self.rcGas.X(wave)

        lineFlambda = lineXx / HbetaXx - 1.0

        return lineFlambda

    def contExtincParams(self, wave, Rv, reddening_law):

        self.rcCont = pn.RedCorr(R_V=Rv, law=reddening_law)

        lineXx = self.rcGas.X(wave)

        return lineXx