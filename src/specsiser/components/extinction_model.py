import pyneb as pn
import numpy as np
import os # TODO update to pathlib only
import pathlib
from scipy.interpolate import interp1d
from uncertainties import unumpy, ufloat
from lmfit.models import LinearModel
from matplotlib import pyplot as plt, rcParams
import mplcursors
from lime import label_decomposition

def flambda_calc(wave, R_V, red_curve):

    ext_obj = ExtinctionModel(R_V, red_curve)

    f_lambda = ext_obj.gasExtincParams(wave)

    return f_lambda

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

    def reddening_correction(self, wave, flux, err_flux=None, reddening_curve=None, cHbeta=None, E_BV=None, R_v=None, normWave=4861.331):

        # By default we perform the calculation using the colour excess
        if E_BV is not None:

            E_BV = E_BV if E_BV is not None else self.Ebv_from_cHbeta(cHbeta, reddening_curve, R_v)

            # Perform reddening correction
            wavelength_range_Xx = self.reddening_Xx(wave, reddening_curve, R_v)
            int_array = flux * np.power(10, 0.4 * wavelength_range_Xx * E_BV)

        else:
            lines_flambda = self.gasExtincParams(wave, R_v=R_v, red_curve=reddening_curve, normWave=normWave)

            if np.isscalar(cHbeta):
                int_array = flux * np.pow(10, cHbeta * lines_flambda)

            else:
                cHbeta = ufloat(cHbeta[0], cHbeta[1]),
                obsFlux_uarray = unumpy.uarray(flux, err_flux)

                int_uarray = obsFlux_uarray * unumpy.pow(10, cHbeta * lines_flambda)
                int_array = (unumpy.nominal_values(int_uarray), unumpy.std_devs(int_uarray))

        return int_array

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

    def cHbeta_from_log(self, line_df, line_labels='all', temp=10000.0, den=100.0, ref_wave='H1_4861A',
                        comp_mode='auto', plot_address=False):

        # Use all hydrogen lines if none are defined
        if np.isscalar(line_labels):
            if line_labels == 'all':
                idcs_H1 = line_df.ion == 'H1'
                line_labels = line_df.loc[idcs_H1].index.values

        assert len(line_labels) > 0, f'- ERROR: No H1 ion transition lines were found in log. Check dataframe data.'

        # Loop through the input lines
        assert ref_wave in line_df.index, f'- ERROR: {ref_wave} not found in input lines log dataframe for c(Hbeta) calculation'

        # Label the lines which are found in the lines log
        idcs_lines = line_df.index.isin(line_labels) & (line_df.intg_flux > 0) & (line_df.gauss_flux > 0)
        line_labels = line_df.loc[idcs_lines].index.values
        ion_ref, waves_ref, latexLabels_ref = label_decomposition(ref_wave, scalar_output=True)
        ion_array, waves_array, latexLabels_array = label_decomposition(line_labels)

        # Observed ratios
        if comp_mode == 'auto':
            Href_flux, Href_err = line_df.loc[ref_wave, 'intg_flux'], line_df.loc[ref_wave, 'intg_err']
            obsFlux, obsErr = np.empty(line_labels.size), np.empty(line_labels.size)
            slice_df = line_df.loc[idcs_lines]
            idcs_intg = slice_df.blended_label == 'None'
            obsFlux[idcs_intg] = slice_df.loc[idcs_intg, 'intg_flux'].values
            obsErr[idcs_intg] = slice_df.loc[idcs_intg, 'intg_err'].values
            obsFlux[~idcs_intg] = slice_df.loc[~idcs_intg, 'gauss_flux'].values
            obsErr[~idcs_intg] = slice_df.loc[~idcs_intg, 'gauss_err'].values
            obsRatio_uarray = unumpy.uarray(obsFlux, obsErr) / ufloat(Href_flux, Href_err) # TODO unumpy this with your own model

        elif comp_mode == 'gauss':
            Href_flux, Href_err = line_df.loc[ref_wave, 'gauss_flux'], line_df.loc[ref_wave, 'gauss_err']
            obsFlux, obsErr = line_df.loc[idcs_lines, 'gauss_flux'], line_df.loc[idcs_lines, 'gauss_err']
            obsRatio_uarray = unumpy.uarray(obsFlux, obsErr) / ufloat(Href_flux, Href_err)

        else:
            Href_flux, Href_err = line_df.loc[ref_wave, 'intg_flux'], line_df.loc[ref_wave, 'intg_err']
            obsFlux, obsErr = line_df.loc[idcs_lines, 'intg_flux'], line_df.loc[idcs_lines, 'intg_err']
            obsRatio_uarray = unumpy.uarray(obsFlux, obsErr) / ufloat(Href_flux, Href_err)

        assert not np.any(np.isnan(obsFlux)) in obsFlux, '- ERROR: nan entry in input fluxes for c(Hbeta) calculation'
        assert not np.any(np.isnan(obsErr)) in obsErr, '- ERROR: nan entry in input uncertainties for c(Hbeta) calculation'

        # Theoretical ratios
        H1 = pn.RecAtom('H', 1)
        refEmis = H1.getEmissivity(tem=temp, den=den, wave=waves_ref)
        emisIterable = (H1.getEmissivity(tem=temp, den=den, wave=wave) for wave in waves_array)
        linesEmis = np.fromiter(emisIterable, float)
        theoRatios = linesEmis / refEmis

        # Reddening law
        rc = pn.RedCorr(R_V=self.R_v, law=self.red_curve)
        Xx_ref, Xx = rc.X(waves_ref), rc.X(waves_array)
        f_lines = Xx / Xx_ref - 1
        f_ref = Xx_ref / Xx_ref - 1

        # cHbeta linear fit values
        x_values = f_lines - f_ref
        y_values = np.log10(theoRatios) - unumpy.log10(obsRatio_uarray)

        # Perform fit
        lineModel = LinearModel()
        y_nom, y_std = unumpy.nominal_values(y_values), unumpy.std_devs(y_values)
        pars = lineModel.make_params(intercept=y_nom.min(), slope=0)
        output = lineModel.fit(y_nom, pars, x=x_values, weights=1/y_std)
        cHbeta, cHbeta_err = output.params['slope'].value, output.params['slope'].stderr
        intercept, intercept_err = output.params['intercept'].value, output.params['intercept'].stderr

        if plot_address:

            STANDARD_PLOT = {'figure.figsize': (14, 7), 'axes.titlesize': 12, 'axes.labelsize': 14,
                             'legend.fontsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10}

            axes_dict = {'xlabel': r'$f_{\lambda} - f_{H\beta}$',
                         'ylabel': r'$ \left(\frac{I_{\lambda}}{I_{\H\beta}}\right)_{Theo} - \left(\frac{F_{\lambda}}{F_{\H\beta}}\right)_{Obs}$',
                         'title': f'Logaritmic extinction coefficient calculation'}

            rcParams.update(STANDARD_PLOT)

            fig, ax = plt.subplots(figsize=(8, 4))
            fig.subplots_adjust(bottom=-0.7)

            # Data ratios
            err_points = ax.errorbar(x_values, y_nom, y_std, fmt='o')

            # Linear fitting
            linear_fit = cHbeta * x_values + intercept
            linear_label = r'$c(H\beta)={:.2f}\pm{:.2f}$'.format(cHbeta, cHbeta_err)
            ax.plot(x_values, linear_fit, linestyle='--', label=linear_label)
            ax.update(axes_dict)

            # Legend
            ax.legend(loc='best')
            ax.set_ylim(-0.5, 0.5)

            # Generate plot
            plt.tight_layout()
            if isinstance(plot_address, (str, pathlib.WindowsPath, pathlib.PosixPath)):
                # crs = mplcursors.cursor(ax, hover=True)
                # crs.connect("add", lambda sel: sel.annotation.set_text(sel.annotation))
                plt.savefig(plot_address, dpi=200, bbox_inches='tight')
            else:
                mplcursors.cursor(ax).connect("add", lambda sel: sel.annotation.set_text(latexLabels_array[sel.target.index]))
                plt.show()

        return cHbeta, cHbeta_err