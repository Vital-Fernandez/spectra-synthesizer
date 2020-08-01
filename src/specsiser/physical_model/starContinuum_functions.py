from collections import OrderedDict
from scipy.signal.signaltools import convolve2d
from scipy.interpolate.interpolate import interp1d
from scipy.optimize import nnls
from numpy import power, max, square, ones, arange, exp, zeros, transpose,  diag, linalg, dot, outer, empty, ceil



# Reddening law from CCM89 # TODO Replace this methodology to an Import of Pyneb
def CCM89_Bal07(Rv, wave):
    x = 1e4 / wave  # Assuming wavelength is in Amstrongs
    ax = zeros(len(wave))
    bx = zeros(len(wave))

    idcs = x > 1.1
    y = (x[idcs] - 1.82)

    ax[idcs] = 1 + 0.17699 * y - 0.50447 * y ** 2 - 0.02427 * y ** 3 + 0.72085 * y ** 4 + 0.01979 * y ** 5 - 0.77530 * y ** 6 + 0.32999 * y ** 7
    bx[idcs] = 1. * y + 2.28305 * y ** 2 + 1.07233 * y ** 3 - 5.38434 * y ** 4 - 0.62251 * y ** 5 + 5.30260 * y ** 6 - 2.09002 * y ** 7
    ax[~idcs] = 0.574 * x[~idcs] ** 1.61
    bx[~idcs] = -0.527 * x[~idcs] ** 1.61

    Xx = ax + bx / Rv  # WARNING better to check this definition

    return Xx

class SspFitter():

    def __init__(self):

        self.ssp_conf_dict = OrderedDict()

    def physical_SED_model(self, bases_wave_rest, obs_wave, bases_flux, Av_star, z_star, sigma_star, Rv_coeff=3.4):

        # Calculate wavelength at object z
        wave_z = bases_wave_rest * (1 + z_star)

        # Kernel matrix
        box = int(ceil(max(3 * sigma_star)))
        kernel_len = 2 * box + 1
        kernel_range = arange(0, 2 * box + 1)
        kernel = empty((1, kernel_len))

        # Filling gaussian values (the norm factor is the sum of the gaussian)
        kernel[0, :] = exp(-0.5 * (square((kernel_range - box) / sigma_star)))
        kernel /= sum(kernel[0, :])

        # Convove bases with respect to kernel for dispersion velocity calculation
        basesGridConvolved = convolve2d(bases_flux, kernel, mode='same', boundary='symm')

        # Interpolate bases to wavelength ranges
        basesGridInterp = (interp1d(wave_z, basesGridConvolved, axis=1, bounds_error=True)(obs_wave)).T

        # Generate final flux model including reddening
        Av_vector = Av_star * ones(basesGridInterp.shape[1])
        obs_wave_resam_rest = obs_wave / (1 + z_star)
        Xx_redd = CCM89_Bal07(Rv_coeff, obs_wave_resam_rest)
        dust_attenuation = power(10, -0.4 * outer(Xx_redd, Av_vector))
        bases_grid_redd = basesGridInterp * dust_attenuation

        return bases_grid_redd

    def ssp_fitting(self, ssp_grid_masked, obs_flux_masked):

        optimize_result = nnls(ssp_grid_masked, obs_flux_masked)

        return optimize_result[0]

    def linfit1d(self, obsFlux_norm, obsFlux_mean, basesFlux, weight):

        nx, ny = basesFlux.shape

        # Case where the number of pixels is smaller than the number of bases
        if nx < ny:
            basesFlux = transpose(basesFlux)
            nx = ny

        A = basesFlux
        B = obsFlux_norm

        # Weight definition #WARNING: Do we need to use the diag?
        if weight.shape[0] == nx:
            weight = diag(weight)
            A = dot(weight, A)
            B = dot(weight, transpose(B))
        else:
            B = transpose(B)

        coeffs_0 = dot(linalg.inv(dot(A.T, A)), dot(A.T, B)) * obsFlux_mean

        return coeffs_0










