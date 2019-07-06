import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import theano.tensor as tt
from theano import function
from pymc3.math import abs_

def storeValueInTensor(idx, value, tensor1D):
    return tt.inc_subtensor(tensor1D[idx], value)

class BilinearInterpTheano():

    def __init__(self):

        self.indexing_bilinear()

        self.operation_bilinear()

        return

    def neighbour_points(self, x0_idx, y0_idx, x_range, y_range, image_matrix):
        # x1, x2 = x_range[x0_idx:x0_idx + 2]
        # y1, y2 = y_range[y0_idx:y0_idx + 2]
        #
        # z11, z12 = image_matrix[y0_idx][x0_idx:x0_idx + 2]
        # z21, z22 = image_matrix[y0_idx + 1][x0_idx:x0_idx + 2]

        x1, x2 = x_range[x0_idx], x_range[x0_idx + 1]
        y1, y2 = y_range[y0_idx], y_range[y0_idx + 1]
        z11, z12 = image_matrix[x0_idx, y0_idx], image_matrix[x0_idx, y0_idx + 1]
        z21, z22 = image_matrix[x0_idx + 1, y0_idx], image_matrix[x0_idx + 1, y0_idx + 1]

        return x1, x2, y1, y2, z11, z12, z21, z22

    def indexing_bilinear(self):
        # Declare variables
        a = tt.dvector()
        b = tt.dscalar()

        # Build symbolic expression
        out = tt.argmin(tt.abs_(a - b))

        # Compile function
        self.tt_index_array = function([a, b], out)

    def indexing_bilinear_pymc3(self, range_array, in_cord):
        abs_dif = abs_(range_array - in_cord)

        out = tt.argmin(abs_dif)

        return out

    def operation_bilinear(self):
        # Declare variables
        te_1, te_2, ne_1, ne_2, emis_11, emis_12, emis_21, emis_22, x_in, y_in = tt.dscalars(10)

        # Output function
        out_bInterp = (emis_11 * (te_2 - x_in) * (ne_2 - y_in) +
                       emis_21 * (x_in - te_1) * (ne_2 - y_in) +
                       emis_12 * (te_2 - x_in) * (y_in - ne_1) +
                       emis_22 * (x_in - te_1) * (y_in - ne_1)) / ((te_2 - te_1) * (ne_2 - ne_1))

        # Compile function
        self.tt_interp_array = function(inputs=[te_1, te_2, ne_1, ne_2, emis_11, emis_12, emis_21, emis_22, x_in, y_in],
                                        outputs=out_bInterp)

    def bilinear_tt_interpolator(self, x, y, x0_idx, y0_idx, x_range, y_range, image_matrix):
        x1, x2, y1, y2, z11, z12, z21, z22 = self.neighbour_points(x0_idx, y0_idx, x_range, y_range, image_matrix)

        interpol_value = self.tt_interp_array(x1, x2, y1, y2, z11, z12, z21, z22, x, y)

        return interpol_value

class EmissivitySurfaceFitter_tensorOps():

    def __init__(self):
        self.ionEmisEq_tt = {'S2_6716A': self.emisEquation_TeDe_tt,
                             'S2_6731A': self.emisEquation_TeDe_tt,
                             'S3_6312A': self.emisEquation_Te_tt,
                             'S3_9069A': self.emisEquation_Te_tt,
                             'S3_9531A': self.emisEquation_Te_tt,
                             'Ar4_4740A': self.emisEquation_Te_tt,
                             'Ar3_7136A': self.emisEquation_Te_tt,
                             'Ar3_7751A': self.emisEquation_Te_tt,
                             'O3_4363A': self.emisEquation_Te_tt,
                             'O3_4959A': self.emisEquation_Te_tt,
                             'O3_5007A': self.emisEquation_Te_tt,
                             'O2_7319A': self.emisEquation_TeDe_tt,
                             'O2_7330A': self.emisEquation_TeDe_tt,
                             'N2_6548A': self.emisEquation_Te_tt,
                             'N2_6584A': self.emisEquation_Te_tt,
                             'H1_4102A': self.emisEquation_HI_tt,
                             'H1_4341A': self.emisEquation_HI_tt,
                             'H1_6563A': self.emisEquation_HI_tt,
                             'He1_3889A': self.emisEquation_HeI_tt,
                             'He1_4026A': self.emisEquation_HeI_tt,
                             'He1_4471A': self.emisEquation_HeI_tt,
                             'He1_5876A': self.emisEquation_HeI_tt,
                             'He1_6678A': self.emisEquation_HeI_tt,
                             'He1_7065A': self.emisEquation_HeI_tt,
                             'He2_4686A': self.emisEquation_HeII_tt}

        self.EmisRatioEq_tt = {'RSII': self.emisEquation_TeDe_tt,
                               'RSIII': self.emisEquation_Te_tt,
                               'ROIII': self.emisEquation_Te_tt}

    def emisEquation_Te_tt(self, xy_space, a, b, c):
        temp_range, den_range = xy_space
        return a + b / (temp_range / 10000.0) + c * tt.log10(temp_range / 10000)

    def emisEquation_TeDe_tt(self, xy_space, a, b, c, d, e):
        temp_range, den_range = xy_space
        return a + b / (temp_range / 10000.0) + c * tt.log10(temp_range / 10000) + tt.log10(1 + e * den_range)

    def emisEquation_HI_tt(self, xy_space, a, b, c):
        temp_range, den_range = xy_space
        return a + b * tt.log10(temp_range) + c * tt.log10(temp_range) * tt.log10(temp_range)

    def emisEquation_HeI_tt(self, xy_space, a, b, c, d):
        temp_range, den_range = xy_space
        return tt.pow(temp_range / 10000.0, a + b * den_range) / (c + d * den_range)

    def emisEquation_HeII_tt(self, xy_space, a, b):
        temp_range, den_range = xy_space
        return a * tt.pow(temp_range / 10000, b)

class EmissionEquations_tensorOps():

    def __init__(self):
        self.fluxEq_tt = {}

    def H1_lines_tt(self, emis_ratio, cHbeta, flambda, abund=None, ftau=None, continuum=None):
        return tt.pow(10, emis_ratio - flambda * cHbeta)

    def He1_lines_tt(self, emis_ratio, cHbeta, flambda, abund, ftau=None, continuum=None):
        return abund * emis_ratio * ftau * tt.pow(10, -1 * flambda * cHbeta) + continuum

    def He2_lines_tt(self, emis_ratio, cHbeta, flambda, abund, ftau=None, continuum=None):
        return abund * emis_ratio * tt.pow(10, -1 * flambda * cHbeta) + continuum

    def metal_lines_tt(self, emis_ratio, cHbeta, flambda, abund, ftau=None, continuum=None):
        return tt.pow(10, abund + emis_ratio - flambda * cHbeta - 12)

class EmissionTensors():

    def __init__(self):
        # Dictionary to store the functions for the fitting
        self.emFlux_ttMethods = dict(H1r=self.H1_emisTensor, He1r=self.He1_emisTensor, He2r=self.He2_emisTensor,
                              metals=self.metal_emisTensor, O2_7319A_b=self.corO2_7319_emisTensor)

        # All input values scalars
        emisRatio, cHbeta, flambda, abund, ftau, continuum, O2_abund, O3_abund, Te_high = tt.dscalars('emisRatio', 'cHbeta', 'flambda', 'abund',
                                                                         'ftau', 'continuum', 'O2_abund', 'O3_abund', 'Te_high')

        # Dictionary to store the compiled functions for the fitting
        self.emFluxTensors = dict(H1r=function(inputs=[emisRatio, cHbeta, flambda, abund, ftau, continuum],
                                               outputs=self.emFlux_ttMethods['H1r'](emisRatio, cHbeta, flambda, abund, ftau, continuum),
                                               on_unused_input='ignore'),
                                  He1r=function(inputs=[emisRatio, cHbeta, flambda, abund, ftau, continuum],
                                               outputs=self.emFlux_ttMethods['He1r'](emisRatio, cHbeta, flambda, abund,ftau, continuum),
                                               on_unused_input='ignore'),
                                  He2r=function(inputs=[emisRatio, cHbeta, flambda, abund, ftau, continuum],
                                               outputs=self.emFlux_ttMethods['He2r'](emisRatio, cHbeta, flambda, abund, ftau, continuum),
                                               on_unused_input='ignore'),
                                  metals=function(inputs=[emisRatio, cHbeta, flambda, abund, ftau, continuum],
                                               outputs=self.emFlux_ttMethods['metals'](emisRatio, cHbeta, flambda, abund, ftau, continuum),
                                               on_unused_input='ignore'),
                                  O2_7319A_b=function(inputs=[emisRatio, cHbeta, flambda, O2_abund, O3_abund, Te_high],
                                               outputs=self.emFlux_ttMethods['O2_7319A_b'](emisRatio, cHbeta, flambda, O2_abund, O3_abund, Te_high),
                                               on_unused_input='ignore')
                                  )

    def H1_emisTensor(self, emis_ratio, cHbeta, flambda, abund, ftau, continuum):
        return tt.pow(10, emis_ratio - flambda * cHbeta) + continuum

    def He1_emisTensor(self, emis_ratio, cHbeta, flambda, abund, ftau, continuum):
        return abund * ftau * tt.pow(10, emis_ratio - cHbeta * flambda) + continuum

    def He2_emisTensor(self, emis_ratio, cHbeta, flambda, abund, ftau, continuum):
        return abund * tt.pow(10, emis_ratio - cHbeta * flambda) + continuum

    def metal_emisTensor(self, emis_ratio, cHbeta, flambda, abund, ftau, continuum):
        return tt.pow(10, abund + emis_ratio - flambda * cHbeta - 12)

    def corO2_7319_emisTensor(self, emis_ratio, cHbeta, flambda, O2_abund, O3_abund, Te_high):
        fluxCorr = tt.pow(10, O2_abund + emis_ratio - flambda * cHbeta - 12) + tt.pow(10, O3_abund + 0.9712758487381 + tt.log10(tt.pow(Te_high / 10000.0, 0.44)) - flambda * cHbeta - 12)
        return fluxCorr