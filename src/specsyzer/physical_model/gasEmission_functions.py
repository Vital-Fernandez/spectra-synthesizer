import numpy as np
import theano.tensor as tt
from theano import function


def ftau_func(tau, temp, den, a, b, c, d):
    return 1 + tau / 2.0 * (a + (b + c * den + d * den * den) * temp / 10000.0)


def assignFluxEq2Label(labelsList, ionsList, recombLabels=['H1r', 'He1r', 'He2r']):

    eqLabelArray = np.copy(ionsList)

    for i in range(eqLabelArray.size):
        if eqLabelArray[i] not in recombLabels:
            # TODO integrate a dictionary to add corrections
            if labelsList[i] != 'O2_7319A_b':
                eqLabelArray[i] = 'metals'
            else:
                eqLabelArray[i] = 'O2_7319A_b'

    return eqLabelArray


def calcEmFluxes(Tlow, Thigh, ne, cHbeta, tau, abund_dict,
                 idx, lineLabel, lineIon, lineFlambda,
                 fluxEq, ftau_coeffs, emisCoeffs, emis_func,
                 indcsLabelLines, He1r_check, HighTemp_check):

    # Appropriate data for the ion
    Te_calc = Thigh if HighTemp_check[idx] else Tlow

    # Line Emissivity
    line_emis = emis_func((Te_calc, ne), *emisCoeffs)

    # Atom abundance
    line_abund = abund_dict[lineIon]

    # ftau correction for HeI lines # TODO This will increase in complexity fast need alternative
    if He1r_check[idx]:
        line_ftau = ftau_func(tau, Te_calc, ne, *ftau_coeffs[lineLabel])
    else:
        line_ftau = None

    # Line flux with special correction:
    if indcsLabelLines['O2_7319A_b'][idx]:
        fluxEq_i = fluxEq(line_emis, cHbeta, lineFlambda, line_abund, abund_dict['O3'], Thigh)

    # Classical line flux
    else:
        fluxEq_i = fluxEq(line_emis, cHbeta, lineFlambda, line_abund, line_ftau, continuum=0.0)

    return fluxEq_i

def calcEmFluxes_IFU(Tlow, Thigh, ne, cHbeta, tau, abund_dict,
                 idx, lineLabel, lineIon, lineFlambda,
                 fluxEq, ftau_coeffs, emisCoeffs, emis_func,
                 indcsLabelLines, He1r_check, HighTemp_check, idx_region):

    # Appropriate data for the ion
    Te_calc = Thigh if HighTemp_check[idx] else Tlow

    # Line Emissivity
    line_emis = emis_func((Te_calc, ne), *emisCoeffs)

    # Atom abundance
    line_abund = abund_dict[lineIon]

    # ftau correction for HeI lines # TODO This will increase in complexity fast need alternative
    if He1r_check[idx]: # if lineIon == 'He1r_0':
        line_ftau = ftau_func(tau, Te_calc, ne, *ftau_coeffs[lineLabel])
    else:
        line_ftau = None

    # Lines flux with special correction: # TODO a dictionary might be better as the number of line increases
    if indcsLabelLines['O2_7319A_b'][idx]: #if lineLabel == 'O2_7319A_b':
        fluxEq_i = fluxEq(line_emis, cHbeta, lineFlambda, line_abund, abund_dict['O3_' + str(idx_region)], Thigh)

    # Classical line flux
    else:
        fluxEq_i = fluxEq(line_emis, cHbeta, lineFlambda, line_abund, line_ftau, continuum=0.0)

    return fluxEq_i


def storeValueInTensor(idx, value, tensor1D):
    return tt.inc_subtensor(tensor1D[idx], value)


class EmissionTensors:

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