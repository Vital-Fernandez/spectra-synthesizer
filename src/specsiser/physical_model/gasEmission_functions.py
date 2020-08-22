import numpy as np
import inspect as insp
import theano.tensor as tt
from theano import function
import exoplanet as xo


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


def gridInterpolatorFunction(interpolatorDict, x_range, y_range, interp_type = 'point'):

    emisInterpGrid = {}

    if interp_type == 'point':
        for line, emisGrid_i in interpolatorDict.items():
            emisInterp_i = xo.interp.RegularGridInterpolator([x_range, y_range], emisGrid_i[:, :, None], nout=1)
            emisInterpGrid[line] = emisInterp_i.evaluate

    if interp_type == 'axis':
        for line, emisGrid_i in interpolatorDict.items():
            emisGrid_i_reshape = emisGrid_i.reshape((x_range.size, y_range.size, -1))
            emisInterp_i = xo.interp.RegularGridInterpolator([x_range, y_range], emisGrid_i_reshape)
            emisInterpGrid[line] = emisInterp_i.evaluate

    # emisInterpGrid = {}
    #
    # gridCord_i = tt.drow('gridCord_i')
    #
    # for line in interpolatorDict:
    #     emisInterpGrid[line] = function(inputs=[gridCord_i],
    #                                     outputs=interpolatorDict[line](gridCord_i),
    #                                     on_unused_input='ignore')
    #
    # return emisInterpGrid


    return emisInterpGrid


def calcEmFluxes_Eq(Tlow, Thigh, ne, cHbeta, tau, abund_dict,
                    idx, lineLabel, lineIon, lineFlambda,
                    fluxEq, ftau_coeffs, emisCoeffs, emis_func,
                    indcsLabelLines, He1r_check, HighTemp_check, region_ext=''):

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
        fluxEq_i = fluxEq(line_emis, cHbeta, lineFlambda, line_abund, abund_dict['O3' + region_ext], Thigh)

    # Classical line flux
    else:
        fluxEq_i = fluxEq(line_emis, cHbeta, lineFlambda, line_abund, line_ftau, continuum=0.0)

    return fluxEq_i


def calcEmFluxes_Grid(Tlow, Thigh, ne, emisCordLow, emisCordHigh,
                      cHbeta, tau, abund_dict,
                      idx, lineLabel, lineIon, lineFlambda,
                      emisInterp, fluxEq, ftau_coeffs,
                      indcsLabelLines, He1r_check, HighTemp_check, region_ext=''):

    # Appropriate data for the ion
    Te_calc = Thigh if HighTemp_check[idx] else Tlow
    emisCoordTe_calc = emisCordHigh if HighTemp_check[idx] else emisCordLow

    # Line Emissivity
    line_emis = emisInterp(emisCoordTe_calc)

    # Atom abundance
    line_abund = abund_dict[lineIon]

    # ftau correction for HeI lines
    # TODO This will increase in complexity fast need alternative
    if He1r_check[idx]:
        line_ftau = ftau_func(tau, Te_calc, ne, *ftau_coeffs[lineLabel])
    else:
        line_ftau = None

    # Line flux with special correction:
    if indcsLabelLines['O2_7319A_b'][idx]:
        fluxEq_i = fluxEq(line_emis[0][0], cHbeta, lineFlambda, line_abund, abund_dict['O3' + region_ext], Thigh)

    # Classical line flux
    else:
        fluxEq_i = fluxEq(line_emis[0][0], cHbeta, lineFlambda, line_abund, line_ftau)

    return fluxEq_i


def storeValueInTensor(idx, value, tensor1D):
    return tt.inc_subtensor(tensor1D[idx], value)


class EmissionFluxModel:

    def __init__(self, label_list, ion_list):

        # Dictionary storing the functions corresponding to each emission line
        self.emFluxEqDict = {}

        # Dictionary storing the parameter list corresponding to each emission line for theano functions
        self.emFluxParamDict = {}

        # Loop through all the observed lines and assign a flux equation
        self.declare_emission_flux_functions(label_list, ion_list)

        # Define dictionary with flux functions with flexible number of arguments
        self.assign_flux_eqtt()

        return

    def declare_emission_flux_functions(self, label_list, ion_list):

        # TODO this could read the attribute fun directly
        # Dictionary storing emission flux equation database (log scale)
        emFluxDb_log = {'H1r': self.ion_H1r_flux_log,
                        'He1r': self.ion_He1r_flux_log,
                        'He2r': self.ion_He2r_flux_log,
                        'metals': self.metals_flux_log,
                        'O2_7319A_b': self.ion_O2_7319A_b_flux_log}

        for i, lineLabel in enumerate(label_list):

            if ion_list[i] in ('H1r', 'He1r', 'He2r'):
                self.emFluxEqDict[lineLabel] = emFluxDb_log[ion_list[i]]
                self.emFluxParamDict[lineLabel] = ['emis_ratio', 'cHbeta', 'flambda', 'abund', 'ftau']

            elif lineLabel == 'O2_7319A_b':
                self.emFluxEqDict[lineLabel] = emFluxDb_log['O2_7319A_b']
                self.emFluxParamDict[lineLabel] = ['emis_ratio', 'cHbeta', 'flambda', 'abund', 'ftau', 'O3', 'T_high']

            else:
                self.emFluxEqDict[lineLabel] = emFluxDb_log['metals']
                self.emFluxParamDict[lineLabel] = ['emis_ratio', 'cHbeta', 'flambda', 'abund', 'ftau']

        return

    def assign_lambda_function(self, linefunction, input_dict, label):

        # Single lines
        if '_b' not in label:
            input_dict[label] = lambda emis_ratio, cHbeta, flambda, abund, ftau, kwargs: \
                linefunction(emis_ratio, cHbeta, flambda, abund, ftau)

        # Blended lines
        # FIXME currently only working for O2_7319A_b the only blended line
        else:
            input_dict[label] = lambda emis_ratio, cHbeta, flambda, abund, ftau, kwargs: \
                linefunction(emis_ratio, cHbeta, flambda, abund, ftau, kwargs['O3'], kwargs['T_high'])

        return

    def assign_flux_eqtt(self):

        for label, func in self.emFluxEqDict.items():
            self.assign_lambda_function(func, self.emFluxEqDict, label)

        return

    def compute_flux(self, lineLabel, emis_ratio, cHbeta, flambda, abund, ftau, **kwargs):
        return self.emFluxEqDict[lineLabel](emis_ratio, cHbeta, flambda, abund, ftau, kwargs)

    def ion_H1r_flux_log(self, emis_ratio, cHbeta, flambda, abund, ftau):
        return emis_ratio - flambda * cHbeta

    def ion_He1r_flux_log(self, emis_ratio, cHbeta, flambda, abund, ftau):
        return abund + ftau + emis_ratio - cHbeta * flambda

    def ion_He2r_flux_log(self, emis_ratio, cHbeta, flambda, abund, ftau):
        return abund + emis_ratio - cHbeta * flambda

    def metals_flux_log(self, emis_ratio, cHbeta, flambda, abund, ftau):
        return abund + emis_ratio - flambda * cHbeta - 12

    def ion_O2_7319A_b_flux_log(self, emis_ratio, cHbeta, flambda, abund, ftau, O3, T_high):
        return np.log10(np.power(10, abund + emis_ratio - flambda * cHbeta - 12) + np.power(10, O3 + T_high -
                                                                                            flambda * cHbeta - 12))


class EmissionTensors(EmissionFluxModel):

    def __init__(self, label_list, ion_list):

        # Inherit Emission flux model
        print('\n- Compiling theano flux equations')
        EmissionFluxModel.__init__(self,  label_list, ion_list)

        # Loop through all the observed lines and assign a flux equation
        self.declare_emission_flux_functions(label_list, ion_list)

        # Compile the theano functions for all the input emission lines
        for label, func in self.emFluxEqDict.items():
            func_params = tt.dscalars(self.emFluxParamDict[label])
            self.emFluxEqDict[label] = function(inputs=func_params,
                                                outputs=func(*func_params),
                                                on_unused_input='ignore')

        # Assign function dictionary with flexible arguments
        self.assign_flux_eqtt()
        print('-- done\n')

        return





        # # Dictionary to store the emission flux tensor functions
        # self.emFluxDb_log = {'H1r': function(inputs=tt_paramsA,
        #                                      outputs=self.ion_H1r_flux_log(*tt_paramsA),
        #                                      on_unused_input='ignore'),
        #
        #                      'He1r': function(inputs=tt_paramsA,
        #                                       outputs=self.ion_He1r_flux_log(*tt_paramsA),
        #                                       on_unused_input='ignore'),
        #
        #                      'He2r': function(inputs=tt_paramsA,
        #                                       outputs=self.ion_He2r_flux_log(*tt_paramsA),
        #                                       on_unused_input='ignore'),
        #
        #                      'metals': function(inputs=tt_paramsA,
        #                                         outputs=self.metals_flux_log(*tt_paramsA),
        #                                         on_unused_input='ignore'),
        #
        #                      'O2_7319A_b': function(inputs=[*tt_paramsA, *tt_paramsB],
        #                                             outputs=self.ion_O2_7319A_b_flux_log(*tt_paramsA, *tt_paramsB),
        #                                             on_unused_input='ignore')}



# Original function
# class EmissionTensors:
#
#     def __init__(self):
#
#         # Dictionary to store the functions for the fitting
#         self.emFlux_ttMethods = dict(H1r=self.H1_emisTensor, He1r=self.He1_emisTensor, He2r=self.He2_emisTensor,
#                               metals=self.metal_emisTensor, O2_7319A_b=self.corO2_7319_emisTensor)
#
#         # All input values scalars
#         emisRatio, cHbeta, flambda, abund, ftau, continuum, O2_abund, O3_abund, Te_high = tt.dscalars('emisRatio',
#                                                                                                       'cHbeta','flambda', 'abund',
#                                                                          'ftau', 'continuum', 'O2_abund', 'O3_abund', 'Te_high')
#
#         # Dictionary to store the compiled functions for the fitting
#         self.emFluxTensors = dict(H1r=function(inputs=[emisRatio, cHbeta, flambda, abund, ftau, continuum],
#                                                outputs=self.emFlux_ttMethods['H1r'](emisRatio, cHbeta, flambda, abund, ftau, continuum),
#                                                on_unused_input='ignore'),
#                                   He1r=function(inputs=[emisRatio, cHbeta, flambda, abund, ftau, continuum],
#                                                outputs=self.emFlux_ttMethods['He1r'](emisRatio, cHbeta, flambda, abund,ftau, continuum),
#                                                on_unused_input='ignore'),
#                                   He2r=function(inputs=[emisRatio, cHbeta, flambda, abund, ftau, continuum],
#                                                outputs=self.emFlux_ttMethods['He2r'](emisRatio, cHbeta, flambda, abund, ftau, continuum),
#                                                on_unused_input='ignore'),
#                                   metals=function(inputs=[emisRatio, cHbeta, flambda, abund, ftau, continuum],
#                                                outputs=self.emFlux_ttMethods['metals'](emisRatio, cHbeta, flambda, abund, ftau, continuum),
#                                                on_unused_input='ignore'),
#                                   O2_7319A_b=function(inputs=[emisRatio, cHbeta, flambda, O2_abund, O3_abund, Te_high],
#                                                outputs=self.emFlux_ttMethods['O2_7319A_b'](emisRatio, cHbeta, flambda, O2_abund, O3_abund, Te_high),
#                                                on_unused_input='ignore')
#                                   )
#
#     def H1_emisTensor(self, emis_ratio, cHbeta, flambda, abund, ftau, continuum):
#         return tt.pow(10, emis_ratio - flambda * cHbeta) + continuum
#
#     def He1_emisTensor(self, emis_ratio, cHbeta, flambda, abund, ftau, continuum):
#         return abund * ftau * tt.pow(10, emis_ratio - cHbeta * flambda) + continuum
#
#     def He2_emisTensor(self, emis_ratio, cHbeta, flambda, abund, ftau, continuum):
#         return abund * tt.pow(10, emis_ratio - cHbeta * flambda) + continuum
#
#     def metal_emisTensor(self, emis_ratio, cHbeta, flambda, abund, ftau, continuum):
#         return tt.pow(10, abund + emis_ratio - flambda * cHbeta - 12)
#
#     def corO2_7319_emisTensor(self, emis_ratio, cHbeta, flambda, O2_abund, O3_abund, Te_high):
#         fluxCorr = tt.pow(10, O2_abund + emis_ratio - flambda * cHbeta - 12) + tt.pow(10, O3_abund + 0.9712758487381 + tt.log10(tt.pow(Te_high / 10000.0, 0.44)) - flambda * cHbeta - 12)
#         return fluxCorr


# class EmissionTensors:
#
#     def __init__(self):
#
#         # Dictionary with emission flux equation (log scale)
#         self.emFluxEq = {'H1r': lambda emis_ratio, cHbeta, flambda, abund, ftau:
#                          emis_ratio - flambda * cHbeta,
#
#                          'He1r': lambda emis_ratio, cHbeta, flambda, abund, ftau:
#                          abund + ftau + emis_ratio - cHbeta * flambda,
#
#                          'He2r': lambda emis_ratio, cHbeta, flambda, abund, ftau:
#                          abund + emis_ratio - cHbeta * flambda,
#
#                          'metals': lambda emis_ratio, cHbeta, flambda, abund, ftau:
#                          abund + emis_ratio - flambda * cHbeta - 12,
#
#                          'O2_7319A_b': lambda emis_ratio, cHbeta, flambda, abund, ftau, O3_abund, Te_high:
#                          tt.pow(10, abund + emis_ratio - flambda * cHbeta - 12)
#                          + tt.pow(10, O3_abund + 0.9712758487381 + tt.log10(tt.pow(Te_high / 10000.0, 0.44))
#                                   - flambda * cHbeta - 12)
#                          }
#
#         # Dictionary to store the functions for the fitting
#         self.emFlux_ttMethods = dict(H1r=self.H1_log_emisFlux,
#                                      He1r=self.He1_log_emisFlux,
#                                      He2r=self.He2_log_emisFlux,
#                                      metals=self.metal_log_emisFlux,
#                                      O2_7319A_b=self.corO2_7319_emisTensor)
#
#         # All input values scalars
#         emisRatio, cHbeta, flambda, abund, ftau = tt.dscalars('emisRatio', 'cHbeta', 'flambda', 'abund', 'ftau')
#         O2_abund, O3_abund, Te_high = tt.dscalars('O2_abund', 'O3_abund', 'Te_high')
#
#         # Dictionary to store the compiled functions for the fitting
#         self.emFluxTensors = dict(H1r=function(inputs=[emisRatio, cHbeta, flambda, abund, ftau],
#                                                outputs=self.emFlux_ttMethods['H1r'](emisRatio, cHbeta, flambda, abund,
#                                                                                     ftau),
#                                                on_unused_input='ignore'),
#
#                                   He1r=function(inputs=[emisRatio, cHbeta, flambda, abund, ftau],
#                                                 outputs=self.emFlux_ttMethods['He1r'](emisRatio, cHbeta, flambda, abund,
#                                                                                       ftau),
#                                                 on_unused_input='ignore'),
#                                   He2r=function(inputs=[emisRatio, cHbeta, flambda, abund, ftau],
#                                                 outputs=self.emFlux_ttMethods['He2r'](emisRatio, cHbeta, flambda, abund,
#                                                                                       ftau),
#                                                 on_unused_input='ignore'),
#                                   metals=function(inputs=[emisRatio, cHbeta, flambda, abund, ftau],
#                                                   outputs=self.emFlux_ttMethods['metals'](emisRatio, cHbeta, flambda,
#                                                                                           abund, ftau),
#                                                   on_unused_input='ignore'),
#                                   O2_7319A_b=function(inputs=[emisRatio, cHbeta, flambda, O2_abund, O3_abund, Te_high],
#                                                       outputs=self.emFlux_ttMethods['O2_7319A_b'](emisRatio, cHbeta,
#                                                                                                   flambda, O2_abund,
#                                                                                                   O3_abund, Te_high),
#                                                       on_unused_input='ignore')
#                                   )
#
#         # Dictionary to store the emission flux tensor functions
#         self.emFluxtt = {'H1r': function(inputs=[emisRatio, cHbeta, flambda, abund, ftau],
#                                          outputs=self.emFluxEq['H1r'](emisRatio, cHbeta, flambda, abund, ftau),
#                                          on_unused_input='ignore'),
#
#                          'He1r': function(inputs=[emisRatio, cHbeta, flambda, abund, ftau],
#                                           outputs=self.emFluxEq['He1r'](emisRatio, cHbeta, flambda, abund, ftau),
#                                           on_unused_input='ignore'),
#
#                          'He2r': function(inputs=[emisRatio, cHbeta, flambda, abund, ftau],
#                                           outputs=self.emFluxEq['He2r'](emisRatio, cHbeta, flambda, abund, ftau),
#                                           on_unused_input='ignore'),
#
#                          'metals': function(inputs=[emisRatio, cHbeta, flambda, abund, ftau],
#                                             outputs=self.emFluxEq['metals'](emisRatio, cHbeta, flambda, abund, ftau),
#                                             on_unused_input='ignore'),
#
#                          'O2_7319A_b': function(inputs=[emisRatio, cHbeta, flambda, abund, ftau, O3_abund, Te_high],
#                                                 outputs=self.emFluxEq['O2_7319A_b'](emisRatio, cHbeta, flambda, abund,
#                                                                                     ftau, O3_abund, Te_high),
#                                                 on_unused_input='ignore')
#                          }
#
#     def H1_log_emisFlux(self, emis_ratio, cHbeta, flambda, abund, ftau):
#         return emis_ratio - flambda * cHbeta
#
#     def He1_log_emisFlux(self, emis_ratio, cHbeta, flambda, abund, ftau):
#         return abund + ftau + emis_ratio - cHbeta * flambda
#
#     def He2_log_emisFlux(self, emis_ratio, cHbeta, flambda, abund, ftau):
#         return abund + emis_ratio - cHbeta * flambda
#
#     def metal_log_emisFlux(self, emis_ratio, cHbeta, flambda, abund, ftau):
#         return abund + emis_ratio - flambda * cHbeta - 12
#
#     def corO2_7319_emisTensor(self, emis_ratio, cHbeta, flambda, O2_abund, O3_abund, Te_high):
#         fluxCorr = tt.pow(10, O2_abund + emis_ratio - flambda * cHbeta - 12) \
#                    + tt.pow(10, O3_abund + 0.9712758487381 + tt.log10(
#             tt.pow(Te_high / 10000.0, 0.44)) - flambda * cHbeta - 12)
#         return fluxCorr