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

    return emisInterpGrid


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
        col_ext = tt.power(10, abund + emis_ratio - flambda * cHbeta - 12)
        recomb = tt.power(10, O3 + 0.9712758 + tt.log10(tt.power(T_high/10000.0, 0.44)) - flambda * cHbeta - 12)
        return tt.log10(col_ext + recomb)


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
