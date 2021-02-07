import re
import numpy as np
import pyneb as pn
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, norm
from src.specsiser.data_printing import label_decomposition
from src.specsiser.data_reading import parseConfDict

import numexpr

# Fernandez et al 2018 correction for the S^3+ fraction in the form: # TODO this should be read from a text file
# log(Ar3/Ar4) = a_S4plus * log(S3/S4) + b_s4plus => logS4 =  (a * logS3 - logAr3 + logAr4 + b) / a
a_S4plus_corr, a_S4plus_corrErr = 1.162, 0.00559
b_S4plus_corr, b_S4plus_corrErr = 0.047, 0.0097

# Ohrs 2016 relation for the OI_SI gradient log(SI/OI) -1.53
OI_SI_grad, OI_SIerr_grad = 0.029512, 0.0039


def trunc_limits(mu, sigma, lower_limit, upper_limit):
    return (lower_limit - mu) / sigma, (upper_limit - mu) / sigma


def TOIII_TSIII_relation(TSIII):
    return (1.0807 * TSIII / 10000.0 - 0.0846) * 10000.0


def pyneb_diag_comp(lineLabels, int_dict):
    ratio_componets = len(lineLabels)

    if ratio_componets == 2:
        f0, f1 = int_dict[lineLabels[0]], int_dict[lineLabels[1]]
        ratio = f0 / f1

    elif ratio_componets == 3:
        f0, f1, f2 = int_dict[lineLabels[0]], int_dict[lineLabels[1]], int_dict[lineLabels[2]]
        ratio = f0 / (f1 + f2)

    elif ratio_componets == 4:
        f0, f1, f2, f3 = int_dict[lineLabels[0]], int_dict[lineLabels[1]], int_dict[lineLabels[2]], int_dict[
            lineLabels[3]]
        ratio = (f0 + f1) / (f2 + f3)

    else:
        exit(f'- ERROR: A diagnostic {lineLabels} with more than 4 components introduced')

    return ratio


def check_density_limit(diag_label, int_ratio, ion_pn, user_check, n_steps):
    if user_check and diag_label == '[SII] 6716/6731':
        mu, sigma, lower, upper = 75.0, 25.0, 10.0, np.infty,
        a, b = trunc_limits(mu, sigma, lower, upper)
        neDist = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n_steps)
        int_ratio = ion_pn.getEmissivity(10000.0, neDist, wave=6717) / ion_pn.getEmissivity(10000.0, neDist, wave=6731)

    return int_ratio


def diag_decomposition(diag_conf):

    diag_waves = re.findall(pattern=r'\d+', string=diag_conf[1])
    diag_labels = [''] * len(diag_waves)

    for i, wave in enumerate(diag_waves):
        diag_labels[i] = f'{diag_conf[0]}_{wave}A'

    return diag_labels


class DirectMethod:

    def __init__(self, linesDF=None, highTempIons=None):

        self.obsAtoms = None
        self.indcsLabelLines = {}
        self.indcsIonLines = {}
        self.indcsHighTemp = None
        self.ionicAbundCheck = {}

        if linesDF is not None:
            self.label_ion_features(linesDF, highTempIons)

    def label_ion_features(self, linesDF, highTempIons=None):

        lineLabels = linesDF.index.values
        lineIons = linesDF.ion.values

        # Establish the ions from the available lines
        self.obsAtoms = np.unique(linesDF.ion.values)

        # Determine the line indeces
        for line in lineLabels:
            self.indcsLabelLines[line] = (lineLabels == line)

        # Determine the lines belonging to observed ions
        for ion in self.obsAtoms:
            self.indcsIonLines[ion] = (lineIons == ion)

        # Establish index of lines which below to high and low ionization zones # TODO increase flexibility for more Te
        if highTempIons is not None:
            self.indcsHighTemp = np.in1d(lineIons, highTempIons)
        else:
            self.indcsHighTemp = np.zeros(lineLabels.size, dtype=bool)

        # Establish the ionic abundance logic from the available lines
        for ion in self.obsAtoms:
            self.ionicAbundCheck[ion] = self.checkIonObservance(ion, self.obsAtoms)

        return

    def checkIonObservance(self, ion, ionList):
        return True if ion in ionList else False

    def elementalChemicalModel(self, infParamDict, ionList, iterations):

        # Convert to natural scale
        tracesDict = {}
        for ion in ionList:
            if ion in ['He1r', 'He2r']:
                tracesDict[ion] = infParamDict[ion]
            else:
                tracesDict[ion] = np.power(10, infParamDict[ion] - 12)

        # Oxygen abundance
        if self.O2Check and self.O3Check:
            infParamDict['O_abund'] = self.oxygenAbundance(tracesDict)

        # Nitrogen abundance
        if self.N2Check and self.O2Check and self.O3Check:
            infParamDict['N_abund'] = self.nitrogenAbundance(tracesDict)

        # Sulfur abundance
        if self.S2Check and self.S3Check:
            infParamDict['S_abund'] = self.sulfurAbundance(tracesDict, iterations)
            if 'S4' in tracesDict:
                infParamDict['ICF_SIV'] = infParamDict['S_abund'] / (tracesDict['S2'] + tracesDict['S3'])

        # Helium abundance
        if self.He1rCheck:
            infParamDict['He_abund'] = self.heliumAbundance(tracesDict)

        # Helium mass fraction by oxygen
        if self.He1rCheck and self.O2Check and self.O3Check:
            infParamDict['Ymass_O'] = self.heliumMassFractionOxygen(infParamDict['He_abund'], infParamDict['O_abund'])

        # Helium mass fraction by sulfur
        if self.He1rCheck and self.S2Check and self.S3Check:
            infParamDict['Ymass_S'] = self.heliumMassFractionSulfur(infParamDict['He_abund'], infParamDict['S_abund'],
                                                                    iterations)

        # Convert metal abundances to 12 + log(X^i+) notation
        for metal in ['O_abund', 'N_abund', 'S_abund']:
            if metal in infParamDict:
                infParamDict[metal] = 12 + np.log10(infParamDict[metal])

        return

    def oxygenAbundance(self, abundDict):

        O_abund = abundDict['O2'] + abundDict['O3']

        return O_abund

    def nitrogenAbundance(self, abundDict):

        NO_ratio = abundDict['N2'] / abundDict['O2']

        N_abund = NO_ratio * (abundDict['O2'] + abundDict['O3'])

        return N_abund

    def sulfurAbundance(self, abundDict, iterations):

        if self.Ar3Check and self.Ar4Check:
            aS3corrArray = np.random.normal(self.a_S3corr, self.a_S3corrErr, size=iterations)
            bS3corrArray = np.random.normal(self.b_S3corr, self.b_S3corrErr, size=iterations)

            S4_abund = (aS3corrArray * np.log10(abundDict['S3']) - np.log10(abundDict['Ar3']) +
                        np.log10(abundDict['Ar4']) + bS3corrArray) / aS3corrArray

            abundDict['S4'] = np.power(10, S4_abund)

            S_abund = abundDict['S2'] + abundDict['S3'] + abundDict['S4']

        else:
            S_abund = abundDict['S2'] + abundDict['S3']

        return S_abund

    def heliumAbundance(self, abundDict):

        if self.He2rCheck:
            He_abund = abundDict['He1r'] + abundDict['He2r']
        else:
            He_abund = abundDict['He1r']

        return He_abund

    def heliumMassFractionOxygen(self, He_abund, O_abund):

        Y_fraction = (4 * He_abund * (1 - 20 * O_abund)) / (1 + 4 * He_abund)

        return Y_fraction

    def heliumMassFractionSulfur(self, He_abund, S_abund, iterations, OI_SI=OI_SI_grad, OI_SIerr=OI_SIerr_grad):

        OI_SIArray = np.random.normal(OI_SI, OI_SIerr, size=iterations)  # TODO check this one

        Y_fraction = (4 * He_abund * (1 - 20 * OI_SIArray * S_abund)) / (1 + 4 * He_abund)

        return Y_fraction


class Standard_DirectMetchod:

    DEN_DIAGNOSTICS = {'[SII] 6716/6731': ('S2', 'L(6716)/L(6731)', 'RMS([E(6716), E(6731)])')}

    TEM_DIAGNOSTICS = {'[NII] 5755/6548': ('N2', 'L(5755)/L(6548)', 'RMS([E(6548), E(5755)])'),
                       '[NII] 5755/6584': ('N2', 'L(5755)/L(6584)', 'RMS([E(6584), E(5755)])'),
                       '[NII] 5755/6584+': ('N2', 'L(5755)/(L(6548)+L(6584))', 'RMS([E(6548)*L(6548)/(L(6548)+L(6584)), E(6584)*L(6584)/(L(6584)+L(6548)), E(5755)])'),

                       '[OII] 3727+/7325+': ('O2', '(L(3726)+L(3729))/(B("7319A+")+B("7330A+"))', 'RMS([E(3726)*L(3726)/(L(3726)+L(3729)), E(3729)*L(3729)/(L(3726)+L(3729)),BE("7319A+")*B("7319A+")/(B("7319A+")+B("7330A+")),BE("7330A+")*B("7330A+")/(B("7319A+")+B("7330A+"))])'),

                       '[OIII] 4363/4959': ('O3', 'L(4363)/L(4959)', 'RMS([E(4959), E(4363)])'),
                       '[OIII] 4363/5007': ('O3', 'L(4363)/L(5007)', 'RMS([E(5007), E(4363)])'),
                       '[OIII] 4363/5007+': ('O3', 'L(4363)/(L(5007)+L(4959))', 'RMS([E(5007)*L(5007)/(L(5007)+L(4959)), E(4959)*L(4959)/(L(5007)+L(4959)), E(4363)])'),

                       '[SIII] 6312/9069': ('S3', 'L(6312)/L(9069)', 'RMS([E(9069), E(6312)])'),
                       '[SIII] 6312/9531': ('S3', 'L(6312)/L(9531)', 'RMS([E(9531), E(6312)])'),
                       '[SIII] 6312/9200+': ('S3', 'L(6312)/(L(9069)+L(9531))', 'RMS([E(9069)*L(9069)/(L(9069)+L(9531)), E(9531)*L(9531)/(L(9069)+L(9531)), E(6312)])'),

                       '[SII] 4069/4076': ('S2', 'L(4069)/L(4076)', 'RMS([E(4069), E(4076)])'),
                       '[SII] 4069/6700+': ('S2', 'L(4069)/(L(6716)/L(6731))', 'RMS([E(6731)*L(6731)/(L(6731)+L(6716)), E(6716)*L(6716)/(L(6731)+L(6716)), E(4069)])')
                       }

    LOW_LIMIT_DENSITY = 1.35

    CALIB_RATIOS = dict(S2_ratio=('S2_6716A', 'S2_6731A'),
                        N2_ratio=('N2_6584A', 'N2_6548A'),
                        O3_ratio=('O3_5007A', 'O3_4959A'),
                        S3_ratio=('S3_9531A', 'S3_9069A'))

    DIAG_RATIOS = dict(R_O3=('O3_4363A', 'O3_4959A', 'O3_5007A'),
                       R_O2_m=('O2_3726A_m', 'O2_7319A_m'),
                       R_O2=('O2_3726A_m', 'O3_7319A', 'O2_7330A'),
                       R_S2=('S2_6716A', 'S2_6731A'),
                       R_S2_dash=('S2_6716A', 'S2_6731A', 'S2_4068A'),
                       R_N2=('N2_6548A', 'N2_6584A', 'N2_5755A'),
                       R_S3=('S3_9531A', 'S3_9069A', 'S3_6312'))

    ABUND_LINES = dict(Ar3=['Ar3_7136A'],
                       Ar4=['Ar4_4740A'],
                       Ne3=['Ne3_3869A'],
                       Fe3=['Fe3_4658A'],
                       He1r=['He1_4471A', 'He1_5876A', 'He1_6678A'],
                       He2r=['He2_4686A'],
                       Cl3=['Cl3_5538A', 'Cl3_5518A'],
                       S2=['S2_6716A', 'S2_6731A'],
                       S3=['S3_9069A', 'S3_9531A'],
                       N2=['N2_6584A', 'N2_6548A'],
                       O2_3700=['O2_3726A_m'],
                       O2_7300=['O2_7319A_m'],
                       O3=['O3_4959A', 'O3_5007A'])

    def __init__(self, n_steps=1000):

        self.ionDict = {}
        self.flux_dict = {}
        self.inten_dict = {}
        self.obsIons = None
        self.electron_params = {}
        self.ionic_abund = {}
        self.obs_ratios = {}
        self.n_steps = n_steps

        # Pyneb diagnostics tool
        self.diags = pn.Diagnostics()

        # Declare main diagnostics
        for den_diag, diag_conf in self.DEN_DIAGNOSTICS.items():
            self.diags.addDiag(den_diag, diag_conf)
        for temp_diag, diag_conf in self.TEM_DIAGNOSTICS.items():
            self.diags.addDiag(temp_diag, diag_conf)

    def get_ions_dict(self, ions_list, atomic_references=pn.atomicData.defaultDict):

        # Check if the atomic dataset is the default one
        if atomic_references == pn.atomicData.defaultDict:
            pn.atomicData.resetDataFileDict()
            pn.atomicData.removeFitsPath()
        else:
            pn.atomicData.includeFitsPath()
            pn.atomicData.setDataFileDict(atomic_references)

        # Generate the dictionary with pyneb ions
        self.ionDict = pn.getAtomDict(ions_list)

        return

    def declare_line_fluxes(self, line_labels, flux_array, err_array):

        # Determine transition properties from labels
        ion_array, wavelength_array, latexLabel_array = label_decomposition(line_labels)

        # Generate line gaussian shaped array for lines based on their uncertainty
        flux_dict = {}
        for i, line in enumerate(line_labels):
            a, b = trunc_limits(flux_array[i], err_array[i], 0.0, np.infty)
            flux_dict[line] = truncnorm.rvs(a, b, loc=flux_array[i], scale=err_array[i], size=self.n_steps)
            assert (flux_dict[line] > 0.0).all(), f'- ERROR {line} results in negative fluxes'

        # Store important line fluxes
        obs_lines = flux_dict.keys()
        for ratio_label, ratio_lines in self.CALIB_RATIOS.items():
            if obs_lines >= set(ratio_lines):
                self.obs_ratios[ratio_label] = flux_dict[ratio_lines[0]]/flux_dict[ratio_lines[1]]

        # Analyse lines available
        self.obsIons = np.unique(ion_array)
        self.get_ions_dict(self.obsIons)

        return flux_dict

    def red_corr(self, flux_dict, cHbeta, cHbeta_err, f_lambda):

        assert cHbeta > 0.0, '-ERROR: Negative logaritmic extinction coefficient in reddening correction'

        a, b = trunc_limits(cHbeta, cHbeta_err, 0.0, np.infty)
        cHbeta_array = truncnorm.rvs(a, b, loc=cHbeta, scale=cHbeta_err, size=self.n_steps)

        intensity_dict = {}
        for i, item in enumerate(flux_dict.items()):
            label, flux = item
            intensity_dict[label] = flux * np.power(10, cHbeta_array * f_lambda[i])

        return intensity_dict

    def electron_diagnostics(self, int_dict, ne_diags={}, Te_diags={}, neSII_limit_check=False, Tlow_diag=None,
                             Thigh_diag=None):

        # Diagnostics
        ne_diags = {**self.DEN_DIAGNOSTICS, **ne_diags}
        Te_diags = {**self.TEM_DIAGNOSTICS, **Te_diags}
        obsLines = int_dict.keys()

        # Store diagnostic line ratios (R_S3, R_O2, R_S2_dash...):
        for ratio_key, ratio_lines in self.DIAG_RATIOS.items():
            if obsLines >= set(ratio_lines):
                ratio_value = pyneb_diag_comp(ratio_lines, int_dict)
                self.electron_params[ratio_key] = ratio_value

        # ---- Loop through the density and temperature diagnostics
        for neDiagKey, neDiagConf in ne_diags.items():
            neDiagLines = diag_decomposition(neDiagConf)

            if obsLines >= set(neDiagLines):
                neDiagRatio = pyneb_diag_comp(neDiagLines, int_dict)
                neDiagRatio = check_density_limit(neDiagKey, neDiagRatio, self.ionDict['S2'], neSII_limit_check, self.n_steps)

                for TeDiagKey, TeDiagConf in Te_diags.items():
                    TeDiagLines = diag_decomposition(TeDiagConf)

                    if obsLines >= set(TeDiagLines):
                        TeDiagRatio = pyneb_diag_comp(TeDiagLines, int_dict)

                        # Compute the temperature - density pair
                        diag_label = f'{TeDiagKey.replace(" ", "_")}-{neDiagKey.replace(" ", "_")}'
                        print(f'-- Diagnostic: {diag_label}')

                        Te, ne = self.diags.getCrossTemDen(diag_den=neDiagKey,
                                                           value_den=neDiagRatio,
                                                           diag_tem=TeDiagKey,
                                                           value_tem=TeDiagRatio)

                        # Store the measurement
                        self.electron_params[diag_label] = np.array([Te, ne])

        # ---- Empirical diagnostics
        # TeSIII_from_TeOIII Epm 2017 from Hagele et al 2006
        if Thigh_diag is not None:
            if Thigh_diag in self.electron_params:
                T_high = self.electron_params[Thigh_diag][0, :]
                m_dist = norm.rvs(loc=1.19, scale=0.08, size=self.n_steps)
                n_dist = norm.rvs(loc=0.32, scale=0.10, size=self.n_steps)
                self.electron_params['TeSIII_from_TeOIII'] = (m_dist * T_high/10000 - n_dist) * 10000.0

        # TeNII_from_TeOIII from Epm and Contini 2009
        if Thigh_diag is not None:
            if Thigh_diag in self.electron_params:
                T_high = self.electron_params[Thigh_diag][0, :]
                a_dist = 1.85
                b_dist = 0.72
                self.electron_params['TeNII_from_TeOIII'] = (a_dist / (T_high / 10000 + b_dist)) * 10000.0

        # TeOIII_from_TSIII from SOMEWHERE
        if Tlow_diag is not None:
            if Tlow_diag in self.electron_params:
                T_low = self.electron_params[Tlow_diag][0, :]
                m_dist = 1.0807
                n_dist = 0.0846
                self.electron_params['TeOIII_from_TSIII'] = (m_dist * T_low / 10000 + n_dist) * 10000.0

        # TeOII_from_TeOIII from Epm and Contini 2009
        if Thigh_diag is not None:
            if Thigh_diag in self.electron_params:
                T_high = self.electron_params[Thigh_diag][0, :]
                a_dist = 1.397
                b_dist = 0.385
                self.electron_params['TeOII_from_TeOIII'] = (a_dist / ((1 / (T_high / 10000)) + b_dist)) * 10000

        # Te_O3_pm2017
        if 'R_O3' in self.electron_params:
            R_O3 = self.electron_params['R_O3']
            self.electron_params['Te_O3_pm2017'] = (0.7840 - 0.0001357 * (1/R_O3) + 48.44/(1/R_O3)) * 10000

        # ne_S2_pm2017
        if 'R_S2' in self.electron_params:
            if 'Te_O3_pm2017' in self.electron_params:
                R_S2 = self.electron_params['R_S2']
                Te = self.electron_params['Te_O3_pm2017']
                a0 = 16.054 - 7.79/Te - 11.32 * Te
                a1 = -22.66 + 11.08/Te + 16.02 * Te
                b0 = -21.61 + 11.89/Te + 14.59 * Te
                b1 = 9.17 - 5.09/Te - 6.18 * Te
                self.electron_params['ne_S2_pm2017'] = 1000 * (R_S2 * a0 + a1) / (R_S2 * b0 + b1)

        # Te_O2_pm2017
        if 'R_O2' in self.electron_params:
            R_O2 = self.electron_params['R_O2']
            ne = self.electron_params['ne_S2_pm2017']
            a0 = 0.2526 - 0.000357*ne - 0.43/ne
            a1 = 0.00136 + 0.00481/ne
            a2 = 35.624 - 0.0172 * ne - 25.12/ne
            self.electron_params['Te_O2_pm2017'] = (a0 - a1 * R_O2 + a2 / R_O2) * 10000

        # T_N2_pm2017
        if 'R_N2' in self.electron_params:
            R_N2 = self.electron_params['R_N2']
            self.electron_params['Te_N2_pm2017'] = (0.6153 - 0.0001529 * R_N2 + 35.3641/R_N2) * 10000

        # T_S3_pm2017
        if 'R_S3' in self.electron_params:
            R_S3 = self.electron_params['R_S3']
            self.electron_params['Te_S3_pm2017'] = (0.5147 + 0.0003187 * R_S3 + 23.64041/R_S3) * 10000

        # T_S2_pm2017
        if 'R_S2_dash' in self.electron_params:
            if 'ne_S2_pm2017' in self.electron_params:
                R_S2_dash = (int_dict['S2_6716A'] + int_dict['S2_6731A']) / (1.333 * int_dict['S2_4068A'])
                ne = self.electron_params['ne_S2_pm2017']
                a0 = 0.99 + 34.79/ne + 321.82/np.power(ne, 2)
                a1 = -0.0087 + 0.628/ne + 5.744/np.power(ne, 2)
                a2 = -7.123 + 926.5/ne - 94.78/np.power(ne, 2)
                a3 = 102.82 + 768.852/ne - 5113.0/np.power(ne, 2)
                self.electron_params['Te_S2_pm2017'] = (a0 - a1 * R_S2_dash + a2/R_S2_dash + a3/np.power(R_S2_dash, 2)) * 10000

        return

    def ionic_abundances(self, int_dict, line_fit_dict={}, chem_model_dict={}, **kwargs):

        # Compute the ionic abundance
        lineLabels = np.array(list(int_dict.keys()))
        ion_array, wavelength_array, latexLabel_array = label_decomposition(lineLabels)

        # -------------------------- Region diagnostics
        for diag_ref in ['ne_diag', 'Te_low_diag', 'Te_high_diag']:

            assert diag_ref in chem_model_dict, f'- ERROR: No {diag_ref} in ionic abundance configuration'
            diag_name = chem_model_dict[diag_ref]
            diag_code = diag_ref[0:diag_ref.find('_diag')]

            assert diag_name in self.electron_params, f'- ERROR: {diag_name} diagnostic not calculated in analysis'

            # Extract the diagnostic array and distinguish between Temp, den and single parameter determinations
            diag_matrix = self.electron_params[diag_name]
            if diag_matrix.ndim == 1:
                diag_array = diag_matrix
            else:
                if diag_code[0:2] == 'Te':
                    diag_array = diag_matrix[0, :]
                else:
                    diag_array = diag_matrix[1, :]
            self.electron_params[diag_code] = diag_array

        # -------------------------- Line abundances
        for ion in self.obsIons:
            if ion != 'H1':  # Exclude hydrogen

                # Establish temperature and density for the corresponding ion
                ion_region = 'low' if ion not in chem_model_dict['high_ionzation_ions_list'] else 'high'
                temp_label = f'Te_{ion_region}' if f'Te_{ion}' not in chem_model_dict else f'Te_{ion}'
                den_label = 'ne' if f'ne_{ion}' not in chem_model_dict else f'ne_{ion}'
                Te, ne = self.electron_params[temp_label], self.electron_params[den_label]

                # Loop throught the observed lines to treat the ionic abundances
                idcs_linesIon = ion_array == ion
                for i, lineLabel in enumerate(lineLabels[idcs_linesIon]):

                    int_ratio = int_dict[lineLabel]

                    # Compute line reference accounting for blended lines
                    if lineLabel in line_fit_dict:
                        line_comps = np.array(line_fit_dict[lineLabel], ndmin=1)
                    else:
                        line_comps = lineLabel
                    ion_i, wavelength_i, latexLabel_i = label_decomposition(line_comps)

                    # Ignore merge lines with different ions
                    if np.unique(ion_i).size < 2:

                        line_ref = "+".join(map("L({:.0f})".format, wavelength_i))

                        # For the Helium add the recombination label i.e. He1r
                        ion_emis = ion if ion not in ('He1', 'He2') else ion + 'r'

                        # Compute the abundance
                        try:
                            ionic_abund = self.ionDict[ion_emis].getIonAbundance(int_ratio,
                                                                                Te, ne,
                                                                                to_eval=line_ref,
                                                                                Hbeta=1)
                            if ion_emis not in ('He1r', 'He2r'):
                                ionic_abund = 12 + np.log10(ionic_abund)

                            self.ionic_abund[lineLabel] = ionic_abund
                        except:
                            self.ionic_abund[lineLabel] = 'Abundance failure'
                            print(f'-- {lineLabel} abundance with ref {line_ref} could not be calculated')

        # -------------------------- Ionic abundances
        obsLineSet = set(self.ionic_abund.keys())
        for ion in self.obsIons:
            if ion != 'H1':  # Exclude hydrogen

                ion_abund = ion if ion not in ('He1', 'He2') else ion + 'r'

                abundLines_ref = f'{ion_abund}_line_list'
                if abundLines_ref in chem_model_dict or ion_abund in self.ABUND_LINES:

                    # User lines have preference over default lines
                    if abundLines_ref in chem_model_dict:
                        abund_lines = chem_model_dict[abundLines_ref]
                    else:
                        abund_lines = set(self.ABUND_LINES[ion_abund])

                    # Check lines have abundances
                    abund_lines = list(obsLineSet & set(abund_lines))

                    if len(abund_lines) > 0:

                        # Compute emissivity reference
                        ion_i, wave_i, latexLabel = label_decomposition(abund_lines)
                        line_ref = "+".join(map("L({:.0f})".format, wave_i))

                        # Compute the lines sum
                        int_ratio = 0
                        for line in abund_lines:
                            int_ratio += int_dict[line]

                        # Compute the abundance
                        try:
                            ionic_abund = self.ionDict[ion_abund].getIonAbundance(int_ratio,
                                                                            Te, ne,
                                                                            to_eval=line_ref,
                                                                            Hbeta=1)
                            if ion_abund not in ('He1r', 'He2r'):
                                ionic_abund = 12 + np.log10(ionic_abund)

                            self.ionic_abund[ion_abund] = ionic_abund
                        except:
                            self.ionic_abund[ion_abund] = f'Ionic abundance failure {abund_lines}'

        return

    def save_measurements(self, file_address, prefix='', nan_max=0.05):

        ratios_output = {}
        for ratio_label, ratio_value in self.obs_ratios.items():
            ratios_output[ratio_label] = ratio_value.mean()

        section_name = f'{prefix}_Line_ratios'
        parseConfDict(file_address, ratios_output, section_name=section_name, clear_section=True)

        # Treat the electron parameter measurements
        e_params_output = {}

        for param, mc_value in self.electron_params.items():
            if param.startswith('R_'): # TODO make your own float formatting tool
                float_format = '0.4f'
            else:
                float_format = '0.2f'

            if not np.isnan(mc_value).all():
                if mc_value.ndim == 1:
                    mean, std = np.nanmean(mc_value), np.nanstd(mc_value)
                    output_entry = f'{mean:{float_format}},{std:{float_format}}'
                else:
                    mean, std = np.nanmean(mc_value, axis=1), np.nanstd(mc_value, axis=1)
                    output_entry = f'{mean[0]:{float_format}},{std[0]:{float_format}},{mean[1]:{float_format}},{std[1]:{float_format}}'
            else:
                output_entry = 'All_values_are_nan'

            # Remove [] from param (case of pyneb diagnostics=
            output_key = param.replace('[', '(').replace(']', ')')

            e_params_output[output_key] = output_entry

            # Check for nan entries
            nan_sum = np.isnan(mc_value).sum()
            nan_check = False if nan_sum < nan_max * self.n_steps else True
            if nan_check:
                e_params_output[output_key + '_nan_entries'] = nan_sum

        # Store the results
        section_name = f'{prefix}_electron_parameters'
        parseConfDict(file_address, e_params_output, section_name, clear_section=True)

        # Treat the ionic abundance measurements
        ionic_abund_output = {}
        for param, mc_value in self.ionic_abund.items():
            output_key = param.replace('[', '(').replace(']', ')')

            if not isinstance(mc_value, str):
                mean, std = np.nanmean(mc_value), np.nanstd(mc_value)
                output_entry = f'{mean:0.5f},{std:0.5f}'

                ionic_abund_output[output_key] = output_entry

                # Check for nan entries
                nan_sum = np.isnan(mc_value).sum()
                nan_check = False if nan_sum < nan_max * self.n_steps else True
                if nan_check:
                    ionic_abund_output[output_key + '_nan_entries'] = nan_sum
            else:
                ionic_abund_output[output_key] = mc_value

        # Store the results
        section_name = f'{prefix}_ionic_Abundances'
        parseConfDict(file_address, ionic_abund_output, section_name, clear_section=True)

        return

    def plot_diag_chart(self, int_dict, plot_address, ne_diags={}, Te_diags={}):

        # Generate pyneb object
        obs = pn.Observation(corrected=True)
        diags = pn.Diagnostics()

        # Declare diagnostics to plot
        ne_diags = {**self.DEN_DIAGNOSTICS, **ne_diags}
        Te_diags = {**self.TEM_DIAGNOSTICS, **Te_diags}
        total_diag = {**ne_diags, **Te_diags}
        for diag_label, diag_conf in total_diag.items():
            diags.addDiag(diag_label, diag_tuple=diag_conf)

        # Loop through the lines and add them to the observation object
        for lineLabel, lineInt in int_dict.items():
            if '_m' in lineLabel:
                lineLabel = lineLabel.replace('_m', '+')
            print(lineLabel)
            lineObj = pn.EmissionLine(label=lineLabel, obsIntens=lineInt.mean(), obsError=lineInt.std())
            obs.addLine(lineObj)

        diags.addDiagsFromObs(obs)
        emisgrids = pn.getEmisGridDict(atomDict=diags.atomDict)

        diags.plot(emisgrids, obs)

        if plot_address is None:
            plt.show()
        else:
            plt.savefig(plot_address, bbox_inches='tight')


        # # Generate the observation grid
        # emisgrids = pn.getEmisGridDict(atomDict=diags.atomDict)
        #
        # # Generate a figure for the plot
        # Fig1 = plt.figure(figsize=(16, 16))
        # Fig1.set_dpi(600)
        # Axis1 = Fig1.add_subplot(111)
        #
        # # Load emissivity grid
        # diags.plot(emisgrids, obs, i_obs=0, ax=Axis1)

        return

    # def determine_ionic_abundance(self, abund_code, atom, diagnos_eval, diagnos_mag, tem, den):
    #
    #     try:
    #         hbeta_flux = self.Hbeta_flux
    #     except AttributeError:
    #         hbeta_flux = self.H1_atom.getEmissivity(tem=tem, den=den, label='4_2', product=False)
    #         print
    #         '--Warning using theoretical Hbeta emissivity'
    #
    #     # Ionic abundance calculation using pyneb
    #     ionic_abund = atom.getIonAbundance(int_ratio=diagnos_mag, tem=tem, den=den, to_eval=diagnos_eval,
    #                                        Hbeta=hbeta_flux)
    #
    #     # Evaluate the nan array
    #     nan_idcs = isnan(ionic_abund)
    #     nan_count = np_sum(nan_idcs)
    #
    #     # Directly save if not nan
    #     if nan_count == 0:
    #         self.abunData[abund_code] = ionic_abund
    #
    #     # Remove the nan entries performing a normal distribution
    #     elif nan_count < 0.90 * self.MC_array_len:
    #         mag, error = nanmean(ionic_abund), nanstd(ionic_abund)
    #
    #         # Generate truncated array to store the data
    #         a, b = (0 - mag) / error, (1000 * mag - mag) / error
    #         new_samples = truncnorm(a, b, loc=mag, scale=error).rvs(size=nan_count)
    #
    #         # Replace nan entries
    #         ionic_abund[nan_idcs] = new_samples
    #         self.abunData[abund_code] = ionic_abund
    #
    #         if nan_count > self.MC_warning_limit:
    #             print
    #             '-- {} calculated with {}'.format(abund_code, nan_count)
    #
    #     return
    #
    # def check_obsLines(self, lines_list, just_one_line=False):
    #
    #     # WARNING it would be better something that reads a standard preference over some.
    #     # Right format for pyneb eval: Ar3_7751A -> L(7751)
    #     eval_lines = map(lambda x: 'L({})'.format(x[x.find('_') + 1:len(x) - 1]), lines_list)
    #     diagnos_eval = None
    #
    #     # Case all lines are there
    #     if self.lines_dict.viewkeys() >= set(lines_list):
    #         diagnos_mag = zeros(self.MC_array_len)
    #         for i in range(len(lines_list)):
    #             diagnos_mag += self.lines_dict[lines_list[i]]
    #         diagnos_eval = '+'.join(eval_lines)
    #
    #     # Case we can use any line: #WARNING last line is favoured
    #     elif just_one_line:
    #         diagnos_mag = zeros(self.MC_array_len)
    #         for i in range(len(lines_list)):
    #             if lines_list[i] in self.lines_dict:
    #                 diagnos_mag = self.lines_dict[lines_list[i]]
    #                 diagnos_eval = eval_lines[i]
    #
    #     # Case none of the lines
    #     if diagnos_eval is None:
    #         diagnos_mag = self.generate_nan_array()
    #         diagnos_eval = '+'.join(eval_lines)
    #
    #     return diagnos_eval, diagnos_mag
    #
    # def argon_abundance_scheme(self, Tlow, Thigh, ne):
    #
    #     # Calculate the Ar_+2 abundance according to the lines observed
    #     Ar3_lines = ['Ar3_7136A', 'Ar3_7751A']
    #     diagnos_eval, diagnos_mag = self.check_obsLines(Ar3_lines, just_one_line=True)
    #     self.determine_ionic_abundance('ArIII_HII', self.Ar3_atom, diagnos_eval, diagnos_mag, Tlow, ne)
    #
    #     # Calculate the Ar_+3 abundance according to the lines observed
    #     Ar4_lines = ['Ar4_4740A', 'Ar4_4711A']
    #     diagnos_eval, diagnos_mag = self.check_obsLines(Ar4_lines, just_one_line=True)
    #     self.determine_ionic_abundance('ArIV_HII', self.Ar4_atom, diagnos_eval, diagnos_mag, Thigh, ne)
    #
    # def oxygen_abundance_scheme(self, Tlow, Thigh, ne):
    #
    #     # Calculate the O_+1 abundances from 3200+ lines
    #     O2_lines = ['O2_3726A+']
    #     diagnos_eval, diagnos_mag = self.check_obsLines(O2_lines)
    #     diagnos_eval = 'L(3726)+L(3729)'
    #     self.determine_ionic_abundance('OII_HII_3279A', self.O2_atom, diagnos_eval, diagnos_mag, Tlow, ne)
    #
    #     # Calculate the O_+1 abundances from 7300+ lines
    #     O2_lines = ['O2_7319A+']
    #     diagnos_eval, diagnos_mag = self.check_obsLines(O2_lines)
    #     diagnos_eval = 'L(7319)+L(7330)'
    #     self.determine_ionic_abundance('OII_HII_7319A', self.O2_atom, diagnos_eval, diagnos_mag, Tlow, ne)
    #
    #     # --Correction for recombination contribution Liu2000
    #     if 'OII_HII_7319A' in self.abunData:
    #
    #         try:
    #             hbeta_flux = self.Hbeta_flux
    #         except AttributeError:
    #             hbeta_flux = self.H1_atom.getEmissivity(tem=Tlow, den=ne, label='4_2', product=False)
    #             print
    #             '--Warning using theoretical Hbeta emissivity'
    #
    #         Lines_Correction = (9.36 * power((Tlow / 10000), 0.44) * self.abunData.OII_HII_7319A) * hbeta_flux
    #         ratio = self.lines_dict['O2_7319A+'] - Lines_Correction
    #         self.determine_ionic_abundance('OII_HII_7319A', self.O2_atom, diagnos_eval, ratio, Tlow, ne)
    #
    #     # Get the ratios for empirical relation between OII lines
    #     if 'O2_3726A+' in self.lines_dict:
    #         self.abunData['O_R3200'] = self.lines_dict['O2_3726A+'] / self.Hbeta_flux
    #         print
    #         'O_R3200', mean(self.abunData['O_R3200'])
    #         print
    #         'OII_HII_3279A', mean(self.abunData['OII_HII_3279A'])
    #         print
    #         'Original flux', mean(self.lines_dict['O2_3726A+'])
    #
    #     if 'O2_7319A+' in self.lines_dict:
    #         self.abunData['O_R7300'] = self.lines_dict['O2_7319A+'] / self.Hbeta_flux
    #         print
    #         'OII_HII_7319A', mean(self.abunData['OII_HII_7319A'])
    #     if self.lines_dict.viewkeys() >= set(['O3_5007A']):
    #         self.abunData['O_R3'] = self.lines_dict['O3_5007A'] / self.Hbeta_flux
    #
    #         # Calculate the abundance from the empirical O_R3200_ffO2
    #     if set(self.abunData.index) >= {'O_R7300', 'O_R3'}:
    #         logRO2 = 1.913 + log10(self.abunData['O_R7300']) - 0.374 * log10(self.abunData['O_R3']) / 0.806
    #         print
    #         'logRO2', mean(logRO2)
    #         RO2 = power(10, logRO2)
    #         self.abunData['O_R3200_ffO2'] = RO2
    #         print
    #         'O_R3200_ffO2', mean(self.abunData['O_R3200_ffO2'])
    #         print
    #         'RO2*Hbeta', mean(RO2 * self.Hbeta_flux)
    #         diagnos_eval = 'L(3726)+L(3729)'
    #         self.determine_ionic_abundance('OII_HII_ffO2', self.O2_atom, diagnos_eval, RO2 * self.Hbeta_flux, Tlow, ne)
    #         print
    #         'OII_HII_ffO2', mean(self.abunData['OII_HII_ffO2'])
    #
    #     # Calculate the O_+2 abundance
    #     O3_lines = ['O3_4959A', 'O3_5007A']
    #     diagnos_eval, diagnos_mag = self.check_obsLines(O3_lines)
    #     self.determine_ionic_abundance('OIII_HII', self.O3_atom, diagnos_eval, diagnos_mag, Thigh, ne)
    #
    #     # Determine the O/H abundance (favoring the value from OII_HII
    #     if set(self.abunData.index) >= {'OII_HII_3279A', 'OIII_HII'}:
    #         self.abunData['OII_HII'] = self.abunData['OII_HII_3279A']
    #         self.abunData['OI_HI'] = self.abunData['OII_HII_3279A'] + self.abunData['OIII_HII']
    #     elif set(self.abunData.index) >= {'OII_HII_7319A', 'OIII_HII'}:
    #         self.abunData['OII_HII'] = self.abunData['OII_HII_7319A']
    #         self.abunData['OI_HI'] = self.abunData['OII_HII_7319A'] + self.abunData['OIII_HII']
    #
    #     if set(self.abunData.index) >= {'OII_HII_ffO2', 'OIII_HII'}:
    #         if set(self.abunData.index) >= {'OII_HII_3279A'}:
    #             self.abunData['OI_HI_ff02'] = self.abunData['OII_HII_3279A'] + self.abunData['OIII_HII']
    #         else:
    #             self.abunData['OI_HI_ff02'] = self.abunData['OII_HII_ffO2'] + self.abunData['OIII_HII']
    #
    #     return
    #
    # def nitrogen_abundance_scheme(self, Tlow, ne):
    #
    #     # Calculate TNII temperature from the CHAOS relation
    #     T_NII = Tlow  # self.m_TNII_correction * Tlow + self.n_TNII_correction
    #
    #     # Calculate the N+1 abundance
    #     N2_lines = ['N2_6548A', 'N2_6584A']
    #     diagnos_eval, diagnos_mag = self.check_obsLines(N2_lines)
    #     self.determine_ionic_abundance('NII_HII', self.N2_atom, diagnos_eval, diagnos_mag, T_NII, ne)
    #
    #     # Calculate NI_HI using the OI_HI
    #     if set(self.abunData.index) >= {'NII_HII', 'OI_HI'}:
    #         # Compute  NI_OI
    #         self.abunData['NI_OI'] = self.abunData['NII_HII'] / self.abunData['OII_HII']
    #         self.abunData['NI_HI'] = self.abunData['NI_OI'] * self.abunData['OI_HI']
    #
    #     #             #Repeat calculation if 5755 line was observed to include the recombination contribution
    #     #             if self.lines_dict.viewkeys() >= {'N2_5755A'}:
    #     #
    #     #                 NIII_HI             = self.abunData.NI_HI - self.abunData['NII_HII']
    #     #                 Lines_Correction    = 3.19 * power((Thigh/10000), 0.30) * NIII_HI * self.Hbeta_flux
    #     #                 self.abunData['TNII'], nSII = self.diags.getCrossTemDen(diag_tem = '[NII] 5755/6584+',
    #     #                                                                         diag_den  = '[SII] 6731/6716',
    #     #                                                                         value_tem = (self.lines_dict['N2_5755A'] - Lines_Correction)/(self.lines_dict['N2_6548A'] + self.lines_dict['N2_6584A']),
    #     #                                                                         value_den = self.lines_dict['S2_6731A']/self.lines_dict['S2_6716A'])
    #     #
    #     #                 Ratio = self.lines_dict['N2_6548A'] + self.lines_dict['N2_6584A']
    #     #                 self.determine_ionic_abundance('NII_HII', self.N2_atom, Ratio, diagnos_mag, self.abunData['TNII'], ne)
    #     #
    #     #                 self.abunData['NI_OI'] = self.abunData['NII_HII'] / self.abunData['OII_HII']
    #     #                 self.abunData['NI_HI'] = self.abunData['NI_OI'] * self.abunData['OI_HI']
    #
    #     return
    #
    # def sulfur_abundance_scheme(self, Tlow, ne, SIII_lines_to_use):
    #
    #     print
    #     'Metiendo esto', SIII_lines_to_use
    #
    #     # Calculate the S+1 abundance
    #     S2_lines = ['S2_6716A', 'S2_6731A']
    #     diagnos_eval, diagnos_mag = self.check_obsLines(S2_lines)
    #     self.determine_ionic_abundance('SII_HII', self.S2_atom, diagnos_eval, diagnos_mag, Tlow, ne)
    #
    #     # Calculate the S+2 abundance
    #     S3_lines = ['S3_9069A', 'S3_9531A'] if SIII_lines_to_use == 'BOTH' else [SIII_lines_to_use]
    #     diagnos_eval, diagnos_mag = self.check_obsLines(S3_lines)
    #     if set(S3_lines) != set(['S3_9069A', 'S3_9531A']):
    #         print
    #         '-- Using SIII lines', diagnos_eval
    #
    #     self.determine_ionic_abundance('SIII_HII', self.S3_atom, diagnos_eval, diagnos_mag, Tlow, ne)
    #
    #     # Calculate the total sulfur abundance
    #     if set(self.abunData.index) >= {'SII_HII', 'SIII_HII'}:
    #
    #         self.abunData['SI_HI'] = self.abunData['SII_HII'] + self.abunData['SIII_HII']
    #
    #         # Add the S+3 component if the argon correction is found
    #         if set(self.abunData.index) >= {'ArIII_HII', 'ArIV_HII'}:
    #
    #             logAr2Ar3 = log10(self.abunData['ArIII_HII'] / self.abunData['ArIV_HII'])
    #             logSIV = log10(self.abunData['SIII_HII']) - (logAr2Ar3 - self.n_SIV_correction) / self.m_SIV_correction
    #             SIV_HII = power(10, logSIV)
    #
    #             # Evaluate the nan array
    #             nan_idcs = isnan(SIV_HII)
    #             nan_count = np_sum(nan_idcs)
    #
    #             # Directly save if not nan
    #             if nan_count == 0:
    #                 self.abunData['SIV_HII'] = SIV_HII
    #
    #             # Remove the nan entries performing a normal distribution
    #             elif nan_count < 0.90 * self.MC_array_len:
    #                 mag, error = nanmean(SIV_HII), nanstd(SIV_HII)
    #
    #                 # Generate truncated array to store the data
    #                 a, b = (0 - mag) / error, (1000 * mag - mag) / error
    #                 new_samples = truncnorm(a, b, loc=mag, scale=error).rvs(size=nan_count)
    #
    #                 # Replace nan entries
    #                 SIV_HII[nan_idcs] = new_samples
    #                 self.abunData['SIV_HII'] = SIV_HII
    #
    #                 if nan_count > self.MC_warning_limit:
    #                     print
    #                     '-- {} calculated with {}'.format('SIV_HII', nan_count)
    #
    #             self.abunData['SI_HI'] = self.abunData['SII_HII'] + self.abunData['SIII_HII'] + self.abunData['SIV_HII']
    #             self.abunData['ICF_SIV'] = self.abunData['SI_HI'] / (
    #                         self.abunData['SII_HII'] + self.abunData['SIII_HII'])
    #
    #     return
    #
    # def helium_abundance_elementalScheme(self, Te, ne, lineslog_frame, metal_ext=''):
    #
    #     # Check temperatures are not nan before starting the treatment
    #     if (not isinstance(Te, float)) and (not isinstance(ne, float)):
    #
    #         # HeI_indices = (lineslog_frame.Ion.str.contains('HeI_')) & (lineslog_frame.index != 'He1_8446A')  & (lineslog_frame.index != 'He1_7818A') & (lineslog_frame.index != 'He1_5016A')
    #         HeI_indices = (lineslog_frame.Ion.str.contains('HeI_')) & (
    #             lineslog_frame.index.isin(['He1_4472A', 'He1_5876A', 'He1_6678A']))
    #         HeI_labels = lineslog_frame.loc[HeI_indices].index.values
    #         HeI_ions = lineslog_frame.loc[HeI_indices].Ion.values
    #
    #         Emis_Hbeta = self.H1_atom.getEmissivity(tem=Te, den=ne, label='4_2', product=False)
    #
    #         # --Generating matrices with fluxes and emissivities
    #         for i in range(len(HeI_labels)):
    #
    #             pyneb_code = float(HeI_ions[i][HeI_ions[i].find('_') + 1:len(HeI_ions[i])])
    #             line_relative_Flux = self.lines_dict[HeI_labels[i]] / self.Hbeta_flux
    #             line_relative_emissivity = self.He1_atom.getEmissivity(tem=Te, den=ne, wave=pyneb_code,
    #                                                                    product=False) / Emis_Hbeta
    #             line_relative_emissivity = self.check_nan_entries(line_relative_emissivity)
    #
    #             if i == 0:
    #                 matrix_HeI_fluxes = copy(line_relative_Flux)
    #                 matrix_HeI_emis = copy(line_relative_emissivity)
    #             else:
    #                 matrix_HeI_fluxes = vstack((matrix_HeI_fluxes, line_relative_Flux))
    #                 matrix_HeI_emis = vstack((matrix_HeI_emis, line_relative_emissivity))
    #
    #         matrix_HeI_fluxes = transpose(matrix_HeI_fluxes)
    #         matrix_HeI_emis = transpose(matrix_HeI_emis)
    #
    #         # Perform the fit
    #         params = Parameters()
    #         params.add('Y', value=0.01)
    #         HeII_HII_array = zeros(len(matrix_HeI_fluxes))
    #         HeII_HII_error = zeros(len(matrix_HeI_fluxes))
    #         for i in range(len(matrix_HeI_fluxes)):
    #             fit_Output = lmfit_minimmize(residual_Y_v3, params, args=(matrix_HeI_emis[i], matrix_HeI_fluxes[i]))
    #             HeII_HII_array[i] = fit_Output.params['Y'].value
    #             HeII_HII_error[i] = fit_Output.params['Y'].stderr
    #
    #         # NO SUMANDO LOS ERRORES CORRECTOS?
    #         # self.abunData['HeII_HII_from_' + metal_ext] = random.normal(mean(HeII_HII_array), mean(HeII_HII_error), size = self.MC_array_len)
    #         ionic_abund = random.normal(mean(HeII_HII_array), mean(HeII_HII_error), size=self.MC_array_len)
    #
    #         # Evaluate the nan array
    #         nan_count = np_sum(isnan(ionic_abund))
    #         if nan_count == 0:
    #             self.abunData['HeII_HII_from_' + metal_ext] = ionic_abund
    #         # Remove the nan entries performing a normal distribution
    #         elif nan_count < 0.90 * self.MC_array_len:
    #             mag, error = nanmean(ionic_abund), nanstd(ionic_abund)
    #             self.abunData['HeII_HII_from_' + metal_ext] = random.normal(mag, error, size=self.MC_array_len)
    #             if nan_count > self.MC_warning_limit:
    #                 print
    #                 '-- {} calculated with {}'.format('HeII_HII_from_' + metal_ext, nan_count)
    #
    #         # Calculate the He+2 abundance
    #         if self.lines_dict.viewkeys() >= {'He2_4686A'}:
    #             # self.abunData['HeIII_HII_from_' + metal_ext] = self.He2_atom.getIonAbundance(int_ratio = self.lines_dict['He2_4686A'], tem=Te, den=ne, wave = 4685.6, Hbeta = self.Hbeta_flux)
    #             self.determine_ionic_abundance('HeIII_HII_from_' + metal_ext, self.He2_atom, 'L(4685)',
    #                                            self.lines_dict['He2_4686A'], Te, ne)
    #
    #         # Calculate elemental abundance
    #         Helium_element_keys = ['HeII_HII_from_' + metal_ext, 'HeIII_HII_from_' + metal_ext]
    #         if set(self.abunData.index) >= set(Helium_element_keys):
    #             self.abunData['HeI_HI_from_' + metal_ext] = self.abunData[Helium_element_keys[0]] + self.abunData[
    #                 Helium_element_keys[1]]
    #         else:
    #             self.abunData['HeI_HI_from_' + metal_ext] = self.abunData[Helium_element_keys[0]]
    #
    #         # Proceed to get the Helium mass fraction Y
    #         Element_abund = metal_ext + 'I_HI'
    #         Y_fraction, Helium_abund = 'Ymass_' + metal_ext, 'HeI_HI_from_' + metal_ext
    #         if set(self.abunData.index) >= {Helium_abund, Element_abund}:
    #             self.abunData[Y_fraction] = (4 * self.abunData[Helium_abund] * (
    #                         1 - 20 * self.abunData[Element_abund])) / (1 + 4 * self.abunData[Helium_abund])
