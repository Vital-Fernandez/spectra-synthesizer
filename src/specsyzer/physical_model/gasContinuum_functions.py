from os import path
from numpy import power, log10, exp, zeros, ceil, interp, loadtxt, digitize
from scipy import interpolate


class NebularContinuaCalculator():

    def __init__(self):

        # Dictionary with the constants
        # self.loadNebCons(data_folder)
        return

    def loadNebCons(self, data_folder):

        # Dictionary with the calculation constants
        self.nebConst = {}

        # Browns and Seaton FF methodology
        self.nebConst['h']          = 6.626068e-27  # erg s,  cm2 g / s  ==  erg s
        self.nebConst['c_CGS']      = 2.99792458e10  # cm / s
        self.nebConst['c_Angs']     = 2.99792458e18  # cm / s
        self.nebConst['eV2erg']     = 1.602177e-12
        self.nebConst['pi']         = 3.141592
        self.nebConst['masseCGS']   = 9.1096e-28
        self.nebConst['e_proton']   = 4.80320425e-10  # statCoulomb = 1 erg^1/2 cm^1/2 # Eperez definition electronCGS = 1.60217646e-19 * 3.e9 # Coulomb  # 1eV = 1.60217646e-19 Jul
        self.nebConst['k']          = 1.3806503e-16  # erg / K
        self.nebConst['H0_ion_Energy']  = 13.6057  # eV
        self.nebConst['nu_0']           = self.nebConst['H0_ion_Energy'] * self.nebConst['eV2erg'] / self.nebConst['h']  # Hz
        self.nebConst['H_Ryd_Energy']   = self.nebConst['h'] * self.nebConst['nu_0']

        # Coefficients for calculating A_2q The total radiative probability 2s -> 1s (s^-1)
        self.nebConst['alpha_A']    = 0.88
        self.nebConst['beta_A']     = 1.53
        self.nebConst['gamma_A']    = 0.8
        self.nebConst['lambda_2q']  = 1215.7  # Angstroms
        self.nebConst['C_A']        = 202.0   # (s^-1)
        self.nebConst['A2q']        = 8.2249  # (s^-1) Transition probability at lambda = 1215.7

        # Free Bound constants
        self.nebConst['Ryd2erg']    = 2.1798723e-11  # Rydberg to erg   # (s^-1) Transition probability at lambda = 1215.7

        # Load files
        self.HI_fb_dict     = self.importErcolanoFBdata(path.join(data_folder, 'HI_t3_elec.ascii'))
        self.HeI_fb_dict    = self.importErcolanoFBdata(path.join(data_folder, 'HeI_t5_elec.ascii'))
        self.HeII_fb_dict   = self.importErcolanoFBdata(path.join(data_folder, 'HeII_t4_elec.ascii'))

        return

    def importErcolanoFBdata(self, file_address):

        dict_ion = {}

        # Reading the text files
        with open(file_address, 'r') as f:

            a = f.readlines()

            dict_ion['nTe'] = int(str.split(a[0])[0])  # number of Te columns
            dict_ion['nEner'] = int(str.split(a[0])[1])  # number of energy points rows
            dict_ion['skip'] = int(1 + ceil(dict_ion['nTe'] / 8.))  # 8 es el numero de valores de Te por fila.
            dict_ion['temps'] = zeros(dict_ion['nTe'])
            dict_ion['lines'] = a

            # Storing temperature range
            for i in range(1, dict_ion['skip']):
                tt = str.split(a[i])
                for j in range(0, len(tt)):
                    dict_ion['temps'][8 * (i - 1) + j] = tt[j]

            # Storing gamma_cross grids
            dict_ion['matrix'] = loadtxt(file_address, skiprows=dict_ion['skip'])

        # Get wavelengths corresponding to table threshold and remove zero entries
        wave_thres = dict_ion['matrix'][:, 0] * dict_ion['matrix'][:, 1]
        idx_zero = (wave_thres == 0)
        dict_ion['wave_thres'] = wave_thres[~idx_zero]

        return dict_ion

    def nebFluxCont(self, wave_rest, cHbeta, flambda_neb, Te, He1_abund, He2_abund, Halpha_Flux):

        nebGammaCont    = self.nebGammaCont(wave_rest, Te, He1_abund, He2_abund)

        neb_int_norm    = self.gContCalib(wave_rest, Te, Halpha_Flux, nebGammaCont)

        neb_flux_norm   = neb_int_norm * power(10, -1 * flambda_neb * cHbeta)

        return neb_flux_norm

    def nebGammaCont(self, wave, Te, HeII_HII, HeIII_HII):

        H_He_frac = 1 + HeII_HII * 4 + HeIII_HII * 4

        # Bound bound continuum
        gamma_2q = self.twoPhotonGammaCont(wave, Te)

        # Free-Free continuum
        gamma_ff = H_He_frac * self.freefreeGammaCont(wave, Te, Z_ion=1.0)

        # Get the wavelength range in ryddbergs for the Ercolano grids
        wave_ryd = (self.nebConst['h'] * self.nebConst['c_Angs']) / (self.nebConst['Ryd2erg'] * wave)

        # Free-Bound continuum
        gamma_fb_HI = self.freeboundGammaCont(wave_ryd, Te, self.HI_fb_dict)
        gamma_fb_HeI = self.freeboundGammaCont(wave_ryd, Te, self.HeI_fb_dict)
        gamma_fb_HeII = self.freeboundGammaCont(wave_ryd, Te, self.HeII_fb_dict)
        gamma_fb = gamma_fb_HI + HeII_HII * gamma_fb_HeI + HeIII_HII * gamma_fb_HeII

        return gamma_2q + gamma_ff + gamma_fb

    def twoPhotonGammaCont(self, wave, Te):

        # Prepare arrays
        idx_limit = (wave > self.nebConst['lambda_2q'])
        gamma_array = zeros(wave.size)

        # Params
        q2 = 5.92e-4 - 6.1e-9 * Te  # (cm^3 s^-1) Collisional transition rate coefficient for protons and electrons
        alpha_eff_2q = 6.5346e-11 * power(Te, -0.72315) # (cm^3 s^-1) Effective Recombination coefficient

        nu_array = self.nebConst['c_Angs'] / wave[idx_limit]
        nu_limit = self.nebConst['c_Angs'] / self.nebConst['lambda_2q']

        y = nu_array / nu_limit

        A_y = self.nebConst['C_A'] * (
                    y * (1 - y) * (1 - (4 * y * (1 - y)) ** self.nebConst['gamma_A']) + self.nebConst['alpha_A'] *
                    (y * (1 - y)) ** self.nebConst['beta_A'] * (4 * y * (1 - y)) ** self.nebConst['gamma_A'])

        g_nu1 = self.nebConst['h'] * nu_array / nu_limit / self.nebConst['A2q'] * A_y
        g_nu2 = alpha_eff_2q * g_nu1 / (1 + q2 / self.nebConst['A2q'])

        gamma_array[idx_limit] = g_nu2[:]

        return gamma_array

    def freefreeGammaCont(self, wave, Te, Z_ion=1.0):

        cte_A = (32 * (Z_ion ** 2) * (self.nebConst['e_proton'] ** 4) * self.nebConst['h']) / (
                    3 * (self.nebConst['masseCGS'] ** 2) * (self.nebConst['c_CGS'] ** 3))

        cte_B = ((self.nebConst['pi'] * self.nebConst['H_Ryd_Energy'] / (3 * self.nebConst['k'] * Te)) ** 0.5)

        cte_Total = cte_A * cte_B

        nu_array = self.nebConst['c_Angs'] / wave

        gamma_Comp1 = exp(((-1 * self.nebConst['h'] * nu_array) / (self.nebConst['k'] * Te)))
        gamma_Comp2 = (self.nebConst['h'] * nu_array) / ((Z_ion ** 2) * self.nebConst['e_proton'] * 13.6057)
        gamma_Comp3 = self.nebConst['k'] * Te / (self.nebConst['h'] * nu_array)

        gff = 1 + 0.1728 * power(gamma_Comp2, 0.33333) * (1 + 2 * gamma_Comp3) - 0.0496 * power(gamma_Comp2, 0.66667) * (
                          1 + 0.66667 * gamma_Comp3 + 1.33333 * power(gamma_Comp3, 2))

        gamma_array = cte_Total * gamma_Comp1 * gff

        return gamma_array

    def freeboundGammaCont(self, wave_ryd, Te, data_dict):

        # Temperature entry
        t4 = Te / 10000.0
        logTe = log10(Te)

        # Interpolating the grid for the right wavelength
        thres_idxbin = digitize(wave_ryd, data_dict['wave_thres'])
        interpol_grid = interpolate.interp2d(data_dict['temps'], data_dict['matrix'][:, 1], data_dict['matrix'][:, 2:], kind='linear')

        ener_low = data_dict['wave_thres'][thres_idxbin - 1] # WARNING: This one could be an issue for wavelength range table limits

        # Interpolate table for the right temperature
        gamma_inter_Te = interpol_grid(logTe, data_dict['matrix'][:, 1])[:, 0]
        gamma_inter_Te_Ryd = interp(wave_ryd, data_dict['matrix'][:, 1], gamma_inter_Te)

        Gamma_fb_f = gamma_inter_Te_Ryd * 1e-40 * power(t4, -1.5) * exp(-15.7887 * (wave_ryd - ener_low) / t4)

        return Gamma_fb_f

    def gContCalib(self, wave, Te, flux_Emline, gNeb_cont_nu, lambda_EmLine=6562.819):

        # Zanstra like calibration for the continuum
        t4 = Te / 10000.0

        # Pequignot et al. 1991
        alfa_eff_alpha = 2.708e-13 * t4 ** -0.648 / (1 + 1.315 * t4 ** 0.523)
        # alfa_eff_beta = 0.668e-13 * t4**-0.507 / (1 + 1.221*t4**0.653)

        fNeb_cont_lambda = gNeb_cont_nu * lambda_EmLine * flux_Emline / (alfa_eff_alpha * self.nebConst['h'] * wave * wave)

        return fNeb_cont_lambda