import os
import numpy as np
import pandas as pd
from os import chdir
from shutil import copyfile
from scipy.optimize import nnls
from scipy.signal.signaltools import convolve2d
from scipy.interpolate.interpolate import interp1d
from subprocess import Popen, PIPE, STDOUT
from matplotlib import pyplot as plt, rcParams, colors, cm, ticker


# Reddening law from CCM89 # TODO Replace this methodology to an Import of Pyneb
def CCM89_Bal07(Rv, wave):

    x = 1e4 / wave  # Assuming wavelength is in Amstrongs
    ax = np.zeros(len(wave))
    bx = np.zeros(len(wave))

    idcs = x > 1.1
    y = (x[idcs] - 1.82)

    ax[idcs] = 1 + 0.17699 * y - 0.50447 * y ** 2 - 0.02427 * y ** 3 + 0.72085 * y ** 4 + 0.01979 * y ** 5 - 0.77530 * y ** 6 + 0.32999 * y ** 7
    bx[idcs] = 1. * y + 2.28305 * y ** 2 + 1.07233 * y ** 3 - 5.38434 * y ** 4 - 0.62251 * y ** 5 + 5.30260 * y ** 6 - 2.09002 * y ** 7
    ax[~idcs] = 0.574 * x[~idcs] ** 1.61
    bx[~idcs] = -0.527 * x[~idcs] ** 1.61

    Xx = ax + bx / Rv  # WARNING better to check this definition

    return Xx


# Line finder for Starlight files
def lineFinder(myFile, myText):

    # THIS IS VERY INNEFFICIENT OPTION
    for i in range(len(myFile)):
        if myText in myFile[i]:
            return i


# Calculate galaxy mass from starlight fitting knowing its distance
def computeSSP_galaxy_mass(mass_uncalibrated, flux_norm, redshift):

    # Model constants
    c_kms = 299792  # km/s ufloat(74.3, 6.0)
    Huble_Constant = 74.3  # (km/s / Mpc)
    mpc_2_cm = 3.086e24  # cm/mpc

    # Compute the distance
    dist_mpc = redshift * (c_kms / Huble_Constant)
    dist_cm = dist_mpc * mpc_2_cm

    # Compute mass in solar masses
    Mass = mass_uncalibrated * flux_norm * 4 * np.pi * np.square(dist_cm) * (1 / 3.826e33)
    logMass = np.log10(Mass)

    return logMass


class SspLinearModel:

    def __init__(self):

        self.ssp_conf_dict = {}

    def physical_SED_model(self, bases_wave_rest, obs_wave, bases_flux, Av_star, z_star, sigma_star, Rv_coeff=3.4):

        # Calculate wavelength at object z
        wave_z = bases_wave_rest * (1 + z_star)

        # Kernel matrix
        box = int(np.ceil(max(3 * sigma_star)))
        kernel_len = 2 * box + 1
        kernel_range = np.arange(0, 2 * box + 1)
        kernel = np.empty((1, kernel_len))

        # Filling gaussian values (the norm factor is the sum of the gaussian)
        kernel[0, :] = np.exp(-0.5 * (np.square((kernel_range - box) / sigma_star)))
        kernel /= sum(kernel[0, :])

        # Convove bases with respect to kernel for dispersion velocity calculation
        basesGridConvolved = convolve2d(bases_flux, kernel, mode='same', boundary='symm')

        # Interpolate bases to wavelength ranges
        basesGridInterp = (interp1d(wave_z, basesGridConvolved, axis=1, bounds_error=True)(obs_wave)).T

        # Generate final flux model including reddening
        Av_vector = Av_star * np.ones(basesGridInterp.shape[1])
        obs_wave_resam_rest = obs_wave / (1 + z_star)
        Xx_redd = CCM89_Bal07(Rv_coeff, obs_wave_resam_rest)
        dust_attenuation = np.power(10, -0.4 * np.outer(Xx_redd, Av_vector))
        bases_grid_redd = basesGridInterp * dust_attenuation

        return bases_grid_redd

    def ssp_fitting(self, ssp_grid_masked, obs_flux_masked):

        optimize_result = nnls(ssp_grid_masked, obs_flux_masked)

        return optimize_result[0]

    def linfit1d(self, obsFlux_norm, obsFlux_mean, basesFlux, weight):

        nx, ny = basesFlux.shape

        # Case where the number of pixels is smaller than the number of bases
        if nx < ny:
            basesFlux = np.transpose(basesFlux)
            nx = ny

        A = basesFlux
        B = obsFlux_norm

        # Weight definition #WARNING: Do we need to use the diag?
        if weight.shape[0] == nx:
            weight = np.diag(weight)
            A = np.dot(weight, A)
            B = np.dot(weight, np.transpose(B))
        else:
            B = np.transpose(B)

        coeffs_0 = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, B)) * obsFlux_mean

        return coeffs_0


class SSPsynthesizer(SspLinearModel):

    def __init__(self):

        SspLinearModel.__init__(self)

        self._basesDB_headers = ['file_name', 'age_yr', 'z_star', 'template_label', 'f_star', 'YAV_flag', 'alpha/Fe',
                                 'wmin', 'wmax']

        return

    def import_STARLIGHT_bases(self, bases_file_address, bases_folder, crop_waves=None, resam_inter=None, norm_waves=None):

        # TODO separate read starlight life from reading the spectra
        print('\n- Importing STARLIGHT library')
        print(f'-- Bases file: {bases_file_address}')
        print(f'-- Bases folder: {bases_folder}')

        bases_df = pd.read_csv(bases_file_address, delim_whitespace=True, names=self._basesDB_headers, skiprows=1)
        n_bases = len(bases_df.index)

        # Add column with min and max wavelength
        for header in ['wmin', 'wmax', 'norm_flux']:
            bases_df[header] = np.nan

        # Loop throught the files and get the wavelengths. They may have different wavelength range
        for i in np.arange(n_bases):

            # Load the data from the text file
            template_file = bases_folder/bases_df.iloc[i]['file_name']
            wave_i, flux_i = np.loadtxt(template_file, unpack=True)

            wmin_i, wmax_i = wave_i[0], wave_i[-1]
            bases_df.loc[bases_df.index[i], 'wmin'] = wmin_i
            bases_df.loc[bases_df.index[i], 'wmax'] = wmax_i

            wave_i, flux_i, normFlux_i = self.treat_input_spectrum(wave_i, flux_i, crop_waves, resam_inter, norm_waves)
            bases_df.loc[bases_df.index[i], 'norm_flux'] = normFlux_i

            # Initiate for the first time according to resampling and cropping size
            if i == 0:
                flux_matrix = np.empty((n_bases, wave_i.size))
            flux_matrix[i, :] = flux_i

        print('--Library imported')

        return bases_df, wave_i, flux_matrix

    def treat_input_spectrum(self, wave, flux, crop_waves=None, resam_inter=None, norm_waves=None):

        # TODO we should remove the nBases requirement by some style which can just read the number of dimensions

        # Establish crop limitts
        crop_waves = (wave[0], wave[-1]) if crop_waves is None else crop_waves

        # Resampling the spectra
        if resam_inter is not None:
            wave_out = np.arange(crop_waves[0], crop_waves[1], resam_inter, dtype=float)
            flux_out = interp1d(wave, flux, bounds_error=True)(wave_out)
        else:
            wave_out = wave
            flux_out = flux

        # Normalizing the spectra
        if norm_waves is not None:
            idx_wmin, idx_wmax = np.searchsorted(wave, norm_waves)
            normFlux_coeff_i = np.mean(flux[idx_wmin:idx_wmax])
            flux_out = flux_out / normFlux_coeff_i
        else:
            normFlux_coeff_i = 1.0

        return wave_out, flux_out, normFlux_coeff_i

    def replace_line(self, file_name, line_num, text):

        Input_File = open(file_name, 'r')
        lines = Input_File.readlines()
        lines[line_num] = text

        Output_File = open(file_name, 'w')
        Output_File.writelines(lines)

        Input_File.close()
        Output_File.close()

    def save_starlight_spectra(self, TableOfValues, FileAddress):

        File = open(FileAddress, "w")

        for i in range(len(TableOfValues[0])):
            Sentence = ''
            for j in range(len(TableOfValues)):
                Sentence = Sentence + ' ' + str(TableOfValues[j][i])

            File.write(Sentence + '\n')

        File.close()

    def ImportDispersionVelocity(self, linesLogDF, c_SI=300000.0):

        O3_5007_sigma = linesLogDF.loc['O3_5007A', 'sigma'] if 'O3_5007A' in linesLogDF.index else None
        Hbeta_sigma = linesLogDF.loc['H1_4861A', 'sigma'] if 'H1_4861A' in linesLogDF.index else None

        if Hbeta_sigma is not None:
            sigma_line = Hbeta_sigma / 4861.0 * c_SI
        elif O3_5007_sigma is not None:
            sigma_line = O3_5007_sigma / 5007.0 * c_SI
        else:
            sigma_line = 50

        return sigma_line

    def generate_starlight_files(self, starlight_Folder, objName, X, Y, regionsDF, v_vector=None, clip_value=3):

        Default_InputFolder = starlight_Folder/'Obs'
        Default_MaskFolder = starlight_Folder/'Masks'
        Default_BasesFolder = starlight_Folder/'Bases'
        Default_OutputFoler = starlight_Folder/'Output'

        # -----------------------     Generating Base File    ----------------------------------------------
        BaseDataFile = 'Dani_Bases_Extra.txt'

        # -----------------------     Generating Configuration File    -------------------------------------
        configFile = f'{objName}_Config_v1.txt'
        copyfile(starlight_Folder/'Sl_Config_v1.txt', starlight_Folder/configFile)

        # Establishing object maximum dispersion velocity
        sigma = self.ImportDispersionVelocity(regionsDF)
        UpperVelLimit = str(round(sigma, 1)) + '      [vd_upp (km/s)]     = upper allowed vd\n'
        self.replace_line(starlight_Folder/configFile, 21, UpperVelLimit)

        #Establishing clip value
        clip_value = f'{clip_value:.1f}          [sig_clip_threshold]             = clip points which deviate > than this # of sigmas\n'
        self.replace_line(starlight_Folder/configFile, 27, clip_value)

        # -----------------------Generating input spectra Textfile---------------------------------
        Interpolation = interp1d(X, Y, kind='slinear')
        Wmin = int(round(X[0], 0))
        Wmax = int(round(X[-1], 0))

        # Interpolate the new spectra to one angstrom per pixel resolution
        X_1Angs = range(Wmin + 1, Wmax - 1, 1)
        Y_1Angs = Interpolation(X_1Angs)

        Sl_Input_Filename = f'{objName}.slInput'
        self.save_starlight_spectra([X_1Angs, Y_1Angs], Default_InputFolder/Sl_Input_Filename)
        print('-- Starlight File:', Default_InputFolder/Sl_Input_Filename)

        # -----------------------     Generating Mask File    ----------------------------------------------

        # Import emision line location from lick indexes file
        labelList, iniWaveList, finWaveList = regionsDF.index.values, regionsDF['w3'].values, regionsDF['w4'].values

        # Loop through the lines and for the special cases increase the thickness
        maskFileName = objName + '_Mask.lineslog'
        maskFile = open(Default_MaskFolder/maskFileName, "w")
        maskFile.write(f'{len(labelList)}\n')
        for k in range(len(labelList)):
            line = f'{iniWaveList[k]:.2f}  {finWaveList[k]:.2f}  0.0  {labelList[k]}\n'
            maskFile.write(line)
        maskFile.close()
        print('-- Mask File:', Default_MaskFolder/maskFileName)

        # -----------------------     Generating output files    -------------------------------------------

        Sl_Output_Filename = f'{objName}.slOutput'
        print('-- Output address:', Default_OutputFoler/Sl_Output_Filename)

        # -----------------------Generating Grid file---------------------------------
        if v_vector is None:
            v_vector = ['FXK', '0.0', str(round(sigma, 1))]

        GridLines = []
        GridLines.append("1")  # "[Number of fits to run]"])
        GridLines.append(f'{Default_BasesFolder}/')  # "[base_dir]"])
        GridLines.append(f'{Default_InputFolder}/')  # "[obs_dir]"])
        GridLines.append(f'{Default_MaskFolder}/')  # "[mask_dir]"])
        GridLines.append(f'{Default_OutputFoler}/')  # "[out_dir]"])
        GridLines.append("-652338184")  # "[your phone number]"])
        GridLines.append("4500.0 ")  # "[llow_SN]   lower-lambda of S/N window"])
        GridLines.append("4550.0")  # "[lupp_SN]   upper-lambda of S/N window"])
        GridLines.append("3400.0")  # "[Olsyn_fin] upper-lambda for fit"])
        GridLines.append("12000.0")  # "[Olsyn_fin] upper-lambda for fit"])
        GridLines.append("1.0")  # "[Odlsyn]    delta-lambda for fit"])
        GridLines.append("1.0")  # "[fscale_chi2] fudge-factor for chi2"])
        GridLines.append(v_vector[0])  # "[FIT/FXK] Fit or Fix kinematics"])
        GridLines.append("0")  # "[IsErrSpecAvailable]  1/0 = Yes/No"])
        GridLines.append("0")  # "[IsFlagSpecAvailable] 1/0 = Yes/No"])

        Redlaw = 'CCM'
        v0_start = v_vector[1]
        vd_start = v_vector[2]

        files_row = [Sl_Input_Filename, configFile, BaseDataFile, maskFileName, Redlaw, v0_start, vd_start, Sl_Output_Filename]

        GridLines.append(files_row)

        Grid_FileName = f'{objName}.slGrid'
        print('-- Grid File:', starlight_Folder/Grid_FileName)

        File = open(starlight_Folder/Grid_FileName, "w")

        for i in range(len(GridLines) - 1):
            Parameter = GridLines[i]
            Element = str(Parameter) + "\n"
            File.write(Element)

        Element = "  ".join(GridLines[-1]) + '\n'
        File.write(Element)
        File.close()

        return Grid_FileName, Sl_Output_Filename, Default_OutputFoler, X_1Angs, Y_1Angs

    def load_starlight_output(self, outputFileAddress):

        DataFile = open(outputFileAddress, "r")
        StarlightOutput = DataFile.readlines()
        DataFile.close()

        # Synthesis Results - Best model #
        Chi2Line = lineFinder(StarlightOutput, "[chi2/Nl_eff]")
        AdevLine = lineFinder(StarlightOutput, "[adev (%)]")
        SumXdevLine = lineFinder(StarlightOutput, "[sum-of-x (%)]")
        Mini_totLine = lineFinder(StarlightOutput, "[Mini_tot (???)]")
        Mcor_totLine = lineFinder(StarlightOutput, "[Mcor_tot (???)]")
        v0_min_Line = lineFinder(StarlightOutput, "[v0_min  (km/s)]")
        vd_min_Line = lineFinder(StarlightOutput, "[vd_min  (km/s)]")
        Av_min_Line = lineFinder(StarlightOutput, "[AV_min  (mag)]")
        Nl_eff_line = lineFinder(StarlightOutput, "[Nl_eff]")
        SignalToNoise_Line = lineFinder(StarlightOutput, "## S/N")
        l_norm_Line = lineFinder(StarlightOutput, "## Normalization info") + 1
        llow_norm_Line = lineFinder(StarlightOutput, "## Normalization info") + 2
        lupp_norm_Line = lineFinder(StarlightOutput, "## Normalization info") + 3
        NormFlux_Line = lineFinder(StarlightOutput, "## Normalization info") + 4

        # Location of my Spectrum in starlight output
        SpecLine = lineFinder(StarlightOutput,"## Synthetic spectrum (Best Model) ##l_obs f_obs f_syn wei")

        # Quality of fit
        Chi2 = float(StarlightOutput[Chi2Line].split()[0])
        Adev = float(StarlightOutput[AdevLine].split()[0])
        SumXdev = float(StarlightOutput[SumXdevLine].split()[0])
        Nl_eff = float(StarlightOutput[Nl_eff_line].split()[0])
        Mini_tot = float(StarlightOutput[Mini_totLine].split()[0])
        Mcor_tot = float(StarlightOutput[Mcor_totLine].split()[0])
        v0_min = float(StarlightOutput[v0_min_Line].split()[0])
        vd_min = float(StarlightOutput[vd_min_Line].split()[0])
        Av_min = float(StarlightOutput[Av_min_Line].split()[0])

        # Signal to noise configuration
        SignalToNoise_lowWave = float(StarlightOutput[SignalToNoise_Line + 1].split()[0])
        SignalToNoise_upWave = float(StarlightOutput[SignalToNoise_Line + 2].split()[0])
        SignalToNoise_magnitudeWave = float(StarlightOutput[SignalToNoise_Line + 3].split()[0])

        # Flux normalization parameters
        l_norm = float(StarlightOutput[l_norm_Line].split()[0])
        llow_norm = float(StarlightOutput[llow_norm_Line].split()[0])
        lupp_norm = float(StarlightOutput[lupp_norm_Line].split()[0])
        FluxNorm = float(StarlightOutput[NormFlux_Line].split()[0])

        # Read continuum results
        Pixels_Number = int(StarlightOutput[SpecLine + 1].split()[0])  # Number of pixels in the spectra
        Ind_i = SpecLine + 2  # First pixel location
        Ind_f = Ind_i + Pixels_Number  # Final pixel location

        Input_Wavelength = np.zeros(Pixels_Number)
        Input_Flux = np.zeros(Pixels_Number)
        Output_Flux = np.zeros(Pixels_Number)
        Output_Mask = np.zeros(Pixels_Number)

        for i in range(Ind_i, Ind_f):
            Index = i - Ind_i
            Line = StarlightOutput[i].split()
            Input_Wavelength[Index] = float(Line[0])
            Input_Flux[Index] = float(Line[1]) * FluxNorm if Line[1] != '**********' else 0.0
            Output_Flux[Index] = float(Line[2]) * FluxNorm
            Output_Mask[Index] = float(Line[3])

        # Read fitting masks
        MaskPixels = [[], []]  # The 0 tag
        ClippedPixels = [[], []]  # The -1 tag
        FlagPixels = [[], []]  # The -2 tag

        for j in range(len(Output_Mask)):
            PixelTag = Output_Mask[j]
            Wave = Input_Wavelength[j]
            if PixelTag == 0:
                MaskPixels[0].append(Wave)
                MaskPixels[1].append(Input_Flux[j])
            if PixelTag == -1:
                ClippedPixels[0].append(Wave)
                ClippedPixels[1].append(Input_Flux[j])
            if PixelTag == -2:
                FlagPixels[0].append(Wave)
                FlagPixels[1].append(Input_Flux[j])

        Parameters = dict(Chi2=Chi2, Adev=Adev, SumXdev=SumXdev, Nl_eff=Nl_eff, v0_min=v0_min, vd_min=vd_min, Av_min=Av_min,
                          SignalToNoise_lowWave=SignalToNoise_lowWave, SignalToNoise_upWave=SignalToNoise_upWave,
                          SignalToNoise_magnitudeWave=SignalToNoise_magnitudeWave, l_norm=l_norm, llow_norm=llow_norm,
                          lupp_norm=lupp_norm, MaskPixels=MaskPixels, ClippedPixels=ClippedPixels, FlagPixels=FlagPixels)

        # Get number of bases
        BasesLine = lineFinder(StarlightOutput, "[N_base]")  # Location of my normalization flux in starlight output
        Bases = int(StarlightOutput[BasesLine].split()[0])

        # Location of my normalization flux in starlight output
        #j,  x_j( %),  Mini_j( %),  Mcor_j( %),  age_j(yr), Z_j, (L / M)_j, YAV?, Mstars, component_j, a / Fe...
        #0   1         2            3             4          5     6         7     8       9            10
        Sl_DataHeader = lineFinder(StarlightOutput, "# j     x_j(%)      Mini_j(%)     Mcor_j(%)     age_j(yr)")
        Ind_i = Sl_DataHeader + 1
        Ind_f = Sl_DataHeader + Bases

        index = []
        x_j = []
        Mini_j, Mcor_j = [], []
        age_j, Z_j = [], []
        LbyM, Mstars = [], []
        component_name = []

        for j in range(Ind_i, Ind_f + 1):
            myDataLine = StarlightOutput[j].split()
            index.append(float(myDataLine[0]))
            x_j.append(float(myDataLine[1]))
            Mini_j.append(float(myDataLine[2]))
            Mcor_j.append(float(myDataLine[3]))
            age_j.append(float(myDataLine[4]))
            Z_j.append(float(myDataLine[5]))
            LbyM.append(float(myDataLine[6]))
            Mstars.append(float(myDataLine[8]))
            component_name.append(myDataLine[9])

        columns = ['j', 'x_j', 'Mini_j', 'Mcor_j', 'age_j', 'Z_j', 'L_to_M', 'Mstars', 'component_j', 'a_to_Fe']
        BasesDF = pd.DataFrame(index=index, columns=columns)
        BasesDF['x_j'] = x_j
        BasesDF['Mini_j'] = Mini_j
        BasesDF['Mcor_j'] = Mcor_j
        BasesDF['age_j'] = age_j
        BasesDF['Z_j'] = Z_j
        BasesDF['L_to_M'] = LbyM
        BasesDF['Mstars'] = Mstars
        BasesDF['component_j'] = component_name

        Parameters = dict(Chi2=Chi2, Adev=Adev, SumXdev=SumXdev, Nl_eff=Nl_eff, v0_min=v0_min, vd_min=vd_min, Av_min=Av_min,
                          SignalToNoise_lowWave=SignalToNoise_lowWave, SignalToNoise_upWave=SignalToNoise_upWave,
                          SignalToNoise_magnitudeWave=SignalToNoise_magnitudeWave, l_norm=l_norm, llow_norm=llow_norm,
                          lupp_norm=lupp_norm, MaskPixels=MaskPixels, ClippedPixels=ClippedPixels, FlagPixels=FlagPixels,
                          index=index, x_j=x_j, Mini_j=Mini_j, Mcor_j=Mcor_j, age_j=age_j, Z_j=Z_j, LbyM=LbyM, Mstars=Mstars,
                          Mini_tot=Mini_tot, Mcor_tot=Mcor_tot, DF=BasesDF)

        return Input_Wavelength, Input_Flux, Output_Flux, Parameters

    def starlight_launcher(self, gridFile, executable_folder='/home/vital/'):

        chdir(executable_folder)
        print(f'-- Runing folder: {os.getcwd()} {str(executable_folder)==os.getcwd()}')
        Command = './StarlightChains_v04.exe < ' + str(gridFile)

        print(f'-- Launching: {Command}')
        p = Popen(Command, shell=True, stdout=PIPE, stderr=STDOUT)

        for line in p.stdout.readlines():
            print(line,)

        retval = p.wait()

        return

    def stellar_fit_comparison_plot(self, objName, wave, flux, nebCompFile, stellarFluxFile, outputFileAddress=None):

        labelsDict = {'xlabel': r'Wavelength $(\AA)$',
                      'ylabel': r'Flux $(erg\,cm^{-2} s^{-1} \AA^{-1})$',
                      'title': f'Galaxy {objName} spectrum components'}

        # # compare adding componennts
        wave_neb, flux_neb = np.loadtxt(nebCompFile, unpack=True)
        wave_star, flux_star = np.loadtxt(stellarFluxFile, unpack=True)

        # Plot spectra components
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(wave, flux, label='Object flux')
        ax.plot(wave_neb, flux_neb, label='Nebular flux')
        ax.plot(wave_star, flux_star, label='Stellar flux')
        ax.plot(wave_star, flux_star + flux_neb, label='Combined continuum', linestyle=':')
        ax.legend()
        ax.set_yscale('log')
        plt.tight_layout()

        if outputFileAddress is None:
            plt.show()
        else:
            plt.savefig(outputFileAddress, bbox_inches='tight')

        plt.close(fig)

        return

    def population_fraction_plots(self, fit_output, objName, parameter, ouputFileAddress, mass_galaxy=None):

        # Extract the data from the starlight output
        index, x_j, Mini_j, Mcor_j, age_j, Z_j, LbyM, Mstars = fit_output['index'], fit_output['x_j'], fit_output['Mini_j'],\
                                                               fit_output['Mcor_j'], fit_output['age_j'], fit_output['Z_j'],\
                                                               fit_output['LbyM'], fit_output['Mstars']

        x_j, age_j, Mcor_j, Z_j = np.array(x_j), np.array(age_j), np.array(Mcor_j), np.array(Z_j)

        # Establish configuration for either mass or light fraction plots
        if parameter == 'Light_fraction':
            fraction_param = x_j
            labelsDict = {'xlabel': r'$log(Age)$',
                          'ylabel': r'Light fraction %',
                          'title': f'Galaxy {objName}' + '\nLight fraction'}

        elif parameter == 'Mass_fraction':
            fraction_param = Mcor_j
            labelsDict = {'xlabel': r'$log(Age)$',
                          'ylabel': r'Mass fraction %',
                          'title': f'Galaxy {objName}' + '\nMass fraction'}

            if mass_galaxy is not None:
                labelsDict['title'] = f'Galaxy {objName}'\
                                     + '\n' \
                                     + r'mass fraction $Log(M_{{\star}})={:.2f}$'.format(mass_galaxy)#

        else:
            print(f'-- ERROR: Fraction parameter: {parameter} not recognize use "Mass_fraction" or "Light_fraction"')

        # Get metallicities from calculation
        zValues = np.sort(np.array(np.unique(Z_j)))

        # Get color list for metallicities
        self.gen_colorList(0, len(zValues))

        # Plot format
        plotConf = {'axes.titlesize': 18, 'axes.labelsize': 16, 'legend.fontsize': 14, 'xtick.labelsize': 14,
                    'ytick.labelsize': 16}
        rcParams.update(plotConf)

        # Plot the age bins per metallicity:
        fig, ax = plt.subplots(figsize=(9, 9))

        # Populations which contribute to the fraction
        idx_param = (fraction_param >= 1.00)

        # Get age of valid stellar populations
        ageBins_HW = 0.20
        log_agej = np.log10(age_j)
        log_ageBins = np.linspace(4.80, 10.60, 30)
        log_age_array = log_agej[idx_param]
        idcs_age_bins_array = np.digitize(log_age_array, log_ageBins) - 1

        # Assign age bin to those populations
        idcs_sort = np.argsort(idcs_age_bins_array)
        idx_idcs_age_bins_sort = idcs_age_bins_array[idcs_sort]
        ics_bins = np.unique(idx_idcs_age_bins_sort)
        bins = log_ageBins[ics_bins]

        for i_bin, bin in enumerate(bins):

            idx_total = log_ageBins[idcs_age_bins_array] == bin

            # Load x and y data for the plot
            z_array = Z_j[idx_param][idx_total]
            param_array = fraction_param[idx_param][idx_total]

            # Combine by metallicities of same values:
            z_unique = np.unique(z_array)
            param_sort = np.zeros(z_unique.size)
            for i_z, z_value in enumerate(z_unique):
                idcs_z = (z_array == z_value)
                param_sort[i_z] = param_array[idcs_z].sum()

            # Sort by decreasing param value
            idx_sort = np.argsort(param_sort)[::-1]
            z_sort = z_unique[idx_sort]
            param_sort = param_sort[idx_sort]

            x = bin

            # Plot the individual bars
            for i in range(len(param_sort)):
                y = param_sort[i]

                # Compute the metallicity star contribution
                z_mag = z_sort[i]
                idx_z_mag = [Z_j == z_mag]
                z_total = fraction_param[idx_z_mag].sum()

                label = f'Z = {z_sort[i]} (total {z_total:.1f} %)'

                idx_color = np.where(zValues == z_sort[i])[0][0]
                color = self.get_color(idx_color)
                ax.bar(x, y, label=label, color=color, width=ageBins_HW/2, fill=True, edgecolor=color, log=True)

            # Plot age bin total
            param_total = sum(param_array)
            ax.bar(x, param_total, label='Age bin total', width=ageBins_HW, fill=False,
                          edgecolor='black', linestyle='--', log=True)

        # Change the axis format to replicate the style of Dani Miralles
        ax.set_ylim([1.0, 100])
        ax.set_xlim([5.5, 10.5])
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

        # Legend configuration
        # Security checks to avoid empty legends
        if ax.get_legend_handles_labels()[1] != None:

            if len(ax.get_legend_handles_labels()[1]) != 0:
                Old_Handles, Old_Labels = ax.get_legend_handles_labels()

                # Actual sorting
                labels, handles = zip(*sorted(zip(Old_Labels, Old_Handles), key=lambda t: t[0]))
                Handles_by_Label = dict((zip(labels, handles)))
                ax.legend(Handles_by_Label.values(), Handles_by_Label.keys(), loc='best', ncol=1)

        # Format titles
        ax.update(labelsDict)
        if ouputFileAddress is None:
            plt.show()
        else:
            plt.savefig(ouputFileAddress, bbox_inches='tight')

        # Clear the image
        plt.close(fig)

        return

    def mask_plot(self, fit_output, objName, objWave, objFlux, Input_Wavelength, stellar_flux, Input_Flux, maskFile, outputAddress=None):

        labelsDict = {'xlabel': r'Wavelength $(\AA)$',
                      'ylabel': r'Flux $(erg\,cm^{-2} s^{-1} \AA^{-1})$',
                      'title': f'Galaxy {objName} stellar continuum fit'}

        # Mask plots
        iniPoints, endPoints, tag = np.loadtxt(maskFile, skiprows=1, usecols=(0, 1, 2), unpack=True)
        label = np.loadtxt(maskFile, skiprows=1, usecols=(3), unpack=True, dtype=str)

        # Plot spectra components
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(objWave, objFlux, label='Object flux no nebular component')
        ax.plot(Input_Wavelength, Input_Flux, label='Starlight input spectrum', color='tab:green', linestyle=':')
        ax.plot(Input_Wavelength, stellar_flux, label='Stellar fit', color='tab:green', linestyle=':')

        for idx in np.arange(iniPoints.size):
            ax.axvspan(iniPoints[idx], endPoints[idx], alpha=0.25, color='tab:orange')

        ax.scatter(fit_output['ClippedPixels'][0], fit_output['ClippedPixels'][1], color='tab:purple', label='Clipped pixels')
        ax.scatter(fit_output['FlagPixels'][0], fit_output['FlagPixels'][1], color='tab:red', label='FlagPixels pixels')

        ax.update(labelsDict)
        ax.legend()
        ax.set_yscale('log')

        if outputAddress is None:
            plt.tight_layout()
            plt.show()
        else:
            plt.savefig(outputAddress, bbox_inches='tight')

        # Clear the image
        plt.close(fig)

        return

    def gen_colorList(self, vmin=0.0, vmax=1.0, color_palette=None):
        self.colorNorm = colors.Normalize(vmin, vmax)
        self.cmap = cm.get_cmap(name=color_palette)

    def get_color(self, idx):
        return self.cmap(self.colorNorm(idx))