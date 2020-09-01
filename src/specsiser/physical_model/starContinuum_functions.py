import numpy as np
import pandas as pd
from collections import OrderedDict
from scipy.signal.signaltools import convolve2d
from scipy.interpolate.interpolate import interp1d
from scipy.optimize import nnls
from shutil import copyfile
from os import chdir
from subprocess import Popen, PIPE, STDOUT

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


class SspFitter:

    def __init__(self):

        self.ssp_conf_dict = OrderedDict()

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


class StarlightWrapper:

    def __init__(self):

        return

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

        if Hbeta_sigma >= 0:
            sigma_line = O3_5007_sigma / 5007.0 * c_SI
        elif O3_5007_sigma >= 0:
            sigma_line = Hbeta_sigma / 4861.0 * c_SI
        else:
            sigma_line = 100

        return sigma_line

    def generate_starlight_files(self, starlight_Folder, objName, X, Y, regionsDF, v_vector=None):

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
            v_vector = ['FXK', '0.0', "10.0"]

        GridLines = []
        GridLines.append("1")  # "[Number of fits to run]"])
        GridLines.append(Default_BasesFolder)  # "[base_dir]"])
        GridLines.append(Default_InputFolder)  # "[obs_dir]"])
        GridLines.append(Default_MaskFolder)  # "[mask_dir]"])
        GridLines.append(Default_OutputFoler)  # "[out_dir]"])
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

        files_row = [Sl_Input_Filename, configFile, BaseDataFile, maskFileName, Redlaw, v0_start, vd_start,
                    Sl_Output_Filename]

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
        v0_min = float(StarlightOutput[v0_min_Line].split()[0])
        vd_min = float(StarlightOutput[vd_min_Line].split()[0])
        Av_min = float(StarlightOutput[Av_min_Line].split()[0])

        # Signal to noise configuration
        SignalToNoise_lowWave = float(StarlightOutput[SignalToNoise_Line + 1].split()[0])
        SignalToNoise_upWave = float(StarlightOutput[SignalToNoise_Line + 2].split()[0])
        SignalToNoise_magnitudeWave = float(StarlightOutput[SignalToNoise_Line + 3].split()[0])

        # Flux normailzation parameters
        l_norm = float(StarlightOutput[l_norm_Line].split()[0])
        llow_norm = float(StarlightOutput[llow_norm_Line].split()[0])
        lupp_norm = float(StarlightOutput[lupp_norm_Line].split()[0])
        FluxNorm = float(StarlightOutput[NormFlux_Line].split()[0])

        # Spectra pixels location
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

        return Input_Wavelength, Input_Flux, Output_Flux, Parameters

    def starlight_launcher(self, gridFile, executable_folder='/home/vital/'):

        chdir(executable_folder)
        Command = './StarlightChains_v04.exe < ' + str(gridFile)
        print(f'Launching: {Command}')
        p = Popen(Command, shell=True, stdout=PIPE, stderr=STDOUT)

        for line in p.stdout.readlines():
            print(line,)

        retval = p.wait()

        return




