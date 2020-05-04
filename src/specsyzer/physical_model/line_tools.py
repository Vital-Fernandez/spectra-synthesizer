import numpy as np
import pandas as pd
import astropy.units as au
from astropy.modeling.polynomial import Polynomial1D
from specutils.manipulation import noise_region_uncertainty
from specutils.fitting import find_lines_threshold, find_lines_derivative, fit_generic_continuum
from numpy import ndarray
from specutils import Spectrum1D, SpectralRegion
from matplotlib import pyplot as plt, rcParams
from scipy import stats, optimize, integrate


STANDARD_PLOT = {'figure.figsize': (20, 14), 'axes.titlesize': 14, 'axes.labelsize': 14, 'legend.fontsize': 12,
                 'xtick.labelsize': 12, 'ytick.labelsize': 12}


# Gaussian curve
def gaussFunc(ind_params, a, mu, sigma):
    x, z = ind_params
    return a * np.exp(-((x - mu) * (x - mu)) / (2 * (sigma * sigma))) + z


# Algorithm to combine line and features mask
def generate_object_mask(lines_DF, wavelength, line_labels):
    # TODO This will not work for a redshifted lines log
    idcs_lineMasks = lines_DF.index.isin(line_labels)
    idcs_spectrumMasks = ~lines_DF.index.isin(line_labels)

    # Matrix mask for integrating the emission lines
    n_lineMasks = idcs_lineMasks.sum()
    boolean_matrix = np.zeros((n_lineMasks, wavelength.size), dtype=bool)

    # Total mask for valid regions in the spectrum
    n_objMasks = idcs_spectrumMasks.sum()
    int_mask = np.ones(wavelength.size, dtype=bool)
    object_mask = np.ones(wavelength.size, dtype=bool)

    # Loop through the emission lines
    wmin, wmax = lines_DF['w3'].loc[idcs_lineMasks].values, lines_DF['w4'].loc[idcs_lineMasks].values
    idxMin, idxMax = np.searchsorted(wavelength, [wmin, wmax])
    for i in range(n_lineMasks):
        idx_currentMask = (wavelength >= wavelength[idxMin[i]]) & (wavelength <= wavelength[idxMax[i]])
        boolean_matrix[i, :] = idx_currentMask
        int_mask = int_mask & ~idx_currentMask

    # Loop through the object masks
    wmin, wmax = lines_DF['w3'].loc[idcs_spectrumMasks].values, lines_DF['w4'].loc[idcs_spectrumMasks].values
    idxMin, idxMax = np.searchsorted(wavelength, [wmin, wmax])
    for i in range(n_objMasks):
        idx_currentMask = (wavelength >= wavelength[idxMin[i]]) & (wavelength <= wavelength[idxMax[i]])
        int_mask = int_mask & ~idx_currentMask
        object_mask = object_mask & ~idx_currentMask

    return boolean_matrix


class LineMeasurer:
    """Class to to measure emission line fluxes and fit them as gaussian curves"""

    wave: ndarray = None
    flux: ndarray = None
    redshift: float = None
    pixelWidth: float = None
    peakWave: float = None
    peakInt: float = None
    lineIntgFlux: float = None
    lineIntgErr: float = None
    eqw: float = None
    eqwErr: float = None
    lineGaussFlux: float = None
    lineGaussErr: float = None
    n_continuum: float = None
    m_continuum: float = None
    std_continuum: float = None
    p1: ndarray = None
    p1_Err: ndarray = None

    paramsConversion = {'lineIntgFlux': 'intg_flux',
                        'lineIntgErr': 'intg_err',
                        'lineGaussFlux': 'gauss_flux',
                        'lineGaussErr': 'gauss_err',
                        'm_continuum': 'm_continuum',
                        'n_continuum': 'n_continuum',
                        'std_continuum': 'std_continuum',
                        'eqw': 'eqw',
                        'eqwErr': 'eqw_err'}

    def __init__(self, input_wave=None, input_flux=None, redshift=None):

        if redshift is not None:
            self.wave = input_wave / (1 + redshift)
            self.redshift = redshift
        else:
            self.wave = input_wave

        self.flux = input_flux

        return

    def define_masks(self, masks_array):

        area_indcs = np.searchsorted(self.wave, masks_array)
        idcsLines = ((self.wave[area_indcs[2]] <= self.wave[None]) & (
                self.wave[None] <= self.wave[area_indcs[3]])).squeeze()
        idcsContinua = (
                ((self.wave[area_indcs[0]] <= self.wave[None]) & (self.wave[None] <= self.wave[area_indcs[1]])) |
                ((self.wave[area_indcs[4]] <= self.wave[None]) & (
                        self.wave[None] <= self.wave[area_indcs[5]]))).squeeze()

        return idcsLines, idcsContinua

    def line_properties(self, idcs_line, idcs_continua, bootstrap_size=100):

        # TODO add mechanic to identify absorption features

        # Get regions data
        lineWave, lineFlux = self.wave[idcs_line], self.flux[idcs_line]
        continuaWave, continuaFlux = self.wave[idcs_continua], self.flux[idcs_continua]

        # Linear continuum linear fit
        self.m_continuum, self.n_continuum, r_value, p_value, std_err = stats.linregress(continuaWave, continuaFlux)
        continuaFit = continuaWave * self.m_continuum + self.n_continuum
        lineContinuumFit = lineWave * self.m_continuum + self.n_continuum

        # Line Characteristics
        peakIdx = np.argmax(lineFlux)
        self.peakWave, self.peakInt = lineWave[peakIdx], lineFlux[peakIdx]
        self.pixelWidth = np.diff(lineWave).mean()
        self.std_continuum = np.std(continuaFlux - continuaFit)

        # Monte Carlo to measure line flux and uncertainty
        normalNoise = np.random.normal(0.0, self.std_continuum, (bootstrap_size, lineWave.size))
        lineFluxMatrix = lineFlux + normalNoise
        areasArray = (lineFluxMatrix.sum(axis=1) - lineContinuumFit.sum()) * self.pixelWidth
        self.lineIntgFlux, self.lineIntgErr = areasArray.mean(), areasArray.std()

        # Equivalent width computation
        lineContinuumMatrix = lineContinuumFit + normalNoise
        eqwMatrix = areasArray / lineContinuumMatrix.mean(axis=1)
        self.eqw, self.eqwErr = eqwMatrix.mean(), eqwMatrix.std()

        return

    def line_fitting(self, idcs_line, idcs_continua, bootstrap_size=1000, verbose=False):

        # Get regions data
        lineWave, lineFlux = self.wave[idcs_line], self.flux[idcs_line]
        continuaWave, continuaFlux = self.wave[idcs_continua], self.flux[idcs_continua]

        # Linear continuum linear fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(continuaWave, continuaFlux)
        continuaFit = continuaWave * slope + intercept
        lineContinuumFit = lineWave * slope + intercept

        # Line Characteristics
        peakIdx = np.argmax(lineFlux)
        self.peakWave, self.peakInt = lineWave[peakIdx], lineFlux[peakIdx]
        self.pixelWidth = np.diff(lineWave).mean()
        self.std_continuum = np.std(continuaFlux - continuaFit)

        # Monte Carlo to fit gaussian curve
        normalNoise = np.random.normal(0.0, self.std_continuum, (bootstrap_size, lineWave.size))
        lineFluxMatrix = lineFlux + normalNoise
        p0_array = np.array([self.peakInt, self.peakWave, 1])

        # Bounds system
        paramBounds = ((0, self.peakWave-5, 0),
                       (self.peakInt*2, self.peakWave+5, 5))

        # paramBounds = ((0, self.peakInt*2),
        #                (self.peakWave-5, self.peakWave+5),
        #                (0, 5))

        # Run the fitting
        try:
            # TODO Add logic for all possible combinations
            # TODO Add logic for very small lines
            p1_matrix = np.empty((bootstrap_size, 3))
            for i in np.arange(bootstrap_size):
                p1_matrix[i], pcov = optimize.curve_fit(gaussFunc,
                                                        (lineWave, lineContinuumFit),
                                                        lineFluxMatrix[i],
                                                        p0=p0_array,
                                                        ftol=0.5,
                                                        xtol=0.5,
                                                        # bounds=paramBounds,
                                                        maxfev=1200)

            self.p1, self.p1_Err = p1_matrix.mean(axis=0), p1_matrix.std(axis=0)
            lineArea = np.sqrt(2 * np.pi * p1_matrix[:, 2] * p1_matrix[:, 2]) * p1_matrix[:, 0]
            self.lineGaussFlux, self.lineGaussErr = lineArea.mean(), lineArea.std()

        except:
            self.p1, self.p1_Err = np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])
            self.lineGaussFlux, self.lineGaussErr = np.nan, np.nan

        return

    def continuum_remover(self, noiseWaveLim, intLineThreshold=3, u_spec=(au.AA, au.erg/au.s/au.cm**2/au.AA), order=4):
        assert self.wave[0] < noiseWaveLim[0] and noiseWaveLim[1] < self.wave[-1]

        # Identify high flux regions # TODO Expand limit to absorption lines
        idcs_noiseRegion = (noiseWaveLim[0] <= self.wave) & (self.wave <= noiseWaveLim[1])
        noise_mean, noise_std = self.flux[idcs_noiseRegion].mean(), self.flux[idcs_noiseRegion].std()
        idcsLineMask = self.flux < intLineThreshold * (noise_mean + noise_std)

        print('noise', noise_mean)

        # Fit the continuum and remove it from the observation
        wave_masked, flux_masked = self.wave[idcsLineMask], self.flux[idcsLineMask]
        spectrum_masked = Spectrum1D(flux_masked * u_spec[1], wave_masked * u_spec[0])
        g1_fit = fit_generic_continuum(spectrum_masked, model=Polynomial1D(order))
        continuum_fit = g1_fit(self.wave * u_spec[0])
        spectrum_noContinuum = Spectrum1D(self.flux * u_spec[1] - continuum_fit, self.wave * u_spec[0])

        return spectrum_noContinuum.flux.value

    def line_finder(self, input_flux, noiseWaveLim, intLineThreshold=3, u_spec=(au.AA, au.erg/au.s/au.cm**2/au.AA),
                    verbose=False):

        assert noiseWaveLim[0] > self.wave[0] or noiseWaveLim[1] < self.wave[-1]

        # Establish noise values
        idcs_noiseRegion = (noiseWaveLim[0] <= self.wave) & (self.wave <= noiseWaveLim[1])
        noise_region = SpectralRegion(noiseWaveLim[0] * u_spec[0], noiseWaveLim[1] * u_spec[0])
        flux_threshold = intLineThreshold * input_flux[idcs_noiseRegion].std()

        input_spectrum = Spectrum1D(input_flux * u_spec[1], self.wave * u_spec[0])
        input_spectrum = noise_region_uncertainty(input_spectrum, noise_region)
        linesTable = find_lines_derivative(input_spectrum, flux_threshold)

        if verbose:
            print(linesTable)

        return linesTable

    def match_lines(self, obsLineTable, theoLineDF, lineType='emission', tol=5):

        # Query the lines from the astropy finder tables # TODO Expand technique for absorption lines
        idcsLineType = obsLineTable['line_type'] == lineType
        idcsLinePeak = np.array(obsLineTable[idcsLineType]['line_center_index'])
        waveObs = self.wave[idcsLinePeak]

        # Theoretical wave values
        waveTheory = theoLineDF.wavelength.values

        # Match the lines with the theoretical emission
        tolerance = np.diff(self.wave).mean() * tol
        theoLineDF['observation'] = 'not detected'
        unidentifiedLine = dict.fromkeys(theoLineDF.columns.values, np.nan)
        for i in np.arange(waveObs.size):
            idx_array = np.where(np.isclose(a=waveTheory, b=waveObs[i], atol=tolerance))
            if len(idx_array[0]) == 0:
                unknownLineLabel = 'xy_{:.0f}A'.format(np.round(waveObs[i]))
                if unknownLineLabel not in theoLineDF.index: # Scheme to avoid repeated lines
                    newRow = unidentifiedLine.copy()
                    newRow.update({'wavelength': waveObs[i], 'w3': waveObs[i]-5, 'w4': waveObs[i]+5,
                                   'observation': 'not identified'})
                    theoLineDF.loc[unknownLineLabel] = newRow

            else:
                row_index = theoLineDF.index[theoLineDF.wavelength == waveTheory[idx_array[0][0]]]
                theoLineDF.loc[row_index, 'observation'] = 'detected'
                # print(i, theoLineDF.loc[row_index, 'Observation'])

        # Sort by wavelength
        theoLineDF.sort_values('wavelength', inplace=True)

        return theoLineDF

    def results_to_database(self, lineLabel, linesDF):

        # Loop through the parameters
        for param in self.paramsConversion:
            linesDF.loc[lineLabel, self.paramsConversion[param]] = self.__getattribute__(param)

        # Gaussian fit parameters
        for idx, val in enumerate(('amp', 'mu', 'sigma')):
            linesDF.loc[lineLabel, val] = self.p1[idx]
            linesDF.loc[lineLabel, val + '_err'] = self.p1_Err[idx]

        return

    def load_lineslog(self, file_address):
        return pd.read_csv(file_address, delim_whitespace=True, header=0, index_col=0)

    def save_lineslog(self, linesDF, file_address):

        with open(file_address, 'wb') as output_file:
            string_DF = linesDF.to_string()
            output_file.write(string_DF.encode('UTF-8'))

        return

    def spectrum_components(self, continuumFlux=None,  obsLinesTable=None, matchedLinesDF=None, noise_region=None,
                            plotConf={}):


        # Plot Configuration
        defaultConf = STANDARD_PLOT.copy()
        defaultConf.update(plotConf)
        rcParams.update(defaultConf)
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot the spectrum
        ax.step(self.wave, self.flux, label='Observed spectrum')

        # Plot the continuum if available
        if continuumFlux is not None:
            ax.step(self.wave, continuumFlux, label='Continuum')

        # Plot astropy detected lines if available
        if obsLinesTable is not None:
            idcs_emission = obsLinesTable['line_type'] == 'emission'
            idcs_linePeaks = np.array(obsLinesTable[idcs_emission]['line_center_index'])
            ax.scatter(self.wave[idcs_linePeaks], self.flux[idcs_linePeaks], label='Detected lines', facecolors='none',
                       edgecolors='tab:purple')

        if matchedLinesDF is not None:
            idcs_foundLines = (matchedLinesDF.observation.isin(('detected', 'not identified'))) &\
                              (matchedLinesDF.wavelength >= self.wave[0]) &\
                              (matchedLinesDF.wavelength <= self.wave[-1])
            lineLatexLabel, lineWave = matchedLinesDF.loc[idcs_foundLines].latexLabel.values, matchedLinesDF.loc[
                idcs_foundLines].wavelength.values
            w3, w4 = matchedLinesDF.loc[idcs_foundLines].w3.values, matchedLinesDF.loc[idcs_foundLines].w4.values
            observation = matchedLinesDF.loc[idcs_foundLines].observation.values

            for i in np.arange(lineLatexLabel.size):
                if observation[i] == 'detected':
                    color_area = 'tab:red' if observation[i] == 'not identified' else 'tab:green'
                    ax.axvspan(w3[i], w4[i], alpha=0.25, color=color_area)
                    ax.text(lineWave[i], 0, lineLatexLabel[i], rotation=270)

            # for i in np.arange(lineLatexLabel.size):
            #     color_area = 'tab:red' if observation[i] == 'not identified' else 'tab:green'
            #     ax.axvspan(w3[i], w4[i], alpha=0.25, color=color_area)
            #     ax.text(lineWave[i], 0, lineLatexLabel[i], rotation=270)

        if noise_region is not None:
            ax.axvspan(noise_region[0], noise_region[1], alpha=0.15, color='tab:cyan', label='Noise region')

        ax.update({'xlabel': r'Wavelength $(\AA)$', 'ylabel': r'Flux $(erg\,cm^{-2} s^{-1} \AA^{-1})$'})
        ax.legend()
        plt.tight_layout()
        plt.show()

        return

    def detected_lines(self, matchedLinesDF, limitPeak=5, plotConf={}, ncols=10, nrows=4):

        # Plot data
        indcsLines = matchedLinesDF.observation.isin(['detected'])
        lineLabels = matchedLinesDF.loc[indcsLines].index.values
        lineLatex, wavelengths = matchedLinesDF.loc[indcsLines, 'latexLabel'], matchedLinesDF.loc[indcsLines, 'wavelength']
        waveLimits = matchedLinesDF.loc[indcsLines, 'w1':'w6'].values
        observation = matchedLinesDF.loc[indcsLines].observation

        # Compute plot grid size
        if 'figure.figsize' not in plotConf:
            nrows = int(np.ceil(lineLatex.size / ncols))
            plotConf['figure.figsize'] = (nrows * 4, 14)

        # Plot format
        defaultConf = STANDARD_PLOT.copy()
        defaultConf.update(plotConf)
        rcParams.update(defaultConf)

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
        axesList = ax.flatten()

        # Generate plot
        for i in np.arange(lineLatex.size):

            w1, w2, w3, w4, w5, w6 = waveLimits[i]

            idcsContinuumLeft = (w1 <= self.wave) & (self.wave <= w2)
            idcsContinuumRight = (w5 <= self.wave) & (self.wave <= w6)
            idcsLinePeak = (wavelengths[i] - limitPeak <= self.wave) & (self.wave <= wavelengths[i] + limitPeak)
            idcsLinePlot = (w1 <= self.wave) & (self.wave <= w6)
            idcsLineArea = (w3 <= self.wave) & (self.wave <= w4)

            waveCentral, fluxCentral = self.wave[idcsLineArea], self.flux[idcsLineArea]
            waveLine, fluxLine = self.wave[idcsLinePlot], self.flux[idcsLinePlot]
            wavePeak, fluxPeak = self.wave[idcsLinePeak], self.flux[idcsLinePeak]

            axesList[i].step(waveLine, fluxLine)
            axesList[i].fill_between(waveCentral, 0, fluxCentral, step="pre", alpha=0.4)
            axesList[i].fill_between(self.wave[idcsContinuumLeft], 0, self.flux[idcsContinuumLeft], facecolor='tab:orange',
                                     step="pre", alpha=0.2)
            axesList[i].fill_between(self.wave[idcsContinuumRight], 0, self.flux[idcsContinuumRight], facecolor='tab:orange',
                                     step="pre", alpha=0.2)
            idxPeakFlux = np.argmax(fluxPeak)

            # Plot format
            axesList[i].yaxis.set_major_locator(plt.NullLocator())
            axesList[i].xaxis.set_major_locator(plt.NullLocator())
            axesList[i].update({'title': f'{lineLatex[i]}'})
            axesList[i].set_ylim(ymin=np.min(fluxLine)/5, ymax=fluxPeak[idxPeakFlux] * 1.25)

            # # Gaussian curve plot
            p1 = matchedLinesDF.loc[lineLabels[i], 'amp':'sigma'].values
            m, n = matchedLinesDF.loc[lineLabels[i], 'm_continuum'], matchedLinesDF.loc[lineLabels[i], 'n_continuum']
            if (p1[0] is not np.nan) and (p1[0] is not None):
                wave_array = np.linspace(waveLine[0], waveLine[-1], 1000)
                cont_array = m * wave_array + n
                flux_array = gaussFunc((wave_array, cont_array), p1[0], p1[1], p1[2])
                axesList[i].plot(wave_array, cont_array, ':', color='tab:orange')
                axesList[i].plot(wave_array, flux_array, ':', color='tab:red')

        plt.tight_layout()
        plt.show()

        return


if __name__ == '__main__':
    # Generate fake data
    wave = np.linspace(4950, 5050)
    m, n, noise = 0.0, 2.0, np.random.normal(0, 0.05, wave.size)
    flux_cont = (m * wave + n) + noise
    ampTrue, muTrue, sigmaTrue = 10, 5007, 2.3
    flux_gauss = gaussFunc((wave, flux_cont), ampTrue, muTrue, sigmaTrue)
    wave_regions = np.array([4960, 4980, 4996, 5015, 5030, 5045])
    areaTrue = np.sqrt(2 * np.pi * sigmaTrue ** 2) * ampTrue

    lm = LineMeasurer()

    # Load observed spectrum
    lm.load_spectrum(wave, flux_gauss)

    # Declare regions data
    idcsLines, idcsContinua = lm.define_masks(wave_regions)

    # Identify line regions
    lm.line_properties(idcsLines, idcsContinua, bootstrap_size=1000)

    # Fit gaussian profile
    lm.line_fitting(idcsLines, idcsContinua, bootstrap_size=1000)

    # Comparing flux integration techniques
    lineWave, lineFlux = lm.wave[idcsLines], lm.flux[idcsLines]
    continuaWave, continuaFlux = lm.wave[idcsContinua], lm.flux[idcsContinua]
    lineContinuumFit = lineWave * lm.m_continuum + lm.n_continuum
    areaSimps = integrate.simps(lineFlux, lineWave) - integrate.simps(lineContinuumFit, lineWave)
    areaTrapz = integrate.trapz(lineFlux, lineWave) - integrate.trapz(lineContinuumFit, lineWave)
    areaIntgPixel = (lm.flux[idcsLines].sum() - lineContinuumFit.sum()) * lm.pixelWidth
    resampleWaveLine = np.linspace(lineWave[0] - 10, lineWave[-1] + 10, 100)
    resampleFluxCont = resampleWaveLine * lm.m_continuum + lm.n_continuum
    gaussianCurve = gaussFunc((resampleWaveLine, resampleFluxCont), *lm.p1)
    areaGauss = (gaussianCurve.sum() - resampleFluxCont.sum()) * np.diff(resampleWaveLine).mean()

    # Print the results
    print(f'True area : {areaTrue}')
    print(f'Simpsons rule: {areaSimps}')
    print(f'Trapezoid rule: {areaTrapz}')
    print(f'Pixel intgr: {areaIntgPixel}')
    print(f'Pixel intgr MC: {lm.lineIntgFlux} +/- {lm.lineIntgErr}')
    print(f'Gauss intgr: {areaGauss}')
    print(f'Gauss intgr MC: {lm.lineGaussFlux} +/- {lm.lineGaussErr}')
    print(f'True amp = {ampTrue}, mu = {muTrue}, sigma = {sigmaTrue}')
    print(f'Fit amp = {lm.p1[0]:2f} +/- {lm.p1_Err[0]:2f}, mu = {lm.p1[1]:2f} +/- {lm.p1_Err[1]:2f}, '
          f'sigma = {lm.p1[2]:2f} +/- {lm.p1_Err[2]:2f}')
    print(f'Eqw MC {lm.eqw} +/- {lm.eqwErr}')

    fig, ax = plt.subplots()
    ax.plot(wave, flux_gauss, label='Observed line')
    ax.scatter(continuaWave, continuaFlux, label='Continuum regions')
    ax.plot(lineWave, lineContinuumFit, label='Observed line', linestyle=':')
    ax.plot(resampleWaveLine, gaussianCurve, label='Gaussian fit', linestyle=':')
    ax.legend()
    ax.update({'xlabel': 'Flux', 'ylabel': 'Wavelength', 'title': 'Gaussian fitting'})
    plt.show()
