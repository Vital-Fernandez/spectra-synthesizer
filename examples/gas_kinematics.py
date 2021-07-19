import pymc3 as pm
import theano.tensor as tt
import numpy as np
import pandas as pd
from pathlib import Path
from src.specsiser.physical_model.line_tools import EmissionFluxModel EmissionFitting, gauss_func
from matplotlib import pyplot as plt, rcParams


def mixture_density_mult(w, mu, sd, x):
    logp = tt.log(w) + pm.Normal.dist(mu, sd).logp(x)
    return tt.sum(tt.exp(logp), axis=1)


# Line treatment object
lm = EmissionFitting()

# Declare data
data_folder, data_file = Path('C:/Users/Vital/OneDrive/Desktop/'), 'test_spec2.txt'
file_to_open = data_folder / data_file
linesLogAdress = data_folder / data_file.replace('.txt', '_linesLog.txt')
linesFile = Path('D:/Pycharm Projects/spectra-synthesizer/src/specsiser/literature_data/lines_data.xlsx') # TODO change to open format to avoid new dependency
linesDF = lm.load_lineslog(linesLogAdress)

# Load spectrum
factor = 1e17
redshift = 1.0046
wave, flux = np.loadtxt(file_to_open, unpack=True)
wave, flux = wave/redshift, flux * 1e-20 * factor
# cropLimits, noiseLimits = (9032, 9200), (9080, 9120)
# cropLimits, noiseLimits = (9032, 9300), (9090, 9200)
cropLimits, noiseLimits = (7000, 9300), (8888, 9000)
cropLimits, noiseLimits = (4600, 7800), (5600, 5700)

# Crop the spectrum
idx = (cropLimits[0] <= wave) & (wave <= cropLimits[1])
wave, flux = wave[idx], flux[idx]
lm.wave, lm.flux = wave, flux

# Remove the continuum
flux_noContinuum = lm.continuum_remover(noiseWaveLim=noiseLimits)
continuumFlux = lm.flux - flux_noContinuum

# Plot the spectrum
lm.plot_spectrum_components(continuumFlux, matchedLinesDF=linesDF, noise_region=noiseLimits)

# Line data:
idcsDF = (linesDF.wavelength >= lm.wave[0]) & (linesDF.wavelength <= lm.wave[-1])
lineLabel = linesDF.loc[idcsDF].index.values
lineWaves = linesDF.loc[idcsDF].wavelength.values
lineRanges = linesDF.loc[idcsDF, 'w1':'w6'].values
print(f'- Fitting {lineLabel.size} lines')

indexSpecEmission = np.zeros(lm.wave.size, dtype=bool)
for i in np.arange(lineLabel.size):
    idxLine = (lineRanges[i][2] <= lm.wave) & (lm.wave <= lineRanges[i][3])
    indexSpecEmission += idxLine

# Define input spectrum
specWave = lm.wave[indexSpecEmission][:,None]
specFlux = lm.flux[indexSpecEmission]
specContinuum = continuumFlux[indexSpecEmission]
pixelErr = linesDF.loc[lineLabel, 'std_cont']

with pm.Model():

    # Model Priors
    amp_array = pm.HalfNormal('amp_array', 10., shape=lineLabel.size)
    my_array = pm.Normal('mu_array', lineWaves, 2., shape=lineLabel.size)
    sigma_array = pm.HalfCauchy('sigma_array', 5., shape=lineLabel.size)
    pixelNoise = pm.HalfCauchy('pixelNoise', 5.)

    # Theoretical line profiles
    theoFlux_i = mixture_density_mult(amp_array, my_array, sigma_array, specWave) + specContinuum

    # Model likelihood
    pm.Normal('emission_Y', theoFlux_i, pixelNoise, observed=specFlux)

    # Run sampler
    trace = pm.sample(draws=1000, tune=1000, chains=2, cores=1)

# Reconstruct from traces the results
amp_trace, mu_trace, sigma_trace = trace['amp_array'], trace['mu_array'], trace['sigma_array']
amp_mean, mu_mean, sigma_mean = amp_trace.mean(axis=0), mu_trace.mean(axis=0), sigma_trace.mean(axis=0)
print(trace['amp_array'].mean(axis=0))

wave_resample = np.linspace(lm.wave[0], lm.wave[-1], lm.wave.size * 20)
cont_resample = np.interp(wave_resample, lm.wave, continuumFlux)
hmc_curve = mixture_density_mult(amp_mean, mu_mean, sigma_mean, wave_resample[:, None]).eval()

# Plot the results
fig, ax = plt.subplots()
ax.step(lm.wave, lm.flux, label='Object spectrum')
ax.scatter(specWave, specFlux, label='Input spectrum', color='tab:orange')
ax.plot(wave_resample, hmc_curve + cont_resample, label='Fitting result',  color='tab:red')
ax.legend()
ax.update({'xlabel':r'Wavelength $(\AA)$', 'ylabel':r'Flux $(erg\,cm^{-2} s^{-1} \AA^{-1}) \cdot 10^{17}$',
           'title':'Gaussian fitting'})
plt.show()

