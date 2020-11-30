import pymc3 as pm
import theano.tensor as tt
import numpy as np
import pandas as pd
import astropy.io.fits as astrofits
from pathlib import Path
from src.specsiser.physical_model.line_tools import EmissionFitting, gauss_func
from matplotlib import pyplot as plt, rcParams
from inference_model import displaySimulationData


def import_fits_data(file_address, frame_idx):

    # Open fits file
    with astrofits.open(data_folder / file_address) as hdul:
        data, header = hdul[frame_idx].data, hdul[frame_idx].header

    # Check instrument
    if 'INSTRUME' in header:
        if 'ISIS' in header['INSTRUME']:
            instrument = 'ISIS'

    # William Herschel Telescope ISIS instrument
    if instrument == 'ISIS':
        w_min = header['CRVAL1']
        dw = header['CD1_1']  # dw = 0.862936 INDEF (Wavelength interval per pixel)
        pixels = header['NAXIS1']  # nw = 3801 number of output pixels
        w_max = w_min + dw * pixels
        wave = np.linspace(w_min, w_max, pixels, endpoint=False)

    return wave, data, header


def mixture_density_mult(w, mu, sd, x):
    logp = tt.log(w) + pm.Normal.dist(mu, sd).logp(x)
    return tt.sum(tt.exp(logp), axis=1)


# Line treatment object
lm = EmissionFitting()

# Declare data
data_folder, data_file = Path('D:/Dropbox/Astrophysics/Data/WHT-Ricardo/'), 'COMBINED_blue.0001.fits'
file_to_open = data_folder / data_file
linesFile = Path('D:/Pycharm Projects/spectra-synthesizer/src/specsiser/literature_data/lines_data.xlsx') # TODO change to open format to avoid new dependency
linesDF = pd.read_excel(linesFile, sheet_name=0, header=0, index_col=0)

# Load spectrum
factor = 1e16
# redshift = 1.0046
# wave, flux = np.loadtxt(file_to_open, unpack=True)
# wave, flux = wave/redshift, flux * 1e-20 * factor
wave_blue, flux_blue, header_blue = import_fits_data(file_to_open, 0)
redshift = 1.1735
wave_blue = wave_blue/redshift  # TODO Add import spectrum function
cropLimits, noiseLimits = (3700, 3750), (3705, 3715)

# Crop the spectrum
idx = (cropLimits[0] <= wave_blue) & (wave_blue <= cropLimits[1])
wave_blue, flux_blue = wave_blue[idx], flux_blue[idx] * factor
lm.wave, lm.flux = wave_blue, flux_blue

# Remove the continuum
flux_noContinuum = lm.continuum_remover(noiseRegionLims=noiseLimits, order=1)
continuumFlux = lm.flux - flux_noContinuum

# Plot the spectrum
lm.plot_spectrum_components(continuumFlux, matchedLinesDF=linesDF, noise_region=noiseLimits)

# Define input spectrum
lineLabels = np.array(['O2_3726A', 'O2_3729A'])
lineWaves = np.array([3726.032, 3728.815])
specWave = lm.wave[:, None]
specFlux = lm.flux
specContinuum = continuumFlux

with pm.Model():

    # Model Priors
    amp_array = pm.HalfNormal('amp_array', 10., shape=lineLabels.size)
    my_array = pm.Normal('mu_array', lineWaves, 2., shape=lineLabels.size)
    sigma_array = pm.HalfCauchy('sigma_array', 5., shape=lineLabels.size)
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
ax.scatter(specWave, specFlux, label='Line', color='tab:orange')
ax.plot(wave_resample, hmc_curve + cont_resample, label='HMC fitting',  color='tab:red')
ax.legend()
ax.update({'xlabel':'Wavelength', 'ylabel':'Flux', 'title':'Gaussian fitting'})
plt.show()

