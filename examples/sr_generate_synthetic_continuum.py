import numpy as np
import pandas as pd
from exoplanet.interp import RegularGridInterpolator
from pathlib import Path
from src.specsiser.physical_model.starContinuum_functions import SSPsynthesizer
from src.specsiser.physical_model.gasContinuum_functions import NebularContinua
from src.specsiser.inference_model import displaySimulationData
import theano.tensor as tt
import pymc3 as pm
import arviz as az


def slice_template_grid(db, idcs_db, wave_array, flux_array):

    age_range = np.unique(db.loc[idcs_db, 'age_yr'].values)
    z_range = np.unique(db.loc[idcs_db, 'z_star'].values)

    flux_matrix = np.empty((z_range.size, age_range.size, wave_array.size))
    flux_matrix[:, :, :] = np.nan
    for i_z, z in enumerate(z_range):
        for j_age, age in enumerate(age_range):
            i_flux = (db.z_star == z) & (db.age_yr == age)
            flux_matrix[i_z, j_age, :] = flux_array[i_flux]

    return z_range, age_range, flux_matrix


data_folder = Path('D:\Dropbox\Astrophysics\Papers\gtc_greenpeas\data')

starCalc = SSPsynthesizer()
nebCalc = NebularContinua()

# Stellar parameter
cropWaves = None
resamInter = 1
normWaves = (5100, 5150)
age_max_grid = 1.5e7
z_min_grid = 0.005

# Nebular parameters
temp_range = np.linspace(5000, 30000, 126)
HeI_range = np.linspace(0.05, 0.20, 51)
HeII_HI = 0.001

# Generate stellar data grids if they are not available:
if not Path.is_file(data_folder/'flux_stellar_bases.npy'):
    basesFolder = Path("D:\Dropbox\Astrophysics\Papers\gtc_greenpeas\data\starlight\Bases")
    basesFile = Path("D:\Dropbox\Astrophysics\Papers\gtc_greenpeas\data\starlight\Dani_Bases_short.txt")
    basesDF, waveBases, fluxBases = starCalc.import_STARLIGHT_bases(basesFile, basesFolder, cropWaves, resamInter, normWaves)
    np.save(data_folder/'waves_stellar_bases', waveBases)
    np.save(data_folder/'flux_stellar_bases', fluxBases)
    basesDF.to_csv(data_folder/'basesDF')

# Generate Nebular data grids if they are not available:
if not Path.is_file(data_folder/'neb_gamma.npy'):
    neb_gamma = np.empty((temp_range.size, HeI_range.size, waveBases.size))
    neb_gamma[:, :, :] = np.nan
    for j, temp in enumerate(temp_range):
        for i, HeI_HI in enumerate(HeI_range):
            neb_gamma[j, i, :] = nebCalc.gamma_spectrum(waveBases, temp, HeI_HI, HeII_HI)
    np.save(data_folder/'gamma_nebular_cont', neb_gamma)

# Load the data
basesDF = pd.read_csv(data_folder/'basesDF', index_col=0)
waveBases = np.load(data_folder/'waves_bases.npy')
fluxBases = np.load(data_folder/'flux_bases.npy')
neb_gamma = np.load(data_folder/'neb_gamma.npy')

# Slice the data grid to the region of interest
idcs_bases = (basesDF.age_yr < age_max_grid) & (basesDF.z_star > z_min_grid)
z_range, age_range, fluxBases = slice_template_grid(basesDF, idcs_bases, waveBases, fluxBases)
logAge_range = np.log10(age_range)

# Create grid interpolators
stellarBases_interp = RegularGridInterpolator([z_range, logAge_range], fluxBases)
nebGamma_interp = RegularGridInterpolator([temp_range, HeI_range], neb_gamma)

# Generate synthetic continuum:
cont_params = dict(Te=12350.0,
                   HeI_HI=0.1055,
                   Halpha=500.0,
                   age=np.array([6.95]),
                   z=np.array([0.0265]),
                   w=np.array([1.25]))

# -- Nebular continuum
coord = np.stack(([cont_params['Te']], [cont_params['HeI_HI']]), axis=-1)
nebGamma_true = nebGamma_interp.evaluate(coord).eval()[0]
flux_neb_inter = nebCalc.zanstra_calibration(waveBases, cont_params['Te'], cont_params['Halpha'], nebGamma_true)

# -- Stellar continuum
n_pop = cont_params['age'].size
flux_stellar_inter = np.zeros(waveBases.size)
stellar_comp = np.zeros((n_pop, waveBases.size))
for i in range(n_pop):
    coord = np.stack(([cont_params['z'][i]], [cont_params['age'][i]]), axis=-1)
    flux_true_i = stellarBases_interp.evaluate(coord).eval()[0]
    stellar_comp[i, :] = flux_true_i
    flux_stellar_inter += flux_true_i * cont_params['w'][i]

# -- Normalize the continuum
flux_obj = flux_neb_inter + flux_stellar_inter
wave_obj, flux_obj_norm, normFlux_const = starCalc.treat_input_spectrum(waveBases, flux_obj, norm_waves=(5100, 5150))

# Fit the observation
pop_range = np.arange(n_pop)
nebular_comp, stellar_comp = True, True
specTensor = tt.zeros(waveBases.size)
err_pixel = 0.05 * np.ones(waveBases.size)

with pm.Model() as model:

    # Nebular component
    if nebular_comp:

        # Priors
        Te_prior = pm.Normal('temp', mu=cont_params['Te'], sigma=50)
        y_plus = pm.Normal('HeI_HI', mu=cont_params['HeI_HI'], sigma=0.005)

        # Synthetic continuum
        coord_temp = tt.stack([[Te_prior], [y_plus]], axis=-1)
        neb_gamma_t = nebGamma_interp.evaluate(coord_temp)[0]
        spec_tensor = nebCalc.zanstra_calibration_tt(wave_obj, Te_prior, cont_params['Halpha'], neb_gamma_t)

    else:
        spec_tensor = specTensor * 0


    if stellar_comp:

        # Priors
        z_prior = pm.Uniform('z_star', lower=np.min(z_range), upper=np.max(z_range), shape=n_pop)
        age_prior = pm.Uniform('age', lower=np.min(logAge_range), upper=np.max(logAge_range), shape=n_pop)
        w_prior = cont_params['w']

        # Synthetic continuum
        for i in pop_range:
            coord_i = tt.stack([[z_prior[i]], [age_prior[i]]], axis=-1)
            spec_i = stellarBases_interp.evaluate(coord_i)[0]
            spec_tensor += w_prior[i] * spec_i

    # Likelihood
    pm.Normal('continuum', spec_tensor/normFlux_const, err_pixel, observed=flux_obj_norm)

    # Check simulation statistics
    displaySimulationData(model)

    # Run sampler
    trace = pm.sample(draws=5000, tune=3000, chains=2, cores=1)

print('True values: ', cont_params)

print(az.summary(trace))
az.plot_trace(trace)
# plt.show()
# az.plot_posterior(trace)
# plt.show()
# az.plot_forest(trace)
# plt.show()
