import pymc3
import theano.tensor as tt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize, integrate
from inference_model import displaySimulationData
from astropy.io import fits

gauss_coef = np.sqrt(2 * np.pi)

# Gaussian curves
def gaussFunc(ind_params, a, mu, sigma):
    x, z = ind_params
    return a * np.exp(-((x - mu) * (x - mu)) / (2 * (sigma * sigma))) + z


def gaussFunc_tt(x, z, a, mu, sigma):
    return a * np.exp(-((x - mu) * (x - mu)) / (2 * (sigma * sigma))) + z


def gaussFuncArea(ind_params, area, mu, sigma):
    x, z = ind_params
    return (area/(sigma*gauss_coef)) * np.exp(-((x - mu) * (x - mu)) / (2 * (sigma * sigma))) + z


def gaussFuncArea_tt(x, z, area, mu, sigma):
    return (area/(sigma*gauss_coef)) * np.exp(-((x - mu) * (x - mu)) / (2 * (sigma * sigma))) + z

# Data location
log_file = 'E:/Dropbox/Astrophysics/Data/WHT_observations/bayesianModel/8/8_linesLog.txt'
spec_file = 'E:/Dropbox/Astrophysics/Data/WHT_observations/objects/8/8_WHT.fits'

# Get data
data_array, Header_0 = fits.getdata(spec_file, header=True)
obj_df = pd.read_csv(log_file, delim_whitespace=True, header=0, index_col=0)

fluxHbeta = obj_df.loc['H1_4861A', 'flux_intg']

listLabels = np.array(['O3_4959A', 'O3_5007A', 'He1_5876A', 'H1_6563A'])
gaussParams = np.ones((listLabels.size, 3))
line_dict, cont_dict, lineCont_dict, std_dict = {}, {}, {}, {}

for i in range(listLabels.size):

    lineLabel = listLabels[i]
    wave, flux = data_array['Wave'], data_array['Int']/fluxHbeta
    wave_loc = obj_df.loc[lineLabel, 'w1':'w6'].values

    # Identify line regions
    area_indcs = np.searchsorted(wave, wave_loc)
    idcsLines = ((wave[area_indcs[2]] <= wave[None]) & (wave[None] <= wave[area_indcs[3]])).squeeze()
    idcsContinua = (((wave[area_indcs[0]] <= wave[None]) & (wave[None] <= wave[area_indcs[1]])) | (
            (wave[area_indcs[4]] <= wave[None]) & (wave[None] <= wave[area_indcs[5]]))).squeeze()

    # Get regions data
    lineWave, lineFlux = wave[idcsLines], flux[idcsLines]
    continuaWave, continuaFlux = wave[idcsContinua], flux[idcsContinua]

    # Linear region fitting
    slope, intercept, r_value, p_value, std_err = stats.linregress(continuaWave, continuaFlux)
    continuaFit = continuaWave * slope + intercept
    std_continuum = np.std(continuaFlux - continuaFit)
    lineContinuumFit = lineWave * slope + intercept
    continuumInt = lineContinuumFit.sum()
    centerWave = lineWave[np.argmax(lineFlux)]
    centerContInt = centerWave * slope + intercept

    # Standard fitting
    p0 = (lineFlux.max() * 1.5 * gauss_coef, lineWave.mean(), 1.0)
    fitParams, cov = optimize.curve_fit(gaussFuncArea, (lineWave, lineContinuumFit), lineFlux, p0=p0)

    # Store the data
    line_dict[lineLabel] = np.vstack((lineWave, lineFlux))
    cont_dict[lineLabel] = np.vstack((continuaWave, continuaFlux))
    lineCont_dict[lineLabel] = np.vstack((lineWave, lineContinuumFit))
    std_dict[lineLabel] = std_continuum
    gaussParams[i, :] = fitParams

print(gaussParams)

# resampleWaveLine = np.linspace(lineWave[0]-10, lineWave[-1]+10, 100)
# resampleWaveCont = resampleWaveLine * slope + intercept
# gaussianCurve = gaussFunc((resampleWaveLine, resampleWaveCont), *fitParams)

# Stack line data arrays
wave_vector = np.array([])
flux_vector = np.array([])
cont_vector = np.array([])
err_vector = np.array([])
idxLine_dict = dict.fromkeys(listLabels, np.array([0, 0]))

for i in range(listLabels.size):

    lineLabel = listLabels[i]
    idxLine_dict[lineLabel][0] = wave_vector.size
    wave_vector = np.hstack((wave_vector, line_dict[lineLabel][0, :]))
    flux_vector = np.hstack((flux_vector, line_dict[lineLabel][1, :]))
    cont_vector = np.hstack((cont_vector, lineCont_dict[lineLabel][1, :]))
    err_vector = np.hstack((err_vector, np.ones(line_dict[lineLabel][0, :].size) * std_dict[lineLabel]))
    idxLine_dict[lineLabel][1] = wave_vector.size

print(idxLine_dict)
#
# fig, ax = plt.subplots()
# ax.plot(wave_vector, flux_vector, label='Observed line')
# ax.errorbar(wave_vector, cont_vector, label='Continuum', yerr=err_vector, fmt='o')
# ax.legend()
# ax.update({'xlabel':'Wavelength', 'ylabel':'Flux', 'title':'Gaussian fitting'})
# plt.show()


# PyMC3 fitting

# Container to store the synthetic line fluxes
# fluxTensor = tt.zeros(wave_vector.size)
# rangeLines = np.arange(listLabels.size)
# with pymc3.Model() as model:
#
#     amp_line = pymc3.Lognormal('amp_line', 0, 1, shape=listLabels.size)
#     mu_line = pymc3.Normal('mu_line', 0, 30, shape=listLabels.size)
#     sigma_line = pymc3.Lognormal('sigma_line', 0, 1, shape=listLabels.size)
#
#     for idx_line in rangeLines:
#         lineLabel = listLabels[idx_line]
#         lineWave = line_dict[lineLabel][0, :]
#         lineFlux = line_dict[lineLabel][1, :]
#         lineContFlux = lineCont_dict[lineLabel][1, :]
#         std_continuum = std_dict[lineLabel]
#
#         lineArea = gaussParams[idx_line, 0]
#         line_mu = mu_line[idx_line] + gaussParams[idx_line, 1]
#
#         fluxline_theo_i = gaussFunc_tt(lineWave, lineContFlux, amp_line[idx_line], line_mu, sigma_line[idx_line])
#
#     fluxTensor = tt.inc_subtensor(fluxTensor[:], fluxline_theo_i)
#
#     likelihood_i = pymc3.Normal(lineLabel, mu=fluxTensor, sd=err_vector, observed=flux_vector)
#
#     displaySimulationData(model)
#
#     trace = pymc3.sample(draws=2000, tune=500, chains=2, cores=1)

# # HMC results
# amp_trace, mu_trace, sigma_trace = trace['amp_line'], trace['mu_line'], trace['sigma_line']
# lineFluxHMC, areaHMC = trace['fluxCurves'], trace['areaHMC']
# gaussianCurveHMC = gaussFunc_tt(resampleWaveLine, resampleWaveCont, amp_trace.mean(), mu_trace.mean(), sigma_trace.mean())
#
# # Testing flux integrations
# AreaSimps = integrate.simps(lineFlux, lineWave) - integrate.simps(lineContinuumFit, lineWave)
# AreaTrapz = integrate.trapz(lineFlux, lineWave) - integrate.trapz(lineContinuumFit, lineWave)
# AreaGauss = np.sqrt(2*np.pi*sigma**2)*amp
#
# print(f'Area sum: {AreaSimps}')
# print(f'Simpsons rule: {AreaSimps}')
# print(f'Trapezoid rule: {AreaTrapz}')
# print(f'Gaussian area: {AreaGauss}')
# print(f'HMC area: {areaHMC.mean()}, {areaHMC.std()}')
#
# fig, ax = plt.subplots()
# ax.plot(wave, flux, label='Observed line')
# ax.scatter(continuaWave, continuaFlux, label='Continuum regions')
# ax.plot(lineWave, lineFlux, label='LineRegion', linestyle=':')
# ax.plot(lineWave, lineContinuumFit, label='Observed line', linestyle=':')
# ax.plot(resampleWaveLine, gaussianCurve, label='Gaussian fit', linestyle=':')
# # ax.plot(lineWave, lineFluxHMC.mean(axis=0), label='HMC fit', linestyle=':')
# ax.plot(resampleWaveLine, gaussianCurveHMC, label='HMC fit resample', linestyle='--')
# ax.set_xlim(continuaWave.min(), continuaWave.max())
# ax.set_ylim(bottom=continuaFlux.mean()/100, top=lineFlux.max() * 1.2)
#
# ax.legend()
# ax.update({'xlabel':'Flux', 'ylabel':'Wavelength', 'title':'Gaussian fitting'})
# plt.show()


# import pymc3
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import stats, optimize, integrate
# from inference_model import displaySimulationData
#
# # Gaussian curves
# def gaussFunc(ind_params, a, mu, sigma):
#     x, z = ind_params
#     return a * np.exp(-((x - mu) * (x - mu)) / (2 * (sigma * sigma))) + z
#
#
# def gaussFunc_tt(x, z, a, mu, sigma):
#     return a * np.exp(-((x - mu) * (x - mu)) / (2 * (sigma * sigma))) + z
#
#
# # Fake data
# wave = np.linspace(4950, 5050)
# flux_cont = 0.0 * wave + 2.0 + np.random.normal(0, 0.5, wave.size)
# ampTrue, muTrue, sigmaTrue = 10, 5007, 2.3
# flux_gauss = gaussFunc((wave, flux_cont), ampTrue, muTrue, sigmaTrue)
# w1, w2, w3, w4, w5, w6 = 4960, 4980, 4996, 5015, 5030, 5045
# wave_regions = np.array([w1, w2, w3, w4, w5, w6])
# areaTrue = np.sqrt(2*np.pi*sigmaTrue**2)*ampTrue
#
# # Identify line regions
# area_indcs = np.searchsorted(wave, wave_regions)
# idcsLines = ((wave[area_indcs[2]] <= wave[None]) & (wave[None] <= wave[area_indcs[3]])).squeeze()
# idcsContinua = (((wave[area_indcs[0]] <= wave[None]) & (wave[None] <= wave[area_indcs[1]])) | (
#         (wave[area_indcs[4]] <= wave[None]) & (wave[None] <= wave[area_indcs[5]]))).squeeze()
#
#
# # Get regions data
# lineWave, lineFlux = wave[idcsLines], flux_gauss[idcsLines]
# continuaWave, continuaFlux = wave[idcsContinua], flux_gauss[idcsContinua]
#
# # Linear region fitting
# slope, intercept, r_value, p_value, std_err = stats.linregress(continuaWave, continuaFlux)
# continuaFit = continuaWave * slope + intercept
# std_continuum = np.std(continuaFlux - continuaFit)
# lineContinuumFit = lineWave * slope + intercept
# continuumInt = lineContinuumFit.sum()
# centerWave = lineWave[np.argmax(lineFlux)]
# centerContInt = centerWave * slope + intercept
#
# # Standard fitting
# p0 = (lineFlux.max(), lineWave.mean(), 1.0)
# fitParams, cov = optimize.curve_fit(gaussFunc, (lineWave, lineContinuumFit), lineFlux, p0=p0)
# amp, mu, sigma = fitParams
#
# resampleWaveLine = np.linspace(lineWave[0]-10, lineWave[-1]+10, 100)
# resampleWaveCont = resampleWaveLine * slope + intercept
# gaussianCurve = gaussFunc((resampleWaveLine, resampleWaveCont), *fitParams)
#
# # PyMC3 fitting
# with pymc3.Model() as model:
#
#     amp_line = pymc3.Normal('amp_line', 0, 20)
#     mu_line = pymc3.Normal('mu_line', 5007, 30)
#     sigma_line = pymc3.Lognormal('sigma_line', 0, 1)
#
#     fluxline_theo = gaussFunc_tt(lineWave, lineContinuumFit, amp_line, mu_line, sigma_line)
#
#     # Store computed fluxes
#     pymc3.Deterministic('fluxCurves', fluxline_theo)
#     pymc3.Deterministic('areaHMC', np.sqrt(2*np.pi*sigma_line*sigma_line)*amp_line)
#
#     likelihood_i = pymc3.Normal('likelihood', mu=fluxline_theo, sd=std_continuum, observed=lineFlux)
#
#     displaySimulationData(model)
#
#     trace = pymc3.sample(draws=2000, tune=500, chains=2, cores=1)
#
# # HMC results
# amp_trace, mu_trace, sigma_trace = trace['amp_line'], trace['mu_line'], trace['sigma_line']
# lineFluxHMC, areaHMC = trace['fluxCurves'], trace['areaHMC']
# gaussianCurveHMC = gaussFunc_tt(resampleWaveLine, resampleWaveCont, amp_trace.mean(), mu_trace.mean(), sigma_trace.mean())
#
# # Testing flux integrations
# AreaSimps = integrate.simps(lineFlux, lineWave) - integrate.simps(lineContinuumFit, lineWave)
# AreaTrapz = integrate.trapz(lineFlux, lineWave) - integrate.trapz(lineContinuumFit, lineWave)
# AreaGauss = np.sqrt(2*np.pi*sigma**2)*amp
#
# print(f'True area: {areaTrue}')
# print(f'Simpsons rule: {AreaSimps}')
# print(f'Trapezoid rule: {AreaTrapz}')
# print(f'Gaussian area: {AreaGauss}')
# print(f'HMC area: {areaHMC.mean()}, {areaHMC.std()}')
#
# fig, ax = plt.subplots()
# ax.plot(wave, flux_gauss, label='Observed line')
# ax.scatter(continuaWave, continuaFlux, label='Continuum regions')
# ax.plot(lineWave, lineContinuumFit, label='Observed line', linestyle=':')
# ax.plot(resampleWaveLine, gaussianCurve, label='Gaussian fit', linestyle=':')
# ax.plot(lineWave, lineFluxHMC.mean(axis=0), label='HMC fit', linestyle=':')
# ax.plot(resampleWaveLine, gaussianCurveHMC, label='HMC fit resample', linestyle='--')
#
# ax.legend()
# ax.update({'xlabel':'Flux', 'ylabel':'Wavelength', 'title':'Gaussian fitting'})
# plt.show()