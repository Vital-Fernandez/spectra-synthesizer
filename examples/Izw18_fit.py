import numpy as np
import pandas as pd
from pathlib import Path
import src.specsiser as sr
from matplotlib import pyplot as plt

# Declare spectrum
fits_address = '/home/vital/Astro-data/Observations/IZW18_Blue_cr_f_t_w_e__singleSlit_fglobal.fits'
wave_array, flux_array, header = sr.import_fits_data(fits_address, instrument='ISIS', frame_idx=0)

z_true = 0.00256
norm_obj = 1e-17

# Declare line masks
input_lines = ['H1_4861A']
input_columns = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6']
mask_df = pd.DataFrame(index=input_lines, columns=input_columns)
mask_df.loc['H1_4861A'] = np.array([4809.8, 4836.1, 4840.6, 4878.6, 4883.1, 4908.4])

# Declare fit configuration
conf_dict = dict(fit_conf={})

# Declare line measuring object
lm = sr.LineMesurer(wave_array, flux_array, redshift=z_true, normFlux=norm_obj)
lm.plot_spectrum()

# Find lines
noise_region = (1 + z_true) * np.array([5400, 5500])
norm_spec = lm.continuum_remover(noiseRegionLims=noise_region)
obsLinesTable = lm.line_finder(norm_spec, noiseWaveLim=noise_region, intLineThreshold=3)
matchedDF = lm.match_lines(obsLinesTable, mask_df)
lm.plot_spectrum(obsLinesTable=obsLinesTable, matchedLinesDF=matchedDF, specLabel=f'Emission line detection')
lm.plot_line_mask_selection(matchedDF, Path.home()/'synthetic_spectrum_new_mask.txt')

# Measure the emission lines
for i, lineLabel in enumerate(matchedDF.index.values):
    wave_regions = matchedDF.loc[lineLabel, 'w1':'w6'].values
    lm.fit_from_wavelengths(lineLabel, wave_regions, user_conf=conf_dict['fit_conf'])
    lm.print_results(show_fit_report=True, show_plot=True, log_scale=True, frame='obs')

# Save to txt file
output_address = Path.home()/'Izw18_fit_results.txt'
lm.save_lineslog(lm.linesDF, output_address)

print(f'Integrated flux: {lm.intg_flux*lm.normFlux:.3e}')
print(f'Gaussian flux: {lm.gauss_flux[0]*lm.normFlux:.3e}')
print(f'Continuum: {lm.cont*lm.normFlux:.3e}')
print(f'Eqw: {lm.eqw[0]:.2f}+/-{lm.eqw_err[0]:.2f}')


def iraf_avg(input_array):
    output_value = np.average([input_array])
    return output_value

def iraf_rms(input_array):
    output_value = np.sqrt(np.mean(np.square(input_array - np.mean(input_array))))
    return output_value

def iraf_snr(avg_array, rms_array):
    output_value = avg_array/rms_array
    return output_value


line_region = np.array([4870, 4880])
cont_region = np.array([4850, 4860])

idcsLine = np.searchsorted(lm.wave, line_region)
idcsCont = np.searchsorted(lm.wave, cont_region)

fig, ax = plt.subplots()
ax.plot(lm.wave, lm.flux, label='Line')
ax.plot(lm.wave[idcsLine[0]:idcsLine[1]], lm.flux[idcsLine[0]:idcsLine[1]], label='Line')
ax.plot(lm.wave[idcsCont[0]:idcsCont[1]], lm.flux[idcsCont[0]:idcsCont[1]], label='Cont')

ax.legend()
ax.update({'xlabel': 'Wavelength', 'ylabel': 'Flux', 'title': 'Gaussian fitting'})
plt.show()


lineFlux = lm.flux[idcsLine[0]:idcsLine[1]] * lm.normFlux
lineCont = lm.flux[idcsCont[0]:idcsCont[1]] * lm.normFlux

print(f'Line region: average {iraf_avg(lineFlux):.3e}, rms {iraf_rms(lineFlux):.3e}, snr {iraf_snr(iraf_avg(lineFlux), iraf_rms(lineFlux)):.3f}')
print(f'Cont region: average {iraf_avg(lineCont):.3e}, rms {iraf_rms(lineCont):.3e}, snr {iraf_snr(iraf_avg(lineCont), iraf_rms(lineCont)):.3f}')
lm.plot_line_velocity()

# # Compare with the true values
# emission_lines_dict = {'H1_4861A': [75.25, 4861.0, 1.123]}
#
# for label, param_list in emission_lines_dict.items():
#     z_g = lm.linesDF.loc[label, 'z_line']
#     area_g, A_g, mu_g, sigma_g = lm.linesDF.loc[label, 'gauss_flux']/norm_obj, lm.linesDF.loc[label, 'amp']/norm_obj, lm.linesDF.loc[label, 'center'], lm.linesDF.loc[label, 'sigma']
#     area_true = param_list[0] * np.sqrt(2 * np.pi) * param_list[2]
#
#     print(f'\n{label}')
#     print(f'Area (flux norm):{area_g:0.3f} | True {area_true:0.3f}')
#     print(f'Amplitude (flux norm): {A_g:0.3f} | True {param_list[0]}')
#     print(f'Center (angs): {mu_g/(1 + z_true):0.3f} | True {param_list[1]}')
#     print(f'sigma (angs): {sigma_g:0.3f} | True {param_list[2]}')
#
#     mu_vel, sigma_vel = lm.linesDF.loc[label, 'v_r'], lm.linesDF.loc[label, 'sigma_vel']
#     print(f'Center (km/s): {mu_vel:0.3f}')
#     print(f'sigma (km/s): {sigma_vel:0.3f}')


# # Save to fits file
# output_address = Path.home()/'synthetic_spectrum.fits'
# col_waves = fits.Column(name='wave', array=wave_obs, format='1E')
# col_flux = fits.Column(name='flux', array=flux_obs, format='1E')
# hdu_spec = fits.BinTableHDU.from_columns([col_waves, col_flux], name='spectrum', header=fits.Header(hdr_dict))
# new_hdul.append(hdu_spec)
# new_hdul.writeto(output_address, overwrite=True, output_verify='fix')
