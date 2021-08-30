import numpy as np
import pandas as pd
from pathlib import Path
import src.specsiser as sr

# Declare spectrum
wave, flux = np.loadtxt(Path.home()/'synthetic_spectrum.txt', unpack=True)
z_true = 0.12345
norm_obj = 1e-17

# Declare line masks
input_lines = ['H1_4861A_b', 'O3_4959A', 'O3_5007A', 'H1_6563A_b']
input_columns = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6']
mask_df = pd.DataFrame(index=input_lines, columns=input_columns)
mask_df.loc['H1_4861A_b'] = np.array([4809.8, 4836.1, 4840.6, 4878.6, 4883.1, 4908.4])
mask_df.loc['O3_4959A'] = np.array([4925.2, 4940.4, 4943.0, 4972.9, 4976.7, 4990.2])
mask_df.loc['O3_5007A'] = np.array([4972.7, 4987.0, 4992.0, 5024.7, 5031.5, 5043.984899])
mask_df.loc['H1_6563A_b'] = np.array([6438.0, 6508.6, 6535.10, 6600.9, 6627.69, 6661.8])

# Declare fit configuration
conf_dict = dict(fit_conf={'H1_4861A_b': 'H1_4861A-H1_4861A_w1',
                           'H1_6563A_b': 'H1_6563A-H1_6563A_w1',
                           'H1_6563A_w1_sigma': {'expr': '>1*H1_6563A_sigma'}})

# Declare line measuring object
lm = sr.LineMesurer(wave, flux, redshift=z_true, normFlux=norm_obj)
# lm.plot_spectrum()

# Find lines
noise_region = (1 + z_true) * np.array([5400, 5500])
norm_spec = lm.continuum_remover(noiseRegionLims=noise_region)
obsLinesTable = lm.line_finder(norm_spec, noiseWaveLim=noise_region, intLineThreshold=3)
matchedDF = lm.match_lines(obsLinesTable, mask_df)
# lm.plot_spectrum(obsLinesTable=obsLinesTable, matchedLinesDF=matchedDF, specLabel=f'Emission line detection')
# lm.plot_line_mask_selection(matchedDF, Path.home()/'synthetic_spectrum_new_mask.txt')

# Measure the emission lines
for i, lineLabel in enumerate(matchedDF.index.values):
    wave_regions = matchedDF.loc[lineLabel, 'w1':'w6'].values
    lm.fit_from_wavelengths(lineLabel, wave_regions, user_conf=conf_dict['fit_conf'])
    # lm.print_results(show_fit_report=True, show_plot=True, log_scale=True, frame='obs')
    lm.plot_line_velocity()

# Save to txt file
output_address = Path.home()/'synthetic_spectrum_fit_results.txt'
lm.save_lineslog(lm.linesDF, output_address)

# Compare with the true values
emission_lines_dict = {'H1_4861A': [75.25, 4861.0, 1.123],
                       'H1_4861A_w1': [7.525, 4861.0, 5.615],
                       'O3_4959A': [150.50, 4959.0, 2.456],
                       'O3_5007A': [451.50, 5007.0, 2.456],
                       'H1_6563A': [225.75, 6563.0, 2.456],
                       'H1_6563A_w1': [225.75, 6566.0, 5.615]}

for label, param_list in emission_lines_dict.items():
    z_g = lm.linesDF.loc[label, 'z_line']
    area_g, A_g, mu_g, sigma_g = lm.linesDF.loc[label, 'gauss_flux']/norm_obj, lm.linesDF.loc[label, 'amp']/norm_obj, lm.linesDF.loc[label, 'center'], lm.linesDF.loc[label, 'sigma']
    area_true = param_list[0] * np.sqrt(2 * np.pi) * param_list[2]

    print(f'\n{label}')
    print(f'Area (flux norm):{area_g:0.3f} | True {area_true:0.3f}')
    print(f'Amplitude (flux norm): {A_g:0.3f} | True {param_list[0]}')
    print(f'Center (angs): {mu_g/(1 + z_true):0.3f} | True {param_list[1]}')
    print(f'sigma (angs): {sigma_g:0.3f} | True {param_list[2]}')

    mu_vel, sigma_vel = lm.linesDF.loc[label, 'v_r'], lm.linesDF.loc[label, 'sigma_vel']
    print(f'Center (km/s): {mu_vel:0.3f}')
    print(f'sigma (km/s): {sigma_vel:0.3f}')


# # Save to fits file
# output_address = Path.home()/'synthetic_spectrum.fits'
# col_waves = fits.Column(name='wave', array=wave_obs, format='1E')
# col_flux = fits.Column(name='flux', array=flux_obs, format='1E')
# hdu_spec = fits.BinTableHDU.from_columns([col_waves, col_flux], name='spectrum', header=fits.Header(hdr_dict))
# new_hdul.append(hdu_spec)
# new_hdul.writeto(output_address, overwrite=True, output_verify='fix')
