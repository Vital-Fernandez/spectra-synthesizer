import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from pathlib import Path


# Gaussian area function
def gauss_area(amp, sigma):
    return amp * np.sqrt(2 * np.pi) * sigma


# Gaussian profile function
def gaussian_model(x, amp, center, sigma):
    return amp * np.exp(-0.5 * (((x-center)/sigma) * ((x-center)/sigma)))


# Generate wavelength range
hdr_dict = {'CRVAL1':4500.0,
           'CD1_1':0.2,
           'NAXIS1':20000}

w_min = hdr_dict['CRVAL1']
dw = hdr_dict['CD1_1']
nPixels = hdr_dict['NAXIS1']
w_max = w_min + dw * nPixels

# Spectrum redshift and normalization
z_obj = 0.12345
norm_obj = 1e-17
hdr_dict.update({'z_obj': z_obj, 'norm_obj': norm_obj})

# Spectrum wavelength range
wave_rest = np.linspace(w_min, w_max, nPixels, endpoint=False)
wave_obs = (1 + z_obj) * wave_rest

# Linear continuum : slope and interception
continuum_lineal = np.array([-0.001, 20.345])

# Gaussian emission lines: Amplitude (height (norm flux)), center (angstroms) and sigma (angstroms)
emission_lines_dict = {'H1_4861A': [75.25, 4861.0, 1.123],
                       'H1_4861A_w1': [7.525, 4861.0, 5.615],
                       'O3_4959A': [150.50, 4959.0, 2.456],
                       'O3_5007A': [451.50, 5007.0, 2.456],
                       'H1_6563A': [225.75, 6563.0, 2.456],
                       'H1_6563A_w1': [225.75, 6566.0, 5.615]}

# Adding coninuum as a linear function
flux_obs = wave_obs * continuum_lineal[0] + continuum_lineal[1]

# Adding emission lines
for lineLabel, gaus_params in emission_lines_dict.items():
    flux_obs += gaussian_model(wave_obs, gaus_params[0], gaus_params[1] * (1 + z_obj), gaus_params[2])
    print(f'{lineLabel} area: {gauss_area(gaus_params[0], gaus_params[2])}')

# Add noise
noise_sigma = 0.05
flux_obs = flux_obs + np.random.normal(0, noise_sigma, size=flux_obs.size)

# Normalise
flux_obs = flux_obs * norm_obj

# Plot
output_address = Path.home()/'synthetic_spectrum.png'
fig, ax = plt.subplots(figsize=(10, 6))
ax.step(wave_obs, flux_obs, label='Synthetic spectrum', where='mid')
ax.legend()
ax.update({'xlabel': r'Wavelength $(\AA)$', 'ylabel': r'Flux $(erg\,cm^{-2} s^{-1} \AA^{-1})$'})
plt.savefig(output_address)

# Save to fits file
new_hdul = fits.HDUList()
new_hdul.append(fits.PrimaryHDU())

# Save to txt file
output_address = Path.home()/'synthetic_spectrum.txt'
np.savetxt(output_address, np.c_[wave_obs, flux_obs])

# Save to fits file
output_address = Path.home()/'synthetic_spectrum.fits'
col_waves = fits.Column(name='wave', array=wave_obs, format='1E')
col_flux = fits.Column(name='flux', array=flux_obs, format='1E')
hdu_spec = fits.BinTableHDU.from_columns([col_waves, col_flux], name='spectrum', header=fits.Header(hdr_dict))
new_hdul.append(hdu_spec)
new_hdul.writeto(output_address, overwrite=True, output_verify='fix')
