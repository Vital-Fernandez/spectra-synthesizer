import numpy as np
import matplotlib.pyplot as plt
from src.specsiser.components.gasContinuum_functions import NebularContinua

nebCalculator = NebularContinua()

wmin, wmax = 1000.0, 20000.0
wavelength_range = np.linspace(int(wmin), int(wmax), int(wmax-wmin))

# Physical conditions
flux_halpha = 1e-14
Te, ne = 10000.0, 100.0
HeII_HII, HeIII_HII = 0.1, 0.001

# Compute the nebular continuum
H_He_frac = 1 + HeII_HII * 4 + HeIII_HII * 4

# Bound bound continuum
gamma_2q = nebCalculator.boundbound_gamma(wavelength_range, Te)

# Free-Free continuum
gamma_ff = H_He_frac * nebCalculator.freefree_gamma(wavelength_range, Te, Z_ion=1.0)

# Free-Bound continuum
gamma_fb_HI = nebCalculator.freebound_gamma(wavelength_range, Te, nebCalculator.HI_fb_dict)
gamma_fb_HeI = nebCalculator.freebound_gamma(wavelength_range, Te, nebCalculator.HeI_fb_dict)
gamma_fb_HeII = nebCalculator.freebound_gamma(wavelength_range, Te, nebCalculator.HeII_fb_dict)
gamma_fb = gamma_fb_HI + HeII_HII * gamma_fb_HeI + HeIII_HII * gamma_fb_HeII

# Plot the spectra
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(wavelength_range, gamma_ff, label='Free-Free component')
ax.plot(wavelength_range, gamma_fb, label='Free-Bound component')
ax.plot(wavelength_range, gamma_2q, label='2 photons component')
ax.update({'xlabel': r'Wavelength $(\AA)$', 'ylabel': r'$\gamma_{\nu} (erg\,cm^{3} s^{-1})$'})
ax.legend()
plt.tight_layout()
plt.show()
