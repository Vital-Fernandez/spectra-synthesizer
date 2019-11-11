
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize, integrate
from inference_model import displaySimulationData

np.random.seed(123)

def do_the_work(coeff_g, mu_g, sigma_g, limit = 4):

    x = np.linspace(-5, 5, num=1000)
    y = coeff_g * np.exp(-0.5 * np.power((x - mu_g) / sigma_g, 2))

    idx_area = ((-limit<= x) & (x <= limit)).squeeze()
    area_pix = y[idx_area].sum() * ((limit*2)/idx_area.sum())

    print('\n')
    print(f'Simpsons rule: { integrate.simps(y, x)}')
    print(f'Trapezoid rule: {integrate.trapz(y, x)}')
    print(f'Gaussian area: {np.sqrt(2 * np.pi * sigma_g ** 2) * coeff_g}')
    print(f'Pixel area: {area_pix}')

    return x, y


# Fake data
mu, sigma = 0, 1
coeff = 1 / (sigma * np.sqrt(2*np.pi))

fig, ax = plt.subplots()
x, y = do_the_work(coeff, mu, sigma)
ax.plot(x, y, label='Observed line')

amp = 3
x, y = do_the_work(amp, mu, sigma)
ax.plot(x, y, label=f'Amp = {amp}')

print('coefficient i', amp/coeff)
print('coefficient ii', 7.51988/(sigma * np.sqrt(2*np.pi)))


amp = 3
norm = 2
x, y = do_the_work(amp, mu, sigma)
y_norm = y/norm
ax.plot(x, y_norm, label=f'y/norm')
print(f'Simpsons rule: {integrate.simps(y_norm, x)}')
print(f'Trapezoid rule: {integrate.trapz(y_norm, x)}')
print('coefficient iii', amp/norm/coeff)
print('coefficient iv', (7.51988/norm)/(sigma * np.sqrt(2*np.pi)))


ax.legend()
ax.update({'xlabel':'Flux', 'ylabel':'Wavelength', 'title':'Gaussian fitting'})
plt.show()
