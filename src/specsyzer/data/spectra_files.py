import numpy as np
import astropy.io.fits as astrofits
from mpdaf.obj import Cube


def import_fits_data(file_address, instrument, frame_idx=0):

    if instrument == 'ISIS':

        # Open fits file
        with astrofits.open(file_address) as hdul:
            data, header = hdul[frame_idx].data, hdul[frame_idx].header

        assert 'ISIS' in header['INSTRUME']

        # William Herschel Telescope ISIS instrument
        if instrument == 'ISIS':
            w_min = header['CRVAL1']
            dw = header['CD1_1']  # dw = 0.862936 INDEF (Wavelength interval per pixel)
            pixels = header['NAXIS1']  # nw = 3801 number of output pixels
            w_max = w_min + dw * pixels
            wave = np.linspace(w_min, w_max, pixels, endpoint=False)

        return wave, data, header

    elif instrument == 'MUSE':

        cube = Cube(filename=str(file_address))
        header = cube.data_header

        cube.wave.info()
        dw = header['CD3_3']
        w_min = header['CRVAL3']
        nPixels = header['NAXIS3']
        w_max = w_min + dw * nPixels
        wave = np.linspace(w_min, w_max, nPixels, endpoint=False)

        return wave, cube, header

    else:

        print('-- WARNING: Instrument not recognize')

        # Open fits file
        with astrofits.open(file_address) as hdul:
            data, header = hdul[frame_idx].data, hdul[frame_idx].header

        return None, data, header
