import numpy as np
import astropy.io.fits as astrofits
from mpdaf.obj import Cube


def import_fits_data(file_address, instrument, frame_idx=None):

    if instrument == 'ISIS':

        # Open fits file
        with astrofits.open(file_address) as hdul:
            data, header = hdul[frame_idx].data, hdul[frame_idx].header

        assert 'ISIS' in header['INSTRUME'], 'Input spectrum instrument '

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

    elif instrument == 'OSIRIS':

        # Default frame index
        if frame_idx is None:
            frame_idx = 0

        # Open fits file
        with astrofits.open(file_address) as hdul:
            data, header = hdul[frame_idx].data, hdul[frame_idx].header

        # assert 'OSIRIS' in header['INSTRUME']

        w_min = header['CRVAL1']
        dw = header['CD1_1']  # dw (Wavelength interval per pixel)
        pixels = header['NAXIS1']  # nw number of output pixels
        w_max = w_min + dw * pixels
        wave = np.linspace(w_min, w_max, pixels, endpoint=False)

        return wave, data, header

    elif instrument == 'SDSS':

        # Open fits file
        with astrofits.open(file_address) as hdul:
            data, header_0, header_2, header_3 = hdul[1].data, hdul[0].header, hdul[2].data, hdul[3].data

        assert 'SDSS 2.5-M' in header_0['TELESCOP']

        wave = 10.0 ** data['loglam']
        SDSS_z = float(header_2["z"][0] + 1)
        wave_rest = wave / SDSS_z

        flux_norm = data['flux']
        flux = flux_norm / 1e17

        headers = (header_0, header_2, header_3)

        # return wave_rest, flux, headers
        return wave, data, headers

    elif instrument == 'xshooter':

        # Default frame index
        if frame_idx is None:
            frame_idx = 1

        # Following the steps at: https://archive.eso.org/cms/eso-data/help/1dspectra.html
        with astrofits.open(file_address) as hdul:
            data, header = hdul[1].data, hdul[1].header

        wave = data[0][0]

        return wave, data[0], header

    else:

        print('-- WARNING: Instrument not recognize')

        # Open fits file
        with astrofits.open(file_address) as hdul:
            data, header = hdul[frame_idx].data, hdul[frame_idx].header

        return None, data, header
