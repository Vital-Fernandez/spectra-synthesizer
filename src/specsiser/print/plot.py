import numpy as np
from matplotlib import pyplot as plt, rcParams
from astropy.visualization import mpl_normalize, SqrtStretch


STANDARD_PLOT = {'figure.figsize': (20, 14), 'axes.titlesize': 14, 'axes.labelsize': 14, 'legend.fontsize': 12,
                 'xtick.labelsize': 12, 'ytick.labelsize': 12}


def spectrum(wave, flux, continuumFlux=None, obsLinesTable=None, matchedLinesDF=None, noise_region=None, fig=None,
             ax=None, fig_conf={}, axes_conf={}):

    # Plot Configuration
    defaultConf = STANDARD_PLOT.copy()
    defaultConf.update(fig_conf)
    rcParams.update(defaultConf)
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the spectrum
    ax.step(wave, flux, label='Observed spectrum')

    # Plot the continuum if available
    if continuumFlux is not None:
        ax.step(wave, continuumFlux, label='Continuum')

    # Plot astropy detected lines if available
    if obsLinesTable is not None:
        idcs_emission = obsLinesTable['line_type'] == 'emission'
        idcs_linePeaks = np.array(obsLinesTable[idcs_emission]['line_center_index'])
        ax.scatter(wave[idcs_linePeaks], flux[idcs_linePeaks], label='Detected lines', facecolors='none',
                   edgecolors='tab:purple')

    if matchedLinesDF is not None:
        idcs_foundLines = (matchedLinesDF.observation.isin(('detected', 'not identified'))) & \
                          (matchedLinesDF.wavelength >= wave[0]) & \
                          (matchedLinesDF.wavelength <= wave[-1])
        lineLatexLabel, lineWave = matchedLinesDF.loc[idcs_foundLines].latexLabel.values, matchedLinesDF.loc[
            idcs_foundLines].wavelength.values
        w3, w4 = matchedLinesDF.loc[idcs_foundLines].w3.values, matchedLinesDF.loc[idcs_foundLines].w4.values
        observation = matchedLinesDF.loc[idcs_foundLines].observation.values

        for i in np.arange(lineLatexLabel.size):
            if observation[i] == 'detected':
                color_area = 'tab:red' if observation[i] == 'not identified' else 'tab:green'
                ax.axvspan(w3[i], w4[i], alpha=0.25, color=color_area)
                ax.text(lineWave[i], 0, lineLatexLabel[i], rotation=270)

        # for i in np.arange(lineLatexLabel.size):
        #     color_area = 'tab:red' if observation[i] == 'not identified' else 'tab:green'
        #     ax.axvspan(w3[i], w4[i], alpha=0.25, color=color_area)
        #     ax.text(lineWave[i], 0, lineLatexLabel[i], rotation=270)

    if noise_region is not None:
        ax.axvspan(noise_region[0], noise_region[1], alpha=0.15, color='tab:cyan', label='Noise region')

    ax.update(axes_conf)
    ax.legend()
    plt.tight_layout()
    plt.show()

    return


def image_frame(flux, wcs=None, slices=('y', 'x', 1), fig=None, ax=None, fig_conf={}, axes_conf={}, output_file=None):

    # Plot Configuration
    defaultConf = STANDARD_PLOT.copy()
    defaultConf.update(fig_conf)
    rcParams.update(defaultConf)

    frame_size = flux.shape
    x, y = np.arange(0, frame_size[1]), np.arange(0, frame_size[0])
    X, Y = np.meshgrid(x, y)

    norm = mpl_normalize.ImageNormalize(stretch=SqrtStretch())

    # Axis set up for WCS
    if wcs is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = plt.figure()
        ax = fig.add_subplot(projection=wcs, slices=('x', 'y', 1))

    lower_limit = np.percentile(flux, 90)

    idcs_negative = flux < lower_limit
    flux_contours = np.ones(flux.shape)
    flux_contours[~idcs_negative] = flux[~idcs_negative]

    # Plot the data
    ax.imshow(np.log10(flux_contours), vmin=np.log10(lower_limit))
    # ax.contour(X, Y, np.log10(flux_contours), vmin=np.log10(lower_limit), cmap=plt.cm.inferno)
    ax.update(axes_conf)

    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, bbox_inches='tight')

    return


def image_contour(flux, wcs=None, slices=('y', 'x', 1), fig=None, ax=None, fig_conf={}, axes_conf={}):

    # Plot Configuration
    defaultConf = STANDARD_PLOT.copy()
    defaultConf.update(fig_conf)
    rcParams.update(defaultConf)

    frame_size = flux.shape
    x, y = np.arange(0, frame_size[1]), np.arange(0, frame_size[0])
    X, Y = np.meshgrid(x, y)

    norm = mpl_normalize.ImageNormalize(stretch=SqrtStretch())

    # Axis set up for WCS
    if wcs is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = plt.figure()
        ax = fig.add_subplot(projection=wcs, slices=('x', 'y', 1))

    lower_limit = np.percentile(flux, 90)

    idcs_negative = flux < lower_limit
    flux_contours = np.ones(flux.shape)
    flux_contours[~idcs_negative] = flux[~idcs_negative]

    # Plot the data
    ax.imshow(np.log10(flux_contours), vmin=np.log10(lower_limit))
    ax.contour(X, Y, np.log10(flux_contours), vmin=np.log10(lower_limit), cmap=plt.cm.inferno)
    ax.update(axes_conf)
    plt.show()

    return