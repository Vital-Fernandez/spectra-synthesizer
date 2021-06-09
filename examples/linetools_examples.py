from pathlib import Path
import numpy as np
import src.specsiser as sr

# Declare the data location
obsFitsFile = Path('./sample_data/gp121903_BR.fits')
lineMaskFile = Path('./sample_data/gp121903_BR_mask.txt')
cfgFile = Path('./sample_data/gtc_greenpeas_data.ini')

# Load the data
obsConf = sr.loadConfData(cfgFile, objList_check=True, group_variables=False)
maskDF = sr.lineslogFile_to_DF(lineMaskFile)
wave, flux, header = sr.import_fits_data(obsFitsFile, instrument='OSIRIS')

# Declare line measuring object
lm = sr.LineMesurer(wave, flux, redshift=obsConf['sample_data']['z_array'][2], normFlux=obsConf['sample_data']['norm_flux'])
lm.plot_spectrum()

# Identify line location
norm_spec = lm.continuum_remover(noiseRegionLims=obsConf['sample_data']['noiseRegion_array'])
obsLinesTable = lm.line_finder(norm_spec, noiseWaveLim=obsConf['sample_data']['noiseRegion_array'], intLineThreshold=3)
matchedDF = lm.match_lines(obsLinesTable, maskDF)
lm.plot_spectrum(obsLinesTable=obsLinesTable, matchedLinesDF=matchedDF, specLabel=f'Emission line detection')

# Improve line region
corrected_mask_file = Path('./sample_data/gp121903_BR_mask_corrected.txt')
lm.plot_line_mask_selection(matchedDF, corrected_mask_file)

# Measure the emission lines
objMaskDF = sr.lineslogFile_to_DF(corrected_mask_file)
for i, lineLabel in enumerate(objMaskDF.index.values):
    wave_regions = objMaskDF.loc[lineLabel, 'w1':'w6'].values
    lm.fit_from_wavelengths(lineLabel, wave_regions, fit_conf=obsConf['gp121903_line_fitting'])
    # lm.plot_fit_components(lm.fit_output)

# Display results
lm.plot_line_grid(lm.linesDF)

# Save the results
output_folder = obsFitsFile.parent
lm.save_lineslog(lm.linesDF, output_folder/'linesLog.txt')
lm.table_fluxes(lm.linesDF, output_folder/'linesTable')


