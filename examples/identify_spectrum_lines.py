import numpy as np
import pandas as pd
from pathlib import Path
from src.specsiser.physical_model.line_tools import EmissionFitting, gauss_func
from matplotlib import pyplot as plt, rcParams

# Get data
data_folder, data_file = Path('C:/Users/Vital/OneDrive/Desktop/'), 'test_spec2.txt'
file_to_open = data_folder / data_file
linesFile = Path('D:/Pycharm Projects/spectra-synthesizer/src/specsiser/literature_data/lines_data.xlsx') # TODO change to open format to avoid new dependency
linesDb = pd.read_excel(linesFile, sheet_name=0, header=0, index_col=0)
idcsLines = ~linesDb.index.str.contains('_b')
linesDb = linesDb[idcsLines]

# Import spectrum
redshift = 1.0046
wave, flux = np.loadtxt(file_to_open, unpack=True)
wave, flux = wave/redshift, flux * 1e-20

lm = EmissionFitting(wave, flux)

# Remove the continuum
flux_noContinuum = lm.continuum_remover(noiseWaveLim=(5600, 5850))

# Find lines
linesTable = lm.line_finder(flux_noContinuum, noiseWaveLim=(5590, 5870), verbose=False)

# Match lines
linesDb = lm.match_lines(linesTable, linesDb)

# Measure line fluxes
idcsObsLines = (linesDb.observation == 'detected')
obsLines = linesDb.loc[idcsObsLines].index.values

# Plot the complete spectrum
lm.plot_spectrum_components(lm.flux - flux_noContinuum, linesTable, linesDb)

for i in np.arange(obsLines.size):

    lineLabel = obsLines[i]
    print(f'- {lineLabel}:')

    # Declare regions data
    wave_regions = linesDb.loc[lineLabel, 'w1':'w6'].values
    idcsLinePeak, idcsContinua = lm.define_masks(wave_regions)

    # Identify line regions
    lm.line_properties(idcsLinePeak, idcsContinua, bootstrap_size=500)

    # Perform gaussian fitting
    lm.gauss_mcfit(idcsLinePeak, idcsContinua, bootstrap_size=500)

    # Store results in database
    lm.results_to_database(lineLabel, linesDb)

# Save dataframe to text file
linesLogAdress = data_folder / data_file.replace('.txt', '_linesLog.txt')
lm.save_lineslog(linesDb.loc[idcsObsLines], linesLogAdress)

# Plot the matched lines:
lm.plot_detected_lines(linesDb)




