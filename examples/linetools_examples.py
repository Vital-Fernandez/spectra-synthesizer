from pathlib import Path
import numpy as np
import src.specsiser as sr
from src.specsiser.physical_model.line_tools import label_decomposition

linesLogAddress = Path('D:/Google drive/Astrophysics/Datos/SDSS-Ricardo/green_peas/flux_analysis/J004743+015440_linesLog.txt')

linesDF = sr.lineslogFile_to_DF(linesLogAddress)

combined_dict = {'O2_3726A_m': 'O2_3726A-O2_3729A',
                'H1_3889A_m': 'H1_3889A-He1_3889A',
                'Ar4_4711A_m':  'Ar4_4711A-He1_4713A',
                'O2_7319A_m':  'O2_7319A-O2_7330A',
                'H1_6563A_b':  'H1_6563A-N2_6584A-N2_6548A'}

ion_array, wave_array, latexLabel_array = sr.label_decomposition(linesDF.index, combined_dict=combined_dict)
for i in np.arange(len(ion_array)):
    print(f'{i}: {ion_array[i]}, {wave_array[i]} {latexLabel_array[i]}')

ion_array, wave_array, latexLabel_array = sr.label_decomposition(linesDF.index.values[0], combined_dict=combined_dict)
print(linesDF.index.values[0])
print(ion_array, wave_array, latexLabel_array)