import numpy as np
from physical_model.gasEmission_functions import gridInterpolatorFunction

# Function to read the ionization data
def load_ionization_grid(log_scale=False, log_zero_value = -1000):

    # grid_file = 'D:/Dropbox/Astrophysics/Tools/HCm-Teff_v5.01/C17_bb_Teff_30-90_pp.dat'

    # TODO make an option to create the lines and
    grid_file = 'D:/Dropbox/Astrophysics/Tools/HCm-Teff_v5.01/C17_bb_Teff_30-90_pp.dat'
    lineConversionDict = dict(O2_3726A_m='OII_3727',
                              O3_5007A='OIII_5007',
                              S2_6716A_m='SII_6717,31',
                              S3_9069A='SIII_9069',
                              He1_4471A='HeI_4471',
                              He1_5876A='HeI_5876',
                              He2_4686A='HeII_4686')

    # Load the data and get axes range
    grid_array = np.loadtxt(grid_file)

    grid_axes = dict(OH=np.unique(grid_array[:, 0]),
                     Teff=np.unique(grid_array[:, 1]),
                     logU=np.unique(grid_array[:, 2]))

    # Sort the array according to 'logU', 'Teff', 'OH'
    idcs_sorted_grid = np.lexsort((grid_array[:, 1], grid_array[:, 2], grid_array[:, 0]))
    sorted_grid = grid_array[idcs_sorted_grid]

    # Loop throught the emission line and abundances and restore the grid
    grid_dict = {}
    for i, item in enumerate(lineConversionDict.items()):
        lineLabel, epmLabel = item

        grid_dict[lineLabel] = np.zeros((grid_axes['logU'].size,
                                         grid_axes['Teff'].size,
                                         grid_axes['OH'].size))

        for j, abund in enumerate(grid_axes['OH']):
            idcsSubGrid = sorted_grid[:, 0] == abund
            lineGrid = sorted_grid[idcsSubGrid, i + 3]
            lineMatrix = lineGrid.reshape((grid_axes['logU'].size, grid_axes['Teff'].size))
            grid_dict[lineLabel][:, :, j] = lineMatrix[:, :]

    if log_scale:
        for lineLabel, lineGrid in grid_dict.items():
            grid_logScale = np.log10(lineGrid)

            # Replace -inf entries by -1000
            idcs_0 = grid_logScale == -np.inf
            if np.any(idcs_0):
                grid_logScale[idcs_0] = log_zero_value

            grid_dict[lineLabel] = grid_logScale

    return grid_dict, grid_axes


class ModelGridWrapper:

    def __init__(self):

        self.grid_LineLabels = None
        self.grid_emissionFluxes = None
        self.grid_emissionFluxErrs = None

        self.gridInterp = None
        self.idx_analysis_lines = None

        return

    def HII_Teff_models(self, obsLines, obsFluxes, obsErr):

        gridLineDict, gridAxDict = load_ionization_grid(log_scale=True)
        self.gridInterp = gridInterpolatorFunction(gridLineDict,
                                                   gridAxDict['logU'],
                                                   gridAxDict['Teff'],
                                                   gridAxDict['OH'],
                                                   interp_type='cube')

        # Add merged lines
        if ('S2_6716A' in obsLines) and ('S2_6731A' in obsLines) and ('S2_6716A_m' not in obsLines):

            # Rename the grid label to match observable
            self.gridInterp['S2_6716A'] = self.gridInterp.pop('S2_6716A_m')

            lines_Grid = np.array(list(self.gridInterp.keys()))
            self.idx_analysis_lines = np.in1d(obsLines, lines_Grid)

            # Use different set of fluxes for direct method and grids
            self.grid_LineLabels = obsLines.copy()
            self.grid_emissionFluxes = obsFluxes.copy()
            self.grid_emissionFluxErrs = obsErr.copy()

            # Compute the merged line
            i_S2_6716A, i_S2_6731A = obsLines == 'S2_6716A', obsLines == 'S2_6731A'
            S2_6716A_m_flux = obsFluxes[i_S2_6716A][0] + obsFluxes[i_S2_6731A][0]
            S2_6716A_m_err = np.sqrt(obsErr[i_S2_6716A][0]**2 + obsErr[i_S2_6731A][0]**2)

            # Replace conflicting flux
            self.grid_emissionFluxes[i_S2_6716A] = S2_6716A_m_flux
            self.grid_emissionFluxErrs[i_S2_6716A] = S2_6716A_m_err

        else:
            lines_Grid = np.array(list(gridLineDict.keys()))
            self.idx_analysis_lines = np.in1d(obsLines, lines_Grid)
            self.grid_LineLabels = obsLines.copy()
            self.grid_emissionFluxes = obsFluxes.copy()
            self.grid_emissionFluxErrs = obsErr.copy()

        return

