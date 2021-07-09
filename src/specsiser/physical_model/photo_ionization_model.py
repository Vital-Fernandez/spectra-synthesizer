import numpy as np
from physical_model.gasEmission_functions import gridInterpolatorFunction
import exoplanet as xo

# Function to read the ionization data
def load_ionization_grid(log_scale=False, log_zero_value = -1000):

    # grid_file = 'D:/Dropbox/Astrophysics/Tools/HCm-Teff_v5.01/C17_bb_Teff_30-90_pp.dat'

    # TODO make an option to create the lines and
    grid_file = '/home/vital/Dropbox/Astrophysics/Tools/HCm-Teff_v5.01/C17_bb_Teff_30-90_pp.dat'
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


def gridInterpolatorFunction(interpolatorDict, x_range, y_range, z_range=None, interp_type='point'):

    emisInterpGrid = {}

    if interp_type == 'point':
        for line, emisGrid_i in interpolatorDict.items():
            emisInterp_i = xo.interp.RegularGridInterpolator([x_range, y_range], emisGrid_i[:, :, None], nout=1)
            emisInterpGrid[line] = emisInterp_i.evaluate

    elif interp_type == 'axis':
        for line, emisGrid_i in interpolatorDict.items():
            emisGrid_i_reshape = emisGrid_i.reshape((x_range.size, y_range.size, -1))
            emisInterp_i = xo.interp.RegularGridInterpolator([x_range, y_range], emisGrid_i_reshape)
            emisInterpGrid[line] = emisInterp_i.evaluate

    elif interp_type == 'cube':
        for line, grid_ndarray in interpolatorDict.items():
            xo_interp = xo.interp.RegularGridInterpolator([x_range,
                                                           y_range,
                                                           z_range], grid_ndarray)
            emisInterpGrid[line] = xo_interp.evaluate

    return emisInterpGrid


class ModelGridWrapper:

    def __init__(self, grid_address=None):

        self.grid_LineLabels = None
        self.grid_emissionFluxes = None
        self.grid_emissionFluxErrs = None

        self.gridInterp = None
        self.idx_analysis_lines = None
        self.grid_array = None

        return

    def ndarray_from_DF(self, grid_DF, axes_columns=None, data_columns='all', sort_axes=True, dict_output=True,
                        empty_value=np.nan):

        if sort_axes:
            assert set(axes_columns).issubset(set(grid_DF.columns.values)), f'- Error: Mesh grid does not include all' \
                                                                            f' input columns {axes_columns}'
            grid_DF.sort_values(axes_columns, inplace=True)

        # Compute axes coordinates for reshaping
        axes_cords = {}
        reshape_array = np.zeros(len(axes_columns)).astype(int)
        for i, ax_name in enumerate(axes_columns):
            axes_cords[ax_name] = np.unique(grid_DF[ax_name].values)
            reshape_array[i] = axes_cords[ax_name].size

        # Declare grid data columns
        if data_columns == 'all':
            data_columns = grid_DF.columns[~grid_DF.columns.isin(axes_columns)].values
        axes_cords['data'] = data_columns
        # Establish output format

        # mesh_dict
        if dict_output:
            output_container = {}
            for i, dataColumn in enumerate(data_columns):
                data_array_flatten = grid_DF[dataColumn].values
                output_container[dataColumn] = data_array_flatten.reshape(reshape_array.astype(int))

        # mesh_array
        else:
            output_container = np.full(np.hstack((reshape_array, len(data_columns))), np.nan)
            for i, dataColumn in enumerate(data_columns):
                data_array_flatten = grid_DF[dataColumn].values
                output_container[..., i] = data_array_flatten.reshape(reshape_array.astype(int))

        return output_container, axes_cords

    def generate_xo_interpolators(self, grid_dict, axes_list, axes_coords, interp_type='point', empty_value=np.nan):

        # Establish interpolation axes: (x_range, y_range, z_range,...)
        ax_range_container = [None] * len(axes_list)
        for i, ax in enumerate(axes_list):
            ax_range_container[i] = axes_coords[ax]

        if interp_type == 'point':

            output_container = {}

            for grid_key, grid_ndarray in grid_dict.items():
                xo_interp = xo.interp.RegularGridInterpolator(ax_range_container, grid_ndarray)
                output_container[grid_key] = xo_interp.evaluate

            return output_container

        if interp_type == 'axis':

            # Generate empty grid from first data element
            grid_shape = list(grid_dict[axes_coords['data'][0]].shape) + [len(axes_coords['data'])]
            data_grid = np.full(grid_shape, empty_value)
            for i, label_dataGrid in enumerate(axes_coords['data']):
                data_grid[..., i] = grid_dict[label_dataGrid]

            # Add additional dimension with -1 for interpolation along axis
            reShapeDataGrid_shape = [len(item) for item in ax_range_container] + [-1]
            xo_interp = xo.interp.RegularGridInterpolator(ax_range_container, data_grid.reshape(reShapeDataGrid_shape))

            return xo_interp.evaluate

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

