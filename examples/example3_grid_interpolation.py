import numpy as np
import pandas as pd
import lime

import src.specsiser as sr
from timeit import default_timer as timer

# Load the fitting configuration
conf_file = './sample_data/HII_CHIM_TRY_grid.cfg'
conf_fit = lime.load_cfg(conf_file)

# Load the photoionization grid
file_address = './sample_data/HII-CHI-mistry_1Myr_grid_O.txt'
grid_3D_DF = pd.read_csv(file_address, delim_whitespace=True, header=0)

# Lines for the fitting
obs_lines = np.array(['O3_4959A', 'O3_5007A',
                      'He1_5876A',
                      'N2_6548A', 'H1_6563A', 'N2_6584A',
                      'S2_6716A', 'S2_6731A',
                      'S3_9069A', 'S3_9531A'])

# Prepare grid interpolators
model_variables = ['logOH', 'logU', 'logNO']
gw = sr.GridWrapper()
grid_dict, axes_cords_a = gw.ndarray_from_DF(grid_3D_DF, axes_columns=model_variables)
grid_interpolators = gw.generate_xo_interpolators(grid_dict, model_variables, axes_cords_a, interp_type='point')

grid_lines = np.array(list(grid_dict.keys()))
idcs_lines = np.isin(obs_lines, grid_lines)
input_lines = obs_lines[idcs_lines]

# Grid to perform the fitting
logOH_range = np.round(np.linspace(7.15, 9.0, 5), 3)
logU_range = np.round(np.linspace(-3.90, -1.40, 5), 3)
logNO_range = np.round(np.linspace(-1.90, -0.01, 5), 3)

# Output file for the fitting
outputFits = './sample_data/grid_logOH_logU_logNO_advi.fits'

# Loop throught the grid of synthetic conditions (around 30 seconds per fit)
start = timer()
i_step, n_steps = 0, logNO_range.size * logOH_range.size * logNO_range.size
for i, logOH in enumerate(logOH_range):
    for j, logU in enumerate(logU_range):
        for k, logNO in enumerate(logNO_range):

            print(f'- Fit {i_step}/{n_steps}: expected time {n_steps*30/3600:0.3f} hours')

            # True value coordinate for interpolation
            coord_true = [[logOH, logU, logNO]]
            header_params = {'logOH': logOH, 'logU': logU, 'logNO': logNO}

            # Output files
            cord_label = f'{logOH*1000:.0f}_{logU*-1000:.0f}_{logNO*-1000:.0f}'

            # Fill the dataframe with integrated flux
            log = pd.DataFrame(index=input_lines, columns=['wavelength', 'ion',	'intg_flux', 'intg_err'])
            for line in input_lines:
                ion, wavelength, latexLabel = lime.label_decomposition(line, scalar_output=True)
                flux = np.power(10, grid_interpolators[line](coord_true).eval()[0])
                log.loc[line, :] = wavelength, ion, flux, flux * 0.05
            lines_flambda = np.zeros(log.index.values.size)

            # Declare sampler
            obj1_model = sr.SpectraSynthesizer(grid_sampling=True, grid_interp=grid_interpolators)

            # Declare region physical model
            obj1_model.define_region(log.index.values, log.intg_flux.values, log.intg_err.values,
                                     lineFlambda=lines_flambda)

            # Declare region physical model
            obj1_model.simulation_configuration(prior_conf_dict=conf_fit['priors_configuration'])

            obj1_model.photoionization_sampling(model_variables)

            obj1_model.run_sampler(1000, 2000, nchains=8, njobs=8, init='advi')

            obj1_model.save_fit(outputFits, cord_label, output_format='fits', user_header=header_params)

            # # Load the results
            # fit_results = sr.load_fit_results(outputFits, ext_name=cord_label, output_format='fits')
            # inLines = fit_results[f'{cord_label}_inputs'][0]['lines_list']
            # inParameters = fit_results[f'{cord_label}_outputs'][0]['parameters_list']
            # inFlux = fit_results[f'{cord_label}_inputs'][0]['line_fluxes']
            # inErr = fit_results[f'{cord_label}_inputs'][0]['line_err']
            # traces_dict = fit_results[f'{cord_label}_traces'][0]
            #
            # print('-- Model parameters posterior diagram')
            # figure_file = f'/home/vital/Astro-data/grid_models/{cord_label}_trace_plot.png'
            # sr.plot_traces(figure_file, inParameters, traces_dict, true_values={'logOH': logOH, 'logU': logU, 'logNO': logNO})
            #

end = timer()
print(f'Working time:{(end-start)/n_steps} seconds per fit')
