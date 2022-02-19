import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm, rcParams, rcParamsDefault, colors, gridspec
from matplotlib.mlab import detrend_mean
from lime import label_decomposition
from lime.plots import PdfMaker
import corner

background_color = np.array((43, 43, 43))/255.0
foreground_color = np.array((179, 199, 216))/255.0
red_color = np.array((43, 43, 43))/255.0
yellow_color = np.array((191, 144, 0))/255.0

latex_labels = {'y_plus': r'$y^{+}$',
             'He1_abund': r'$y^{+}$',
             'He2_abund': r'$y^{++}$',
             'Te': r'$T_{e}$',
             'T_low': r'$T_{low}(K)$',
             'T_LOW': r'$T_{low}(K)$',
             'T_high': r'$T_{high}(K)$',
             'T_HIGH': r'$T_{high}(K)$',
             'T_He': r'$T_{He}$',
             'n_e': r'$n_{e}(cm^{-3})$',
             'cHbeta': r'$c(H\beta)$',
             'tau': r'$\tau$',
             'xi': r'$\xi$',
             'ChiSq': r'$\chi^{2}$',
             'ChiSq_Recomb': r'$\chi^{2}_{Recomb}$',
             'ChiSq_Metals': r'$\chi^{2}_{Metals}$',
             'ChiSq_O': r'$\chi^{2}_{O}$',
             'ChiSq_S': r'$\chi^{2}_{S}$',
             'S2_abund': r'$S^{+}$',
             'He1r': r'$y^{+}$',
             'He2r': r'$y^{2+}$',
             'He1': r'$y^{+}$',
             'He2': r'$y^{2+}$',
             'log(He1r)': r'$log(y^{+})$',
             'log(He2r)': r'$log(y^{2+})$',
             'OH': r'$\frac{O}{H}$',
             'OH_err': r'$O/H\,err$',
             'S3_abund': r'$S^{2+}$',
             'O2_abund': r'$O^{+}$',
             'O3_abund': r'$O^{2+}$',
             'S3_abund': r'$S^{2+}$',
             'O2_abund': r'$O^{+}$',
             'O3_abund': r'$O^{2+}$',
             'N2_abund': r'$N^{+}$',
             'Ar3_abund': r'$Ar^{2+}$',
             'Ar4_abund': r'$Ar^{3+}$',
             'S2': r'$\frac{S^{+}}{H^{+}}$',
             'S3': r'$\frac{S^{2+}}{H^{+}}$',
             'S4': r'$\frac{S^{3+}}{H^{+}}$',
             'O2': r'$\frac{O^{+}}{H^{+}}$',
             'O3': r'$\frac{O^{2+}}{H^{+}}$',
             'Ni3': r'$\frac{Ni^{2+}}{H^{+}}$',
             'NI3': r'$\frac{Ni^{2+}}{H^{+}}$',
             'Cl3': r'$\frac{Cl^{2+}}{H^{+}}$',
             'CL3': r'$\frac{Cl^{2+}}{H^{+}}$',
             'Ne3': r'$\frac{Ne^{2+}}{H^{+}}$',
             'NE3': r'$\frac{Ne^{2+}}{H^{+}}$',
             'Fe3': r'$\frac{Fe^{2+}}{H^{+}}$',
             'FE3': r'$\frac{Fe^{2+}}{H^{+}}$',
             'N2': r'$\frac{N^{+}}{H^{+}}$',
             'Ar3': r'$\frac{Ar^{2+}}{H^{+}}$',
             'AR3': r'$\frac{Ar^{2+}}{H^{+}}$',
             'Ar4': r'$\frac{Ar^{3+}}{H^{+}}$',
             'AR4': r'$\frac{Ar^{3+}}{H^{+}}$',
             'Cl4': r'$\frac{Cl^{3+}}{H^{+}}$',
             'CL4': r'$\frac{Cl^{3+}}{H^{+}}$',
             'Ar_abund': r'$\frac{ArI}{HI}$',
             'He_abund': r'$\frac{HeI}{HI}$',
             'O_abund': r'$\frac{OI}{HI}$',
             'N_abund': r'$\frac{NI}{HI}$',
             'S_abund': r'$\frac{SI}{HI}$',
             'Ymass_O': r'$Y_{O}$',
             'Ymass_S': r'$Y_{S}$',
             'Ar': r'$\frac{Ar}{H}$',
             'He': r'$\frac{He}{H}$',
             'O': r'$\frac{O}{H}$',
             'N': r'$\frac{N}{H}$',
             'S': r'$\frac{S}{H}$',
             'Ymass_O': r'$Y_{O}$',
             'Ymass_S': r'$Y_{S}$',
             'NO': r'$\frac{N}{O}$',
             'calcFluxes_Op': 'Line fluxes',
             'z_star': r'$z_{\star}$',
             'sigma_star': r'$\sigma_{\star}$',
             'Av_star': r'$Av_{\star}$',
             'chiSq_ssp': r'$\chi^{2}_{SSP}$',
             'x': r'x interpolator$',
             'ICF_SIV': r'$ICF\left(S^{3+}\right)$',
             'logU': r'$log(U)$',
             'logOH': r'$log(O/H)$',
             'logNO': r'$log(N/O)$',
             'Teff': r'$T_{eff}$',
             'TEFF': r'$T_{eff}$',
             'X_i+': r'$X^{i+}$',
             'log(X_i+)': r'$12+log\left(X^{i+}\right)$',
             'redNoise': r'$\Delta(cH\beta)$'}

DARK_PLOT = {'figure.figsize': (14, 7),
             'axes.titlesize': 14,
             'axes.labelsize': 14,
             'legend.fontsize': 12,
             'xtick.labelsize': 12,
             'ytick.labelsize': 12,
             'text.color': foreground_color,
             'figure.facecolor': background_color,
             'axes.facecolor': background_color,
             'axes.edgecolor': foreground_color,
             'axes.labelcolor': foreground_color,
             'xtick.color': foreground_color,
             'ytick.color': foreground_color,
             'legend.edgecolor': 'inherit',
             'legend.facecolor': 'inherit'}


def numberStringFormat(value, cifras = 4):
    if value > 0.001:
        newFormat = f'{value:.{cifras}f}'
    else:
        newFormat = f'{value:.{cifras}e}'

    return newFormat


def _histplot_bins(column, bins=100):
    """Helper to get bins for histplot."""
    col_min = np.min(column)
    col_max = np.max(column)
    return range(col_min, col_max + 2, max((col_max - col_min) // bins, 1))


def plot_traces(plot_address, params_list, traces_dict, true_values=None, plot_conf={}, dark_mode=True):

    if true_values is not None:
        trace_true_dict = {}
        for param in params_list:
            if param in true_values:
                trace_true_dict[param] = true_values[param]
    n_traces = len(params_list)

    # Plot format
    if dark_mode:
        defaultConf = DARK_PLOT.copy()
        defaultConf.update(plot_conf)
        rcParams.update(defaultConf)
    else:
        defaultConf = {'axes.titlesize': 14, 'axes.labelsize': 14, 'legend.fontsize': 10,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8}
        defaultConf.update(plot_conf)
    rcParams.update(defaultConf)

    fig = plt.figure(figsize=(8, n_traces))

    colorNorm = colors.Normalize(0, n_traces)
    cmap = cm.get_cmap(name=None)

    gs = gridspec.GridSpec(n_traces * 2, 4)
    gs.update(wspace=0.2, hspace=1.8)

    for i in range(n_traces):

        trace_code = params_list[i]
        trace_array = traces_dict[trace_code]

        mean_value = np.mean(trace_array)
        std_dev = np.std(trace_array)

        axTrace = fig.add_subplot(gs[2 * i:2 * (1 + i), :3])
        axPoterior = fig.add_subplot(gs[2 * i:2 * (1 + i), 3])

        # Label for the plot
        if mean_value > 10: # TODO need a special rutine
            label = r'{} = ${:.0f}$$\pm${:.0f}'.format(latex_labels[trace_code], mean_value, std_dev)
        else:
            label = r'{} = ${:.3f}$$\pm${:.3f}'.format(latex_labels[trace_code], mean_value, std_dev)

        # Plot the traces
        axTrace.plot(trace_array, label=label, color=cmap(colorNorm(i)))
        axTrace.axhline(y=mean_value, color=cmap(colorNorm(i)), linestyle='--')
        axTrace.set_ylabel(latex_labels[trace_code])

        # Plot the histograms
        axPoterior.hist(trace_array, bins=50, histtype='step', color=cmap(colorNorm(i)), align='left')

        # Plot the axis as percentile
        median, percentile16th, percentile84th = np.median(trace_array), np.percentile(trace_array, 16), np.percentile(trace_array, 84)

        # Add true value if available
        if true_values is not None:
            if trace_code in trace_true_dict:
                value_param = trace_true_dict[trace_code]

                # Nominal value and uncertainty
                if isinstance(value_param, (list, tuple, np.ndarray)):
                    nominal_value, std_value = value_param[0], 0.0 if len(value_param) == 1 else value_param[1]
                    axPoterior.axvline(x=nominal_value, color='black', linestyle='solid')
                    axPoterior.axvspan(nominal_value - std_value, nominal_value + std_value, alpha=0.5, color=cmap(colorNorm(i)))

                # Nominal value only
                else:
                    nominal_value = value_param
                    axPoterior.axvline(x=nominal_value, color='black', linestyle='solid')

        # Add legend
        axTrace.legend(loc=7)

        # Remove ticks and labels
        if i < n_traces - 1:
            axTrace.get_xaxis().set_visible(False)
            axTrace.set_xticks([])

        axPoterior.yaxis.set_major_formatter(plt.NullFormatter())
        axPoterior.set_yticks([])

        axPoterior.set_xticks([percentile16th, median, percentile84th])
        round_n = 0 if median > 10 else 3
        axPoterior.set_xticklabels(['', numberStringFormat(median, round_n), ''])

        axTrace.set_yticks((percentile16th, median, percentile84th))
        round_n = 0 if median > 10 else 3
        axTrace.set_yticklabels((numberStringFormat(percentile16th,round_n), '', numberStringFormat(percentile84th, round_n)))

    if plot_address is not None:
        plt.savefig(plot_address, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        # plt.tight_layout()
        plt.show()

    rcParams.update(rcParamsDefault)

    return


def plot_flux_grid(plot_address, input_lines, inFlux, inErr, trace_dict, n_columns=8, combined_dict={},
                   plot_conf={}, user_labels={}):

    # Input data
    ion_array, wave_array, latexLabel_array = label_decomposition(input_lines, comp_dict=combined_dict,
                                                                  user_format=user_labels)

    # Declare plot grid size
    n_lines = len(input_lines)
    n_rows = int(np.ceil(float(n_lines)/float(n_columns)))
    n_cells = n_rows * n_columns

    # Declare figure format
    size_dict = {'figure.figsize': (22, 9), 'axes.titlesize': 14, 'axes.labelsize': 10, 'legend.fontsize': 10,
                 'xtick.labelsize': 8, 'ytick.labelsize': 3}
    size_dict.update(plot_conf)
    rcParams.update(size_dict)

    #self.FigConf(plotSize=size_dict, Figtype='Grid', n_columns=n_columns, n_rows=n_rows)
    fig, axes = plt.subplots(n_rows, n_columns)
    axes = axes.ravel()

    # Generate the color dict
    obsIons = np.unique(ion_array)
    colorNorm = colors.Normalize(0, obsIons.size)
    cmap = cm.get_cmap(name=None)
    colorDict = dict(zip(obsIons, np.arange(obsIons.size)))

    # Plot individual traces
    for i in range(n_cells):

        if i < n_lines:

            # Current line
            label = input_lines[i]
            ion = ion_array[i]
            trace = trace_dict[label]
            median_flux = np.median(trace)

            label_mean = 'Mean value: {}'.format(np.around(median_flux, 4))
            axes[i].hist(trace, histtype='stepfilled', bins=35, alpha=.7, color=cmap(colorNorm(colorDict[ion])), density=False)

            label_true = 'True value: {}'.format(np.around(inFlux[i], 3))
            axes[i].axvline(x=inFlux[i], label=label_true, color='black', linestyle='solid')
            axes[i].axvspan(inFlux[i] - inErr[i], inFlux[i] + inErr[i], alpha=0.5, color='grey')
            axes[i].get_yaxis().set_visible(False)
            axes[i].set_yticks([])

            # Plot wording
            axes[i].set_title(latexLabel_array[i])

        else:
            fig.delaxes(axes[i])

    if plot_address is not None:
        plt.savefig(plot_address, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        # plt.tight_layout()
        plt.show()

    rcParams.update(rcParamsDefault)

    return


def plot_corner(plot_address, params_list, traces_dict, true_values=None):

    # Reshape plot data
    n_traces = len(params_list)
    list_arrays, labels_list = [], []
    for i in range(n_traces):
        trace_code = params_list[i]
        trace_array = traces_dict[trace_code]
        list_arrays.append(trace_array)
        labels_list.append(latex_labels[trace_code])
    traces_array = np.array(list_arrays).T

    # Prepare True values
    traceTrueValues = [None] * n_traces
    if true_values is not None:
        for i, param in enumerate(params_list):
            if param in true_values:
                param_value = true_values[param]
                if np.isscalar(param_value):
                    traceTrueValues[i] = param_value
                else:
                    traceTrueValues[i] = param_value[0]

    # Dark model
    # # Declare figure format
    # background = np.array((43, 43, 43)) / 255.0
    # foreground = np.array((179, 199, 216)) / 255.0
    #
    # figConf = {'text.color': foreground,
    #            'figure.figsize': (16, 10),
    #            'figure.facecolor': background,
    #            'axes.facecolor': background,
    #            'axes.edgecolor': foreground,
    #            'axes.labelcolor': foreground,
    #            'axes.labelsize': 30,
    #            'xtick.labelsize': 12,
    #            'ytick.labelsize': 12,
    #            'xtick.color': foreground,
    #            'ytick.color': foreground,
    #            'legend.edgecolor': 'inherit',
    #            'legend.facecolor': 'inherit',
    #            'legend.fontsize': 16,
    #            'legend.loc': "center right"}
    # rcParams.update(figConf)
    # # Generate the plot
    # mykwargs = {'no_fill_contours':True, 'fill_contours':True}
    # self.Fig = corner.corner(traces_array[:, :], fontsize=30, labels=labels_list, quantiles=[0.16, 0.5, 0.84],
    #                          show_titles=True, title_args={"fontsize": 200},
    #                          truth_color='#ae3135', title_fmt='0.3f', color=foreground, **mykwargs)#, hist2d_kwargs = {'cmap':'RdGy',
    #                                                                                    #'fill_contours':False,
    #                                                                                    #'plot_contours':False,
    #                                                                                    #'plot_datapoints':False})


    # # Generate the plot
    fig = corner.corner(traces_array[:, :], fontsize=30, labels=labels_list, quantiles=[0.16, 0.5, 0.84],
                             show_titles=True, title_args={"fontsize": 200}, truths=traceTrueValues,
                             truth_color='#ae3135', title_fmt='0.3f')

    plt.savefig(plot_address, dpi=100, bbox_inches='tight')
    plt.close(fig)

    return


def table_fluxes(table_address, input_lines, inFlux, inErr, traces_dict, combined_dict={}, file_type='table',
                 user_labels={}, theme='white'):

    # Table headers
    headers_pdf = ['Line', 'Observed flux', 'Fit Mean', 'Standard deviation', 'Median', r'$16^{th}$ $percentil$',
                   r'$84^{th}$ $percentil$', r'$Difference\,\%$']

    headers_txt = ['Observed_flux', 'Observed_err', 'Fit_Mean', 'Fit_Standard_deviation', 'Median', 'Per_16th',
                   'Per_84th', 'Percentage_difference']

    # Create containers
    tableDF = pd.DataFrame(columns=headers_txt)
    pdf = PdfMaker()
    pdf.create_pdfDoc(pdf_type=file_type, theme=theme)
    pdf.pdf_insert_table(headers_pdf)

    # Output data
    flux_matrix = np.array([traces_dict[lineLabel] for lineLabel in input_lines])
    mean_line_values = flux_matrix.mean(axis=1)
    std_line_values = flux_matrix.std(axis=1)
    median_line_values = np.median(flux_matrix, axis=1)
    p16th_line_values = np.percentile(flux_matrix, 16, axis=1)
    p84th_line_values = np.percentile(flux_matrix, 84, axis=1)

    # Array wih true error values for flux
    diff_Percentage = np.round((1 - (median_line_values / inFlux)) * 100, 2)
    diff_Percentage = list(map(str, diff_Percentage))

    ion_array, wave_array, latexLabel_array = label_decomposition(input_lines, comp_dict=combined_dict,
                                                                  user_format=user_labels)

    for i in range(inFlux.size):
        # label = label_formatting(inputLabels[i])
        flux_obs = r'${:0.3}\pm{:0.3}$'.format(inFlux[i], inErr[i])

        row_i = [latexLabel_array[i], flux_obs, mean_line_values[i], std_line_values[i], median_line_values[i],
                 p16th_line_values[i], p84th_line_values[i], diff_Percentage[i]]

        row_txt = [inFlux[i], inErr[i], mean_line_values[i], std_line_values[i], median_line_values[i],
                   p16th_line_values[i], p84th_line_values[i], diff_Percentage[i]]

        pdf.addTableRow(row_i, last_row=False if input_lines[-1] != input_lines[i] else True)
        tableDF.loc[input_lines[i]] = row_txt

    pdf.generate_pdf(table_address)

    # Save the table as a dataframe.
    with open(f'{table_address}.txt', 'wb') as output_file:
        string_DF = tableDF.to_string()
        output_file.write(string_DF.encode('UTF-8'))


def table_params(table_address, parameter_list, trace_dict, true_values=None, file_type='table', theme='white'):

    # Table headers
    headers = ['Parameter', 'Mean', 'Standard deviation', 'Number of points', 'Median',
               r'$16^{th}$ percentil', r'$84^{th}$ percentil']

    if true_values is not None:
        headers.insert(1, 'True value')
        headers.append(r'Difference $\%$')

    # Generate containers
    pdf = PdfMaker()
    pdf.create_pdfDoc(pdf_type=file_type, theme=theme)
    pdf.pdf_insert_table(headers)
    tableDF = pd.DataFrame(columns=headers[1:])

    # Loop around the parameters
    for param in parameter_list:

        trace_i = trace_dict[param]

        label = latex_labels[param]
        mean_value = np.mean(trace_i)
        std = np.std(trace_i)
        n_traces = trace_i.size
        median = np.median(trace_i)
        p_16th = np.percentile(trace_i, 16)
        p_84th = np.percentile(trace_i, 84)

        true_value, perDif = 'None', 'None'

        if true_values is not None:
            if param in true_values:
                value_param = true_values[param]
                if isinstance(value_param, (list, tuple, np.ndarray)):
                    true_value = r'${}$ $\pm${}'.format(value_param[0], value_param[1])
                    perDif = str(np.round((1 - (value_param[0] / mean_value)) * 100, 2))

                else:
                    true_value = value_param
                    perDif = str(np.round((1 - (true_value / mean_value)) * 100, 2))

            row_i = [label, true_value, mean_value, std, n_traces, median, p_16th, p_84th, perDif]

        else:
            row_i = [label, mean_value, std, n_traces, median, p_16th, p_84th]

        pdf.addTableRow(row_i, last_row=False if parameter_list[-1] != param else True)
        tableDF.loc[row_i[0]] = row_i[1:]

    pdf.generate_pdf(output_address=table_address)

    # Save the table as a dataframe.
    with open(f'{table_address}.txt', 'wb') as output_file:
        string_DF = tableDF.to_string()
        output_file.write(string_DF.encode('UTF-8'))

    return


def emissivitySurfaceFit_2D(self, line_label, emisCoeffs, emisGrid, funcEmis, te_ne_grid, denRange, tempRange):

    # Plot format
    size_dict = {'figure.figsize': (20, 14), 'axes.titlesize': 16, 'axes.labelsize': 16, 'legend.fontsize': 18}
    rcParams.update(size_dict)

    # Generate figure
    fig, ax = plt.subplots(1, 1)

    # Generate fitted surface points
    surface_points = funcEmis(te_ne_grid, *emisCoeffs)

    # Plot plane
    plt.imshow(surface_points.reshape((denRange.size, tempRange.size)), aspect=0.03,
               extent=(te_ne_grid[1].min(), te_ne_grid[1].max(), te_ne_grid[0].min(), te_ne_grid[0].max()))

    # Compare pyneb values with values from fitting
    percentage_difference = (1 - surface_points / emisGrid.flatten()) * 100

    # Points with error below 1.0 are transparent:
    idx_interest = percentage_difference < 1.0
    x_values, y_values = te_ne_grid[1][idx_interest], te_ne_grid[0][idx_interest]
    ax.scatter(x_values, y_values, c="None", edgecolors='black', linewidths=0.35, label='Error below 1%')

    if idx_interest.sum() < emisGrid.size:
        # Plot grid points
        plt.scatter(te_ne_grid[1][~idx_interest], te_ne_grid[0][~idx_interest],
                    c=percentage_difference[~idx_interest],
                    edgecolors='black', linewidths=0.1, cmap=cm.OrRd, label='Error above 1%')

        # Color bar
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('% difference', rotation=270, fontsize=15)

    # Trim the axis
    ax.set_xlim(te_ne_grid[1].min(), te_ne_grid[1].max())
    ax.set_ylim(te_ne_grid[0].min(), te_ne_grid[0].max())

    # Add labels
    ax.update({'xlabel': 'Density ($cm^{-3}$)', 'ylabel': 'Temperature $(K)$', 'title': line_label})

    return


def emissivitySurfaceFit_3D(self, line_label, emisCoeffs, emisGrid, funcEmis, te_ne_grid, denRange, tempRange):

    # Plot format
    size_dict = {'figure.figsize': (20, 14), 'axes.titlesize': 16, 'axes.labelsize': 16, 'legend.fontsize': 18}
    rcParams.update(size_dict)

    # Plot the grid points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # # Generate fitted surface points
    # matrix_edge = int(np.sqrt(te_ne_grid[0].shape[0]))
    #
    # # Plotting pyneb emissivities
    # x_values, y_values = te_ne_grid[0].reshape((matrix_edge, matrix_edge)), te_ne_grid[1].reshape((matrix_edge, matrix_edge))
    # ax.plot_surface(x_values, y_values, emisGrid.reshape((matrix_edge, matrix_edge)), color='g', alpha=0.5)

    # Generate fitted surface points
    x_values = te_ne_grid[0].reshape((denRange.size, tempRange.size))
    y_values = te_ne_grid[1].reshape((denRange.size, tempRange.size))
    ax.plot_surface(x_values, y_values, emisGrid.reshape((denRange.size, tempRange.size)), color='g',
                    alpha=0.5)

    # Plotting emissivity parametrization
    fit_points = funcEmis(te_ne_grid, *emisCoeffs)
    ax.scatter(te_ne_grid[0], te_ne_grid[1], fit_points, color='r', alpha=0.5)

    # Add labels
    ax.update({'ylabel': 'Density ($cm^{-3}$)', 'xlabel': 'Temperature $(K)$', 'title': line_label})

    return