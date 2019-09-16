import corner
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib import rcParams
from matplotlib import colors
from matplotlib.mlab import detrend_mean
from numpy import reshape, percentile, median
from pylatex import Document, Figure, NewPage, NoEscape, Package, Tabular, Section, Tabu, Table, LongTable
from functools import partial
from collections import Sequence

latex_labels = {'y_plus': r'$y^{+}$',
             'He1_abund': r'$y^{+}$',
             'He2_abund': r'$y^{++}$',
             'Te': r'$T_{e}$',
             'T_low': r'$T_{low}$',
             'T_high': r'$T_{high}$',
             'T_He': r'$T_{He}$',
             'n_e': r'$n_{e}$',
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
             'S3_abund': r'$S^{2+}$',
             'O2_abund': r'$O^{+}$',
             'O3_abund': r'$O^{2+}$',
             'S3_abund': r'$S^{2+}$',
             'O2_abund': r'$O^{+}$',
             'O3_abund': r'$O^{2+}$',
             'N2_abund': r'$N^{+}$',
             'Ar3_abund': r'$Ar^{2+}$',
             'Ar4_abund': r'$Ar^{3+}$',
             'S2': r'$S^{+}$',
             'S3': r'$S^{2+}$',
             'O2': r'$O^{+}$',
             'O3': r'$O^{2+}$',
             'N2': r'$N^{+}$',
             'Ar3': r'$Ar^{2+}$',
             'Ar4': r'$Ar^{3+}$',
             'Ar_abund': r'$\frac{ArI}{HI}$',
             'He_abund': r'$\frac{HeI}{HI}$',
             'O_abund': r'$\frac{OI}{HI}$',
             'N_abund': r'$\frac{NI}{HI}$',
             'S_abund': r'$\frac{SI}{HI}$',
             'Ymass_O': r'$Y_{O}$',
             'Ymass_S': r'$Y_{S}$',
             'calcFluxes_Op': 'Line fluxes',
             'z_star': r'$z_{\star}$',
             'sigma_star': r'$\sigma_{\star}$',
             'Av_star': r'$Av_{\star}$',
             'chiSq_ssp': r'$\chi^{2}_{SSP}$',
             'ICF_SIV': r'$ICF\left(S^{3+}\right)$'
             }

def label_formatting(line_label):
    label = line_label.replace('_', '\,\,')
    if label[-1] == 'A':
        label = label[0:-1] + '\AA'
    label = '$' + label + '$'

    return label


def _histplot_bins(column, bins=100):
    """Helper to get bins for histplot."""
    col_min = np.min(column)
    col_max = np.max(column)
    return range(col_min, col_max + 2, max((col_max - col_min) // bins, 1))


def numberStringFormat(value, cifras = 4):
    if value > 0.001:
        newFormat = str(round(value, cifras))
    else:
        newFormat = r'${:.3e}$'.format(value)

    return newFormat


def printSimulationData(model, priorsDict, lineLabels, lineFluxes, lineErr, lineFitErr):

    print('\n- Simulation configuration')

    # Print input lines and fluxes
    print('\n-- Input lines')
    for i in range(lineLabels.size):
        warnLine = '{}'.format('|| WARNING obsLineErr = {:.4f}'.format(lineErr[i]) if lineErr[i] != lineFitErr[i] else '')
        displayText = '{} flux = {:.4f} +/- {:.4f} || err % = {:.5f} {}'.format(lineLabels[i], lineFluxes[i], lineFitErr[i], lineFitErr[i] / lineFluxes[i], warnLine)
        print(displayText)

    # Present the model data
    print('\n-- Priors design:')
    for prior in priorsDict:
        displayText = '{} : mu = {}, std = {}'.format(prior, priorsDict[prior][0], priorsDict[prior][1])
        print(displayText)

    # Check test_values are finite
    print('\n-- Test points:')
    model_var = model.test_point
    for var in model_var:
        displayText = '{} = {}'.format(var, model_var[var])
        print(displayText)

    # Checks log probability of random variables
    print('\n-- Log probability variable:')
    print(model.check_test_point())

    return


class FigConf:

    def __init__(self):

        # Default sizes for computer
        self.defaultFigConf = {'figure.figsize': (14, 8), 'legend.fontsize': 15, 'axes.labelsize': 20,
                               'axes.titlesize': 24, 'xtick.labelsize': 14, 'ytick.labelsize': 14}

        # rcParams.update(sizing_dict)

    def gen_colorList(self, vmin=0.0, vmax=1.0, color_palette=None):

        colorNorm = colors.Normalize(vmin, vmax)
        cmap = cm.get_cmap(name=color_palette)
        # return certain color
        # self.cmap(self.colorNorm(idx))

        return colorNorm, cmap

    def FigConf(self, plotStyle=None, plotSize='medium', Figtype='Single', AxisFormat=111, n_columns=None, n_rows=None,
                n_colors=None, color_map=None, axis_not=None):

        # Set the figure format before creating it
        self.define_format(plotStyle, plotSize)

        if Figtype == 'Single':

            if AxisFormat == 111:
                self.Fig = plt.figure()
                self.Axis = self.Fig.add_subplot(AxisFormat)
            else:
                self.Fig, self.Axis = plt.subplots(n_rows, n_columns)
                self.Axis = self.Axis.ravel()

        elif Figtype == 'Posteriors':
            self.Fig = plt.figure()
            AxisFormat = int(str(n_columns) + '11')
            self.Axis = self.Fig.add_subplot(AxisFormat)

        elif Figtype == 'grid':
            self.Fig, self.Axis = plt.subplots(n_rows, n_columns)
            if (n_rows * n_columns) != 1:
                self.Axis = self.Axis.ravel()
            else:
                self.Axis = [self.Axis]

        elif Figtype == 'tracePosterior':
            self.Fig, self.Axis = plt.subplots(n_rows, n_columns)

        elif Figtype == 'Grid':
            frame1 = plt.gca()
            frame1.axes.xaxis.set_visible(False)
            frame1.axes.yaxis.set_visible(False)
            gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
            self.Fig, self.Axis = plt.subplots(n_rows, n_columns)
            self.Axis = self.Axis.ravel()

        elif Figtype == 'Grid_size':
            self.Fig = plt.figure()
            gs = gridspec.GridSpec(n_rows, n_columns, height_ratios=[2.5, 1])
            self.ax1 = self.Fig.add_subplot(gs[0, :])
            self.ax2 = self.Fig.add_subplot(gs[1, :])

        if axis_not == 'sci':
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        return

    def FigWording(self, xlabel, ylabel, title, loc='best', Expand=False, XLabelPad=0.0, YLabelPad=0.0, Y_TitlePad=1.02,
                   cb_title=None, sort_legend=False, ncols_leg=1, graph_axis=None):

        if graph_axis == None:
            Axis = self.Axis
        else:
            Axis = graph_axis

        Axis.set_xlabel(xlabel)
        Axis.set_ylabel(ylabel)
        Axis.set_title(title, y=Y_TitlePad)

        if (XLabelPad != 0) or (YLabelPad != 0):
            Axis.xaxis.labelpad = XLabelPad
            Axis.yaxis.labelpad = YLabelPad

        if cb_title != None:
            self.cb.set_label(cb_title, fontsize=18)

        self.legend_conf(Axis, loc, sort_legend=sort_legend, ncols=ncols_leg)

    def legend_conf(self, Axis=None, loc='best', sort_legend=False, ncols=1):

        if Axis == None:
            Axis = self.Axis

        Axis.legend(loc=loc, ncol=ncols)

        # Security checks to avoid empty legends
        if Axis.get_legend_handles_labels()[1] != None:

            if len(Axis.get_legend_handles_labels()[1]) != 0:
                Old_Handles, Old_Labels = Axis.get_legend_handles_labels()

                if sort_legend:
                    labels, handles = zip(*sorted(zip(Old_Labels, Old_Handles), key=lambda t: t[0]))
                    Handles_by_Label = dict(zip(labels, handles))
                    Axis.legend(Handles_by_Label.values(), Handles_by_Label.keys(), loc=loc, ncol=ncols)
                else:
                    Handles_by_Label = dict(zip(Old_Labels, Old_Handles))
                    Axis.legend(Handles_by_Label.values(), Handles_by_Label.keys(), loc=loc, ncol=ncols)

        return

    def bayesian_legend_conf(self, Axis, loc='best', fontize=None, edgelabel=False):
        # WARNING: THIS DOES NOT WORK WITH LEGEND RAVELIN

        if Axis.get_legend_handles_labels()[1] != None:
            Old_Handles, Old_Labels = Axis.get_legend_handles_labels()
            Handles_by_Label = dict(zip(Old_Labels, Old_Handles))

            Hl = zip(Handles_by_Label.values(), Handles_by_Label.keys())

            New_Handles, New_labels = zip(*Hl)

            myLegend = Axis.legend(New_Handles, New_labels, loc=loc, prop={'size': 12}, scatterpoints=1, numpoints=1)

            if fontize != None:

                Leg_Frame = myLegend.get_frame()
                Leg_Frame.set_facecolor(self.Color_Vector[0])
                Leg_Frame.set_edgecolor(self.Color_Vector[1])

                for label in myLegend.get_texts():
                    label.set_fontsize('large')

                for label in myLegend.get_lines():
                    label.set_linewidth(1.5)

                for text in myLegend.get_texts():
                    text.set_color(self.Color_Vector[1])

            if edgelabel:
                Leg_Frame = myLegend.get_frame()
                Leg_Frame.set_edgecolor('black')

    def savefig(self, output_address, extension='.png', reset_fig=True, pad_inches=0.2, resolution=300.0):

        plt.savefig(output_address + extension, dpi=resolution, bbox_inches='tight')

        return


class PdfPrinter():

    def __init__(self):

        self.pdf_type = None
        self.pdf_geometry_options = {'right': '1cm',
                                     'left': '1cm',
                                     'top': '1cm',
                                     'bottom': '2cm'}

        # TODO add dictionary with numeric formats for tables depending on the variable

    def create_pdfDoc(self, fname, pdf_type='graphs', geometry_options=None, document_class=u'article'):

        # TODO it would be nicer to create pdf object to do all these things

        self.pdf_type = pdf_type

        # Update the geometry if necessary (we coud define a dictionary distinction)
        if pdf_type == 'graphs':
            pdf_format = {'landscape': 'true'}
            self.pdf_geometry_options.update(pdf_format)

        elif pdf_type == 'table':
            pdf_format = {'landscape': 'true',
                          'paperwidth': '30in',
                          'paperheight': '30in'}
            self.pdf_geometry_options.update(pdf_format)

        if geometry_options is not None:
            self.pdf_geometry_options.update(geometry_options)

        # Generate the doc
        self.pdfDoc = Document(fname, documentclass=document_class, geometry_options=self.pdf_geometry_options)

        if pdf_type == 'table':
            self.pdfDoc.packages.append(Package('preview', options=['active', 'tightpage', ]))
            self.pdfDoc.packages.append(Package('hyperref', options=['unicode=true', ]))
            self.pdfDoc.append(NoEscape(r'\pagenumbering{gobble}'))
            self.pdfDoc.packages.append(Package('nicefrac'))
            self.pdfDoc.packages.append(
                Package('color', options=['usenames', 'dvipsnames', ]))  # Package to crop pdf to a figure

        elif pdf_type == 'longtable':
            self.pdfDoc.append(NoEscape(r'\pagenumbering{gobble}'))

    def pdf_create_section(self, caption, add_page=False):

        with self.pdfDoc.create(Section(caption)):
            if add_page:
                self.pdfDoc.append(NewPage())

    def add_page(self):

        self.pdfDoc.append(NewPage())

        return

    def pdf_insert_image(self, image_address, fig_loc='htbp', width=r'1\textwidth'):

        with self.pdfDoc.create(Figure(position='h!')) as fig_pdf:
            fig_pdf.add_image(image_address, NoEscape(width))

        return

    def pdf_insert_table(self, column_headers=None, table_format=None, addfinalLine=True):

        # Set the table format
        if table_format is None:
            table_format = 'l' + 'c' * (len(column_headers) - 1)

        # Case we want to insert the table in a pdf
        if self.pdf_type != None:

            if self.pdf_type == 'table':
                self.pdfDoc.append(NoEscape(r'\begin{preview}'))

                # Initiate the table
                with self.pdfDoc.create(Tabu(table_format)) as self.table:
                    if column_headers != None:
                        self.table.add_hline()
                        self.table.add_row(map(str, column_headers), escape=False, strict=False)
                        if addfinalLine:
                            self.table.add_hline()

            elif self.pdf_type == 'longtable':

                # Initiate the table
                with self.pdfDoc.create(LongTable(table_format)) as self.table:
                    if column_headers != None:
                        self.table.add_hline()
                        self.table.add_row(map(str, column_headers), escape=False)
                        if addfinalLine:
                            self.table.add_hline()

        # Table .tex without preamble
        else:
            self.table = Tabu(table_format)
            if column_headers != None:
                self.table.add_hline()
                self.table.add_row(map(str, column_headers), escape=False)
                if addfinalLine:
                    self.table.add_hline()

    def pdf_insert_longtable(self, column_headers=None, table_format=None):

        # Set the table format
        if table_format is None:
            table_format = 'l' + 'c' * (len(column_headers) - 1)

        # Case we want to insert the table in a pdf
        if self.pdf_type != None:

            if self.pdf_type == 'table':
                self.pdfDoc.append(NoEscape(r'\begin{preview}'))

                # Initiate the table
            with self.pdfDoc.create(Tabu(table_format)) as self.table:
                if column_headers != None:
                    self.table.add_hline()
                    self.table.add_row(map(str, column_headers), escape=False)
                    self.table.add_hline()

                    # Table .tex without preamble
        else:
            self.table = LongTable(table_format)
            if column_headers != None:
                self.table.add_hline()
                self.table.add_row(map(str, column_headers), escape=False)
                self.table.add_hline()

    def addTableRow(self, input_row, row_format='auto', rounddig=4, rounddig_er=None, last_row=False):

        # Default formatting
        if row_format == 'auto':
            mapfunc = partial(self.format_for_table, rounddig=rounddig)
            output_row = map(mapfunc, input_row)

        # Append the row
        self.table.add_row(output_row, escape=False, strict=False)

        # Case of the final row just add one line
        if last_row:
            self.table.add_hline()

    def format_for_table(self, entry, rounddig=4, rounddig_er=2, scientific_notation=False, nan_format='-'):

        if rounddig_er == None: #TODO declare a universal tool
            rounddig_er = rounddig

        # Check None entry
        if entry != None:

            # Check string entry
            if isinstance(entry, (str, bytes)):
                formatted_entry = entry

            # Case of Numerical entry
            else:

                # Case of an array
                scalarVariable = True
                if isinstance(entry, (Sequence, np.ndarray)):

                    # Confirm is not a single value array
                    if len(entry) == 1:
                        entry = entry[0]
                    # Case of an array
                    else:
                        scalarVariable = False
                        formatted_entry = '_'.join(entry)  # we just put all together in a "_' joined string

                # Case single scalar
                if scalarVariable:

                    # Case with error quantified # TODO add uncertainty protocol for table
                    # if isinstance(entry, UFloat):
                    #     formatted_entry = round_sig(nominal_values(entry), rounddig,
                    #                                 scien_notation=scientific_notation) + r'$\pm$' + round_sig(
                    #         std_devs(entry), rounddig_er, scien_notation=scientific_notation)

                    # Case single float
                    if np.isnan(entry):
                        formatted_entry = nan_format

                    # Case single float
                    else:
                        formatted_entry = numberStringFormat(entry, rounddig)
        else:
            # None entry is converted to None
            formatted_entry = 'None'

        return formatted_entry

    def fig_to_pdf(self, label=None, fig_loc='htbp', width=r'1\textwidth', add_page=False, *args, **kwargs):

        with self.pdfDoc.create(Figure(position=fig_loc)) as plot:
            plot.add_plot(width=NoEscape(width), placement='h', *args, **kwargs)

            if label is not None:
                plot.add_caption(label)

        if add_page:
            self.pdfDoc.append(NewPage())

    def generate_pdf(self, clean_tex=True, output_address=None):
        if output_address == None:
            if self.pdf_type == 'table':
                self.pdfDoc.append(NoEscape(r'\end{preview}'))
                # self.pdfDoc.generate_pdf(clean_tex = clean_tex) # TODO this one does not work in windows
            self.pdfDoc.generate_pdf(clean_tex=clean_tex, compiler='pdflatex')
        else:
            self.table.generate_tex(output_address)

        return


class MCOutputDisplay(FigConf, PdfPrinter):

    def __init__(self):

        # Classes with plotting tools
        FigConf.__init__(self)
        PdfPrinter.__init__(self)

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

    def emissivitySurfaceFit_3D(self, line_label, emisCoeffs, emisGrid, funcEmis, te_ne_grid):

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
        x_values = te_ne_grid[0].reshape((self.denRange.size, self.tempRange.size))
        y_values = te_ne_grid[1].reshape((self.denRange.size, self.tempRange.size))
        ax.plot_surface(x_values, y_values, emisGrid.reshape((self.denRange.size, self.tempRange.size)), color='g',
                        alpha=0.5)

        # Plotting emissivity parametrization
        fit_points = funcEmis(te_ne_grid, *emisCoeffs)
        ax.scatter(te_ne_grid[0], te_ne_grid[1], fit_points, color='r', alpha=0.5)

        # Add labels
        ax.update({'ylabel': 'Density ($cm^{-3}$)', 'xlabel': 'Temperature $(K)$', 'title': line_label})

        return

    def traces_plot(self, traces_list, stats_dic):

        # Remove operations from the parameters list
        traces = traces_list[
            [i for i, v in enumerate(traces_list) if ('_Op' not in v) and ('_log__' not in v) and ('w_i' not in v)]]

        # Number of traces to plot
        n_traces = len(traces)

        # Declare figure format
        size_dict = {'figure.figsize': (14, 20), 'axes.titlesize': 26, 'axes.labelsize': 24, 'legend.fontsize': 18}
        self.FigConf(plotSize=size_dict, Figtype='Grid', n_columns=1, n_rows=n_traces)

        # Generate the color map
        self.gen_colorList(0, n_traces)

        # Plot individual traces
        for i in range(n_traces):

            # Current trace
            trace_code = traces[i]
            trace_array = stats_dic[trace_code]['trace']

            # Label for the plot
            mean_value = stats_dic[trace_code]['mean']
            std_dev = stats_dic[trace_code]['standard deviation']
            if mean_value > 0.001:
                label = r'{} = ${}$ $\pm${}'.format(self.labels_latex_dic[trace_code], np.round(mean_value, 4),
                                                    np.round(std_dev, 4))
            else:
                label = r'{} = ${:.3e}$ $\pm$ {:.3e}'.format(self.labels_latex_dic[trace_code], mean_value, std_dev)

            # Plot the data
            self.Axis[i].plot(trace_array, label=label, color=self.get_color(i))
            self.Axis[i].axhline(y=mean_value, color=self.get_color(i), linestyle='--')
            self.Axis[i].set_ylabel(self.labels_latex_dic[trace_code])

            if i < n_traces - 1:
                self.Axis[i].set_xticklabels([])

            # Add legend
            self.legend_conf(self.Axis[i], loc=2)

        return

    def tracesPosteriorPlot(self, params_list, stats_dic, idx_region=0, true_values=None):

        # Remove operations from the parameters list # TODO addapt this line to discremenate better
        traces_list = stats_dic.keys()
        region_ext = f'_{idx_region}'
        #traces = [item for item in params_list if item in traces_list]
        #traces = [item for item in params_list if (item in traces_list) or (item + region_ext in traces_list)]

        traces, traceTrueValuse = [], {}
        for param_name in params_list:

            paramExt_name = param_name + region_ext
            if param_name in stats_dic:
                ref_name = param_name
            elif paramExt_name in stats_dic:
                ref_name = paramExt_name
            traces.append(ref_name)

            if param_name in true_values:
                traceTrueValuse[ref_name] = true_values[param_name]

        # Number of traces to plot
        n_traces = len(traces)

        # Declare figure format
        size_dict = {'axes.titlesize': 20, 'axes.labelsize': 20, 'legend.fontsize': 10, 'xtick.labelsize':8, 'ytick.labelsize':8}
        rcParams.update(size_dict)
        fig = plt.figure(figsize=(8, n_traces))

        # # Generate the color map
        colorNorm, cmap = self.gen_colorList(0, n_traces)
        gs = gridspec.GridSpec(n_traces * 2, 4)
        gs.update(wspace=0.2, hspace=1.8)

        # Loop through the parameters and print the traces
        for i in range(n_traces):

            # Creat figure axis
            axTrace = fig.add_subplot(gs[2 * i:2 * (1 + i), :3])
            axPoterior = fig.add_subplot(gs[2 * i:2 * (1 + i), 3])

            # Current trace
            trace_code = traces[i]
            trace_array = stats_dic[trace_code]
            print(i, trace_code)

            # Label for the plot
            mean_value = np.mean(stats_dic[trace_code])
            std_dev = np.std(stats_dic[trace_code])
            traceLatexRef = trace_code.replace(region_ext, '')

            if mean_value > 0.001:
                label = r'{} = ${}$ $\pm${}'.format(latex_labels[traceLatexRef], np.round(mean_value, 4), np.round(std_dev, 4))
            else:
                label = r'{} = ${:.3e}$ $\pm$ {:.3e}'.format(latex_labels[traceLatexRef], mean_value, std_dev)

            # Plot the traces
            axTrace.plot(trace_array, label=label, color=cmap(colorNorm(i)))
            axTrace.axhline(y=mean_value, color=cmap(colorNorm(i)), linestyle='--')
            axTrace.set_ylabel(latex_labels[traceLatexRef])

            # Plot the histograms
            axPoterior.hist(trace_array, bins=50, histtype='step', color=cmap(colorNorm(i)), align='left')

            # Plot the axis as percentile
            median, percentile16th, percentile84th = np.median(trace_array), np.percentile(trace_array, 16), np.percentile(trace_array, 84)

            # Add true value if available
            if true_values is not None:
                if trace_code in traceTrueValuse:
                    value_param = traceTrueValuse[trace_code]
                    print(trace_code, value_param)
                    if isinstance(value_param, (list, tuple, np.ndarray)):
                        nominal_value, std_value = value_param[0], 0.0 if len(value_param) == 1 else value_param[1]
                        axPoterior.axvline(x=nominal_value, color=cmap(colorNorm(i)), linestyle='solid')
                        axPoterior.axvspan(nominal_value - std_value, nominal_value + std_value, alpha=0.5, color=cmap(colorNorm(i)))
                    else:
                        nominal_value = value_param
                        axPoterior.axvline(x=nominal_value, color=cmap(colorNorm(i)), linestyle='solid')

            # Add legend
            axTrace.legend(loc=7)

            # Remove ticks and labels
            if i < n_traces - 1:
                axTrace.get_xaxis().set_visible(False)
                axTrace.set_xticks([])

            axPoterior.yaxis.set_major_formatter(plt.NullFormatter())
            axPoterior.set_yticks([])

            axPoterior.set_xticks([percentile16th, median, percentile84th])
            axPoterior.set_xticklabels(['',numberStringFormat(median),''])
            axTrace.set_yticks((percentile16th, median, percentile84th))
            axTrace.set_yticklabels((numberStringFormat(percentile16th), '', numberStringFormat(percentile84th)))

        return

    def posteriors_plot(self, traces_list, stats_dic):

        # Remove operations from the parameters list
        traces = traces_list[[i for i, v in enumerate(traces_list) if ('_Op' not in v) and ('_log__' not in v) and ('w_i' not in v)]]

        # Number of traces to plot
        n_traces = len(traces)

        # Declare figure format
        size_dict = {'figure.figsize': (14, 20), 'axes.titlesize': 22, 'axes.labelsize': 22, 'legend.fontsize': 14}
        self.FigConf(plotSize=size_dict, Figtype='Grid', n_columns=1, n_rows=n_traces)

        # Generate the color map
        self.gen_colorList(0, n_traces)

        # Plot individual traces
        for i in range(len(traces)):

            # Current trace
            trace_code = traces[i]
            mean_value = stats_dic[trace_code]['mean']
            trace_array = stats_dic[trace_code]['trace']

            # Plot HDP limits
            HDP_coords = stats_dic[trace_code]['95% HPD interval']
            for HDP in HDP_coords:

                if mean_value > 0.001:
                    label_limits = 'HPD interval: {} - {}'.format(numberStringFormat(HDP_coords[0], 4),
                                                                  numberStringFormat(HDP_coords[1], 4))
                    label_mean = 'Mean value: {}'.format(numberStringFormat(mean_value, 4))
                else:
                    label_limits = 'HPD interval: {:.3e} - {:.3e}'.format(HDP_coords[0], HDP_coords[1])
                    label_mean = 'Mean value: {:.3e}'.format(mean_value)

                self.Axis[i].axvline(x=HDP, label=label_limits, color='grey', linestyle='dashed')

            self.Axis[i].axvline(x=mean_value, label=label_mean, color='grey', linestyle='solid')
            self.Axis[i].hist(trace_array, histtype='stepfilled', bins=35, alpha=.7, color=self.get_color(i),
                              normed=False)

            # Add true value if available
            if 'true_value' in stats_dic[trace_code]:
                value_true = stats_dic[trace_code]['true_value']
                if value_true is not None:
                    label_true = 'True value {:.3e}'.format(value_true)
                    self.Axis[i].axvline(x=value_true, label=label_true, color='black', linestyle='solid')

            # Figure wording
            self.Axis[i].set_ylabel(self.labels_latex_dic[trace_code])
            self.legend_conf(self.Axis[i], loc=2)

    def fluxes_distribution(self, lines_list, ions_list, function_key, db_dict, obsFluxes=None, obsErr=None):

        # Declare plot grid size
        n_columns = 3
        n_lines = len(lines_list)
        n_rows = int(np.ceil(float(n_lines)/float(n_columns)))

        # Declare figure format
        size_dict = {'figure.figsize': (9, 22), 'axes.titlesize': 14, 'axes.labelsize': 10, 'legend.fontsize': 10,
                     'xtick.labelsize': 8, 'ytick.labelsize': 3}
        self.FigConf(plotSize=size_dict, Figtype='Grid', n_columns=n_columns, n_rows=n_rows)

        # Generate the color dict
        self.gen_colorList(0, 10)
        colorDict = dict(H1r=0, O2=1, O3=2, N2=3, S2=4, S3=5, Ar3=6, Ar4=7, He1r=8, He2r=9)

        # Flux statistics
        traces_array = db_dict[function_key]
        median_values = median(db_dict[function_key], axis=0)
        p16th_fluxes = percentile(db_dict[function_key], 16, axis=0)
        p84th_fluxes = percentile(db_dict[function_key], 84, axis=0)

        # Plot individual traces
        for i in range(n_lines):

            # Current line
            label = lines_list[i]
            trace = traces_array[:, i]
            median_flux = median_values[i]

            label_mean = 'Mean value: {}'.format(round_sig(median_flux, 4))
            self.Axis[i].hist(trace, histtype='stepfilled', bins=35, alpha=.7, color=self.get_color(colorDict[ions_list[i]]), normed=False)

            if obsFluxes is not None:
                true_value, fitErr = obsFluxes[i], obsErr[i]
                label_true = 'True value: {}'.format(round_sig(true_value, 3))
                self.Axis[i].axvline(x=true_value, label=label_true, color='black', linestyle='solid')
                self.Axis[i].axvspan(true_value - fitErr, true_value + fitErr, alpha=0.5, color='grey')
                self.Axis[i].get_yaxis().set_visible(False)
                self.Axis[i].set_yticks([])

            # Plot wording
            self.Axis[i].set_title(r'{}'.format(self.linesDb.loc[label, 'latex_code']))

        return

    def acorr_plot(self, traces_list, stats_dic, n_columns=4, n_rows=2):

        # Remove operations from the parameters list
        traces = traces_list[
            [i for i, v in enumerate(traces_list) if ('_Op' not in v) and ('_log__' not in v) and ('w_i' not in v)]]

        # Number of traces to plot
        n_traces = len(traces)

        # Declare figure format
        size_dict = {'figure.figsize': (14, 14), 'axes.titlesize': 20, 'legend.fontsize': 10}
        self.FigConf(plotSize=size_dict, Figtype='Grid', n_columns=n_columns, n_rows=n_rows)

        # Generate the color map
        self.gen_colorList(0, n_traces)

        # Plot individual traces
        for i in range(n_traces):

            # Current trace
            trace_code = traces[i]

            label = self.labels_latex_dic[trace_code]

            trace_array = stats_dic[trace_code]['trace']

            if trace_code != 'ChiSq':
                maxlags = min(len(trace_array) - 1, 100)
                self.Axis[i].acorr(x=trace_array, color=self.get_color(i), detrend=detrend_mean, maxlags=maxlags)

            else:
                # Apano momentaneo
                chisq_adapted = reshape(trace_array, len(trace_array))
                maxlags = min(len(chisq_adapted) - 1, 100)
                self.Axis[i].acorr(x=chisq_adapted, color=self.get_color(i), detrend=detrend_mean, maxlags=maxlags)

            self.Axis[i].set_xlim(0, maxlags)
            self.Axis[i].set_title(label)

        return

    def corner_plot(self, params_list, stats_dic, true_values=None):

        # Remove operations from the parameters list
        traces_list = stats_dic.keys()
        traces = [item for item in params_list if item in traces_list]

        # Number of traces to plot
        n_traces = len(traces)

        # Set figure conf
        sizing_dict = {}
        sizing_dict['figure.figsize'] = (14, 14)
        sizing_dict['legend.fontsize'] = 30
        sizing_dict['axes.labelsize'] = 30
        sizing_dict['axes.titlesize'] = 15
        sizing_dict['xtick.labelsize'] = 12
        sizing_dict['ytick.labelsize'] = 12

        rcParams.update(sizing_dict)

        # Reshape plot data
        list_arrays, labels_list = [], []
        for trace_code in traces:
            trace_array = stats_dic[trace_code]
            list_arrays.append(trace_array)
            if trace_code == 'Te':
                labels_list.append(r'$T_{low}$')
            else:
                labels_list.append(latex_labels[trace_code])
        traces_array = np.array(list_arrays).T

        # # Reshape true values
        # true_values_list = [None] * len(traces)
        # for i in range(len(traces)):
        #     reference = traces[i] + '_true'
        #     if reference in true_values:
        #         value_param = true_values[reference]
        #         if isinstance(value_param, (list, tuple, np.ndarray)):
        #             true_values_list[i] = value_param[0]
        #         else:
        #             true_values_list[i] = value_param
        #
        # # Generate the plot
        # self.Fig = corner.corner(traces_array[:, :], fontsize=30, labels=labels_list, quantiles=[0.16, 0.5, 0.84],
        #                          show_titles=True, title_args={"fontsize": 200}, truths=true_values_list,
        #                          truth_color='#ae3135', title_fmt='0.3f')

        # Generate the plot
        self.Fig = corner.corner(traces_array[:, :], fontsize=30, labels=labels_list, quantiles=[0.16, 0.5, 0.84],
                                 show_titles=True, title_args={"fontsize": 200},
                                 truth_color='#ae3135', title_fmt='0.3f')

        return

    def table_mean_outputs(self, table_address, db_dict, true_values=None):

        # Table headers
        headers = ['Parameter', 'True value', 'Mean', 'Standard deviation', 'Number of points', 'Median',
                   r'$16^{th}$ percentil', r'$84^{th}$ percentil', r'Difference $\%$']

        # Generate pdf
        self.create_pdfDoc(table_address, pdf_type='table')
        self.pdf_insert_table(headers)

        # Loop around the parameters
        parameters_list = list(db_dict.keys())

        for param in parameters_list:

            if ('_Op' not in param) and param not in ['w_i']:
                print(param)
                # Label for the plot
                label       = latex_labels[param]
                mean_value  = np.mean(db_dict[param])
                std         = np.std(db_dict[param])
                n_traces    = db_dict[param].size
                median      = np.median(db_dict[param])
                p_16th      = np.percentile(db_dict[param], 16)
                p_84th      = np.percentile(db_dict[param], 84)

                true_value, perDif = 'None', 'None'
                if param + '_true' in true_values:
                    value_param = true_values[param + '_true']
                    if isinstance(value_param, (list, tuple, np.ndarray)):
                        true_value = value_param[0]
                    else:
                        true_value = value_param

                    perDif = (1 - (true_value / median)) * 100

                self.addTableRow([label, true_value, mean_value, std, n_traces, median, p_16th, p_84th, perDif],
                                 last_row=False if parameters_list[-1] != param else True)

        self.generate_pdf(clean_tex=True)
        # self.generate_pdf(output_address=table_address)

        return

    def table_line_fluxes(self, table_address, lines_list, function_key, db_dict, true_data=None):

        # Generate pdf
        self.create_pdfDoc(table_address, pdf_type='table')

        # Table headers
        headers = ['Line Label', 'Observed flux', 'Mean', 'Standard deviation', 'Median', r'$16^{th}$ $percentil$',
                   r'$84^{th}$ $percentil$', r'$Difference\,\%$']
        self.pdf_insert_table(headers)

        # Data for table
        true_values = ['None'] * len(lines_list) if true_data is None else true_data
        mean_line_values = db_dict[function_key].mean(axis=0)
        std_line_values = db_dict[function_key].std(axis=0)
        median_line_values = median(db_dict[function_key], axis=0)
        p16th_line_values = percentile(db_dict[function_key], 16, axis=0)
        p84th_line_values = percentile(db_dict[function_key], 84, axis=0)
        diff_Percentage = ['None'] * len(lines_list) if true_data is None else (1 - (median_line_values / true_values)) * 100

        for i in range(len(lines_list)):

            label = label_formatting(lines_list[i])

            row_i = [label, true_values[i], mean_line_values[i], std_line_values[i], median_line_values[i], p16th_line_values[i],
                     p84th_line_values[i], diff_Percentage[i]]

            self.addTableRow(row_i, last_row=False if lines_list[-1] != lines_list[i] else True)

        self.generate_pdf(clean_tex=True)

# class MCMC_printer(Basic_plots, Basic_tables):
#
#     def __init__(self):
#
#         # Supporting classes
#         Basic_plots.__init__(self)
#         Basic_tables.__init__(self)
#
#     def plot_emisFits(self, linelabels, emisCoeffs_dict, emisGrid_dict, output_folder):
#
#         te_ne_grid = (self.tempGridFlatten, self.denGridFlatten)
#
#         for i in range(linelabels.size):
#             lineLabel = linelabels[i]
#             print('--Fitting surface', lineLabel)
#
#             # 2D Comparison between PyNeb values and the fitted equation
#             self.emissivitySurfaceFit_2D(lineLabel, emisCoeffs_dict[lineLabel], emisGrid_dict[lineLabel],
#                                          self.ionEmisEq[lineLabel], te_ne_grid, self.denRange, self.tempRange)
#
#             output_address = '{}{}_{}_temp{}-{}_den{}-{}'.format(output_folder, 'emissivityTeDe2D', lineLabel,
#                                                                 self.tempGridFlatten[0], self.tempGridFlatten[-1],
#                                                                 self.denGridFlatten[0], self.denGridFlatten[-1])
#
#             self.savefig(output_address, resolution=200)
#             plt.clf()
#
#             # # 3D Comparison between PyNeb values and the fitted equation
#             # self.emissivitySurfaceFit_3D(lineLabel, emisCoeffs_dict[lineLabel], emisGrid_dict[lineLabel],
#             #                              self.ionEmisEq[lineLabel], te_ne_grid)
#             #
#             # output_address = '{}{}_{}_temp{}-{}_den{}-{}'.format(output_folder, 'emissivityTeDe3D', lineLabel,
#             #                                                      self.tempGridFlatten[0], self.tempGridFlatten[-1],
#             #                                                      self.denGridFlatten[0], self.denGridFlatten[-1])
#             # self.savefig(output_address, resolution=200)
#             # plt.clf()
#
#         return
#
#     def plot_emisRatioFits(self, diagnoslabels, emisCoeffs_dict, emisGrid_array, output_folder):
#
#         # Temperature and density meshgrids
#         X, Y = np.meshgrid(self.tem_grid_range, self.den_grid_range)
#         XX, YY = X.flatten(), Y.flatten()
#         te_ne_grid = (XX, YY)
#
#         for i in range(diagnoslabels.size):
#             lineLabel = diagnoslabels[i]
#             print('--Fitting surface', lineLabel)
#
#             # 2D Comparison between PyNeb values and the fitted equation
#             self.emissivitySurfaceFit_2D(lineLabel, emisCoeffs_dict[lineLabel], emisGrid_array[:, i],
#                                          self.EmisRatioEq_fit[lineLabel], te_ne_grid)
#
#             output_address = '{}{}_{}'.format(output_folder, 'emissivityTeDe2D', lineLabel)
#             self.savefig(output_address, resolution=200)
#             plt.clf()
#
#             # 3D Comparison between PyNeb values and the fitted equation
#             self.emissivitySurfaceFit_3D(lineLabel, emisCoeffs_dict[lineLabel], emisGrid_array[:, i],
#                                          self.EmisRatioEq_fit[lineLabel], te_ne_grid)
#
#             output_address = '{}{}_{}'.format(output_folder, 'emissivityTeDe3D', lineLabel)
#             self.savefig(output_address, resolution=200)
#             plt.clf()
#
#         return
#
#     def plotOuputData(self, database_address, db_dict, model_params):
#
#         if self.stellarCheck:
#             self.continuumFit(db_dict)
#             self.savefig(database_address + '_ContinuumFit', resolution=200)
#
#         if self.emissionCheck:
#
#             # Table mean values
#             print('-- Model parameters table')
#             self.table_mean_outputs(database_address + '_meanOutput', db_dict, self.obj_data)
#
#             # Line fluxes values
#             print('-- Line fluxes table')
#             self.table_line_fluxes(database_address + '_LineFluxes', self.lineLabels, 'calcFluxes_Op', db_dict, true_data=self.obsLineFluxes)
#             self.fluxes_distribution(self.lineLabels, self.lineIons, 'calcFluxes_Op', db_dict, obsFluxes=self.obsLineFluxes, obsErr=self.fitLineFluxErr)
#             self.savefig(database_address + '_LineFluxesPosteriors', resolution=200)
#
#             # Traces and Posteriors
#             print('-- Model parameters posterior diagram')
#             self.tracesPosteriorPlot(model_params, db_dict)
#             self.savefig(database_address + '_ParamsTracesPosterios', resolution=200)
#
#             # Corner plot
#             print('-- Scatter plot matrix')
#             self.corner_plot(model_params, db_dict, self.obj_data)
#             self.savefig(database_address + '_CornerPlot', resolution=50)
#
#         return
