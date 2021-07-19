import corner
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib import rcParams
from matplotlib import colors
from matplotlib.mlab import detrend_mean
from pylatex import Document, Figure, NewPage, NoEscape, Package, Tabular, Tabularx, Section, Tabu, Table, LongTable, MultiColumn, MultiRow, utils
from functools import partial
from collections import Sequence
from scipy import stats as st

latex_labels = {'y_plus': r'$y^{+}$',
             'He1_abund': r'$y^{+}$',
             'He2_abund': r'$y^{++}$',
             'Te': r'$T_{e}$',
             'T_low': r'$T_{low}(K)$',
             'T_high': r'$T_{high}(K)$',
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
             'Cl3': r'$\frac{Cl^{2+}}{H^{+}}$',
             'Ne3': r'$\frac{Ne^{2+}}{H^{+}}$',
             'Fe3': r'$\frac{Fe^{2+}}{H^{+}}$',
             'N2': r'$\frac{N^{+}}{H^{+}}$',
             'Ar3': r'$\frac{Ar^{2+}}{H^{+}}$',
             'Ar4': r'$\frac{Ar^{3+}}{H^{+}}$',
             'Cl4': r'$\frac{Cl^{3+}}{H^{+}}$',
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
             'X_i+': r'$X^{i+}$',
             'log(X_i+)': r'$12+log\left(X^{i+}\right)$',
             'redNoise': r'$\Delta(cH\beta)$'}

VAL_LIST = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
SYB_LIST = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]


background_color = np.array((43, 43, 43))/255.0
foreground_color = np.array((179, 199, 216))/255.0
red_color = np.array((43, 43, 43))/255.0
yellow_color = np.array((191, 144, 0))/255.0

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


def label_formatting(line_label):
    label = line_label.replace('_', '\,\,')
    if label[-1] == 'A':
        label = label[0:-1] + '\AA'
    label = '$' + label + '$'

    return label


def int_to_roman(num):
    i, roman_num = 0, ''
    while num > 0:
        for _ in range(num // VAL_LIST[i]):
            roman_num += SYB_LIST[i]
            num -= VAL_LIST[i]
        i += 1
    return roman_num


def label_decomposition(input_lines, recombAtoms=('H1', 'He1', 'He2'), combined_dict={}, scalar_output=False, user_format={}):

    # Confirm input array has one dimension
    input_lines = np.array(input_lines, ndmin=1)

    # Containers for input data
    ion_dict, wave_dict, latexLabel_dict = {}, {}, {}

    for lineLabel in input_lines:
        if lineLabel not in user_format:
            # Check if line reference corresponds to blended component
            mixture_line = False
            if '_b' in lineLabel or '_m' in lineLabel:
                mixture_line = True
                if lineLabel in combined_dict:
                    lineRef = combined_dict[lineLabel]
                else:
                    lineRef = lineLabel[:-2]
            else:
                lineRef = lineLabel

            # Split the components if they exists
            lineComponents = lineRef.split('-')

            # Decomponse each component
            latexLabel = ''
            for line_i in lineComponents:

                # Get ion:
                if 'r_' in line_i: # Case recombination lines
                    ion = line_i[0:line_i.find('_')-1]
                else:
                    ion = line_i[0:line_i.find('_')]

                # Get wavelength and their units # TODO add more units and more facilities for extensions
                ext_n = line_i.count('_')
                if (line_i.endswith('A')) or (ext_n > 1):
                    wavelength = line_i[line_i.find('_') + 1:line_i.rfind('A')]
                    units = '\AA'
                    ext = f'-{line_i[line_i.rfind("_")+1:]}' if ext_n > 1 else ''
                else:
                    wavelength = line_i[line_i.find('_') + 1:]
                    units = ''
                    ext = ''

                # Get classical ion notation
                atom, ionization = ion[:-1], int(ion[-1])
                ionizationRoman = int_to_roman(ionization)

                # Define the label
                if ion in recombAtoms:
                    comp_Label = wavelength + units + '\,' + atom + ionizationRoman + ext
                else:
                    comp_Label = wavelength + units + '\,' + '[' + atom + ionizationRoman + ']' + ext

                # In the case of a mixture line we take the first entry as the reference
                if mixture_line:
                    if len(latexLabel) == 0:
                        ion_dict[lineRef] = ion
                        wave_dict[lineRef] = float(wavelength)
                        latexLabel += comp_Label
                    else:
                        latexLabel += '+' + comp_Label

                # This logic will expand the blended lines, but the output list will be larger than the input one
                else:
                    ion_dict[line_i] = ion
                    wave_dict[line_i] = float(wavelength)
                    latexLabel_dict[line_i] = '$'+comp_Label+'$'

            if mixture_line:
                latexLabel_dict[lineRef] = '$'+latexLabel +'$'

        else:
            ion_dict[lineLabel], wave_dict[lineLabel], latexLabel_dict[lineLabel] = user_format[lineLabel]

    # Convert to arrays
    label_array = np.array([*ion_dict.keys()], ndmin=1)
    ion_array = np.array([*ion_dict.values()], ndmin=1)
    wavelength_array = np.array([*wave_dict.values()], ndmin=1)
    latexLabel_array = np.array([*latexLabel_dict.values()], ndmin=1)

    assert label_array.size == wavelength_array.size, 'Output ions do not match wavelengths size'
    assert label_array.size == latexLabel_array.size, 'Output ions do not match labels size'

    if ion_array.size == 1 and scalar_output:
        return ion_array[0], wavelength_array[0], latexLabel_array[0]
    else:
        return ion_array, wavelength_array, latexLabel_array


def _histplot_bins(column, bins=100):
    """Helper to get bins for histplot."""
    col_min = np.min(column)
    col_max = np.max(column)
    return range(col_min, col_max + 2, max((col_max - col_min) // bins, 1))


def numberStringFormat(value, cifras = 4):
    if value > 0.001:
        newFormat = f'{value:.{cifras}f}'
    else:
        newFormat = f'{value:.{cifras}e}'

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


def format_for_table(entry, rounddig=4, rounddig_er=2, scientific_notation=False, nan_format='-'):

    if rounddig_er == None: #TODO declare a universal tool
        rounddig_er = rounddig

    # Check None entry
    if entry != None:

        # Check string entry
        if isinstance(entry, (str, bytes)):
            formatted_entry = entry

        elif isinstance(entry, (MultiColumn, MultiRow, utils.NoEscape)):
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

        plt.savefig(str(output_address) + extension, dpi=resolution, bbox_inches='tight')

        return


class PdfPrinter():

    def __init__(self):

        self.pdf_type = None
        self.pdf_geometry_options = {'right': '1cm',
                                     'left': '1cm',
                                     'top': '1cm',
                                     'bottom': '2cm'}
        self.table = None

        # TODO add dictionary with numeric formats for tables depending on the variable

    def create_pdfDoc(self, pdf_type=None, geometry_options=None, document_class=u'article'):

        # TODO integrate this into the init
        # Case for a complete .pdf or .tex
        if pdf_type is not None:

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
            self.pdfDoc = Document(documentclass=document_class, geometry_options=self.pdf_geometry_options)

            if pdf_type == 'table':
                self.pdfDoc.packages.append(Package('preview', options=['active', 'tightpage', ]))
                self.pdfDoc.packages.append(Package('hyperref', options=['unicode=true', ]))
                self.pdfDoc.append(NoEscape(r'\pagenumbering{gobble}'))
                self.pdfDoc.packages.append(Package('nicefrac'))
                self.pdfDoc.packages.append(Package('siunitx'))
                self.pdfDoc.packages.append(Package('makecell'))
                self.pdfDoc.packages.append(Package('color', options=['usenames', 'dvipsnames', ]))  # Package to crop pdf to a figure
                self.pdfDoc.packages.append(Package('colortbl', options=['usenames', 'dvipsnames', ]))  # Package to crop pdf to a figure
                self.pdfDoc.packages.append(Package('xcolor', options=['table']))

            elif pdf_type == 'longtable':
                self.pdfDoc.append(NoEscape(r'\pagenumbering{gobble}'))


        return

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

    def pdf_insert_table(self, column_headers=None, table_format=None, addfinalLine=True, color_font=None, color_background=None):

        # Set the table format
        if table_format is None:
            table_format = 'l' + 'c' * (len(column_headers) - 1)

        # Case we want to insert the table in a pdf
        if self.pdf_type != None:

            if self.pdf_type == 'table':
                self.pdfDoc.append(NoEscape(r'\begin{preview}'))

                # Initiate the table
                with self.pdfDoc.create(Tabular(table_format)) as self.table:
                    if column_headers != None:
                        self.table.add_hline()
                        # self.table.add_row(list(map(str, column_headers)), escape=False, strict=False)
                        output_row = list(map(partial(format_for_table), column_headers))

                        if color_font is not None:
                            for i, item in enumerate(output_row):
                                output_row[i] = NoEscape(r'\color{{{}}}{}'.format(color_font, item))

                        if color_background is not None:
                            for i, item in enumerate(output_row):
                                output_row[i] = NoEscape(r'\cellcolor{{{}}}{}'.format(color_background, item))

                        self.table.add_row(output_row, escape=False, strict=False)
                        if addfinalLine:
                            self.table.add_hline()

            elif self.pdf_type == 'longtable':

                # Initiate the table
                with self.pdfDoc.create(LongTable(table_format)) as self.table:
                    if column_headers != None:
                        self.table.add_hline()
                        self.table.add_row(list(map(str, column_headers)), escape=False)
                        if addfinalLine:
                            self.table.add_hline()

        # Table .tex without preamble
        else:
            self.table = Tabu(table_format)
            if column_headers != None:
                self.table.add_hline()
                # self.table.add_row(list(map(str, column_headers)), escape=False, strict=False)
                output_row = list(map(partial(format_for_table), column_headers))
                self.table.add_row(output_row, escape=False, strict=False)
                if addfinalLine:
                    self.table.add_hline()



            # self.table = Tabu(table_format)
            # if column_headers != None:
            #     self.table.add_hline()
            #     self.table.add_row(list(map(str, column_headers)), escape=False)
            #     if addfinalLine:
            #         self.table.add_hline()

        return

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
                self.table.add_row(list(map(str, column_headers)), escape=False)
                self.table.add_hline()

    def addTableRow(self, input_row, row_format='auto', rounddig=4, rounddig_er=None, last_row=False, color_font=None,
                    color_background=None):

        # Default formatting
        if row_format == 'auto':
            output_row = list(map(partial(format_for_table, rounddig=rounddig), input_row))

        if color_font is not None:
            for i, item in enumerate(output_row):
                output_row[i] = NoEscape(r'\color{{{}}}{}'.format(color_font, item))

        if color_background is not None:
            for i, item in enumerate(output_row):
                output_row[i] = NoEscape(r'\cellcolor{{{}}}{}'.format(color_background, item))

        # Append the row
        self.table.add_row(output_row, escape=False, strict=False)

        # Case of the final row just add one line
        if last_row:
            self.table.add_hline()

    def fig_to_pdf(self, label=None, fig_loc='htbp', width=r'1\textwidth', add_page=False, *args, **kwargs):

        with self.pdfDoc.create(Figure(position=fig_loc)) as plot:
            plot.add_plot(width=NoEscape(width), placement='h', *args, **kwargs)

            if label is not None:
                plot.add_caption(label)

        if add_page:
            self.pdfDoc.append(NewPage())

    def generate_pdf(self, output_address, clean_tex=True):

        if self.pdf_type is None:
            self.table.generate_tex(str(output_address))

        else:
            if self.pdf_type == 'table':
                self.pdfDoc.append(NoEscape(r'\end{preview}'))
            self.pdfDoc.generate_pdf(filepath=str(output_address), clean_tex=clean_tex, compiler='pdflatex')

        # if output_address == None:
        #     if self.pdf_type == 'table':
        #         self.pdfDoc.append(NoEscape(r'\end{preview}'))
        #         # self.pdfDoc.generate_pdf(clean_tex = clean_tex) # TODO this one does not work in windows
        #     self.pdfDoc.generate_pdf(clean_tex=clean_tex, compiler='pdflatex')
        # else:
        #     self.table.generate_tex(output_address)

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

    def tracesPosteriorPlot(self, plot_address, params_list, traces_dict, true_values=None, plot_conf={}):

        if true_values is not None:
            trace_true_dict = {}
            for param in params_list:
                if param in true_values:
                    trace_true_dict[param] = true_values[param]
        n_traces = len(params_list)

        # Plot format
        size_dict = {'axes.titlesize': 14, 'axes.labelsize': 14, 'legend.fontsize': 10,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8}
        size_dict.update(plot_conf)
        rcParams.update(size_dict)

        fig = plt.figure(figsize=(8, n_traces))
        colorNorm, cmap = self.gen_colorList(0, n_traces)
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

        return

    def tracesPriorPostComp(self, params_list, stats_dic, idx_region=0, true_values=None):

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
        background = np.array((43, 43, 43)) / 255.0
        foreground = np.array((179, 199, 216)) / 255.0

        figConf = {'text.color': foreground,
                   'figure.figsize': (10, 10),
                   'figure.facecolor': background,
                   'axes.facecolor': background,
                   'axes.edgecolor': foreground,
                   'axes.labelcolor': foreground,
                   'axes.labelsize': 18,
                   'xtick.labelsize': 16,
                   'ytick.labelsize': 16,
                   'xtick.color': foreground,
                   'ytick.color': foreground,
                   'legend.edgecolor': 'inherit',
                   'legend.facecolor': 'inherit',
                   'legend.fontsize': 16,
                   'legend.loc': "center right"}
        rcParams.update(figConf)

        fig = plt.figure()
        ax = fig.add_subplot()

        # Generate the color map
        colorNorm, cmap = self.gen_colorList(0, n_traces)
        gs = gridspec.GridSpec(n_traces * 2, 4)
        gs.update(wspace=0.2, hspace=1.8)

        i = 2
        trace_code = traces[i]
        trace_array = stats_dic[trace_code]
        traceLatexRef = trace_code.replace(region_ext, '')
        print(i, trace_code, latex_labels[traceLatexRef])

        priorTrace = np.random.normal(15000.0, 5000.0, size=trace_array.size)
        ax.hist(trace_array, label= '$T_{high}$ posterior', bins=50, histtype='step', color=cmap(colorNorm(i)), align='left')
        ax.hist(priorTrace, label='$T_{high}$ prior', bins=50, histtype='step', color=foreground, align='left')
        ax.legend()
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Counts')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # plt.show()


        # # Loop through the parameters and print the traces
        # for i in range(n_traces):
        #
        #     # Creat figure axis
        #     axTrace = fig.add_subplot(gs[2 * i:2 * (1 + i), :3])
        #     axPoterior = fig.add_subplot(gs[2 * i:2 * (1 + i), 3])
        #
        #     # Current trace
        #     trace_code = traces[i]
        #     trace_array = stats_dic[trace_code]
        #     print(i, trace_code)
        #
        #     # Label for the plot
        #     mean_value = np.mean(stats_dic[trace_code])
        #     std_dev = np.std(stats_dic[trace_code])
        #     traceLatexRef = trace_code.replace(region_ext, '')
        #
        #     if mean_value > 0.001:
        #         label = r'{} = ${}$ $\pm${}'.format(latex_labels[traceLatexRef], np.round(mean_value, 4), np.round(std_dev, 4))
        #     else:
        #         label = r'{} = ${:.3e}$ $\pm$ {:.3e}'.format(latex_labels[traceLatexRef], mean_value, std_dev)
        #
        #     # Plot the traces
        #     axTrace.plot(trace_array, label=label, color=cmap(colorNorm(i)))
        #     axTrace.axhline(y=mean_value, color=cmap(colorNorm(i)), linestyle='--')
        #     axTrace.set_ylabel(latex_labels[traceLatexRef])
        #
        #     # Plot the histograms
        #     axPoterior.hist(trace_array, bins=50, histtype='step', color=cmap(colorNorm(i)), align='left')
        #
        #     # Plot the axis as percentile
        #     median, percentile16th, percentile84th = np.median(trace_array), np.percentile(trace_array, 16), np.percentile(trace_array, 84)
        #
        #     # Add true value if available
        #     if true_values is not None:
        #         if trace_code in traceTrueValuse:
        #             value_param = traceTrueValuse[trace_code]
        #             print(trace_code, value_param)
        #             if isinstance(value_param, (list, tuple, np.ndarray)):
        #                 nominal_value, std_value = value_param[0], 0.0 if len(value_param) == 1 else value_param[1]
        #                 axPoterior.axvline(x=nominal_value, color=cmap(colorNorm(i)), linestyle='solid')
        #                 axPoterior.axvspan(nominal_value - std_value, nominal_value + std_value, alpha=0.5, color=cmap(colorNorm(i)))
        #             else:
        #                 nominal_value = value_param
        #                 axPoterior.axvline(x=nominal_value, color=cmap(colorNorm(i)), linestyle='solid')
        #
        #     # Add legend
        #     axTrace.legend(loc=7)
        #
        #     # Remove ticks and labels
        #     if i < n_traces - 1:
        #         axTrace.get_xaxis().set_visible(False)
        #         axTrace.set_xticks([])
        #
        #     axPoterior.yaxis.set_major_formatter(plt.NullFormatter())
        #     axPoterior.set_yticks([])
        #
        #     axPoterior.set_xticks([percentile16th, median, percentile84th])
        #     axPoterior.set_xticklabels(['',numberStringFormat(median),''])
        #     axTrace.set_yticks((percentile16th, median, percentile84th))
        #     axTrace.set_yticklabels((numberStringFormat(percentile16th), '', numberStringFormat(percentile84th)))

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

    def fluxes_distribution(self, plot_address, input_lines, inFlux, inErr, trace_dict, n_columns=8, combined_dict={},
                            plot_conf={}, user_labels={}):

        # Input data
        ion_array, wave_array, latexLabel_array = label_decomposition(input_lines, combined_dict=combined_dict,
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
        colorNorm, cmap = self.gen_colorList(0, obsIons.size)
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
                chisq_adapted = np.reshape(trace_array, len(trace_array))
                maxlags = min(len(chisq_adapted) - 1, 100)
                self.Axis[i].acorr(x=chisq_adapted, color=self.get_color(i), detrend=detrend_mean, maxlags=maxlags)

            self.Axis[i].set_xlim(0, maxlags)
            self.Axis[i].set_title(label)

        return

    def corner_plot(self, plot_address, params_list, traces_dict, true_values=None):

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

    def table_mean_outputs(self, table_address, parameter_list, trace_dict, true_values=None, file_type='table'):

        # Table headers
        headers = ['Parameter', 'Mean', 'Standard deviation', 'Number of points', 'Median',
                   r'$16^{th}$ percentil', r'$84^{th}$ percentil']

        if true_values is not None:
            headers.insert(1, 'True value')
            headers.append(r'Difference $\%$')

        # Generate containers
        self.create_pdfDoc(pdf_type=file_type)
        self.pdf_insert_table(headers)
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

            self.addTableRow(row_i, last_row=False if parameter_list[-1] != param else True)
            tableDF.loc[row_i[0]] = row_i[1:]

        self.generate_pdf(output_address=table_address)

        # Save the table as a dataframe.
        with open(f'{table_address}.txt', 'wb') as output_file:
            string_DF = tableDF.to_string()
            output_file.write(string_DF.encode('UTF-8'))

        return

    def table_line_fluxes(self, table_address, input_lines, inFlux, inErr, traces_dict, combined_dict={}, file_type='table',
                          user_labels={}):

        # Table headers
        headers = ['Line', 'Observed flux', 'Mean', 'Standard deviation', 'Median', r'$16^{th}$ $percentil$',
                   r'$84^{th}$ $percentil$', r'$Difference\,\%$']

        # Create containers
        tableDF = pd.DataFrame(columns=headers[1:])
        self.create_pdfDoc(pdf_type=file_type)
        self.pdf_insert_table(headers)

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

        ion_array, wave_array, latexLabel_array = label_decomposition(input_lines, combined_dict=combined_dict, user_format=user_labels)

        for i in range(inFlux.size):

            # label = label_formatting(inputLabels[i])
            flux_obs = r'${:0.3}\pm{:0.3}$'.format(inFlux[i], inErr[i])

            row_i = [latexLabel_array[i], flux_obs, mean_line_values[i], std_line_values[i], median_line_values[i],
                     p16th_line_values[i], p84th_line_values[i], diff_Percentage[i]]

            self.addTableRow(row_i, last_row=False if input_lines[-1] != input_lines[i] else True)
            tableDF.loc[input_lines[i]] = row_i[1:]

        self.generate_pdf(table_address)

        # Save the table as a dataframe.
        with open(f'{table_address}.txt', 'wb') as output_file:
            string_DF = tableDF.to_string()
            output_file.write(string_DF.encode('UTF-8'))
