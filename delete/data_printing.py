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

def int_to_roman(num):
    i, roman_num = 0, ''
    while num > 0:
        for _ in range(num // VAL_LIST[i]):
            roman_num += SYB_LIST[i]
            num -= VAL_LIST[i]
        i += 1
    return roman_num


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


