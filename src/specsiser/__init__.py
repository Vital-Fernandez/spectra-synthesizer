"""
SpecSyzer - python package for spectra synthesis of astronomical bodies
"""

import os
import sys
import pandas as pd
import configparser


# Get python version being used
__python_version__ = sys.version_info

# Get specsiser setup configuration
_dir_path = os.path.dirname(os.path.realpath(__file__))
setup_path = os.path.abspath(os.path.join(_dir_path, os.path.join(os.pardir, os.pardir)))
_setup_cfg = configparser.ConfigParser()
_setup_cfg.optionxform = str
_setup_cfg.read(os.path.join(setup_path, 'setup.cfg'))

# Read specsiser version
__version__ = _setup_cfg['metadata']['version']

# Load package libraries
from .data_reading import *
from .physical_model.extinction_model import ExtinctionModel
from .physical_model.atomic_model import IonEmissivity, compute_emissivity_grid
from .physical_model.chemical_model import DirectMethod, TOIII_TSIII_relation
from .physical_model.gasEmission_functions import EmissionTensors, assignFluxEq2Label,\
    gridInterpolatorFunction, EmissionFluxModel
from inference_model import SpectraSynthesizer
from .physical_model.line_tools import EmissionFitting, LineMesurer, label_decomposition
from .data.spectra_files import import_fits_data
from .print import plot

# Get default configuration settings
_default_cfg = loadConfData(os.path.join(_dir_path, 'config.ini'))

# Declare default data folder
_literatureDataFolder = os.path.join(_dir_path, _default_cfg['external_data_folder'])

# Load library databases
linesDatabasePath = os.path.join(_literatureDataFolder, _default_cfg['lines_data_file'])
_linesDb = pd.read_excel(linesDatabasePath, sheet_name=0, header=0, index_col=0)



