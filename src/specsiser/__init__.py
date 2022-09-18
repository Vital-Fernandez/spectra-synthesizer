"""
SpecSyzer - python package for spectra synthesis of astronomical bodies
"""

import os
import sys
import configparser
from lime import load_cfg

from .inout import safeConfData, load_fit_results
from .models import EmissionTensors
from .components.extinction_model import flambda_calc, ExtinctionModel
from .grids import GridWrapper
from .treatment import emissivity_grid_calc, SpectraSynthesizer

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

# Get default configuration settings
_default_cfg = load_cfg(os.path.join(_dir_path, 'default.cfg'))


