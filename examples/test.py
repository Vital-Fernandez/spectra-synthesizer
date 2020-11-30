# First, we have to be sure that the packages are in default mode
#
# Within pyraf, write
# unlearn gscrspec
# unlearn gspecshift
# unlearn findgaps
# unlearn fndblocks
# unlearn qecorr
# unlearn ifuproc
# unlearn gfreduce
# unlearn gfextract
# unlearn gftransform
# unlearn gfskysub
# unlearn gfbkgsub
# unlearn gscombine
# unlearn gscombine_mod
# unlearn gfapsum
# unlearn gsappwave
# unlearn hselect
# unlearn gdisplay
# unlearn imcombine
# unlearn gsfquick
# unlearn scombine
#
# gemini
# unlearn gemini
# gemtools
# unlearn gemtools
# gmos
# unlearn gmos

# GMOSAIC and LACOS_SPEC must be in the scripts folder
# task gmosaic= "/Users/Dania/Documents/Proyectos/J0838-cubo/gemini_data/scripts/gmosaic.cl"
# task lacos_spec.cl= "/Users/Dania/Documents/Proyectos/J0838-cubo/gemini_data/scripts/lacos_spec.cl"
# set stdimage = imtgmos

# iraf.gmos.logfile = "J0838.log"
# iraf.gemtools.logfile = "J0838.log"
# iraf.gmos.gdisplay.fl_paste=yes

# Data processing from here
# CCD reduction for GMOS-IFU based on David Sanmartim's script (From Las Campanas Observtory)
import argparse
import os
import sys
import glob
import copy
import warnings

from IPython import get_ipython
from astropy import log

# Loading the required iraf packages
from pyraf import iraf
from pyraf.iraf import gemini, gemtools, gmos

__author__ = 'Dania Munoz'
__date__ = '2020-11'
__version__ = "0.1"
__email__ = "daniamunozv@gmail.com"


class Main:

    def __init__(self):

        # About warnings
        warnings.filterwarnings('ignore')
        log.propagate = False

        # Set variables used globally
        self.bias_list = []
        self.flat_list = []
        self.arc_list = []
        self.std_list = []
        self.sci_list = []

        self.masterbias = 'BIAS'
        self.masterflat = 'FLAT'

        self.instrument = 'GMOS-N'

        self.ccdbin = '1 1'

        self.grating = 'B1200+_%'
        # self.grating = 'R831+OG515'

        self.apmask = '1.0arcsec'

        self.object = 'J0838'

        self.std = 'Feige66'

        self.date_obs = '2020-01-23'

        self.data_obs_range = '2019-01-23:2020-01-25'

        self.centwave = 502.0
        # self.centwave = 700.0

        self.regions = ['Full']

        # Constructing Query Dictionary

        self.qd = {'Full': {'use_me': 1,
                            'Instrument': self.instrument,
                            'CcdBin': self.ccdbin,
                            'RoI': 'Full',
                            'Disperser': self.grating,
                            'CentWave': self.centwave,
                            'AperMask': self.apmask,
                            'Object': self.object,
                            'DateObs': self.date_obs}}

        # Taking some args from argparse method
        self.red_path = '../red'
        self.raw_path = '../raw'
        self.scripts_path = '../scripts'

        # Observing log database
        # Make a logfile containing the BIAS
        self.db_file = str(self.raw_path + '/obsLog.sqlite3')

        ### Parameter definitions
        #
        self.bias_flags = {
            'rawpath': self.raw_path, 'fl_over': 'yes', 'fl_trim': 'yes', 'biasrows': 'default', 'fl_inter': 'no',
            'order': 11, 'low_reject': 3.0, 'high_reject': 3.0, 'niterate': 5.0, 'fl_vardq': 'yes',
            'logfile': 'J0838.log', 'verbose': 'no'
        }

        self.flat_flags = {
            'fl_nodshuffle': 'no', 'fl_inter': 'yes', 'fl_vardq': 'yes', 'fl_addmdf': 'yes',
            'fl_over': 'yes', 'fl_trim': 'yes', 'fl_bias': 'yes', 'fl_gscrrej': 'no', 'fl_fulldq': 'yes',
            'rawpath': self.raw_path, 'mdfdir': self.scripts_path, 'biasrows': 'default', 'order': 11,
            'low_reject': 3.0, 'high_reject': 3.0, 'niterate': 5
        }

        # Science
        self.sci_flags = {
            'fl_nodshuffle': 'no', 'fl_inter': 'yes', 'fl_vardq': 'yes', 'fl_addmdf': 'yes',
            'fl_over': 'yes', 'fl_trim': 'yes', 'fl_bias': 'yes', 'fl_gscrrej': 'no', 'fl_extract': 'no',
            'fl_gsappwave': 'no', 'fl_wavtran': 'no', 'fl_skysub': 'no', 'fl_fluxcal': 'no', 'fl_fulldq': 'yes',
            'rawpath': self.raw_path, 'mdfdir': self.scripts_path, 'biasrows': 'default', 'order': 11,
            'low_reject': 3.0, 'high_reject': 3.0, 'niterate': 5
        }

        # Standard
        # self.std_flags = copy.deepcopy(self.sci_flags)
        # self.std_flags.update({'fl_fixpix': 'yes', 'fl_vardq': 'no', 'fl_fulldq': 'no',
        #                       'fl_gscrrej': 'yes', 'fl_crspec': 'no',
        #                       'cr_xorder': 9, 'cr_sigclip': 4.5, 'cr_sigfrac': 0.5, 'cr_objlim': 4.0, 'cr_niter': 4
        #                       })

        # Arc
        self.arc_flags = copy.deepcopy(self.sci_flags)

    def __call__(self, *args, **kwargs):

        # cleaning up the reduction dir
        self.clean_path(self.red_path, ext_list=['txt', 'log', 'lis'])

        # verify raw and red paths
        self.verify_paths(create_obs_log='True')

        # Creating master bias
        self.create_master_bias(raw_path=self.raw_path, db_file=self.db_file,
                                query_dict=self.qd, data_obs_range='*')

        # Creating master gcal flat
        self.create_gcalflat(raw_path=self.raw_path, db_file=self.db_file, query_dict=self.qd,
                             flat_flags=self.flat_flags, comb_flats='False', date_obs='*')

        # Reducing science observations
        self.reduce_sci(raw_path=self.raw_path, db_file=self.db_file, query_dict=self.qd,
                        sci_flags=self.sci_flags, date_obs='*')

        # Reducing std observations
        self.reduce_std(raw_path=self.raw_path, db_file=self.db_file, query_dict=self.qd,
                        std_flags=self.std_flags, date_obs='*')

        # Reducing acr observations
        self.reduce_arc(raw_path=self.raw_path, db_file=self.db_file, query_dict=self.qd,
                        arc_flags=self.arc_flags, date_obs='*')

        return

    def verify_paths(self, create_obs_log='True'):

        # not equal
        if self.raw_path == self.red_path:
            log.error('raw_path may not be equal to red_path')
        else:
            pass

        # create paths if they do not exist
        if not os.path.isdir(self.red_path):
            os.mkdir(self.red_path)

        if not os.path.isdir(self.raw_path):
            os.mkdir(self.raw_path)

        # go to red_path
        if os.getcwd() is not self.red_path:
            os.chdir(self.red_path)

        # Downloading python files
        dict_files = {'obslog.py': 'http://ast.noao.edu/sites/default/files/GMOS_Cookbook/_downloads/obslog.py',
                      'fileSelect.py': 'http://ast.noao.edu/sites/default/files/GMOS_Cookbook/_downloads/fileSelect.py'}

        # Test if python files already exist and download
        for file in dict_files.keys():
            try:
                if os.path.isfile(file) is False:
                    # print ('%s downloaded.' % (file))
                    os.system('wget %s .' % dict_files[file])
                else:
                    pass
                    # print ("%s already exist!" % (file))
            except:
                pass

        if create_obs_log == 'True':
            if os.getcwd() == self.red_path:
                os.system('cp obslog.py ' + self.raw_path)
                os.chdir(self.raw_path)
                log.info('Wait a minute... creating sqlite3 database file: obsLog.sqlite3')
                if os.path.isfile('obsLog.sqlite3') is True:
                    os.remove('obsLog.sqlite3')
                os.system('python obslog.py obsLog.sqlite3')
                log.info('Log file obsLog.sqlite3 has been created')
                os.chdir(self.red_path)
            else:
                log.warning('\n Script obslog.py has not been run at ' + self.raw_path + ' directory')

        if os.getcwd() == self.red_path:
            log.info('Downloading LACOSMIC...')
            if os.path.isfile('lacos_spec.cl') is False:
                os.system('wget http://www.astro.yale.edu/dokkum/lacosmic/lacos_spec.cl .')
            iraf.task(lacos_spec=self.red_path + '/lacos_spec.cl')

    @staticmethod
    def clean_path(path, ext_list):
        """
        Clean up files in a directoy. It's not recursive.
        """
        if os.path.exists(path):
            log.info('Cleaning up the data reduction directory')
            iraf.imdelete('*tmp*', verify='no')
            for ext in ext_list:
                for _file in glob.glob(os.path.join(path, '*.' + str(ext))):
                    os.remove(_file)

    def create_master_bias(self, raw_path, db_file, query_dict, data_obs_range):

        gemtools.gemextn.unlearn()  # Disarm a bug in gbias
        gmos.gbias.rawpath = self.raw_path

        bias_flags = {'logfile': 'J0838.log', 'rawpath': raw_path, 'fl_over': 'yes', 'fl_vardq': 'yes',
                      'verbose': 'yes'}

        for r in self.regions:

            # The following SQL generates the list of full-frame files to process.
            query_dict[r].update({'DateObs': data_obs_range})
            SQL = fs.createQuery('bias', query_dict[r])
            bias_files = fs.fileListQuery(db_file, SQL, query_dict[r])
            print
            bias_files
            if len(bias_files) > 1:
                with open('bias.list', 'w') as f:
                    [f.write(x + '\n') for x in bias_files]
                log.info('Creating master bias for calibration: ' + self.masterbias + r + '.fits\n')
                gmos.gbias('@bias.lis', self.masterbias + r, **bias_flags)

            # Removing auxiliary files created in this step
            os.remove('bias.lis')

            # gN for Norht and gS for South
            iraf.imdelete('g' + self.instrument[-1] + '*.fits', verify='no')


if __name__ == '__main__':

    # Parsing Arguments ---
    parser = argparse.ArgumentParser(description="PyGoodman CCD Reduction - CCD reductions for "
                                                 "Goodman spectroscopic data")

    parser.add_argument('raw_path', metavar='raw_path', type=str, nargs=1,
                        help="Full path to raw data (e.g. /home/jamesbond/GN-2017A-FT-19/RAW/)")

    parser.add_argument('red_path', metavar='red_path', type=str, nargs=1,
                        help="Full path to reduced data (e.g. /home/jamesbond/GN-2017A-FT-19/RED/RAW)")

    main = Main()
    main()

else:
    print('gmos_ls_reducrtion.py is not being executed as main.')

