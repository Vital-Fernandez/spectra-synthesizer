import numpy as np
import pandas as pd
from pathlib import Path
import src.specsiser as sr
import scipy as spy
import exoplanet as xo

import time
from astropy.io import fits
from astropy.table import Table
from astro.data.muse.common_methods import grid_columns
from astro.papers.gtc_greenpeas.common_methods import epm_HII_CHI_mistry


# Declare data and files location
file_conf = '/home/vital/PycharmProjects/vital_tests/astro/data/muse/muse_greenpeas.ini'
obsData = sr.loadConfData(file_conf, group_variables=False)
objList = obsData['data_location']['object_list']
fileList = obsData['data_location']['file_list']
fitsFolder = Path(obsData['data_location']['fits_folder'])
dataFolder = Path(obsData['data_location']['data_folder'])
resultsFolder = Path(obsData['data_location']['results_folder'])

z_objs = obsData['sample_data']['z_array']
pertil_array = obsData['sample_data']['percentil_array']
noise_region = obsData['sample_data']['noiseRegion_array']
norm_flux = obsData['sample_data']['norm_flux']

# Load the data
grid_file = Path('/home/vital/Dropbox/Astrophysics/Data/muse-Ricardo/Data/HII-CHI-mistry_1Myr_grid.csv')
grid_DF = pd.read_csv(grid_file, skiprows=1, names=grid_columns.values())
grid_DF.logNO = np.round(grid_DF.logNO.values, decimals=3)

# Remove carbon dimension
idcs_rows = grid_DF.carbon == 'O'
idcs_columns = ~grid_DF.columns.isin(['carbon'])
grid_3D_DF = grid_DF.loc[idcs_rows, idcs_columns]

model_variables = ['logOH', 'logU', 'logNO']
gw = sr.ModelGridWrapper()
grid_dict, axes_cords_a = gw.ndarray_from_DF(grid_3D_DF, axes_columns=model_variables)
grid_array, axes_cords_b = gw.ndarray_from_DF(grid_3D_DF, axes_columns=model_variables, dict_output=False)
grid_interpolators = gw.generate_xo_interpolators(grid_dict, model_variables, axes_cords_a, interp_type='point')
axis_interpolator = gw.generate_xo_interpolators(grid_dict, model_variables, axes_cords_a, interp_type='axis')

idx = 1
print('logOH', axes_cords_a['logOH'][idx])
print('logU', axes_cords_a['logU'][idx])
print('logNO', axes_cords_a['logNO'][idx])

print('Dictionary grid', grid_dict['O2_3729A'][idx, idx, idx])
print('array grid', grid_array[idx, idx, idx, 1])

data_grid = grid_dict['O2_3729A']
axes_container = (axes_cords_a['logOH'], axes_cords_a['logU'], axes_cords_a['logNO'])
coord = [[axes_cords_a['logOH'][idx], axes_cords_a['logU'][idx], axes_cords_a['logNO'][idx]]]

spy_RGridInterp = spy.interpolate.RegularGridInterpolator(axes_container, data_grid)
print('scipy', spy_RGridInterp(coord)[0])

exop_interpAxis = xo.interp.RegularGridInterpolator(axes_container, data_grid)
print('Exoplanet', exop_interpAxis.evaluate(coord).eval()[0])

xo_interp_gen = grid_interpolators['O2_3729A']
print('Exoplanet mine', xo_interp_gen(coord).eval())

print('Exoplanet mine Axis', axis_interpolator(coord).eval())




# --------------------------- 4 dimensinos fitting
# grid_file = Path('/home/vital/Dropbox/Astrophysics/Data/muse-Ricardo/Data/HII-CHI-mistry_1Myr_grid.csv')
# grid_DF = pd.read_csv(grid_file, skiprows=1, names=grid_columns.values())
# grid_DF.logNO = np.round(grid_DF.logNO.values, decimals=3)
#
# gw = sr.ModelGridWrapper()
# model_variables = ['logOH', 'logU', 'logNO', 'carbon']
# grid_dict, axes_cords_a = gw.ndarray_from_DF(grid_DF, axes_columns=model_variables)
# grid_array, axes_cords_b = gw.ndarray_from_DF(grid_DF, axes_columns=model_variables, dict_output=False)
#
# axes_cords_a['carbon'] = np.array([0, 1])
#
# idx = 1
# print('logOH', axes_cords_a['logOH'][idx])
# print('logU', axes_cords_a['logU'][idx])
# print('logNO', axes_cords_a['logNO'][idx])
# print('carbon', axes_cords_a['carbon'][idx])
#
# print('Dictionary grid', grid_dict['O2_3729A'][idx, idx, idx, idx])
# print('array grid', grid_array[idx, idx, idx, idx, 1])
#
# data_grid = grid_dict['O2_3729A']
# axes_container = (axes_cords_a['logOH'], axes_cords_a['logU'], axes_cords_a['logNO'], axes_cords_a['carbon'])
# coord = [[axes_cords_a['logOH'][idx], axes_cords_a['logU'][idx], axes_cords_a['logNO'][idx], axes_cords_a['carbon'][idx]]]
#
# coordB = np.stack(([axes_cords_a['logOH'][idx]],
#                    [axes_cords_a['logU'][idx]],
#                    [axes_cords_a['logNO'][idx]],
#                    [axes_cords_a['carbon'][idx]]), axis=-1)
#
# spy_RGridInterp = spy.interpolate.RegularGridInterpolator(axes_container, data_grid)
# print('scipy', spy_RGridInterp(coord))
#
# exop_interpAxis = xo.interp.RegularGridInterpolator(axes_container, data_grid)
# print('Exoplanet', exop_interpAxis.evaluate(coord).eval())


# grid_interpolators = gw.generate_xo_interpolators(grid_dict, model_variables, axes_cords_a, interp_type='point')

# print('New interpolator', grid_interpolators['O2_3729A'](coord).eval())


# Python program to demonstrate working of
# issubset().





# Halpha = np.power(10, grid_DF['H1_4341A'].values)
#
# w = np.unique(grid_DF.logZ.values)
# x = np.unique(grid_DF.logU.values)
# y = np.unique(grid_DF.logNO.values)
# z = np.unique(grid_DF.carbon.values)
#
# file_matrix = grid_DF['O2_3726A'].values
# file_mdimArray = file_matrix.reshape(len(w), len(x), len(y), len(z))
#
# idx_w = 2
# idx_x = 3
# idx_y = 15
# idx_z = 0
# print(w[idx_w], x[idx_x], y[idx_y], z[idx_z])
#
# print(file_mdimArray[idx_w, idx_x, idx_y, idx_z])

# import pandas as pd
# import pymysql
# import matplotlib.pyplot as plt
# import os
#
# os.environ['MdB_HOST'] = '3mdb.astro.unam.mx'
# os.environ['MdB_USER'] = 'oiii5007'
# os.environ['MdB_PASSWD'] = 'OVN_user'
# os.environ['MdB_PORT'] = '3306'
# os.environ['MdB_DB_17'] = '3MdB_17'
# os.environ['MdB_DBs'] = '3MdBs'
# os.environ['MdB_DBp'] = '3MdB'
#
# # co = pymysql.connect(host=os.environ['MdB_HOST'],
# #                      db=os.environ['MdB_DB_17'],
# #                      user=os.environ['MdB_USER'],
# #                      passwd=os.environ['MdB_PASSWD'])
# co = pymysql.connect(host='3mdb.astro.unam.mx', db='3MdB_17', user='OVN_user', passwd='oiii5007')
# # co = pymysql.connect(host='3mdb.astro.unam.mx',
# #                      db='3MdB_17',
# #                      user='oiii5007',
# #                      passwd='OVN_user')
# res = pd.read_sql("select log10(N__2_658345A/H__1_656281A) as n2, log10(O__3_500684A/H__1_486133A) as o3, OXYGEN as O from tab_17 where ref = 'BOND'", con=co)
# co.close()
#
# f, ax = plt.subplots()
# sc = ax.scatter(res['n2'], res['o3'], c=12+res['O'], edgecolor='')
# ax.set_xlabel("log [NII]/Ha")
# ax.set_ylabel("log [OIII]/Hb")
# cb = f.colorbar(sc, ax=ax)
# cb.set_label("12 + log O/H")
# plt.show()