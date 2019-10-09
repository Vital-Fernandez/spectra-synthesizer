import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams, colors, cm
from data_printing import latex_labels
import src.specsyzer as ss

def gen_colorList(vmin=0.0, vmax=1.0, color_palette=None):

    colorNorm = colors.Normalize(vmin, vmax)
    cmap = cm.get_cmap(name=color_palette)
    # return certain color
    # self.cmap(self.colorNorm(idx))
    return colorNorm, cmap



# Search for the data in the default user folder
n_objs = 10
data_folder = 'C:/Users/Vital/Documents/IFU_results'
output_db = f'{data_folder}/{n_objs}IFU_fitting_db'

# # Declare sampler
# obj1_model = ss.SpectraSynthesizer()

candidate_params = ['n_e', 'T_low', 'T_high', 'cHbeta', 'tau', 'He1r', 'He2r',
                    'Ar3', 'Ar4', 'N2', 'O2', 'O3', 'S2', 'S3']
# ç

with open(output_db, 'rb') as trace_restored:
    db = pickle.load(trace_restored)

trace = db['trace']
trace_params = trace.varnames

# Extract database traces in dictionary ordered by regions
n_traces = 0
dict_traces = {}
list_params = []
for item in candidate_params:
    if item in trace_params:
        n_traces += 1
        list_params.append(item)
        for idx_region in range(n_objs):
            item_region = item + f'_{idx_region}'
            dict_traces[item_region] = trace[item][:, idx_region]
    else:
        if item + '_0' in trace_params:
            n_traces += 1
            list_params.append(item)
            for idx_region in range(n_objs):
                item_region = item + f'_{idx_region}'
                dict_traces[item_region] = trace[item_region]

# Read true data
true_data = {}
for idx_region in range(n_objs):

    linesLogAddress = f'{data_folder}/{n_objs}IFU_region{idx_region}_linesLog.txt'
    simulationData_file = f'{data_folder}/{n_objs}IFU_region{idx_region}_config.txt'
    objParams = ss.loadConfData(simulationData_file)
    region_ext = f'_{idx_region}'

    for item in candidate_params:
        if item +'_true' in objParams:
            true_data[item + region_ext] = objParams[item +'_true']

# Figure configuration
size_dict = {'axes.titlesize': 15, 'axes.labelsize': 15, 'legend.fontsize': 10, 'xtick.labelsize': 10,
             'ytick.labelsize': 8, 'figure.figsize': (12, 12)}
rcParams.update(size_dict)
ncols = 2

# Generate the color map
colorNorm, cmap = gen_colorList(0, n_objs)

fig = plt.figure()
gs = gridspec.GridSpec(nrows=int(n_traces/ncols), ncols=ncols)

for idx_param in range(len(list_params)):
    param = list_params[idx_param]
    axes = fig.add_subplot(gs[idx_param])

    for idx_region in range(n_objs):
        extRegion = f'_{idx_region}'
        paramRegion = param + extRegion
        paramTrace = dict_traces[paramRegion]

        axes.hist(paramTrace, bins=50, histtype='step', color=cmap(colorNorm(idx_region)), align='left')

        axes.axvline(x=true_data[paramRegion], color=cmap(colorNorm(idx_region)), linestyle='solid')

        if param == 'He2r':
            print(idx_region, true_data[paramRegion], paramTrace.mean)

        axes.yaxis.set_major_formatter(plt.NullFormatter())
        axes.set_yticks([])
        axes.set_ylabel(latex_labels[param])

plt.tight_layout()
plt.show()



# fig = plt.figure()
# spec = fig.add_gridspec(ncols=1, nrows=n_traces)
#
# for idx_param in range(len(list_params)):
#     param = list_params[idx_param]
#     axes = fig.add_subplot(spec[idx_param, 0])
#
#     for idx_region in range(n_objs):
#         extRegion = f'_{idx_region}'
#         paramRegion = param + extRegion
#         paramTrace = dict_traces[paramRegion]
#
#         axes.hist(paramTrace, bins=50, histtype='step', color=cmap(colorNorm(idx_region)), align='left')
#
#         axes.axvline(x=true_data[paramRegion], color=cmap(colorNorm(idx_region)), linestyle='solid')
#
#         axes.yaxis.set_major_formatter(plt.NullFormatter())
#         axes.set_yticks([])
#         axes.set_ylabel(latex_labels[param])
#
# plt.tight_layout()
# plt.show()

# Generate figure
# gs = gridspec.GridSpec(n_traces, 1, width_ratios=[3, 1])
# fig, axes = plt.subplots(n_traces, 1)
# axes = axes.ravel()

# fig, axes = plt.subplots(n_traces, 1)
#
# print(dict_traces.keys())
# print(n_traces)
#
# for idx_param in range(len(list_params)):
#     param = list_params[idx_param]
#     for idx_region in range(n_objs):
#         extRegion = f'_{idx_region}'
#         paramRegion = param + extRegion
#         paramTrace = dict_traces[paramRegion]
#
#         axes[idx_param].hist(paramTrace, bins=50, histtype='step', color=cmap(colorNorm(idx_region)), align='left')
#
#         axes[idx_param].yaxis.set_major_formatter(plt.NullFormatter())
#         axes[idx_param].set_yticks([])
#         axes[idx_param].set_ylabel(latex_labels[param])
#
# plt.tight_layout()
# plt.show()
