import copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable


def highlight_cells(x, y, xwidth=1, ywidth=1, ax=None, **kwargs):
    rect = plt.Rectangle((x - .5, y - .5), xwidth, ywidth, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect


data = pd.read_json("../outputs/confounds_reference.json", orient="records")
data["bias"] = data["tau_med"] - data["treat"]

## CONFOUNDING MAJOR ORG
data_refined = data[["model", "w_i_data_form", "w_i_model_form", "w_c_data_form", "w_c_model_form", "tau_med", "tau_var", "bias"]]
mapper = {"none": 1, "binary": 2, "distance": 3, "region": 4}
data_refined["w_i_data_form"] = data_refined["w_i_data_form"].map(mapper)
data_refined["w_i_model_form"] = data_refined["w_i_model_form"].map(mapper)
data_refined["w_c_data_form"] = data_refined["w_c_data_form"].map(mapper)
data_refined["w_c_model_form"] = data_refined["w_c_model_form"].map(mapper)

data_refined.sort_values(by=["model", "w_c_model_form", "w_i_model_form", "w_c_data_form", "w_i_data_form"], inplace=True)

rev_mapper = {1: "none", 2: "binary", 3: "distance", 4: "region"}
data_refined["w_i_data_form"] = data_refined["w_i_data_form"].map(rev_mapper)
data_refined["w_i_model_form"] = data_refined["w_i_model_form"].map(rev_mapper)
data_refined["w_c_data_form"] = data_refined["w_c_data_form"].map(rev_mapper)
data_refined["w_c_model_form"] = data_refined["w_c_model_form"].map(rev_mapper)


conf_var_matrix = np.zeros((28, 16))
conf_bias_matrix = np.zeros((28, 16))
model_intfs = ["none", "binary", "distance", "region"]
wgt_names = {"none": "N", "binary": "B", "distance": "D", "region": "R"}
sp_confs = ["binary", "distance", "region"]

# OLS
ylabels = []
rowctr = 0
ols_data = data_refined[data_refined["model"] == "ols"]
for model_intf in model_intfs:
    conf_bias_matrix[rowctr, :] = ols_data[ols_data["w_i_model_form"] == model_intf]["bias"].values
    conf_var_matrix[rowctr, :] = ols_data[ols_data["w_i_model_form"] == model_intf]["tau_var"].values

    if rowctr == 0:
        ylabels.append("OLS N N")
    else:
        ylabels.append(f"{wgt_names[model_intf]}")
    rowctr += 1

sp_data = data_refined[data_refined["model"] != "ols"]
for model in ["car", "joint"]:
    for model_conf in sp_confs:
        for model_intf in model_intfs:
            conf_bias_matrix[rowctr, :] = sp_data[(sp_data["model"] == model) & (sp_data["w_i_model_form"] == model_intf) & (sp_data["w_c_model_form"] == model_conf)]["bias"].values
            conf_var_matrix[rowctr, :] = sp_data[(sp_data["model"] == model) & (sp_data["w_i_model_form"] == model_intf) & (sp_data["w_c_model_form"] == model_conf)]["tau_var"].values

            if rowctr == 4:
                ylabels.append("CAR B N")
            elif rowctr == 16:
                ylabels.append("Joint B N")
            elif rowctr % 4 == 0:
                ylabels.append(f"{wgt_names[model_conf]} {wgt_names[model_intf]}")
            else:
                ylabels.append(f"{wgt_names[model_intf]}")
            rowctr += 1

intf_strs = ["N", "B", "D", "R"]
conf_strs = ["N", "B", "D", "R"]
sp_conf_strs = ["B", "D", "R"]
xlabels = []
for conf in conf_strs:
    for intf in intf_strs:
        if intf == "N":
            xlabels.append(intf + "\n" + conf)
        else:
            xlabels.append(intf)

# ylabels = [*(f"OLS ({intf}, N)" for intf in intf_strs)] + \
          # [*(f"CAR ({intf}, {conf})" for conf in sp_conf_strs for intf in intf_strs)] + \
          # [*(f"Joint ({intf}, {conf})" for conf in sp_conf_strs for intf in intf_strs)]

fig, ax = plt.subplots(ncols=2, figsize=(12, 9), sharey=True)
bias_im = ax[0].imshow(np.abs(conf_bias_matrix)/1.5, cmap="Reds")
# plt.colorbar(bias_im)
ax[0].set_yticks(list(range(len(ylabels))))
ax[0].set_yticklabels(ylabels)
#ax[0].xaxis.tick_top()
ax[0].set_xticks(list(range(len(xlabels))))
# ax[0].xaxis.set_ticks_position('none')
ax[0].set_xticklabels(xlabels)
ax[0].set_title("(a): Bias to effect size ratio")
ax[0].set_xlabel("Data scenarios")
ax[0].set_ylabel("Models")
# ax[0].set_xlabel("models (interference varies more)")
divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(bias_im, cax=cax, orientation='vertical')
ax[0].plot([-0.5, 15.5], [3.5, 3.5], 'k', linewidth=1)
ax[0].plot([-0.5, 15.5], [15.5, 15.5], 'k', linewidth=1)
# highlight_cells(0, 4, xwidth=4, ywidth=12, ax=ax[0], color="black", zorder=2, linewidth=5) # 2
# highlight_cells(0, 16, xwidth=16, ywidth=8, ax=ax[0], color="black", zorder=2, linewidth=5) # 3
highlight_cells(0, 12, xwidth=16, ywidth=4, ax=ax[0], color="black", zorder=2, linewidth=5) # 4
highlight_cells(0, 24, xwidth=16, ywidth=4, ax=ax[0], color="black", zorder=2, linewidth=5) # 4

var_im = ax[1].imshow(np.log(conf_var_matrix), cmap="Blues")
# plt.colorbar()
ax[1].set_yticks(list(range(len(ylabels))))
ax[1].set_yticklabels(ylabels)
#ax[1].xaxis.tick_top()
ax[1].set_xticks(list(range(len(xlabels))))
ax[1].set_xticklabels(xlabels)
ax[1].set_title("(b): Log variance")
# ax[1].set_ylabel("Data scenarios")
ax[1].set_xlabel("Data scenarios")
divider = make_axes_locatable(ax[1])
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(var_im, cax=cax, orientation='vertical')
cbar.set_ticks([])
ax[1].plot([-0.5, 15.5], [3.5, 3.5], 'k', linewidth=1)
ax[1].plot([-0.5, 15.5], [15.5, 15.5], 'k', linewidth=1)
plt.text(0.075, 0.88, "Model", rotation=45, fontsize=10, transform=fig.transFigure)
plt.text(0.0925, 0.88, "Confounding", rotation=45, fontsize=10, transform=fig.transFigure)
plt.text(0.11, 0.88, "Interference", rotation=45, fontsize=10, transform=fig.transFigure)
plt.text(0.05, 0.0875, "Interference", fontsize=10, transform=fig.transFigure)
plt.text(0.05, 0.07, "Confounding", fontsize=10, transform=fig.transFigure)
# highlight_cells(0, 4, xwidth=4, ywidth=12, ax=ax[1], color="black", zorder=2, linewidth=5)
# highlight_cells(0, 16, xwidth=16, ywidth=8, ax=ax[1], color="black", zorder=2, linewidth=5)
highlight_cells(0, 12, xwidth=16, ywidth=4, ax=ax[1], color="black", zorder=2, linewidth=5)
highlight_cells(0, 24, xwidth=16, ywidth=4, ax=ax[1], color="black", zorder=2, linewidth=5)
# plt.text(0.50, 0.09, "Interference", fontsize=10, transform=fig.transFigure)
# plt.text(0.50, 0.075, "Confounding", fontsize=10, transform=fig.transFigure)
# plt.tight_layout()
plt.show()


## MASKED PLOTS FOR SLIDES: 1
# This is FAR from the best way to do this
biasmat = np.abs(conf_bias_matrix)/1.5
varmat = np.log(conf_var_matrix)
bvmin = biasmat.min()
bvmax = biasmat.max()
vvmin = varmat.min()
vvmax = varmat.max()

bias_cm = copy.copy(get_cmap("Reds"))
bias_cm.set_bad(color='lightgray')
var_cm = copy.copy(get_cmap("Blues"))
var_cm.set_bad(color='lightgray')

mask1 = biasmat.copy()
mask1_r = [0]*16 + [1]*16 + [2]*16 + [3]*16 + \
          [4]*12 + [5]*12 + [6]*12 + [7]*12 + \
          [8]*12 + [9]*12 + [10]*12 + [11]*12 + \
          [12]*12 + [13]*12 + [14]*12 + [15]*12 + \
          [16]*16 + [17]*16 + [18]*16 + [19]*16 + \
          [20]*16 + [21]*16 + [22]*16 + [23]*16 + \
          [24]*16 + [25]*16 + [26]*16 + [27]*16
mask1_c = 4*list(range(16)) + 12*list(range(4, 16)) + 12*list(range(16))
mask1[mask1_r, mask1_c] = np.nan

mask2 = biasmat.copy()
mask2_r = [0]*16 + [1]*16 + [2]*16 + [3]*16 + \
          [4]*16 + [5]*16 + [6]*16 + [7]*16 + \
          [8]*16 + [9]*16 + [10]*16 + [11]*16 + \
          [12]*16 + [13]*16 + [14]*16 + [15]*16 + \
          [24]*16 + [25]*16 + [26]*16 + [27]*16
mask2_c = 20*list(range(16))
mask2[mask2_r, mask2_c] = np.nan

mask3 = biasmat.copy()
mask3_r = [0]*16 + [1]*16 + [2]*16 + [3]*16 + \
          [4]*16 + [5]*16 + [6]*16 + [7]*16 + \
          [8]*16 + [9]*16 + [10]*16 + [11]*16 + \
          [16]*16 + [17]*16 + [18]*16 + [19]*16 + \
          [20]*16 + [21]*16 + [22]*16 + [23]*16
mask3_c = 20*list(range(16))
mask3[mask3_r, mask3_c] = np.nan


biasmat1 = np.ma.array(biasmat, mask=np.isnan(mask1))
biasmat2 = np.ma.array(biasmat, mask=np.isnan(mask2))
biasmat3 = np.ma.array(biasmat, mask=np.isnan(mask3))
varmat1 = np.ma.array(varmat, mask=np.isnan(mask1))
varmat2 = np.ma.array(varmat, mask=np.isnan(mask2))
varmat3 = np.ma.array(varmat, mask=np.isnan(mask3))

fig, ax = plt.subplots(ncols=2, figsize=(12, 9), sharey=True)
bias_im = ax[0].imshow(biasmat1, vmin=bvmin, vmax=bvmax, cmap=bias_cm)
ax[0].set_yticks(list(range(len(ylabels))))
ax[0].set_yticklabels(ylabels)
ax[0].set_xticks(list(range(len(xlabels))))
ax[0].set_xticklabels(xlabels)
ax[0].set_title("(a): Bias to effect size ratio")
ax[0].set_xlabel("Data scenarios")
ax[0].set_ylabel("Models")
divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(bias_im, cax=cax, orientation='vertical')
ax[0].plot([-0.5, 15.5], [3.5, 3.5], 'k', linewidth=1)
ax[0].plot([-0.5, 15.5], [15.5, 15.5], 'k', linewidth=1)

var_im = ax[1].imshow(varmat1, vmin=vvmin, vmax=vvmax, cmap=var_cm)
ax[1].set_yticks(list(range(len(ylabels))))
ax[1].set_yticklabels(ylabels)
ax[1].set_xticks(list(range(len(xlabels))))
ax[1].set_xticklabels(xlabels)
ax[1].set_title("(b): Log variance")
ax[1].set_xlabel("Data scenarios")
divider = make_axes_locatable(ax[1])
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(var_im, cax=cax, orientation='vertical')
cbar.set_ticks([])
ax[1].plot([-0.5, 15.5], [3.5, 3.5], 'k', linewidth=1)
ax[1].plot([-0.5, 15.5], [15.5, 15.5], 'k', linewidth=1)
plt.text(0.075, 0.88, "Model", rotation=45, fontsize=10, transform=fig.transFigure)
plt.text(0.0925, 0.88, "Confounding", rotation=45, fontsize=10, transform=fig.transFigure)
plt.text(0.11, 0.88, "Interference", rotation=45, fontsize=10, transform=fig.transFigure)
plt.text(0.05, 0.0875, "Interference", fontsize=10, transform=fig.transFigure)
plt.text(0.05, 0.07, "Confounding", fontsize=10, transform=fig.transFigure)
plt.show()


## MASK 2

fig, ax = plt.subplots(ncols=2, figsize=(12, 9), sharey=True)
bias_im = ax[0].imshow(biasmat2, vmin=bvmin, vmax=bvmax, cmap=bias_cm)
ax[0].set_yticks(list(range(len(ylabels))))
ax[0].set_yticklabels(ylabels)
ax[0].set_xticks(list(range(len(xlabels))))
ax[0].set_xticklabels(xlabels)
ax[0].set_title("(a): Bias to effect size ratio")
ax[0].set_xlabel("Data scenarios")
ax[0].set_ylabel("Models")
divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(bias_im, cax=cax, orientation='vertical')
ax[0].plot([-0.5, 15.5], [3.5, 3.5], 'k', linewidth=1)
ax[0].plot([-0.5, 15.5], [15.5, 15.5], 'k', linewidth=1)

var_im = ax[1].imshow(varmat2, vmin=vvmin, vmax=vvmax, cmap=var_cm)
ax[1].set_yticks(list(range(len(ylabels))))
ax[1].set_yticklabels(ylabels)
ax[1].set_xticks(list(range(len(xlabels))))
ax[1].set_xticklabels(xlabels)
ax[1].set_title("(b): Log variance")
ax[1].set_xlabel("Data scenarios")
divider = make_axes_locatable(ax[1])
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(var_im, cax=cax, orientation='vertical')
cbar.set_ticks([])
ax[1].plot([-0.5, 15.5], [3.5, 3.5], 'k', linewidth=1)
ax[1].plot([-0.5, 15.5], [15.5, 15.5], 'k', linewidth=1)
plt.text(0.075, 0.88, "Model", rotation=45, fontsize=10, transform=fig.transFigure)
plt.text(0.0925, 0.88, "Confounding", rotation=45, fontsize=10, transform=fig.transFigure)
plt.text(0.11, 0.88, "Interference", rotation=45, fontsize=10, transform=fig.transFigure)
plt.text(0.05, 0.0875, "Interference", fontsize=10, transform=fig.transFigure)
plt.text(0.05, 0.07, "Confounding", fontsize=10, transform=fig.transFigure)
plt.show()


## MASK 3

fig, ax = plt.subplots(ncols=2, figsize=(12, 9), sharey=True)
bias_im = ax[0].imshow(biasmat3, vmin=bvmin, vmax=bvmax, cmap=bias_cm)
ax[0].set_yticks(list(range(len(ylabels))))
ax[0].set_yticklabels(ylabels)
ax[0].set_xticks(list(range(len(xlabels))))
ax[0].set_xticklabels(xlabels)
ax[0].set_title("(a): Bias to effect size ratio")
ax[0].set_xlabel("Data scenarios")
ax[0].set_ylabel("Models")
divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(bias_im, cax=cax, orientation='vertical')
ax[0].plot([-0.5, 15.5], [3.5, 3.5], 'k', linewidth=1)
ax[0].plot([-0.5, 15.5], [15.5, 15.5], 'k', linewidth=1)

var_im = ax[1].imshow(varmat3, vmin=vvmin, vmax=vvmax, cmap=var_cm)
ax[1].set_yticks(list(range(len(ylabels))))
ax[1].set_yticklabels(ylabels)
ax[1].set_xticks(list(range(len(xlabels))))
ax[1].set_xticklabels(xlabels)
ax[1].set_title("(b): Log variance")
ax[1].set_xlabel("Data scenarios")
divider = make_axes_locatable(ax[1])
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(var_im, cax=cax, orientation='vertical')
cbar.set_ticks([])
ax[1].plot([-0.5, 15.5], [3.5, 3.5], 'k', linewidth=1)
ax[1].plot([-0.5, 15.5], [15.5, 15.5], 'k', linewidth=1)
plt.text(0.075, 0.88, "Model", rotation=45, fontsize=10, transform=fig.transFigure)
plt.text(0.0925, 0.88, "Confounding", rotation=45, fontsize=10, transform=fig.transFigure)
plt.text(0.11, 0.88, "Interference", rotation=45, fontsize=10, transform=fig.transFigure)
plt.text(0.05, 0.0875, "Interference", fontsize=10, transform=fig.transFigure)
plt.text(0.05, 0.07, "Confounding", fontsize=10, transform=fig.transFigure)
plt.show()
