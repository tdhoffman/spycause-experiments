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
xlabels = []
rowctr = 0
ols_data = data_refined[data_refined["model"] == "ols"]
for model_intf in model_intfs:
    conf_bias_matrix[rowctr, :] = ols_data[ols_data["w_i_model_form"] == model_intf]["bias"].values
    conf_var_matrix[rowctr, :] = ols_data[ols_data["w_i_model_form"] == model_intf]["tau_var"].values

    xlabels.append(f"{wgt_names[model_intf]}")
    if rowctr == 0:
        xlabels[rowctr] += "\nN\nOLS"
    rowctr += 1

sp_data = data_refined[data_refined["model"] != "ols"]
for model in ["car", "joint"]:
    for model_conf in sp_confs:
        for model_intf in model_intfs:
            conf_bias_matrix[rowctr, :] = sp_data[(sp_data["model"] == model) & (sp_data["w_i_model_form"] == model_intf) & (sp_data["w_c_model_form"] == model_conf)]["bias"].values
            conf_var_matrix[rowctr, :] = sp_data[(sp_data["model"] == model) & (sp_data["w_i_model_form"] == model_intf) & (sp_data["w_c_model_form"] == model_conf)]["tau_var"].values

            xlabels.append(f"{wgt_names[model_intf]}")
            if rowctr % 4 == 0:
                xlabels[rowctr] += f"\n{wgt_names[model_conf]}"
            if rowctr == 4:
                xlabels[rowctr] += "\nCAR"
            if rowctr == 16:
                xlabels[rowctr] += "\nJoint"
            rowctr += 1

intf_strs = ["N", "B", "D", "R"]
conf_strs = ["N", "B", "D", "R"]
sp_conf_strs = ["B", "D", "R"]
ylabels = []
for conf in conf_strs:
    for intf in intf_strs:
        if intf == "N":
            ylabels.append(conf + " " + intf)
        else:
            ylabels.append(intf)

# ylabels = [*(f"OLS ({intf}, N)" for intf in intf_strs)] + \
          # [*(f"CAR ({intf}, {conf})" for conf in sp_conf_strs for intf in intf_strs)] + \
          # [*(f"Joint ({intf}, {conf})" for conf in sp_conf_strs for intf in intf_strs)]

fig, ax = plt.subplots(nrows=2, sharex=True)
bias_im = ax[0].imshow(np.abs(conf_bias_matrix.T)/1.5, cmap="Reds")
# plt.colorbar(bias_im)
ax[0].set_yticks(list(range(len(ylabels))))
ax[0].set_yticklabels(ylabels)
#ax[0].xaxis.tick_top()
ax[0].set_xticks(list(range(len(xlabels))))
# ax[0].xaxis.set_ticks_position('none')
ax[0].set_xticklabels(xlabels)
ax[0].set_title("Bias:effect size")
ax[0].set_ylabel("Data scenarios")
# ax[0].set_xlabel("models (interference varies more)")
divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(bias_im, cax=cax, orientation='vertical')
ax[0].plot([3.5, 3.5], [-0.5, 15.5], 'k', linewidth=1)
ax[0].plot([15.5, 15.5], [-0.5, 15.5], 'k', linewidth=1)
plt.text(0.3425, 0.855, "Confounding", rotation=45, fontsize=10, transform=fig.transFigure)
plt.text(0.3525, 0.855, "Interference", rotation=45, fontsize=10, transform=fig.transFigure)

var_im = ax[1].imshow(np.log(conf_var_matrix.T), cmap="Blues")
# plt.colorbar()
ax[1].set_yticks(list(range(len(ylabels))))
ax[1].set_yticklabels(ylabels)
#ax[1].xaxis.tick_top()
ax[1].set_xticks(list(range(len(xlabels))))
ax[1].set_xticklabels(xlabels)
ax[1].set_title("Log variance")
ax[1].set_ylabel("Data scenarios")
ax[1].set_xlabel("Models")
divider = make_axes_locatable(ax[1])
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(var_im, cax=cax, orientation='vertical')
cbar.set_ticks([])
ax[1].plot([3.5, 3.5], [-0.5, 15.5], 'k', linewidth=1)
ax[1].plot([15.5, 15.5], [-0.5, 15.5], 'k', linewidth=1)
plt.text(0.3, 0.17, "Interference", fontsize=10, transform=fig.transFigure)
plt.text(0.3, 0.155, "Confounding", fontsize=10, transform=fig.transFigure)
plt.text(0.3, 0.14, "Model", fontsize=10, transform=fig.transFigure)
plt.tight_layout()
plt.show()
