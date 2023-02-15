import copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

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


conf_bias_matrix = np.zeros((28, 16))
model_intfs = ["none", "binary", "distance", "region"]
sp_confs = ["binary", "distance", "region"]

# OLS
ylabels = []
rowctr = 0
ols_data = data_refined[data_refined["model"] == "ols"]
for model_intf in model_intfs:
    conf_bias_matrix[rowctr, :] = ols_data[ols_data["w_i_model_form"] == model_intf]["bias"].values
    ylabels.append(f"OLS ({model_intf}, none)")
    rowctr += 1

sp_data = data_refined[data_refined["model"] != "ols"]
for model in ["car", "joint"]:
    for model_conf in sp_confs:
        for model_intf in model_intfs:
            conf_bias_matrix[rowctr, :] = sp_data[(sp_data["model"] == model) & (sp_data["w_i_model_form"] == model_intf) & (sp_data["w_c_model_form"] == model_conf)]["bias"].values

            if model == "car":
                ylabels.append(f"CAR ({model_intf}, {model_conf})")
            if model == "joint":
                ylabels.append(f"Joint ({model_intf}, {model_conf})")
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

_, ax = plt.subplots()
plt.imshow(np.abs(conf_bias_matrix)/1.5, cmap="Reds")
plt.colorbar()
ax.set_yticks(list(range(28)))
ax.set_yticklabels(ylabels)
#ax.xaxis.tick_top()
ax.set_xticks(list(range(16)))
ax.set_xticklabels(xlabels)
plt.title("Bias:effect size (interference, confounding)")
plt.xlabel("data scenarios (interference varies more)")
plt.ylabel("models")
plt.tight_layout()
plt.show()


## Make a version of this that has everything grayed out except matched scenarios
matched_inds_r = list(range(28))
matched_inds_c = list(range(16)) + list(range(4, 16))
unmatched_conf_bias = conf_bias_matrix.copy()
unmatched_conf_bias[matched_inds_r, matched_inds_c] = np.nan
unmatched_conf_bias = np.ma.array(unmatched_conf_bias, mask=np.isnan(unmatched_conf_bias))
matched_conf_bias = np.ma.array(conf_bias_matrix, mask=~unmatched_conf_bias.mask)

vmin = 0  # (np.abs(conf_bias_matrix)/1.5).min()
vmax = 1.5  # (np.abs(conf_bias_matrix)/1.5).max()
fig, ax = plt.subplots(ncols=2)
cm = copy.copy(get_cmap("hot"))
cm.set_bad(color='lightgray')
ax[0].imshow(np.abs(unmatched_conf_bias)/1.5, cmap=cm, vmin=vmin, vmax=vmax)
# plt.colorbar()
ax[0].set_yticks(list(range(28)))
ax[0].set_yticklabels(ylabels)
#ax[0].xaxis.tick_top()
ax[0].set_xticks(list(range(16)))
ax[0].set_xticklabels(xlabels)
ax[0].set_title("Bias:effect size among unmatched scenarios (interference, confounding)")
ax[0].set_xlabel("data scenarios (interference varies more)")
ax[0].set_ylabel("models")

im = ax[1].imshow(np.abs(matched_conf_bias)/1.5, cmap=cm, vmin=vmin, vmax=vmax)
# plt.colorbar()
ax[1].set_yticks(list(range(28)))
ax[1].set_yticklabels(ylabels)
#ax[1].xaxis.tick_top()
ax[1].set_xticks(list(range(16)))
ax[1].set_xticklabels(xlabels)
ax[1].set_title("Bias:effect size among matched scenarios (interference, confounding)")
ax[1].set_xlabel("data scenarios (interference varies more)")
ax[1].set_ylabel("models")
# plt.tight_layout()
fig.colorbar(im, ax=ax.ravel().tolist())
plt.show()


## Descriptives
bias_ratio_matrix = np.abs(conf_bias_matrix)/1.5
(bias_ratio_matrix >= 0.5).sum()
np.median(bias_ratio_matrix)
bias_ratio_matrix.max()
bias_ratio_matrix.min()

matched_bias_ratio = np.abs(matched_conf_bias)/1.5
unmatched_bias_ratio = np.abs(unmatched_conf_bias)/1.5
matched_bias_ratio.max()
np.ma.median(matched_bias_ratio)
matched_bias_ratio.min()
unmatched_bias_ratio.max()
np.ma.median(unmatched_bias_ratio)
unmatched_bias_ratio.min()

_, ax = plt.subplots(ncols=2)
sns.histplot(matched_bias_ratio.flatten(), ax=ax[0])
sns.histplot(unmatched_bias_ratio.flatten(), ax=ax[1])
plt.show()

plt.boxplot([matched_bias_ratio[~matched_bias_ratio.mask], unmatched_bias_ratio[~unmatched_bias_ratio.mask]], vert=False)
plt.show()

_, ax = plt.subplots()
sns.boxplot(data=[matched_bias_ratio[~matched_bias_ratio.mask], unmatched_bias_ratio[~unmatched_bias_ratio.mask]], orient="h")
ax.set_yticklabels(["Matched", "Unmatched"])
ax.set_xlabel("Bias:effect size ratio")
plt.tight_layout()
plt.show()

## Subplots
# Confounding and no interference
_, ax = plt.subplots(ncols=2)
ax[0].imshow(np.abs(conf_bias_matrix[:, :4])/1.5, cmap="Reds")
#plt.colorbar()
ax[0].set_yticks(list(range(28)))
ax[0].set_yticklabels(ylabels)
#ax[0].xaxis.tick_top()
ax[0].set_xticks(list(range(4)))
ax[0].set_xticklabels(["N\nN", "B", "D", "R"])
ax[0].set_title("Bias:effect size (no interference)")
ax[0].set_xlabel("data scenarios (confounding varies more)")
ax[0].set_ylabel("models")

# Interference and no confounding
ax[1].imshow(np.abs(conf_bias_matrix[:, ::4])/1.5, cmap="Reds")
#plt.colorbar()
ax[1].set_yticks(list(range(28)))
ax[1].set_yticklabels(ylabels)
#ax[1].xaxis.tick_top()
ax[1].set_xticks(list(range(4)))
ax[1].set_xticklabels(["N\nN", "B\nN", "D\nN", "R\nN"])
ax[1].set_title("Bias:effect size (no confounding")
ax[1].set_xlabel("data scenarios (confounding varies more)")
ax[1].set_ylabel("models")
plt.suptitle("(confounding, interference)")
plt.tight_layout()
plt.show()


## Model/data matrices for each model
car_data = data_refined[data_refined["model"] == "car"]
joint_data = data_refined[data_refined["model"] == "joint"]

ols_matrix = np.zeros((4, 4))
for mintf in model_intfs:
    for dintf in model_intfs:
        ols_matrix[




matched_data = data[(data["w_i_data_form"] == data["w_i_model_form"]) & (data["w_c_data_form"] == data["w_c_model_form"])]
unmatched_data = data[(data["w_i_data_form"] != data["w_i_model_form"]) | (data["w_c_data_form"] != data["w_c_model_form"])]

## INTEFERENCE MAJOR ORG
data_refined = data[["model", "w_i_data_form", "w_i_model_form", "w_c_data_form", "w_c_model_form", "tau_med", "tau_var", "bias"]]
mapper = {"none": 1, "binary": 2, "distance": 3, "region": 4}
data_refined["w_i_data_form"] = data_refined["w_i_data_form"].map(mapper)
data_refined["w_i_model_form"] = data_refined["w_i_model_form"].map(mapper)
data_refined["w_c_data_form"] = data_refined["w_c_data_form"].map(mapper)
data_refined["w_c_model_form"] = data_refined["w_c_model_form"].map(mapper)

data_refined.sort_values(by=["model", "w_i_model_form", "w_c_model_form", "w_i_data_form", "w_c_data_form"], inplace=True)

rev_mapper = {1: "none", 2: "binary", 3: "distance", 4: "region"}
data_refined["w_i_data_form"] = data_refined["w_i_data_form"].map(rev_mapper)
data_refined["w_i_model_form"] = data_refined["w_i_model_form"].map(rev_mapper)
data_refined["w_c_data_form"] = data_refined["w_c_data_form"].map(rev_mapper)
data_refined["w_c_model_form"] = data_refined["w_c_model_form"].map(rev_mapper)


bias_matrix = np.zeros((28, 16))
model_intfs = ["none", "binary", "distance", "region"]
sp_confs = ["binary", "distance", "region"]

# OLS
rowctr = 0
ols_data = data_refined[data_refined["model"] == "ols"]
for model_intf in model_intfs:
    bias_matrix[rowctr, :] = ols_data[ols_data["w_i_model_form"] == model_intf]["bias"].values
    rowctr += 1

sp_data = data_refined[data_refined["model"] != "ols"]
for model in ["car", "joint"]:
    for model_intf in model_intfs:
        for model_conf in sp_confs:
            bias_matrix[rowctr, :] = sp_data[(sp_data["model"] == model) & (sp_data["w_i_model_form"] == model_intf) & (sp_data["w_c_model_form"] == model_conf)]["bias"].values
            rowctr += 1

intf_strs = ["N", "B", "D", "R"]
conf_strs = ["N", "B", "D", "R"]
xlabels = []
for conf in conf_strs:
    for intf in intf_strs:
        if intf == "N":
            xlabels.append(intf + "\n" + conf)
        else:
            xlabels.append(intf)

xlabels = [f"{intf}" for conf in model_intfs for intf in intf_strs]
ylabels = [*(f"OLS ({intf}, none)" for intf in model_intfs)] + \
          [*(f"CAR ({intf}, {conf})" for intf in model_intfs for conf in sp_confs)] + \
          [*(f"Joint ({intf}, {conf})" for intf in model_intfs for conf in sp_confs)]

_, ax = plt.subplots()
plt.imshow(np.abs(bias_matrix))
plt.colorbar()
ax.set_yticks(list(range(28)))
ax.set_yticklabels(ylabels)
#ax.xaxis.tick_top()
ax.set_xticks(list(range(16)))
ax.set_xticklabels(xlabels, ha='right')
plt.title("Biases (interference, confounding)")
plt.xlabel("data scenarios (confounding varies more)")
plt.ylabel("models")
plt.show()
