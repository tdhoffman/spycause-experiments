import copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable

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


## Organize matrix for an ANOVA on confounding
data_refined["bias_ratio"] = np.abs(data_refined["bias"])/1.5
data_refined["log_var"] = np.log(data_refined["tau_var"])
bias_lm = ols("bias_ratio ~ C(w_c_data_form)", data=data_refined).fit()
var_lm = ols("log_var ~ C(w_c_data_form)", data=data_refined).fit()

bias_table = sm.stats.anova_lm(bias_lm, typ=1)
print(bias_table)

var_table = sm.stats.anova_lm(var_lm, typ=1)
print(var_table)

## Organize matrix for an ANOVA on interference
bias_lm = ols("bias_ratio ~ C(w_i_data_form)", data=data_refined).fit()
var_lm = ols("log_var ~ C(w_i_data_form)", data=data_refined).fit()

bias_table = sm.stats.anova_lm(bias_lm, typ=1)
print(bias_table)

var_table = sm.stats.anova_lm(var_lm, typ=1)
print(var_table)

# Boxplots
_, ax = plt.subplots(ncols=2, nrows=2, sharex=True, sharey='row')
sns.boxplot(data=data_refined, y="bias_ratio", x="w_c_data_form", orient="v", ax=ax[0, 0])
ax[0, 0].set_xlabel("Form of confounding in data")
ax[0, 0].set_ylabel("Bias:effect size")
sns.boxplot(data=data_refined, y="log_var", x="w_c_data_form", orient="v", ax=ax[1, 0])
ax[1, 0].set_xlabel("Form of confounding in data")
ax[1, 0].set_ylabel("Log variance")
sns.boxplot(data=data_refined, y="bias_ratio", x="w_i_data_form", orient="v", ax=ax[0, 1])
ax[0, 1].set_xlabel("Form of interference in data")
ax[0, 1].set_ylabel("Bias:effect size")
sns.boxplot(data=data_refined, y="log_var", x="w_i_data_form", orient="v", ax=ax[1, 1])
ax[1, 1].set_xlabel("Form of interference in data")
ax[1, 1].set_ylabel("Log variance")
plt.show()
