import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_json("../outputs/confounds_reference.json", orient="records")
noint_data = data[data["w_i_data_form"] == "none"]
noint_noconf = noint_data[noint_data["w_c_data_form"] == "none"]
data["bias"] = data["tau_med"] - data["treat"]

sns.histplot(data["tau_med"], kde=True)
plt.show()

sns.histplot(data["tau_var"], kde=True, stat='probability')
plt.show()

sns.histplot(noint_data["tau_med"], kde=True)
plt.show()

sns.histplot(noint_data[noint_data["w_c_data_form"] == "binary"]["tau_med"], kde=True)
plt.show()

sns.histplot(data[data["w_i_data_form"] == "binary"]["tau_med"], kde=True)
plt.show()

sns.histplot(data[(data["w_i_data_form"] == "none") & (data["w_c_data_form"] == "none") & (data["w_i_model_form"] == "none")]["tau_med"], kde=True, bins=100)
plt.show()

sns.histplot(noint_noconf[noint_noconf["interf"] == data["interf"][0]]["tau_med"], kde=True)
plt.show()

sns.histplot(noint_data[noint_data["ucar_str"] == 0.99]["tau_med"], kde=True)
plt.show()

sns.histplot(noint_data[noint_data["interf"] == -2]['tau_med'], kde=True)
plt.show()


binint = data[data["w_i_data_form"] == "binary"]
sns.histplot(binint["tau_med"], kde=True)
plt.show()


matched_data = data[(data["w_i_data_form"] == data["w_i_model_form"]) & (data["w_c_data_form"] == data["w_c_model_form"])]
unmatched_data = data[(data["w_i_data_form"] != data["w_i_model_form"]) | (data["w_c_data_form"] != data["w_c_model_form"])]

sns.histplot(matched_data["bias"])
plt.show()

sns.histplot(unmatched_data["bias"])
plt.show()


matched_data_outs = matched_data[["model", "w_i_data_form", "w_i_model_form", "w_c_data_form", "w_c_model_form", "tau_med", "tau_var", "bias"]]
mapper = {"none": 1, "binary": 2, "distance": 3, "region": 4}
matched_data_outs["w_i_data_form"] = matched_data_outs["w_i_data_form"].map(mapper)
matched_data_outs["w_i_model_form"] = matched_data_outs["w_i_model_form"].map(mapper)
matched_data_outs["w_c_data_form"] = matched_data_outs["w_c_data_form"].map(mapper)
matched_data_outs["w_c_model_form"] = matched_data_outs["w_c_model_form"].map(mapper)

matched_data_outs.sort_values(by=["model", "w_i_data_form", "w_c_data_form"], inplace=True)

rev_mapper = {1: "none", 2: "binary", 3: "distance", 4: "region"}
matched_data_outs["w_i_data_form"] = matched_data_outs["w_i_data_form"].map(rev_mapper)
matched_data_outs["w_i_model_form"] = matched_data_outs["w_i_model_form"].map(rev_mapper)
matched_data_outs["w_c_data_form"] = matched_data_outs["w_c_data_form"].map(rev_mapper)
matched_data_outs["w_c_model_form"] = matched_data_outs["w_c_model_form"].map(rev_mapper)

data["bias"] = data["tau_med"] - data["treat"]
sns.histplot(data["bias"])
plt.show()
