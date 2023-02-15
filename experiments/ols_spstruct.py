# 4*3*3*4*4*4 = 2304
# to save:
# - model.results_["tau.1"].values: np.ndarray
# - [model.results_[f"y_pred.{i+1}"].values for i in range(N)]: list of np.ndarray
# - model.waic()["waic"]: float
# - model.results_["divergent__"].sum(): float
# - (model.results_["treedepth__"] == max_depth).sum(): int
# - model.ess_dict: dict
# - model.rhats[<param>].values: xarray.Dataset
# - model.bfmi: np.ndarray

import json
import fcntl
import pickle
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from libpysal.cg import KDTree
from libpysal.weights import lat2W, KNN, W, fill_diagonal
from concurrent.futures import ProcessPoolExecutor
from networkx import from_numpy_array, empty_graph

# Get rundown on the path
import sys
sys.path.insert(0, '/home/tdhoffman/Documents/rundown')
import rundown as rd

## Set up hyperparameters
Nlat        = 30
N           = Nlat**2
D           = 1
nsamples    = 4000
nwarmup     = 1000
save_warmup = False
nchains     = 1
delta       = 0.8
max_depth   = 10
n_nbs       = 6
n_regs      = 6

# Generate weights matrices (binary, distance, and region)
binary = lat2W(Nlat, Nlat, rook=False)

coords = [(i, j) for i in range(Nlat) for j in range(Nlat)]
kd = KDTree(np.array(coords))
distance = KNN(kd, k=n_nbs)

data = np.vstack((np.hstack((np.ones((Nlat//2, Nlat//2)), 2*np.ones((Nlat//2, Nlat//2)))),
                    np.hstack((3*np.ones((Nlat//2, Nlat//2)), 4*np.ones((Nlat//2, Nlat//2))))))
region = np.zeros((N, N))

for p in range(Nlat**2):
    i1, j1 = np.unravel_index(p, (Nlat, Nlat))
    for q in range(Nlat**2):
        i2, j2 = np.unravel_index(q, (Nlat, Nlat))
        if data[i1, j1] == data[i2, j2]:
            region[p, q] = 1

region = W.from_networkx(from_numpy_array(region))
region = fill_diagonal(region, 0)

nowgt = W.from_networkx(from_numpy_array(np.eye(N)))
nowgt = fill_diagonal(nowgt, 0)

## Create parameter space
treat    = 0.5
z_conf   = 1  # these control the shape of the arrows in the triangle DAG
y_conf   = 1  # set both to 1 to assume nonspatial confounding is present.
interf   = np.linspace(start=-2, stop=2, num=4)  # no interference is covered by the nowgt option
x_sd     = 0.5
y_sd     = 0.5
x_sp     = 0.9
ucar_str = np.array([0.5, 0.9, 0.99])
vcar_str = np.array([0.5, 0.9, 0.99])
ucar_sd  = 1
vcar_sd  = 1
balance  = 0.5  # no spatial confounding is covered by the nowgt option
w_I      = {"binary": binary, "distance": distance, "region": region, "none": nowgt}
w_C      = {"binary": binary, "distance": distance, "region": region, "none": nowgt}

# params = list(itertools.product(treat, z_conf, y_conf, interf, ucar_str, vcar_str,
                                # balance, w_I.keys(), w_I.keys(), w_C.keys()))
params = list(itertools.product(interf, ucar_str, vcar_str, w_I.keys(), w_I.keys(), w_C.keys()))

## Loop
ncores = 32
ntasks = len(params)
chunksize = ntasks//(ncores - 1)
outfile = "../outputs/ols_confounds_spstruct.json"


def chunk_sim_run(params):
    interf, ucar_str, vcar_str, w_I_data_form, w_I_model_form, w_C_data_form = params

    if w_I_data_form == "none":
        intval = None
    else:
        intval = w_I[w_I_data_form]

    sim = rd.CARSimulator(Nlat, D, sp_confound=w_C[w_C_data_form], interference=intval)
    X, Y, Z = sim.simulate(treat=treat, z_conf=z_conf, y_conf=y_conf, interf=interf,
                           x_sd=x_sd, y_sd=y_sd, x_sp=x_sp, ucar_str=ucar_str,
                           vcar_str=vcar_str, ucar_sd=ucar_sd, vcar_sd=vcar_sd,
                           balance=balance)

    if w_I_model_form == "none":
        Zint = Z
    else:
        intadj = rd.InterferenceAdj(w=w_I[w_I_model_form])
        Zint = intadj.transform(Z)

    model = rd.BayesOLS(fit_intercept=False)
    model = model.fit(X, Y, Zint, nsamples=nsamples, nwarmup=nwarmup, save_warmup=save_warmup,
                      nchains=nchains, delta=delta, max_depth=max_depth, simulation=True)
    model.diagnostics()

    treedepth = (model.results_["treedepth__"] == max_depth).sum()
    results_dict = {
        "treat": treat,
        "z_conf": z_conf,
        "y_conf": y_conf,
        "interf": interf,
        "x_sd": x_sd,
        "y_sd": y_sd,
        "x_sp": x_sp,
        "ucar_str": ucar_str,
        "vcar_str": vcar_str,
        "ucar_sd": ucar_sd,
        "vcar_sd": vcar_sd,
        "balance": balance,
        "tau_med": model.ate_[0],
        "tau_var": model.results_["tau[1]"].var(),
        "tau_025": model.results_["tau[1]"].quantile(0.025),
        "tau_975": model.results_["tau[1]"].quantile(0.975),
        "w_I_data_form": w_I_data_form,
        "w_I_model_form": w_I_model_form,
        "w_C_data_form": w_C_data_form,
        "divergences": model.results_["divergent__"].sum(),
        "treedepth": int(treedepth),
        "bfmi": list(model.bfmi)
    }

    # Save and dump results from memory
    # number at the end of the file is the last iteration saved
    with open(outfile, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        json.dump(results_dict, f, indent=4)
        f.write(",\n")
        fcntl.flock(f, fcntl.LOCK_UN)


with open(outfile, "a") as f:
    f.write("[")

# give each process a chunk of iterations to work on
with ProcessPoolExecutor() as executor:
    for i, _ in enumerate(executor.map(chunk_sim_run, params, chunksize=chunksize)):
        print(f"completed iteration {i}")

with open(outfile, "a") as f:
    f.write("]")
