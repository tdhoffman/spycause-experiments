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
sys.path.insert(0, '/home/tdh/research/rundown')
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
distance.symmetrize(inplace=True)

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

## Create parameter space
treat    = 1.5
z_conf   = 1  # these control the shape of the arrows in the triangle DAG
y_conf   = 1  # set both to 1 to assume nonspatial confounding is present.
interf   = 0.5  # no interference is covered by the nowgt option
x_sd     = 0.5
y_sd     = 0.5
x_sp     = 0.9
ucar_str = 0.9
vcar_str = 0.9
ucar_sd  = 1
vcar_sd  = 1
balance  = 0.5  # no spatial confounding is covered by the nowgt option
w_i      = {"binary": binary, "distance": distance, "region": region, "none": None}
w_c      = {"binary": binary, "distance": distance, "region": region, "none": None}
model_str    = ["joint"]

params = list(itertools.product(w_i.keys(), w_i.keys(), w_c.keys(), w_c.keys(), model_str))

## Loop
ncores = 32
ntasks = len(params)
print(ntasks)
chunksize = ntasks // (ncores - 1)
outfile = "../outputs/joint_confounds_spstruct.json"
reffile = "../outputs/confounds_reference.json"

reference = pd.read_json(reffile, orient="records")
jointref = reference[reference["model"] == "joint"]

finished_params = list(map(lambda a, b, c, d, e: (a, b, c, d, e),
                           jointref["w_i_data_form"].values,
                                         jointref["w_i_model_form"].values,
                                         jointref["w_c_data_form"].values,
                                         jointref["w_c_model_form"].values,
                                         148*["joint"]))

# Preemptively delete no confounding scenarios (only applicable for joint model)
for param in params:
    w_i_data_form, w_i_model_form, w_c_data_form, w_c_model_form, model_str = param
    if w_c_model_form == "none":
        params.remove(param)

# Final set of tasks to do!!
candidates = set(params) - set(finished_params)
print(len(candidates))


def chunk_sim_run(params):
    w_i_data_form, w_i_model_form, w_c_data_form, w_c_model_form, model_str = params

    # CAR and Joint must model confounding; OLS cannot model confounding
    if w_c_model_form == "none" and (model_str == "car" or model_str == "joint"):
        return
    if w_c_model_form != "none" and model_str == "ols":
        return

    sim = rd.CARSimulator(Nlat, D, sp_confound=w_c[w_c_data_form], interference=w_i[w_i_data_form])
    X, Y, Z = sim.simulate(treat=treat, z_conf=z_conf, y_conf=y_conf, interf=interf,
                           x_sd=x_sd, y_sd=y_sd, x_sp=x_sp, ucar_str=ucar_str,
                           vcar_str=vcar_str, ucar_sd=ucar_sd, vcar_sd=vcar_sd,
                           balance=balance)

    if w_i_model_form == "none":
        Zint = Z
    else:
        intadj = rd.InterferenceAdj(w=w_i[w_i_model_form])
        Zint = intadj.transform(Z)

    if model_str == "ols":
        model = rd.BayesOLS(fit_intercept=False)
    elif model_str == "car":
        model = rd.CAR(w=w_c[w_c_model_form], fit_intercept=False)
    elif model_str == "joint":
        model = rd.Joint(w=w_c[w_c_model_form], fit_intercept=False)

    model = model.fit(X, Y, Zint, nsamples=nsamples, nwarmup=nwarmup, save_warmup=save_warmup,
                      nchains=nchains, delta=delta, max_depth=max_depth, simulation=True)
    model.diagnostics()

    treedepth = (model.results_["treedepth__"] == max_depth).sum()
    results_dict = {
        "model": model_str,
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
        "w_i_data_form": w_i_data_form,
        "w_i_model_form": w_i_model_form,
        "w_c_data_form": w_c_data_form,
        "w_c_model_form": w_c_model_form,
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
with ProcessPoolExecutor(max_workers=ncores - 4) as executor:
    for i, _ in enumerate(executor.map(chunk_sim_run, candidates, chunksize=chunksize)):
        print(f"completed iteration {i}")

with open(outfile, "a") as f:
    f.write("]")
