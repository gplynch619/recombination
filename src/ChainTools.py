import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from getdist import plots, MCSamples, loadMCSamples
import classy as Class
import getdist
from scipy.interpolate import CubicSpline
import pickle as pkl
import itertools
import sys
sys.path.insert(0, '/Users/gabe/projects/emulators/src')
from TrainedEmulator import *

DATA_DIR = "/Users/gabe/projects/recombination/data/chains/"
MODREC_PARAMS = ['omega_b','omega_cdm','n_s','tau_reio','ln10^{10}A_s','H0','q_1','q_2','q_3','q_4','q_5']
LCDM_PARAMS = ['omega_b','omega_cdm','n_s','tau_reio','ln10^{10}A_s','H0']


def get_individual_chains(sample_name, sample):
    base_samples = np.split(sample.samples,sample.chain_offsets[1:-1])
    base_chains = []

    rename_dict = {}

    sample_dir = os.path.join(DATA_DIR, "{}/chains/".format(sample_name))

    with open(os.path.join(sample_dir, "{}.paramnames".format(sample_name)), "r") as f:
        lines = [l.strip() for l in f.readlines()]
        for i,name in enumerate(lines):
            rename_dict["param{}".format(i+1)] = name

    sum=0
    for samp in base_samples:
        chain = MCSamples()
        sum+=len(samp)
        chain.loadChains(root=sample_dir, files_or_samples=samp)
        chain.setParamNames(os.path.join(sample_dir, "{}.paramnames".format(sample_name)))
        base_chains.append(chain)

    return base_chains

def traceplot(chain_name, chain, params_to_plot=LCDM_PARAMS):
   
    base_chains = get_individual_chains(chain_name, chain)

    num_rows = int(np.ceil(len(params_to_plot)/3))
    fig,ax=plt.subplots(nrows=num_rows, ncols=3, figsize = (6*num_rows, 15))

    for i,p in enumerate(params_to_plot):
        if p=="ln10^{10}A_s":
            name = "logA"
        else:
            name = chain.getParamNames().parWithName(p).name
        ax.flatten()[i].set_ylabel(name)
        for chain in base_chains:
            samples = chain.getParams().__dict__[name]
            ax.flatten()[i].plot(samples, alpha=0.5)
    
    return fig,ax

def find_index_of(array, values):
    indices_return = []
    for value in values:
        test = np.where(array<value, 1, 0)
        ind = np.argwhere(np.diff(test)!=0)[0][0]
        indices_return.append(ind)
    return indices_return

def CI(vals, level):
    num = vals.shape[0]
    tail_frac = (1-level)/2.
    tail_num = int(np.ceil(num*tail_frac))
    sortedargs = np.argsort(vals)
    lower = vals[sortedargs[tail_num]]
    upper = vals[sortedargs[-1*tail_num]]
    return lower,upper

#each row in array_of_funcs should be the  y-values of a single function
def function_ci(array_of_funcs, level):
    upper = []
    lower = []
    for col in array_of_funcs.T:
        l,u = CI(col, level)
        upper.append(u)
        lower.append(l)
    upper = np.array(upper)
    lower = np.array(lower)
    return np.vstack([upper, lower])

def get_predictions_for_selection(chain, em, criteria='none', N=5000):
    chain_as_dict = chain.getParams().__dict__

    size_of_total = len(chain_as_dict["H0"])
    if criteria=='none':
        idx_of_subsample = np.arange(size_of_total)
    elif criteria=="random":
        idx_of_subsample = chain.random_single_samples_indices(max_samples=N)
    elif isinstance(criteria, dict):
        criteria_string = ""
        for quantity, range in criteria.items():
            criteria_string += '(chain_as_dict["{0}"] > {1}) & (chain_as_dict["{2}"] < {3})'.format(quantity, range[0], quantity, range[1])
            if quantity != list(criteria)[-1]:
                criteria_string += ' & '
        print(criteria_string)
        idx_of_subsample = np.where(eval(criteria_string))

    param_dict={}
    for p in em.output_info["input_names"]:
        if p=='ln10^{10}A_s':
            param_dict[p]=chain_as_dict["logA"][idx_of_subsample]
        elif p.startswith("q_"):
            if p in chain_as_dict.keys():
                param_dict[p]=chain_as_dict[p][idx_of_subsample]
            else:
                param_dict[p]=np.zeros(len(idx_of_subsample))
        else:
            name = chain.getParamNames().parWithName(p).name
            param_dict[p]=chain_as_dict[name][idx_of_subsample]

    return em.get_predictions_dict(param_dict), param_dict

def get_selection_from_predictions(chain, predictions, criteria='none', N=5000, params_to_select = MODREC_PARAMS):
    chain_as_dict = chain.getParams().__dict__

    size_of_total = len(chain_as_dict["H0"])
    if criteria=='none':
        idx_of_subsample = np.arange(size_of_total)
    elif criteria=='random':
        idx_of_subsample = chain.random_single_samples_indices(max_samples=N)
    elif isinstance(criteria, dict):
        criteria_string = ""
        for quantity, range in criteria.items():
            criteria_string += '(chain_as_dict["{0}"] > {1}) & (chain_as_dict["{2}"] < {3})'.format(quantity, range[0], quantity, range[1])
            if quantity != list(criteria)[-1]:
                criteria_string += ' & '
        print(criteria_string)
        idx_of_subsample = np.where(eval(criteria_string))

    param_dict={}
    for p in params_to_select:
        if p=='ln10^{10}A_s':
            param_dict[p]=chain_as_dict["logA"][idx_of_subsample]
        elif p.startswith("q_"):
            if p in chain_as_dict.keys():
                param_dict[p]=chain_as_dict[p][idx_of_subsample]
            else:
                param_dict[p]=np.zeros(len(idx_of_subsample))
        else:
            name = chain.getParamNames().parWithName(p).name
            param_dict[p]=chain_as_dict[name][idx_of_subsample]

    selection_predictions = {k:v[idx_of_subsample] for k,v in predictions.items()}

    return selection_predictions, param_dict

def extract_trajectories_from_predictions(predictions, em, quantity, n=1000):

    if quantity in em.output_info["output_Cl"]:
        pivot_grid = em.ell
    elif quantity in em.output_info["output_z"]:
        pivot_grid = em.output_info["output_z_grids"][quantity]

    outputs = predictions[quantity]

    fine_grid = np.linspace(min(pivot_grid), max(pivot_grid), n)

    trajectories = np.vstack([CubicSpline(pivot_grid, y)(fine_grid) for y in outputs])

    return trajectories, fine_grid

def construct_trajectory_heatmap(list_of_trajectories, y_range, num_bins=1000):
    
    ## one version

    #heatmap = map(lambda x: np.histogram(x, bins=num_bins, range=y_range)[0][0], list_of_trajectories.T)
    #heatmap = [h/np.max(h) for h in heatmap]
    #_, be = np.histogram(list_of_trajectories.T[0], bins=num_bins, range=y_range)[0]
    
    # or easier to read
    heatmap= []
    for i in np.arange(0, list_of_trajectories.shape[1]):
        h, be = np.histogram(list_of_trajectories[:,i], bins=1000, range=y_range)
        if np.max(h)!=0:
            heatmap.append(h/np.max(h))
        else:
            eps=1e-15
            heatmap.append(h/1e-15)

    return np.array(heatmap).T, be

def plot_model_points_on_triangle(getdist_plotter, plotted_params, list_of_models):
    param_combos = list(itertools.combinations(plotted_params, 2))

    for c in param_combos:
        ax = getdist_plotter.get_axes_for_params(c[0], c[1])
        c0 = c[0]
        c1 = c[1]
        if (c[0]=="logA"):
            c0 = 'ln10^{10}A_s'
        elif (c[1]=="logA"):
            c1='ln10^{10}A_s'
        ax.scatter(list_of_models[c0], list_of_models[c1], s=1, marker=".", color="black", alpha=0.5)

def get_emulator_quantity_from_chain(em, chain, emulator_param):
    
    chain_as_dict = chain.getParams().__dict__
    param_dict={}
    for p in em.output_info["input_names"]:
        if p=='ln10^{10}A_s':
            param_dict[p]=chain_as_dict["logA"]
        elif p.startswith("q_"):
            if p in chain_as_dict.keys():
                param_dict[p]=chain_as_dict[p]
            else:
                param_dict[p]=np.zeros(len(chain.getParams().H0))
        else:
            name = chain.getParamNames().parWithName(p).name
            param_dict[p]=chain_as_dict[name]
    
    pred = em.get_predictions_dict(param_dict)
    return pred[emulator_param]