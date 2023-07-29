# working with data
import numpy as np
import pandas as pd
from scipy import signal, stats

# plotting
import seaborn as sns
import matplotlib.pyplot as plt

# system
from pathlib import Path
import os
import time
from tqdm import tqdm

def create_folder_structure():
    os.makedirs("../data/", exist_ok=True)
    os.makedirs("../data/rf", exist_ok=True)
    os.makedirs("../data/spikes", exist_ok=True)


def load_data(path):
    data = np.load(path, allow_pickle=True).item()
    return data


def print_info(data):
    data_iter = ((k, type(v), v.shape) for k, v in data.items())
    l = [f"[{k}] - {t}, - {s}" for k, t, s in data_iter]
    print("\n".join(l) + "\n")
    
    
def shape2str(arr):
    return "(" + "-".join(map(str, arr)) + ")"


def find_index(arr, val):
    """ 
    Returns the exact index of a number val in array arr or the closes one possible.
    Does not work when such index is not uniquely defined.
    If val is a single number returns single index
    If val is iterable returns iterable of indices of the same shape   
    """
    # check if val is iterable
    try:
        val[0] # if not iterable raises an error
    except:
        # if not iterable - return the single index
        index = np.argmin(np.abs(np.array(arr) - val))
        return index
    # if got to this point it is definitely iterable
    indices = []
    for v in val:
        indices.append(np.argmin(np.abs(np.array(arr) - v)))
    return indices


def est_max(dff, neuron):
    return np.max(dff[neuron])

def est_mean(dff, neuron):
    return np.mean(dff[neuron])

def est_median(dff, neuron):
    return np.median(dff[neuron])

def est_quantile(dff, neuron, q=0.99):
    return np.quantile(dff[neuron], q)

def est_std(dff, neuron):
    return np.std(dff[neuron])

def est_robust_std(dff, neuron):
    x = dff[neuron]
    sigma = np.median(np.abs(x - np.mean(x)) / 0.6745)
    return sigma

def est_percent_activity(dff, neuron, sd_multiplier=5):
    sigma = est_robust_std(dff, neuron) * sd_multiplier
    return np.sum(dff[neuron] > sigma) / dff.shape[1]
    
def est_num_events(dff, neuron, sd_multiplier=5):
    sigma = est_robust_std(dff, neuron) * sd_multiplier
    islarger = dff[neuron] > sigma
    
    # crossing are calculated using np.diff
    # islarger = [False False True ... True False False ...]
    # True - False = True
    # False - True = True
    # Thus np.diff will result in True values when threshold crossing took place
    
    thresh_crossed = np.diff(islarger)
    n_events = int(np.sum(thresh_crossed) / 2)
    return n_events


def get_stim_spike_data(spikes, stim_table, n_neurons=189):
    """
    Calculate expected amount of spikes within each stimulus timeframe.
    Gets spike time series and stim_table with indexes of start and end of the stimulus.
    Returns expected spike counts in shape (n_stimuli, n_neurons)
    """
    istim_start = stim_table['start'].values
    istim_end   = stim_table['end'].values

    n_stim = len(istim_start)

    stim_spike_data = np.empty((n_stim, n_neurons))
    for i in range(n_stim):
        # cut out spikes outside of trial times
        spikes_frame = spikes[:, istim_start[i]:istim_end[i]]
        # sum over time dimension to get expected spike count within a trial
        stim_spike_data[i] = np.sum(spikes_frame, axis=1) 
    return stim_spike_data


def load_spikes(model_name, fillnan="zeros"):
    """
    Expected spike directory: ../data/spikes/
    Fills NaNs in the data with fillnan. Available options:
    - zeros
    - mean
    - median
    
    """  
    spikes = np.load(f"../data/spikes/spikes_{model_name}.npy")
    n_neurons = spikes.shape[0]
    
    fillvals = {
        "zeros": [0 for _ in range(n_neurons)], 
        "median": np.median(spikes, axis=1), 
        "mean": np.nanmean(spikes, axis=1)
               }
    for neuron in range(n_neurons):
        spikes[neuron, np.isnan(spikes[neuron])] = fillvals[fillnan][neuron]
    return spikes
    
    
def gen_norm():
    return stats.norm.rvs()
    