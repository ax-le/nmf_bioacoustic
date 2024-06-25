"""
Created on June 2024

@author: a23marmo

Defines the experiment for counting the number of sources in the AnuraSet dataset, based on the NMF outputs.
It uses the package SACRED to log the results and the parameters of the experiment.
See the docs of SACRED for more details https://sacred.readthedocs.io/en/stable/
"""

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from sacred import Experiment
from sacred.observers import FileStorageObserver
from scipy.special import kl_div
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

from nn_fac.nmf import nmf as nn_fac_nmf

import nmf_bioacoustic.tasks.source_count as species_count
from nmf_bioacoustic.datasets.anuraset import AnuraSet
from nmf_bioacoustic.utils.signal_to_spectrogram import FeatureObject

anuraset_path = "/home/a23marmo/datasets/anuraset" # TODO: change this path to the correct one

ex = Experiment("NMF on Anuraset for source count estimation")

save_log_folder = "experiments/jjba/source_count/running"
ex.observers.append(FileStorageObserver(save_log_folder))

@ex.config
def default_config():
    # Base configuration
    sr = 16000 # Sampling rate
    hop_length = 512 # Hop length
    n_fft = hop_length * 4 # FFT window size
    feature = "nn_log_mel" # Feature to use
    fmin = 0 # Minimum frequency
    fmax = None # Maximum frequency
    n_mels = 80 # Number of mel bands

    n_nmf = 10 # Number of components in the NMF
    beta = 2 # Beta parameter of the NMF, defining the loss function. 2 for Euclidean, 1 for Kullback-Leibler, 0 for Itakura-Saito

    default_var_divide = 10 # Variance divide parameter for considering some components as noise.
    default_eps = 0.8 # Epsilon parameter for the DBSCAN clustering
    default_metric = "correlation" # Metric for the DBSCAN clustering

    scoring = "accuracy" # Scoring for the GridSearchCV
    cv = 4 # Number of folds for the GridSearchCV

    param_grid = {"eps": np.linspace(0.3, 1.3, 11)} # Grid for the DBSCAN clustering

    subset = "INCT17" # Subset of the dataset to use

@ex.capture
def compute_all_nmf_on_this_dataset(n_nmf, beta, subset, sr, feature, hop_length, n_fft, fmin, fmax, n_mels):
    """
    Computes the NMF on all the signals of the dataset, and returns the H matrices of the NMF (used for clustering) and the annotations.

    Parameters
    ----------
    See the configuration of the experiment (default_config).
    """
    feature_object = FeatureObject(sr, feature, hop_length, n_fft=n_fft, fmin = fmin, fmax=fmax, mel_grill = False, n_mels=n_mels)

    dataset = AnuraSet(
        audio_path=f"{anuraset_path}/raw_data/{subset}",
        annotations_file=f"{anuraset_path}/weak_labels.csv",
        feature_object=feature_object,
    )
    all_annot = []
    all_H = []

    if beta == 2:
        nmf_algo = "hals"
    else:
        nmf_algo = "mu"

    for idx_song in tqdm.tqdm(range(len(dataset))):
        spec, annotations = dataset[idx_song]
        all_annot.append(annotations)

        _, H = nn_fac_nmf(
            spec,
            rank=n_nmf,
            init="nndsvd",
            n_iter_max=100,
            tol=1e-8,
            update_rule=nmf_algo,
            beta=beta,
            normalize=[False, True],
            verbose=False,
            return_costs=False,
            deterministic=True,
        )

        all_H.append(H)

    return all_H, all_annot


@ex.capture
def grid_search_DBSCAN(
    all_H,
    all_annot,
    param_grid,
    default_var_divide,
    default_metric,
    default_eps,
    scoring,
    cv,
):
    """
    Grid search for the DBSCAN clustering, based on the NMF outputs.

    In general, the parameter grid is used to find the best epsilon parameter for the DBSCAN clustering.

    Parameters
    ----------
    all_H : list of np.array
        List of the H matrices of the NMF.
    all_annot : list of int
        List of the annotations of the number of species in the signals.
    param_grid : dict
        Dictionary of the parameter grid for the DBSCAN clustering.
    default_var_divide : float
        Variance divide parameter for considering some components as noise.
    default_metric : str
        Metric for the DBSCAN clustering.
    default_eps : float
        Epsilon parameter for the DBSCAN clustering.
    scoring : str   
        Scoring for the GridSearchCV.
    cv : int
        Number of folds for the GridSearchCV.
    """

    default_counter = species_count.SourceCountEstimator(
        var_divide=default_var_divide, eps=default_eps, metric=default_metric
    )
    clf = GridSearchCV(default_counter, param_grid=param_grid, cv=cv, scoring=scoring)
    clf.fit(all_H, all_annot)
    fitted_estimation = clf.predict(all_H)
    accuracy = clf.score(all_H, all_annot)
    print(f"Best parameters: {clf.best_params_}")
    return fitted_estimation, accuracy


@ex.automain
def expe_nmf_main(_run, _config):
    """
    Main function of the experiment.
    """

    exp_id = _run.meta_info['options']['--id']

    # Compute the NMF on the dataset
    all_H, all_annot = compute_all_nmf_on_this_dataset()

    # Grid search for the DBSCAN clustering
    fitted_estimation, accuracy = grid_search_DBSCAN(all_H=all_H, all_annot=all_annot)
    
    # Log the results
    # Log the number of samples first
    _run.log_scalar("Number samples", len(all_annot))
    
    # Log the accuracy
    print(f"Accuracy: {accuracy}")
    _run.log_scalar("Accuracy", accuracy)

    # Log the average difference between the annotations and the estimation 
    difference = np.mean(np.abs(np.subtract(fitted_estimation, all_annot)))
    print(f"Difference: {difference}")
    _run.log_scalar("Difference", difference)

    # Log the KL divergence between the annotations and the estimation
    kl = np.mean(kl_div(all_annot, fitted_estimation))
    print(f"KL div: {kl}")
    _run.log_scalar("KL div", kl)


    # Log all the fitted estimation, if needed
    _run.log_scalar("Estimation", fitted_estimation)

    # Plots the confusion matrix, and save it
    cm_display = ConfusionMatrixDisplay.from_predictions(all_annot, fitted_estimation)
    plt.gca().invert_yaxis()  # Get the current axes and invert the y-axis
    plt.xlabel("Predicted number of sources")
    plt.ylabel("Annotated number of species")
    plt.savefig(
        f"{save_log_folder}/{exp_id}/confusion_matrix_{exp_id}.png"
    )
    plt.close()

    # Plots the histogram of the annotations and the estimation, and save it
    fig, ax = plt.subplots()
    ax.hist(
        fitted_estimation, bins=np.arange(0.5, 11.5, 1), alpha=0.5, label="estimation"
    )
    ax.hist(all_annot, bins=np.arange(0.5, 11.5, 1), alpha=0.5, label="annotation")
    ax.legend()
    ax.set_xlabel("Number of species/sources")
    #ax.set_title(f"Beta = {_config['beta']}")
    plt.savefig(f"{save_log_folder}/{exp_id}/histogram_{exp_id}.png")

    plt.close()

if __name__ == "__main__":

    ## Run all the experiments

    # for subset in ["INCT4", "INCT17", "INCT41", "INCT20955"]:
    #     for feature in ["mel", "nn_log_mel"]:
    #         for beta in [2,1,0]:
    #             id_experiment = f"beta{beta}_nmf10_feature{feature}_subset{subset}"
    #             r = ex.run(config_updates={'beta':beta, 'feature':feature, 'subset':subset}, options={'--id':id_experiment})

    ## Run one case
    beta = 1
    feature = "nn_log_mel"
    subset="INCT17"
    id_experiment = f"beta{beta}_nmf10_feature{feature}_subset{subset}"
    r = ex.run(config_updates={'beta':beta, 'feature':feature, 'subset':subset}, options={'--id':id_experiment})
