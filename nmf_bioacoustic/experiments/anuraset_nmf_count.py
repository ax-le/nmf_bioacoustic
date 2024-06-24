import matplotlib.pyplot as plt
import numpy as np
import tqdm
from nn_fac.nmf import nmf as nn_fac_nmf
from sacred import Experiment
from sacred.observers import FileStorageObserver
from scipy.special import kl_div
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

import nmf_bioacoustic.tasks.source_count as species_count
from nmf_bioacoustic.datasets.anuraset import AnuraSet
from nmf_bioacoustic.utils.signal_to_spectrogram import FeatureObject

anuraset_path = "/home/a23marmo/datasets/anuraset"

ex = Experiment("NMF on Anuraset for source count estimation")

save_log_folder = "experiments/jjba/source_count/running"
ex.observers.append(FileStorageObserver(save_log_folder))


@ex.config
def default_config():
    sr = 16000
    hop_length = 512
    n_fft = hop_length * 4
    feature = "nn_log_mel"
    fmin = 0
    fmax = None
    n_mels = 80

    n_nmf = 10
    beta = 2

    default_var_divide = 10
    default_eps = 0.8
    default_metric = "correlation"

    scoring = "accuracy"
    cv = 4

    param_grid = {"eps": np.linspace(0.3, 1.3, 11)}

    subset = "INCT17"


@ex.capture
def compute_all_nmf_on_this_dataset(n_nmf, beta, subset, sr, feature, hop_length, n_fft, fmin, fmax, n_mels):
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
            normalize=[False, True],  # sparsity_coefficients=[0, 10],
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
    exp_id = _run.meta_info['options']['--id']
    all_H, all_annot = compute_all_nmf_on_this_dataset()
    fitted_estimation, accuracy = grid_search_DBSCAN(all_H=all_H, all_annot=all_annot)
    _run.log_scalar("Number samples", len(all_annot))
    print(f"Accuracy: {accuracy}")
    _run.log_scalar("Accuracy", accuracy)

    difference = np.mean(np.abs(np.subtract(fitted_estimation, all_annot)))
    print(f"Difference: {difference}")
    _run.log_scalar("Difference", difference)

    kl = np.mean(kl_div(all_annot, fitted_estimation))
    print(f"KL div: {kl}")
    _run.log_scalar("KL div", kl)

    _run.log_scalar("Estimation", fitted_estimation)

    # ConfusionMatrixDisplay.from_predictions(all_annot, fitted_estimation).plot()
    # plt.savefig(f'jjba_experiments/confusion_matrix_beta{_config['beta']}.png')
    # plt.show()

    cm_display = ConfusionMatrixDisplay.from_predictions(all_annot, fitted_estimation)
    # cm_display.plot()
    plt.gca().invert_yaxis()  # Get the current axes and invert the y-axis
    plt.xlabel("Predicted number of sources")
    plt.ylabel("Annotated number of species")
    plt.savefig(
        f"{save_log_folder}/{exp_id}/confusion_matrix_{exp_id}.png"
    )
    plt.close()

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

    # for subset in ["INCT4", "INCT17", "INCT41", "INCT20955"]:
    #     for feature in ["mel", "nn_log_mel"]:
    #         for beta in [2,1,0]:
    #             id_experiment = f"beta{beta}_nmf10_feature{feature}_subset{subset}"
    #             r = ex.run(config_updates={'beta':beta, 'feature':feature, 'subset':subset}, options={'--id':id_experiment})

    beta = 1
    feature = "nn_log_mel"
    subset="INCT17"
    id_experiment = f"beta{beta}_nmf10_feature{feature}_subset{subset}"
    r = ex.run(config_updates={'beta':beta, 'feature':feature, 'subset':subset}, options={'--id':id_experiment})
