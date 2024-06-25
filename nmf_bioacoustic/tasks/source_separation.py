"""
Created on June 2024

@author: a23marmo

Base code for the task of source separation in a signal, using NMF.
This is only qualitative, as the annotations of independent sources are not provided. 
"""
import numpy as np
import nmf_bioacoustic.utils.audio_helper as audio_helper

# Plotting functions
from librosa.display import specshow
import matplotlib.pyplot as plt

def evaluate_source_separation(W, H, feature_object, time_limit=None, phase_retrieval="original_phase", phase=None, plot_specs=False):
    """
    Evaluates the source separation of the NMF.
    Plots the audio of the mixture and the separated sources.
    It can also plots the spectrograms of the mixture and the separated sources.
    
    Parameters
    ----------
    W : np.array
        W matrix of the NMF.
    H : np.array
        H matrix of the NMF.
    feature_object : FeatureObject
        Feature object, defining the important parameters to compute spectrograms.
    time_limit : int
        Time limit to evaluate the source separation, and limit the size of the audio and spectrograms.
    phase_retrieval : str
        Method to retrieve the phase of the audio. It can be 'original_phase' or 'griffin_lim'.
    phase : np.array
        Phase of the original audio, to be used in the phase retrieval.
        Only used if phase_retrieval is 'original_phase'.
    plot_specs : bool
        If True, plots the spectrograms of the mixture and the separated sources.

    Returns
    -------
    source_list : list of np.array
        List of the separated sources, as spectrograms.
    """
    if phase_retrieval == "original_phase":
        assert phase is not None, "You need to provide the phase of the original audio to evaluate the source separation"

    if time_limit is None:
        time_limit = H.shape[1]

    source_list = []

    # Listen to the whole mixture
    print("Whole mixture:")
    audio_helper.listen_to_this_spectrogram(W@H[:,:time_limit], feature_object=feature_object, phase_retrieval = phase_retrieval, original_phase = phase[:,:time_limit])
    
    # Plots the spectrogram of the whole mixture
    if plot_specs:
        fig, ax = plt.subplots()
        img = specshow(W@H[:,:time_limit], sr=feature_object.sr, hop_length=feature_object.hop_length, y_axis="log", x_axis="time", vmax=10) # specshow(W@H, sr=sr, hop_length=hop_length, y_axis="log")
        ax.set_title("Whole mixture")
        plt.savefig(f"imgs/source_separation/whole_mixture.png", transparent = True)
        plt.show()

    # Listen to the separated sources
    for i in range(0, H.shape[0]):
        print(f"Source: {i}")
        audio_helper.listen_to_this_spectrogram(W[:,i][:,np.newaxis]@H[i,:time_limit][np.newaxis,:], feature_object=feature_object, phase_retrieval = phase_retrieval, original_phase = phase[:,:time_limit])
        source_list.append(W[:,i][:,np.newaxis]@H[i,:time_limit][np.newaxis,:])

        # Plots the spectrogram of the separated source
        if plot_specs:
            fig, ax = plt.subplots()
            img = specshow(source_list[-1], sr=feature_object.sr, hop_length=feature_object.hop_length, y_axis="log", x_axis="time", vmax=10) # specshow(W@H, sr=sr, hop_length=hop_length, y_axis="log")
            ax.set_title(f"Source {i}")
            plt.savefig(f"imgs/source_separation/source_{i}.png", transparent = True)
            plt.show()

    return source_list