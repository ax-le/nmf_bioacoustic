import numpy as np
import nmf_bioacoustic.utils.audio_helper as audio_helper

# Plotting functions
from librosa.display import specshow
import matplotlib.pyplot as plt

def evaluate_source_separation(W, H, feature_object, time_limit=None, phase_retrieval="original_phase", phase=None, plot_specs=False):
    if phase_retrieval == "original_phase":
        assert phase is not None, "You need to provide the phase of the original audio to evaluate the source separation"

    if time_limit is None:
        time_limit = H.shape[1]

    source_list = []

    print("Whole mixture:")
    audio_helper.listen_to_this_spectrogram(W@H[:,:time_limit], feature_object=feature_object, phase_retrieval = phase_retrieval, original_phase = phase[:,:time_limit])
    if plot_specs:
        fig, ax = plt.subplots()
        img = specshow(W@H[:,:time_limit], sr=feature_object.sr, hop_length=feature_object.hop_length, y_axis="log", x_axis="time", vmax=10) # specshow(W@H, sr=sr, hop_length=hop_length, y_axis="log")
        ax.set_title("Whole mixture")
        plt.savefig(f"imgs/source_separation/whole_mixture.png", transparent = True)
        plt.show()
    for i in range(0, H.shape[0]):
        print(f"Source: {i}")
        audio_helper.listen_to_this_spectrogram(W[:,i][:,np.newaxis]@H[i,:time_limit][np.newaxis,:], feature_object=feature_object, phase_retrieval = phase_retrieval, original_phase = phase[:,:time_limit])
        source_list.append(W[:,i][:,np.newaxis]@H[i,:time_limit][np.newaxis,:])

        if plot_specs:
            fig, ax = plt.subplots()
            img = specshow(source_list[-1], sr=feature_object.sr, hop_length=feature_object.hop_length, y_axis="log", x_axis="time", vmax=10) # specshow(W@H, sr=sr, hop_length=hop_length, y_axis="log")
            ax.set_title(f"Source {i}")
            plt.savefig(f"imgs/source_separation/source_{i}.png", transparent = True)
            plt.show()

    return source_list