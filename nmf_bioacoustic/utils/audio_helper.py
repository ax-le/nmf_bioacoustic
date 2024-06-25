"""
Created on April 2024

@author: a23marmo

Code to listen to audio based on spectrograms, and to compute the SDR between two audio signals.
"""

import IPython.display as ipd
import mir_eval
import numpy as np
import nmf_bioacoustic.utils.spectrogram_to_signal as spec_to_sig

def listen_to_this_signal(signal, sr=44100):
    return ipd.display(ipd.Audio(data=signal, rate=sr))

def listen_to_this_spectrogram(spectrogram, feature_object, phase_retrieval = "griffin_lim", original_phase = None):
    """
    Inverts the spectrogram using the istft method, and plots the audio using IPython.diplay.audio.

    Parameters
    ----------
    spectrogram : numpy array
        The spectrogram to be inverted.
    feature_object : FeatureObject
        Feature object, defining the important parameters to compute spectrograms.
    phase_retrieval : str
        Method to retrieve the phase of the audio. It can be 'original_phase' or 'griffin_lim'.
        If set to 'original_phase', the original_phase parameter is used as an estimation of the phase.
        If set to 'griffin_lim', the phase is estimated using the Griffin-Lim algorithm.
    original_phase : numpy array
        Phase of the original audio, to be used in the phase retrieval.
        Only used if phase_retrieval is 'original_phase'.

    Returns
    -------
    IPython.display audio
        The audio signal of the song, reconstructed from NTD
        (the one the SDR was computed on).

    """
    signal = spec_to_sig.spectrogram_to_audio_signal(spectrogram, feature_object=feature_object, phase_retrieval = phase_retrieval, original_phase=original_phase)
    return listen_to_this_signal(signal, feature_object.sr)

# %% Audio evaluation
def compute_sdr(audio_ref, audio_estimate):
    """
    Function encapsulating the SDR computation in mir_eval.
    SDR is computed between 'audio_ref' and 'audio_estimate'.
    """
    if (audio_estimate == 0).all():
        return -np.inf
    return mir_eval.separation.bss_eval_sources(np.array([audio_ref]), np.array([audio_estimate]))[0][0]