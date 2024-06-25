"""
Created on April 2024

@author: a23marmo

Code to transform spectrograms to audio signals.
"""

import nmf_bioacoustic.utils.errors as err
import nmf_bioacoustic.utils.signal_to_spectrogram as signal_to_spectrogram

import IPython.display as ipd
import librosa


# TODO: handle CQT

# %% Audio to signal conversion
def spectrogram_to_audio_signal(spectrogram, feature_object, phase_retrieval = "griffin_lim", original_phase=None):
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
        The audio signal of the song, reconstructed from NTD.
    """
    if feature_object.feature in ["stft", "stft_complex"]:
        spectrogram_stft = spectrogram
    elif feature_object.feature in ["mel", "log_mel", "nn_log_mel"]:
        spectrogram_stft = feature_object.get_stft_from_mel(spectrogram)
    else:
        raise err.InvalidArgumentValueException(f"Feature representation not handled for audio reconstruction: {feature_object.feature}.")

    if phase_retrieval == "griffin_lim":
        return librosa.griffinlim(spectrogram_stft, hop_length = feature_object.hop_length, random_state = 0)
        
    elif phase_retrieval == "original_phase":
        assert original_phase is not None, "Original phase is necessary for phase retrieval using the original_phase."
        assert original_phase.shape == spectrogram_stft.shape, "Original phase and spectrogram must have the same shape."
        complex_spectrogram_to_inverse = spectrogram_stft * original_phase
        return complex_stft_to_audio(complex_spectrogram_to_inverse, feature_object.hop_length)

    else:
        raise err.InvalidArgumentValueException(f"Phase retrieval method not understood: {phase_retrieval}.")
    
def complex_stft_to_audio(stft_to_inverse, hop_length):
    """
    Inverts the complex spectrogram using the istft method.
    """
    return librosa.istft(stft_to_inverse, hop_length = hop_length)

