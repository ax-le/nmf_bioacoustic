import nmf_bioacoustic.utils.errors as err
import nmf_bioacoustic.utils.signal_to_spectrogram as signal_to_spectrogram

import IPython.display as ipd
import librosa


# TODO: handle CQT

# %% Audio to signal conversion
def spectrogram_to_audio_signal(spectrogram, feature_object, phase_retrieval = "griffin_lim", original_phase=None):
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
    return librosa.istft(stft_to_inverse, hop_length = hop_length)

