# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:54:59 2020

@author: amarmore

Computing spectrogram in different feature description.

Note that Mel (and variants of Mel) spectrograms follow the particular definition of [1].

[1] Grill, T., & Schlüter, J. (2015, October). 
Music Boundary Detection Using Neural Networks on Combined Features and Two-Level Annotations. 
In ISMIR (pp. 531-537).
"""

import numpy as np
import librosa.core
import librosa.feature
import librosa.effects
from math import inf
import nmf_bioacoustic.utils.errors as err
import IPython.display as ipd

mel_power = 2

class FeatureObject():
    def __init__(self, sr, feature, hop_length, n_fft=2048, fmin = 0, fmax=None, mel_grill = True, n_mels=80):
        self.sr = sr
        self.feature = feature.lower()
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.fmin = fmin
        self.fmax = fmax
        self.mel_grill = mel_grill
        self.n_mels = n_mels

        match self.feature:
            case "pcp":
                self.frequency_dimension = 12
            case "cqt":
                self.frequency_dimension = 84
            case "mel" | "log_mel" | "nn_log_mel" | "padded_log_mel" | "minmax_log_mel":
                self.frequency_dimension = self.n_mels
            case "stft" | "stft_complex":
                self.frequency_dimension = self.n_fft // 2 + 1
            case _:
                raise err.InvalidArgumentValueException(f"Unknown signal representation: {self.feature}.")


    # TODO: add MFCC, maybe tonnetz
    def get_spectrogram(self, signal):
        """
        Returns a spectrogram, from the signal of a song.
        Different types of spectrogram can be computed, which are specified by the argument "feature".
        All these spectrograms are computed with the toolbox librosa [1].
        
        Parameters
        ----------
        signal : numpy array
            Signal of the song.
        sr : float
            Sampling rate of the signal, (typically 44100Hz).
        feature : String
            The types of spectrograms to compute.
                TODO

        hop_length : integer
            The desired hop_length, which is the step between two frames (ie the time "discretization" step)
            It is expressed in terms of number of samples, which are defined by the sampling rate.
        fmin : integer, optional
            The minimal frequence to consider, used for denoising.
            The default is 98.
        n_mfcc : integer, optional
            Number of mfcc features.
            The default is 20 (as in librosa).

        Raises
        ------
        InvalidArgumentValueException
            If the "feature" argument is not presented above.

        Returns
        -------
        numpy array
            Spectrogram of the signal.
            
        References
        ----------
        [1] McFee, B., Raffel, C., Liang, D., Ellis, D. P., McVicar, M., Battenberg, E., & Nieto, O. (2015, July).
        librosa: Audio and music signal analysis in python. 
        In Proceedings of the 14th python in science conference (Vol. 8).
        
        [2] Grill, T., & Schlüter, J. (2015, October). 
        Music Boundary Detection Using Neural Networks on Combined Features and Two-Level Annotations. 
        In ISMIR (pp. 531-537).
        """
        match self.feature:
            case "pcp":
                return self._compute_pcp(signal)
        
            case "cqt":
                return self._compute_cqt(signal)
        
            # For Mel spectrograms, by default we use the same parameters as the ones of [2].
            # [2] Grill, Thomas, and Jan Schlüter. "Music Boundary Detection Using Neural Networks on Combined Features and Two-Level Annotations." ISMIR. 2015.
            case "mel":
                return self._compute_mel_spectrogram(signal)
        
            case "log_mel" | "nn_log_mel" | "padded_log_mel" | "minmax_log_mel":
                mel_spectrogram = self._compute_mel_spectrogram(signal)
                return get_log_mel_from_mel(mel_spectrogram, self.feature)
            
            case "stft":
                return self._compute_stft(signal, complex = False)
            case "stft_complex":
                return self._compute_stft(signal, complex = True)
        
            case _:
                raise err.InvalidArgumentValueException(f"Unknown signal representation: {self.feature}.")
        
    def _compute_pcp(self, signal):
        norm=inf # Columns normalization
        win_len_smooth=82 # Size of the smoothign window
        n_octaves=6
        bins_per_chroma = 3
        bins_per_octave=bins_per_chroma * 12
        fmin=98
        return librosa.feature.chroma_cens(y=signal,sr=self.sr,hop_length=self.hop_length,
                                    fmin=fmin, n_chroma=12, n_octaves=n_octaves, bins_per_octave=bins_per_octave,
                                    norm=norm, win_len_smooth=win_len_smooth)

    def _compute_cqt(self, signal):
        constant_q_transf = librosa.cqt(y=signal, sr = self.sr, hop_length = self.hop_length)
        return np.abs(constant_q_transf)

    def _compute_mel_spectrogram(self, signal):
        if self.mel_grill:
            mel = librosa.feature.melspectrogram(y=signal, sr = self.sr, n_fft=2048, hop_length = self.hop_length, n_mels=80, fmin=80.0, fmax=16000, power=mel_power)
        else:
            mel = librosa.feature.melspectrogram(y=signal, sr = self.sr, n_fft=self.n_fft, hop_length = self.hop_length, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax, power=mel_power)
        return np.abs(mel)
        
    def _compute_stft(self, signal, complex):
        stft = librosa.stft(y=signal, hop_length=self.hop_length,n_fft=self.n_fft)
        if complex:
            mag, phase = librosa.magphase(stft, power = 1)
            return mag, phase
        else:
            return np.abs(stft)
        
    def get_stft_from_mel(self, mel_spectrogram, feature=None):
        if feature is None: # Recursive function, so it takes the feature as an argument
            feature = self.feature # Default case takes the object feature as the feature to compute

        match feature: 
            case "mel":
                if self.mel_grill:
                    return librosa.feature.inverse.mel_to_stft(M=mel_spectrogram, sr=self.sr, n_fft=2048, power=mel_power, fmin=80.0, fmax=16000) # Fixed as such
                else:
                    return librosa.feature.inverse.mel_to_stft(M=mel_spectrogram, sr=self.sr, n_fft=self.n_fft, power=mel_power, fmin=self.fmin, fmax=self.fmax)
        
            case "log_mel":
                mel = librosa.db_to_power(S_db=mel_spectrogram, ref=1)
                return self.get_stft_from_mel(mel, "mel")

            case "nn_log_mel":
                mel = librosa.db_to_power(S_db=mel_spectrogram, ref=1) - np.ones(mel_spectrogram.shape)
                return self.get_stft_from_mel(mel, "mel")

            case _:
                raise err.InvalidArgumentValueException("Unknown feature representation.")

def get_log_mel_from_mel(mel_spectrogram, feature):
    """
    Computes a variant of a Mel spectrogram (typically Log Mel).

    Parameters
    ----------
    mel_spectrogram : numpy array
        Mel spectrogram of the signal.
    feature : string
        Desired feature name (must be a variant of a Mel spectrogram).

    Raises
    ------
    err.InvalidArgumentValueException
        Raised in case of unknown feature name.

    Returns
    -------
    numpy array
        Variant of the Mel spectrogram of the signal.

    """
    match feature:
        case "log_mel":
            return librosa.power_to_db(np.abs(mel_spectrogram), ref=1)
    
        case "nn_log_mel":
            mel_plus_one = np.abs(mel_spectrogram) + np.ones(mel_spectrogram.shape)
            nn_log_mel = librosa.power_to_db(mel_plus_one, ref=1)
            return nn_log_mel
    
        case "padded_log_mel":
            log_mel = get_log_mel_from_mel(mel_spectrogram, "log_mel")
            return log_mel - np.amin(log_mel) * np.ones(log_mel.shape)
        
        case "minmax_log_mel":        
            padded_log_mel = get_log_mel_from_mel(mel_spectrogram, "padded_log_mel")
            return np.divide(padded_log_mel, np.amax(padded_log_mel))
    
        case _:
            raise err.InvalidArgumentValueException("Unknown feature representation.")

