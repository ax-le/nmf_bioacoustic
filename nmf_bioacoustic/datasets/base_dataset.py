"""
Created on June 2024

@author: a23marmo

Defines a base dataset class, from which will inherit the individual and specialized dataset classes.
"""

# Imports
import librosa
import os
import pandas as pd

class GeneralDataset():
    # This class is a general dataset class, from which will inherit the specialized dataset classes.
    def __init__(self, audio_path, annotations_file, feature_object):
        """
        Initializes the dataset object.

        Parameters
        ----------
        audio_path : str
            Path to the audio files.
        annotations_file : str
            Path to the annotations file.
        feature_object : FeatureObject
            Feature object, defining the important parameters to compute spectrograms.
        """
        self.audio_path = audio_path
        self.all_signals = list(os.listdir(self.audio_path))
        self.feature_object = feature_object

    def __len__(self):
        """
        Returns the number of signals in the dataset.
        """
        return len(self.all_signals)
    
    def __getitem__(self, idx):
        """
        Returns the spectrogram and the annotations of the idx-th signal in the dataset.
        
        Parameters
        ----------
        idx : int
            Index of the signal in the dataset.

        Returns
        -------
        spec : numpy array
            Spectrogram of the signal.
        annot_this_file : not defined
            Annotations of the signal, the type of which will depend on the dataset.
        """
        signal, orig_sr = librosa.load(os.path.join(self.audio_path, self.all_signals[idx]), sr = self.feature_object.sr)
        if self.feature_object.sr != orig_sr:
            signal = librosa.resample(signal, orig_sr, self.feature_object.sr)
        spec = self.feature_object.get_spectrogram(signal)
        annot_this_file = self.get_annotations(idx)

        return spec, annot_this_file
    
    def get_annotations(self, idx):
        """
        Returns the annotations of the idx-th signal in the dataset.
        Base function, which should be implemented in the child class.

        Parameters
        ----------
        idx : int
            Index of the signal in the dataset.
        """
        raise NotImplementedError("This method should be implemented in the child class")
    
    def get_item_of_id(self, audio_id):
        """
        Returns the spectrogram and the annotations of the signal with the given id.
        
        Parameters
        ----------
        audio_id : str
            Id of the signal in the dataset.

        Returns
        -------
        spec : numpy array
            Spectrogram of the signal.
        annot_this_file : not defined
            Annotations of the signal, the type of which will depend on the dataset.
        """
        index = self.all_signals.index(audio_id)
        return self.__getitem__(index)

