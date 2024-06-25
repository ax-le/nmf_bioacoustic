"""
Created on June 2024

@author: a23marmo

Defines the dataset class for the NARW (North Atlantic Right Whale) dataset. Inherits from the base dataset class.
"""

from nmf_bioacoustic.datasets.base_dataset import *

class NARWDataset(GeneralDataset):

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
        super().__init__(audio_path, annotations_file, feature_object)
        self.annotations = pd.read_csv(annotations_file)
    
    def get_annotations(self, idx):
        """
        Returns the annotations of the idx-th signal in the dataset.
        Annotations consist here of the time and types of the calls.

        Parameters
        ----------
        idx : int
            Index of the signal in the dataset.

        Returns
        -------
        annot_this_file : pandas Series
            Time and type of the calls.
        """ 
        annot_this_file = self.annotations[self.annotations["filename"] == self.all_signals[idx]]
        annotated_detections = list(annot_this_file["timestamp"])
        return annotated_detections