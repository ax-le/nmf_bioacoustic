"""
Created on June 2024

@author: a23marmo

Defines the dataset class for the dataset of humpback whale. Inherits from the base dataset class.

TODO: add reference for the dataset, ask Dorian Cazau.
"""

from nmf_bioacoustic.datasets.base_dataset import *

class WhaleSing(GeneralDataset):

    def __init__(self, audio_path, feature_object):
        """
        Initializes the dataset object, exactly following tehe general dataset.
        """
        super().__init__(audio_path, None, feature_object)

    def get_annotations(self, idx):
        """
        Void function, as the dataset does not have annotations.
        """
        return None
