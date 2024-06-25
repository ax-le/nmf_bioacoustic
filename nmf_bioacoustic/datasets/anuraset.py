"""
Created on June 2024

@author: a23marmo

Defines the dataset class for the AnuraSet dataset. Inherits from the base dataset class.
Annotations are simplified to the number of species present in the signal.
Species are counted as individual species, hence the number of species may be different from the number of calls.
"""

from nmf_bioacoustic.datasets.base_dataset import *

class AnuraSet(GeneralDataset):

    def __init__(self, audio_path, annotations_file, feature_object):
        super().__init__(audio_path, annotations_file, feature_object)
        self.annotations = pd.read_csv(annotations_file)
        if 'nonzero_species_count' not in self.annotations.columns:
            self.annotations['nonzero_species_count'] = self.annotations.filter(regex='^SPECIES').apply(lambda row: (row != 0).sum(), axis=1)

    def get_annotations(self, idx):
        """
        Returns the annotations of the idx-th signal in the dataset.
        Annotations consist here of the number of species present in the signal.
        This is a simplification of the original dataset, where the annotations are more complex.
        Species are counted as individual species, hence the number of species may be different from the number of calls.

        Parameters
        ----------
        idx : int
            Index of the signal in the dataset.

        Returns
        -------
        annot_this_file : int
            Number of species present in the signal.
        """ 
        number_of_species = self.annotations['nonzero_species_count'][self.annotations["AUDIO_FILE_ID"] == self.all_signals[idx].split(".")[0]].values[0]
        return number_of_species