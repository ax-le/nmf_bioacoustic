from nmf_bioacoustic.datasets.base_dataset import *

class AnuraSet(GeneralDataset):
    def __init__(self, audio_path, annotations_file, feature_object):
        super().__init__(audio_path, annotations_file, feature_object)
        self.annotations = pd.read_csv(annotations_file)
        if 'nonzero_species_count' not in self.annotations.columns:
            self.annotations['nonzero_species_count'] = self.annotations.filter(regex='^SPECIES').apply(lambda row: (row != 0).sum(), axis=1)

    def get_annotations(self, idx):
        number_of_species = self.annotations['nonzero_species_count'][self.annotations["AUDIO_FILE_ID"] == self.all_signals[idx].split(".")[0]].values[0]
        return number_of_species