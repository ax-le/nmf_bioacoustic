from nmf_bioacoustic.datasets.base_dataset import *

class WhaleSing(GeneralDataset):
    def __init__(self, audio_path, feature_object):
        super().__init__(audio_path, None, feature_object)

    def get_annotations(self, idx):
        return None
