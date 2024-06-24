from nmf_bioacoustic.datasets.base_dataset import *

class NARWDataset(GeneralDataset):
    def __init__(self, audio_path, annotations_file, feature_object):
        super().__init__(audio_path, annotations_file, feature_object)
        self.annotations = pd.read_csv(annotations_file)
    
    def get_annotations(self, idx):
        annot_this_file = self.annotations[self.annotations["filename"] == self.all_signals[idx]]
        annotated_detections = list(annot_this_file["timestamp"])
        return annotated_detections