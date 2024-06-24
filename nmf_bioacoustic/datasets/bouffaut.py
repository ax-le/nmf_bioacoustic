from nmf_bioacoustic.datasets.base_dataset import *
import numpy as np

class Bouffaut(GeneralDataset):
    def __init__(self, audio_path, annotations_file, feature_object):
        super().__init__(audio_path, annotations_file, feature_object)
        self.annotations = pd.read_csv(annotations_file, delimiter='\t')

    def get_annotations(self, idx):
        annot_this_file = self.annotations[self.annotations["File"] == self.all_signals[idx].split(".")[0]]
        type = annot_this_file['Type']
        begin = annot_this_file['Begin Time (s)']
        end = annot_this_file['End Time (s) ']
        return type, begin, end
    
    def crop_annotations(self, annotations, time_limit_s):
        type, begin, end = annotations
        indices_ok = begin[begin < time_limit_s].keys()
        type = type[indices_ok]
        begin = begin[indices_ok]
        end = end[indices_ok]
        return type, begin, end
    
def create_one_annotation_file(dataset_path, annotations_1="Annotation.txt", annotations_2="Annotation_RR48_2013_D151.txt"):
    annot_1 = pd.read_csv(f"{dataset_path}/{annotations_1}", delimiter='\t')
    annot_2 = pd.read_csv(f"{dataset_path}/{annotations_2}", delimiter='\t')
    pd.concat([annot_1, annot_2]).to_csv(f"{dataset_path}/merged_annotations.txt", sep='\t', index=False)