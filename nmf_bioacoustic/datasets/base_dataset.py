import pandas as pd
import librosa
import os
import nmf_bioacoustic.utils.signal_to_spectrogram as sig_to_spec

class GeneralDataset():
    def __init__(self, audio_path, annotations_file, feature_object):
        self.audio_path = audio_path
        self.all_signals = list(os.listdir(self.audio_path))
        self.feature_object = feature_object

    def __len__(self):
        return len(self.all_signals)
    
    def __getitem__(self, idx):
        signal, orig_sr = librosa.load(os.path.join(self.audio_path, self.all_signals[idx]), sr = self.feature_object.sr)
        if self.feature_object.sr != orig_sr:
            signal = librosa.resample(signal, orig_sr, self.feature_object.sr)
        spec = self.feature_object.get_spectrogram(signal)
        annot_this_file = self.get_annotations(idx)

        return spec, annot_this_file
    
    def get_annotations(self, idx):
        raise NotImplementedError("This method should be implemented in the child class")
    
    def get_item_of_id(self, audio_id):
        index = self.all_signals.index(audio_id)
        return self.__getitem__(index)

