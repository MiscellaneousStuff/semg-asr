# MIT License
# 
# Copyright (c) 2022 Tada Makepeace
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""This module encapsulates the two different datasets used for training
the sEMG ASR system."""

import csv
import torch
import torchaudio


class SilentSpeechDataset(torch.utils.data.Dataset):
    """Regular speech recognition dataset which contains audio wave forms as
    in the input and text transcriptions as the target from the silent speech
    dataset."""
    def __init__(self, metadata_path, dataset_type=None):
        with open(metadata_path) as metadata:
            flist = csv.reader(metadata, delimiter="|", quotechar="'", quoting=csv.QUOTE_MINIMAL)
            self._flist = list(flist)
            fis = []
            if dataset_type:
                for fi in self._flist:
                    line = fi
                    _, _, cur_dataset_type, _, _, _ = line
                    if cur_dataset_type == dataset_type:
                        fis.append(fi)
            else:
                Exception("No dataset type specified for SilentSpeech() dataset.""")
            self._flist = fis

    def get_exact(self, book, sentence_idx):
        lines = [fi for fi in self._flist
                 if fi[-2] == book and fi[-1] == sentence_idx]
        line = lines[0]
        cur_path, text, dataset_type, _, _, _ = line
        waveform, sr = torchaudio.load(cur_path)
        return (waveform, sr, text, dataset_type)

    def __getitem__(self, n):
        line = self._flist[n]
        cur_path, text, dataset_type, _, _, _ = line
        waveform, sr = torchaudio.load(cur_path)
        return (waveform, sr, text, dataset_type)

    def __len__(self):
        return len(self._flist)


class SilentSpeechPredDataset(torch.utils.data.Dataset):
    """Custom speech recognition dataset which contains predicted mel
    spectrograms from the transduction model as the input and the text
    transcriptions as the target."""
    def __init__(self, metadata_path, dataset_type=None, \
        silent_only=False, voiced_only=False):
        with open(metadata_path) as metadata:
            flist = csv.reader(metadata, delimiter="|", \
                quotechar="'", quoting=csv.QUOTE_MINIMAL)
            self._flist = list(flist)
            fis = []
            if dataset_type:
                for fi in self._flist:
                    line = fi
                    _, _, cur_dataset_type, modality, _, _ = line
                    if cur_dataset_type == dataset_type:
                        if silent_only and modality == "silent":
                            fis.append(fi)
                        elif voiced_only and modality == "voiced":
                            fis.append(fi)
                        elif not silent_only and not voiced_only:
                            fis.append(fi)
                        else:
                            Exception("You've selected silent only and voiced only.")
            else:
                Exception("No dataset type specified for SilentSpeechPred() dataset.""")

            self._flist = fis

    def get_item_vis(self, n):
        line = self._flist[n]
        cur_path, text, dataset_type, _, book, sentence_idx = line
        mel_spectrogram = torch.load(cur_path)
        return (mel_spectrogram, text, dataset_type, book, sentence_idx)

    def __getitem__(self, n):
        line = self._flist[n]
        cur_path, text, dataset_type, _, _, _ = line
        mel_spectrogram = torch.load(cur_path)
        return (mel_spectrogram, text, dataset_type)

    def __len__(self):
        return len(self._flist)