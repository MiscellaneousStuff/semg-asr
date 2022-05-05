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
"""This module contains utility functions for preprocessing the speech
recognition datasets."""

import jiwer
import torchaudio

import torch.nn as nn

transformation = jiwer.Compose(\
    [jiwer.RemovePunctuation(), jiwer.ToLowerCase()])

# NOTE: Hyperparameters are set to match the transduction model
train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(
        sample_rate=16_000,
        n_mels=128,
        hop_length=160,
        win_length=432,
        n_fft=512,
        center=False),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    torchaudio.transforms.TimeMasking(time_mask_param=35))

valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

def data_processing(data, encoder, data_type="train"):
    """Function used to pre-process individual utterances from a ground truth
    audio dataset. Also supports collecting multiple mel spectrograms
    and padding them for training in a recurrent neural network."""
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    for cur in data:
        waveform, _, utterance, dataset_type = cur

        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        elif data_type == "valid":
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            raise Exception('data_type should be train or valid')
        spectrograms.append(spec)

        label = transformation(utterance)
        label = encoder.batch_encode(utterance.lower())
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths

def data_processing_preds(data, encoder):
    """Function used to pre-process individual utterances from a dataset
    made from predicted mel spectrograms from the transduction model.
    Also supports collecting multiple mel spectrograms and padding them
    for training in a recurrent neural network."""
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    for cur in data:
        mel_spectrogram, utterance, _ = cur

        spectrograms.append(mel_spectrogram)

        label = transformation(utterance)
        label = encoder.batch_encode(utterance.lower())
        labels.append(label)
        input_lengths.append(mel_spectrogram.shape[0]//2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)

    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths