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
"""This module contains functions to visualise the mel spectrograms of
ground truth audio files along with their predicted mel spectrograms from
the transduction model."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import librosa

import torch
import torch.utils.data as data

from absl import flags
from absl import app

from datasets import SilentSpeechDataset, SilentSpeechPredDataset
from preprocessing import valid_audio_transforms

FLAGS = flags.FLAGS
flags.DEFINE_string("pred_dataset_path", None, \
    "Path to pred *.csv file which defines the dataset to evaluate")
flags.DEFINE_string("ground_dataset_path", None, \
    "Path to ground *.csv file which defines the dataset to evaluate")
flags.DEFINE_string("testset_path", None, "Path to transduction model testset.json")
flags.DEFINE_boolean("closed_only", False, \
    "(Optional) Evaluate only on the closed vocabulary slice of the dataset")
flags.DEFINE_integer("max_examples", 10, "Number of testset examples to visualise")
flags.mark_flag_as_required("pred_dataset_path")
flags.mark_flag_as_required("ground_dataset_path")

def stack_mel_spectrogram(data):
    # Loop over each second of `audio_features`
    new_data = data[0]
    for i in range(1, data.shape[0]):
        new_data = np.vstack((new_data, data[i]))
    
    return new_data

def plot_mel_spectrograms(pred, y, text):
    fig, ax = plt.subplots(2) # nrows=1, ncols=2)

    # ax[0].set_title(f"Mel Spectogram (Predicted)")
    pred = np.swapaxes(pred, 0, 1)
    cax = ax[0].imshow(pred, interpolation='nearest', cmap=cm.coolwarm, origin='lower')

    # ax[1].set_title(f"Mel Spectogram (Ground Truth)")
    y = np.swapaxes(y, 0, 1)
    cax = ax[1].imshow(y, interpolation='nearest', cmap=cm.coolwarm, origin='lower')

    return fig, ax

def main(unused_argv):
    g_dataset_path    = FLAGS.ground_dataset_path
    p_dataset_path    = FLAGS.pred_dataset_path
    closed_only     = FLAGS.closed_only

    # get desired book, sentence_idx

    # ground truth
    # voiced pred
    # silent pred

    pred_test_dataset = SilentSpeechPredDataset(\
        p_dataset_path, dataset_type="test", silent_only=True)
    
    ground_test_dataset = SilentSpeechDataset(\
        g_dataset_path, dataset_type="test")

    for i in range(len(pred_test_dataset))[0:min(len(pred_test_dataset)-1, FLAGS.max_examples)]:
        p_mel_spectrogram, p_text, \
            _, book, sentence_idx = pred_test_dataset.get_item_vis(i)
        waveform, sr, g_text, _   = ground_test_dataset.get_exact(book, sentence_idx)

        g_mel_spectrogram = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)

        g_mel_spectrogram = torch.log(g_mel_spectrogram+1e-5)

        fig, ax = plot_mel_spectrograms(\
            stack_mel_spectrogram(p_mel_spectrogram),
            stack_mel_spectrogram(g_mel_spectrogram),
            g_text)
        
        print(g_text, sentence_idx)

        ax1 = ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted()).expanded(1.2, 1.3)
        ax2 = ax[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted()).expanded(1.2, 1.3)
        fig.savefig(f"./testset_visuals/{sentence_idx}_p.png", bbox_inches=ax1)
        fig.savefig(f"./testset_visuals/{sentence_idx}_g.png", bbox_inches=ax2)

def entry_point():
    app.run(main)

if __name__ == "__main__":
    app.run(main)