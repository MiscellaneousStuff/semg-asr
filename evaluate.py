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
"""Evaluates the performance of either a ground truth audio model to get
a baseline word-error rate or a model trained on the mel spectrograms of the
transduction model to get the word-error rate on the silent speech testset."""

import random
import numpy as np
from jiwer import wer, cer

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

from absl import flags
from absl import app

from datasets import SilentSpeechDataset, SilentSpeechPredDataset
from preprocessing import data_processing, data_processing_preds
from hparams import get_hparams
from model import SpeechRecognitionModel
from decoder import closed_vocab_encoder, open_vocab_encoder, GreedyDecoder

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_path", None, "Path to the pre-trained DeepSpeech2 model")
flags.DEFINE_boolean("semg_eval", False, \
    "(Optional) Evaluate an ASR model on predicted mel spectrograms from the transducer."
    "Otherwise evaluate the ground truth audio files.")
flags.DEFINE_integer("random_seed", 7, \
    "(Optional) Set a different random seed if you train a different model."
    "The models trained along with this release used a random seed of 7 by default.")
flags.DEFINE_string("dataset_path", None, \
    "Path to *.csv file which defines the dataset to evaluate")
flags.DEFINE_integer("batch_size", 5, "Sets the batch size for the evaluation")
flags.DEFINE_boolean("closed_only", False, \
    "(Optional) Evaluate only on the closed vocabulary slice of the dataset")
flags.DEFINE_integer("print_top", 3, \
    "(Optional) Set number of most accurate predictions to print")
flags.mark_flag_as_required("checkpoint_path")
flags.mark_flag_as_required("dataset_path")

def evaluate(model, test_loader, device, criterion, encoder):
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []

    # Print the most accurate transcription
    scored_preds = []

    with torch.no_grad():
        for i, _data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths = _data 
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)

            test_loss += loss.item() / len(test_loader)
            decoded_preds, decoded_targets = \
                GreedyDecoder(
                    output.transpose(0, 1), labels, label_lengths, encoder)
            
            for j in range(len(decoded_preds)):
                cur_ground = decoded_targets[j]
                cur_pred   = decoded_preds[j]
                cur_wer = wer(cur_ground, cur_pred)
                cur_cer = cer(cur_ground, cur_pred)
                test_cer.append(cur_cer)
                test_wer.append(cur_wer)

                scored_preds.append([cur_ground, cur_pred, cur_wer])

    avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)

    print(\
        'Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'\
        .format(test_loss, avg_cer, avg_wer))

    sorted_preds = sorted(scored_preds, key=lambda pred: pred[2])

    print_top = FLAGS.print_top
    if print_top > 0:
        sorted_preds = \
            sorted_preds[0:min(print_top, len(sorted_preds) - 1)]
        for i, pred in enumerate(sorted_preds):
            # Get rid of unknown tokens in final output
            ground     = pred[0].replace("<unk>", "")
            prediction = pred[1].replace("<unk>", "")

            score  = pred[2]
            print(f"{i+1}.\n Target:     {ground}\n Prediction: {prediction}\n WER: {score:4f}")

def main(unused_argv):
    checkpoint_path = FLAGS.checkpoint_path
    semg_eval       = FLAGS.semg_eval
    seed            = FLAGS.random_seed
    dataset_path    = FLAGS.dataset_path
    batch_size      = FLAGS.batch_size
    closed_only     = FLAGS.closed_only

    # NOTE: All of the original experiments used a random_seed := 7
    # Fix the seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    encoder = closed_vocab_encoder if closed_only else open_vocab_encoder

    if semg_eval:
        test_dataset = SilentSpeechPredDataset(\
            dataset_path, dataset_type="test", silent_only=True)
        test_loader = data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: data_processing_preds(x, encoder),
            **kwargs)
    else:
        test_dataset = SilentSpeechDataset(\
            dataset_path, dataset_type="test", silent_only=True)
        test_loader = data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: data_processing(x, 'valid', encoder),
            **kwargs)

    blank   = 0 if closed_only else 28
    hparams = get_hparams(n_class=len(encoder.vocab))
    model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path))

    criterion = nn.CTCLoss(blank=blank).to(device)
    evaluate(model, test_loader, device, criterion, encoder)

def entry_point():
    app.run(main)

if __name__ == "__main__":
    app.run(main)