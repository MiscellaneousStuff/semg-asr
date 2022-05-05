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
"""Train a DeepSpeech2-lie speech recognition model either using ground
truth audio files on the predicted mel spectrograms from the transduction
model."""

import random
import numpy as np
from dotenv import dotenv_values
from jiwer import wer, cer

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.cuda.amp.grad_scaler import GradScaler

from absl import flags
from absl import app

from datasets import SilentSpeechDataset, SilentSpeechPredDataset
from preprocessing import data_processing, data_processing_preds
from hparams import get_hparams
from model import SpeechRecognitionModel
from decoder import closed_vocab_encoder, open_vocab_encoder, GreedyDecoder

import neptune.new as neptune

FLAGS = flags.FLAGS
flags.DEFINE_integer("n_epochs", 100,\
     "Recommended epochs is 200 for closed vocab and 100 for the others")
flags.DEFINE_integer("random_seed", 7, \
    "(Optional) Set a different random seed if you train a different model.\n"
    "The models trained along with this release used a random seed of 7 by default.")
flags.DEFINE_boolean("semg_train", False, \
    "(Optional) Train an ASR model on predicted mel spectrograms from the transducer.\n"
    "Otherwise train on the ground truth audio files.")
flags.DEFINE_boolean("silent_only", False, \
    "(Optional) Transduction dataset only.\n"
    "Train only on the mel spectrograms predicted from EMG signals during silent speech")
flags.DEFINE_boolean("voiced_only", False, \
    "(Optional) Transduction dataset only.\n"
    "Train only on the mel spectrograms predicted from EMG signals during vocalised speech")
flags.DEFINE_boolean("amp", False, \
    "(Optional) Train using Automatic Mixed Precision (AMP)")
flags.DEFINE_string("checkpoint_path", None, "Start training from pre-trained model")
flags.DEFINE_string("dataset_path", None, \
    "Path to *.csv file which defines the dataset to train on")
flags.DEFINE_integer("batch_size", 5, "Number of examples to train on per mini-batch")
flags.DEFINE_boolean("closed_only", False, \
    "(Optional) Evaluate only on the closed vocabulary slice of the dataset")
flags.mark_flag_as_required("dataset_path")

def get_dataloaders(
    semg_train, silent_only, voiced_only, hparams, kwargs, dataset_path, encoder):
    if semg_train:
        train_dataset = SilentSpeechPredDataset(\
            dataset_path,
            dataset_type="train",
            silent_only=silent_only,
            voiced_only=voiced_only)
        test_dataset  = SilentSpeechPredDataset(\
            dataset_path,
            dataset_type="test",
            silent_only=True)
        train_loader = data.DataLoader(dataset=train_dataset,
            batch_size=hparams['batch_size'],
            shuffle=True,
            collate_fn=lambda x: data_processing_preds(x, encoder),
            **kwargs)
        test_loader = data.DataLoader(dataset=test_dataset,
            batch_size=hparams['batch_size'],
            shuffle=False,
            collate_fn=lambda x: data_processing_preds(x, encoder),
            **kwargs)
    else:
        train_dataset = SilentSpeechDataset(\
            dataset_path,
            dataset_type="train")
        test_dataset  = SilentSpeechDataset(\
            dataset_path,
            dataset_type="test")
        train_loader = data.DataLoader(dataset=train_dataset,
            batch_size=hparams['batch_size'],
            shuffle=True,
            collate_fn=lambda x: data_processing(x, 'train'),
            **kwargs)
        test_loader = data.DataLoader(dataset=test_dataset,
            batch_size=hparams['batch_size'],
            shuffle=False,
            collate_fn=lambda x: data_processing(x, 'valid'),
            **kwargs)
    return train_loader, test_loader

def test(model, device, test_loader, criterion, run, encoder):
    print('\nevaluating...')
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []

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
                GreedyDecoder(output.transpose(0, 1), labels, label_lengths, encoder)

            for j in range(len(decoded_preds)):
                test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

    avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)

    if run:
        run["test_loss"].log(test_loss)
        run["cer"].log(avg_cer)
        run["wer"].log(avg_wer)
    
    print(
        'Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'\
        .format(test_loss, avg_cer, avg_wer))

    return test_loss, avg_wer

def train(model, device, train_loader, criterion, optimizer, scheduler, epoch, run):
    model.train()
    data_len = len(train_loader.dataset)

    scaler = GradScaler()
    amp_enabled = FLAGS.amp

    for batch_idx, _data in enumerate(train_loader):
        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        with torch.autocast(
            enabled=amp_enabled,
            dtype=torch.bfloat16,
            device_type="cuda"):

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if run:
            run["train_loss"].log(loss.item())
            run["learning_rate"].log(scheduler.get_last_lr())

        scheduler.step()

        if batch_idx % 100 == 0 or batch_idx == data_len:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(spectrograms), data_len,
                100. * batch_idx / len(train_loader), loss.item()))

def main(unused_argv):
    # Setup Neptune.ai logging if credentials are set in .env
    config = dotenv_values(".env")
    if "NEPTUNE_PROJECT" in config and "NEPTUNE_TOKEN" in config:
        neptune_project = config["NEPTUNE_PROJECT"]
        neptune_token   = config["NEPTUNE_TOKEN"]

        run = neptune.init(project=neptune_project,
                        api_token=neptune_token)
    else:
        run = None
    
    # Training settings
    epochs          = FLAGS.n_epochs
    checkpoint_path = FLAGS.checkpoint_path
    seed            = FLAGS.random_seed
    dataset_path    = FLAGS.dataset_path
    batch_size      = FLAGS.batch_size
    closed_only     = FLAGS.closed_only
    semg_train      = FLAGS.semg_train
    silent_only     = FLAGS.silent_only
    voiced_only     = FLAGS.voiced_only

    # NOTE: All of the original experiments used a random_seed := 7
    # Fix the seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    encoder = closed_vocab_encoder if closed_only else open_vocab_encoder

    # Load the model using defaut hyperparameters
    blank   = 0 if closed_only else 28
    hparams = get_hparams(n_class=len(encoder.vocab))
    hparams["batch_size"] = batch_size
    hparams["epochs"] = epochs

    if run:
        run["hparams"] = hparams
        run["device"] = device

    model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
    ).to(device)
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
        if run:
            run["checkpoint_path"] = checkpoint_path

    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    criterion = nn.CTCLoss(blank=blank).to(device)

    train_loader, test_loader = \
        get_dataloaders(semg_train,
                        silent_only,
                        voiced_only,
                        hparams, 
                        kwargs,
                        dataset_path,
                        encoder)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=hparams['learning_rate'], 
        steps_per_epoch=int(len(train_loader)),
        epochs=hparams['epochs'],
        anneal_strategy='linear')

    best_avg_wer = float("inf")

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, criterion, optimizer, scheduler, epoch, run)
        test_loss, avg_wer = test(model, device, test_loader, criterion, run, encoder)

        if avg_wer < best_avg_wer:
            torch.save(
                model.state_dict(),
                f"./models/ds2_DATASET_SILENT_SPEECH_EPOCHS_{epoch}_TEST_LOSS_{test_loss}_WER_{avg_wer}")
            best_avg_wer = avg_wer

def entry_point():
    app.run(main)

if __name__ == "__main__":
    app.run(main)