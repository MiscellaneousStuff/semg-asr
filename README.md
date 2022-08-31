# sEMG Silent Speech - Automatic Speech Recognition (ASR)

## About

This module is produced as part of the requirement for my Final Year Project
for my BSc Computer Science degree requirement at the University of Portsmouth.
This module represents
the second round of experimentation for my research project which involves
producing a speech recognition system for sEMG silent speech.

## Inspiration

The inspiration for this project largely comes from the landmark sEMG silent speech
synthesis papers "Digital Voicing of Silent Speech" at EMNLP 2020 and
"An Improved Model for Voicing Silent Speech" at ACL 2021. These papers provided
a method to transduce silent speech surface EMG (sEMG) signals directly
into speech features (either MFCCs or mel spectrograms). This project goes one step
further and uses the predicted speech features (mel spectrograms)
from the transduction model to directly perform speech recognition. To perform
this process with the current SOTA transduction model, you would have to use
the vocoder inbetween the transduction model and the deepspeech-0.7.0 model
used for the evaluations.

## Performance

The speech recognition trained uses 115 times less training data than using
the evaluation function in the original "An Improved Model for Voicing Silent Speech"
at ACL 2021 paper.

## Data

The dataset for this project is based on the open-source dataset released
with the "Digital Voicing of Silent Speech" at EMNLP 2020 paper with
the additional force-aligned phonemes released along with the
"An Improved Model for Voicing Silent Speech" at ACL 2021 paper.

The EMG and audio data can be downloaded from
[https://doi.org/10.5281/zenodo.4064408](https://doi.org/10.5281/zenodo.4064408).

And the force-aligned phonemes for the dataset (not including the closed
vocabulary portion of the dataset) can be downloaded from
[https://github.com/dgaddy/silent_speech_alignments/raw/main/text_alignments.tar.gz](https://github.com/dgaddy/silent_speech_alignments/raw/main/text_alignments.tar.gz).

## Logging

This project supports optional Neptune.ai based logging by using `.env` with
these settings:

```
NEPTUNE_PROJECT=<neptune_project_name>
NEPTUNE_TOKEN=<neptune_account_token>
```

# Quick Start Guide

## Clone the Repository

You can clone this repository by doing the following:

```bash
git clone https://github.com/MiscellaneousStuff/semg_asr.git
git submodule init
git submodule update
```

## Train

To train an sEMG Silent Speech speech recognition model, use

```bash
python3 train.py \
    --dataset_path "path_to_dataset.csv" \
    --semg_train
```

## Create Dataset

There are two main types of datasets which can be used with this module,
the first one is a regular ASR model which performs speech recognition
on the ground truth audio files. To create a dataset like this, use:

```bash
python3 create_dataset.py \
    --emg_dir "./silent_speech/emg_data" \
    --testset_path "./silent_speech/testset_largedev.json"
```

Whereas if you want to create a dataset which that uses the predicted
mel spectrograms from the transduction model which you have
already generated, use:

```bash
python3 create_dataset.py \
    --emg_dir "./silent_speech/emg_data" \
    --testset_path "./silent_speech/testset_largedev.json" \
    --semg_preds_path "./silent_speech/pred_audio"
```

## Evaluate

To evaluate the best trained model released with the report,
download the model from
[Google Drive](https://drive.google.com/file/d/1O8jIWV1v0orE4kOVA6IG-FgYyFO8OMDH/view?usp=sharing)
into this directory.
Then create a dataset of the full EMG data predictions using
the above instructions and run the following code:

```bash
python3 evaluate.py \
    --checkpoint_path "ds2_DATASET_SILENT_SPEECH_EPOCHS_10_TEST_LOSS_1.8498832106590273_WER_0.6825681123095443" \
    --dataset_path "path_to_dataset.csv" \
    --print_top 10 \
    --semg_eval
```

There are a large number of models and different datasets which have
been evaluated in the report, to find the full list of evaluation conditions
and how to run them, run:

```bash
python3 evaluate.py --help
```