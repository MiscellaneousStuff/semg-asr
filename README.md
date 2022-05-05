# sEMG Silent Speech - Automatic Speech Recognition (ASR)

## About

This module is produced as part of the requirement for my Final Year Project for my BSc Computer Science degree requirement at the University of Portsmouth. This module represents
the second round of experimentation for my research project which involves
producing a speech recognition system for sEMG silent speech.

## Inspiration

The inspiration for this project largely comes from the landmark sEMG silent speech
synthesis papers "Digital Voicing of Silent Speech" at EMNLP 2020 and
"An Improved Model for Voicing Silent Speech" at ACL 2021. These papers provided
a method to transduce silent speech surface EMG (sEMG) signals directly
into speech features (either MFCCs or mel spectrograms). This project goes one step
further and uses the predicted mel spectrograms from the transduction model and
directly performs speech recognition on these speech features to produce a highly
efficient silent speech recognition system.

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

## Evaluate

To evaluate the best trained model released with the report, run the
following code:

```bash
python3 evaluate.py \
    --checkpoint_path "path_to_pretrained_model/ds2_DATASET_SILENT_SPEECH_EPOCHS_10_TEST_LOSS_1.8498832106590273_WER_0.6825681123095443" \
    --dataset_path "path_to_dataset.csv" \
    --semg_eval
```

There are a large number of models and different datasets which have
been evaluated in the report, to find the full list of evaluation conditions
and how to run them, run:

```bash
python3 evaluate.py --help
```