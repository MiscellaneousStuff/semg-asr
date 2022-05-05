# sEMG Silent Speech - Automatic Speech Recognition (ASR)

## About

This module is produced as part of the requirement for my Final Year Project for my BSc Computer Science degree requirement at the University of Portsmouth.

## Inspiration

The inspiration for this project largely comes from the landmark sEMG silent speech
synthesis papers "Digital Voicing of Silent Speech" at EMNLP 2020 and
"An Improved Model for Voicing Silent Speech" at ACL 2021. These papers provided
a method to transduce silent speech surface EMG (sEMG) signals directly
into speech features (either MFCCs or mel spectrograms). This project goes one step
further and uses the predicted mel spectrograms from the transduction model and
directly performs speech recognition on these speech features.

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