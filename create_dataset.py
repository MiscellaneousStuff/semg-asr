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
"""Creates a CSV file which contains the absolute path, text transcription,
dataset and speech modality for each utterance."""

import argparse
import os
import json
import csv
from unidecode import unidecode

from absl import flags
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_string("emg_dir", None, \
    "Absolute directory containing EMG dataset along with audio and text")
flags.DEFINE_string("semg_preds_path", "",\
    "(Optional) Path to sEMG Silent Speech mel spectrogram predictions")
flags.DEFINE_boolean("closed_only", False, \
    "(Optional) Only include data from the closed vocabulary dataset")
flags.DEFINE_boolean("remove_nonparallel", False, \
    "(Optional) Removes voiced_non_parallel data from dataset.\n"
    "Included for ablation studies.")
flags.DEFINE_string("testset_path", None, "Path to transduction model testset.json")
flags.mark_flag_as_required("emg_dir")
flags.mark_flag_as_required("testset_path")

def main(unused_argv):
    base_dir           = FLAGS.emg_dir
    closed_only        = FLAGS.closed_only
    remove_nonparallel = FLAGS.remove_nonparallel
    testset_path       = FLAGS.testset_path
    semg_preds_path    = FLAGS.semg_preds_path

    if closed_only:
        top_dirs = [
            ["closed_vocab/silent/", "silent"],
            ["closed_vocab/voiced/", "voiced"]]
    else:
        top_dirs = [
            ["voiced_parallel_data", "voiced"],
            ["silent_parallel_data", "silent"]]
        if not remove_nonparallel:
            top_dirs += [["nonparallel_data", "voiced"]]

    with open(testset_path) as f:
        datasets = json.loads(f.read())

    if semg_preds_path:
        csv_path = "metadata_dgaddy_preds.csv"
    else:
        csv_path = "metadata_dgaddy.csv"
    
    with open(csv_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(
            csvfile,
            delimiter="|",
            quotechar="'",
            quoting=csv.QUOTE_MINIMAL)
        for top_dir, modality in top_dirs:
            sub_dirs = os.listdir(os.path.join(base_dir, top_dir))
            for sub_dir in sub_dirs:
                cur_fis   = os.listdir(os.path.join(base_dir, top_dir, sub_dir))
                cur_infos = list(filter(lambda fi: fi.endswith(".json"), cur_fis))
                for info_fi in cur_infos:
                    info_path = os.path.join(base_dir, top_dir, sub_dir, info_fi)
                    with open(info_path) as f:
                        info = json.loads(f.read())
                        sentence_idx = info["sentence_index"]
                        book = info["book"]

                        if sentence_idx != -1:
                            file_idx = info_fi.split("_")[0]
                            fname = f"{file_idx}_audio_clean.flac"
                            audio_path = \
                                os.path.join(base_dir, top_dir, sub_dir, fname)
                            
                            cur_pair = [book, sentence_idx]
                            dataset = "train"
                            dataset = "valid" if cur_pair in datasets["dev"]  else "train"
                            dataset = "test"  if cur_pair in datasets["test"] else "train"

                            valid_idxs = [d[1] for d in datasets["dev"]]

                            text = unidecode(info["text"])

                            if semg_preds_path:
                                if book == "books/War_of_the_Worlds.txt":
                                    base_pred_dir = os.path.join(\
                                        semg_preds_path, "open_vocab_parallel")
                                elif book == "books/Sherlock_Holmes.txt":
                                    base_pred_dir = os.path.join(\
                                        semg_preds_path, "open_vocab_non_parallel")
                                elif book == "datedata.txt":
                                    base_pred_dir = os.path.join(\
                                        semg_preds_path, "closed_vocab")
                                else:
                                    raise ValueError(
                                        "This module was not built to handle data from"
                                        f" this book {book}!")

                                mel_spectrogram_path = \
                                    os.path.join(base_pred_dir, \
                                                 modality, \
                                                 str(sentence_idx))

                                audio_path = mel_spectrogram_path

                            if modality == "voiced":
                                if sentence_idx in valid_idxs:
                                    dataset = "valid"  
                            
                            csv_writer.writerow([audio_path, text, dataset, modality])
def entry_point():
    app.run(main)

if __name__ == "__main__":
    app.run(main)