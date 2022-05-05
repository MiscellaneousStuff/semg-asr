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
"""Creates a hyperparameter dictionary to be used to set the
SpeechRecognitionModel hyperparameters and to make it easier to log the
hyperparameters during training."""

def get_hparams(n_class,
                learning_rate=5e-4,
                n_cnn_layers=3,
                n_rnn_layers=5,
                rnn_dim=512,
                n_feats=128,
                stride=2,
                dropout=0.1):
    hparams = {
        "n_cnn_layers":  n_cnn_layers,
        "n_rnn_layers":  n_rnn_layers,
        "rnn_dim":       rnn_dim,
        "n_class":       n_class,
        "n_feats":       n_feats,
        "stride":        stride,
        "dropout":       dropout,
        "learning_rate": learning_rate,
    }
    return hparams