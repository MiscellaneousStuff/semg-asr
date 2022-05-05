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
"""This module contains different functions used to decode the predictions from
the speech recognition network. It currently only contains a greedy decoder which
picks the most likely character at each time step, rather than considering longer
sequences."""

import torch

def GreedyDecoder(output, labels, label_lengths, encoder, blank_label=28,\
                  collapse_repeated=True):
    """Decodes the predicted text transcription by picking the character
    with the highest probability at each timestep. This decoding method
    has the fastest runtime but also the lowest accuracy, however it is
    also the simplest to implement."""

    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        cur_target = labels[i][:label_lengths[i]]
        if len(cur_target) > 0:
            cur_target = \
                "".join(encoder.batch_decode(torch.tensor(cur_target)))
        else:
            cur_target = ""
        targets.append(cur_target)

        # Greedy decoding process
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index)
        cur_decode = decode
        
        if len(cur_decode) > 0:
            cur_decode = \
                "".join(encoder.batch_decode(torch.tensor(cur_decode)))
        else:
            cur_decode = ""
        decodes.append(cur_decode)

    return decodes, targets