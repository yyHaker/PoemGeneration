#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: baseRNN.py
@time: 2018/10/8 20:29
"""
import torch.nn as nn


class BaseRNN(nn.Module):
    """
    A base class for RNN.
    Applies a multi-layer RNN to an input sequence.
    Note:
        Do not use this class directly, use one of the sub classes.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): maximum allowed length for the sequence to be processed
        hidden_size (int): number of features in the hidden state `h`
        input_dropout_p (float): dropout probability for the input sequence
        output_dropout_p (float): dropout probability for the output sequence
        n_layers (int): number of recurrent layers
        rnn_cell (str): type of RNN cell (Eg. 'LSTM' , 'GRU')
    Inputs: ``*args``, ``**kwargs``
        - ``*args``: variable length argument list.
        - ``**kwargs``: arbitrary keyword arguments.
    Attributes:
        SYM_MASK: masking symbol
        SYM_EOS: end-of-sequence symbol
    """
    SYM_MASK = "MASK"
    SYM_EOS = "EOS"

    def __init__(self, vocab_size, max_len, hidden_size, input_dropout_p,
                 output_dropout_p, n_layers, rnn_cell):
        super(BaseRNN, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.output_dropout_p = output_dropout_p
        self.n_layers = n_layers

        if rnn_cell.lower() == "lstm":
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == "gru":
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {}".format(rnn_cell))

    def forward(self, *args, **kwargs):
        raise NotImplementedError
