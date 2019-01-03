import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np


class PointerNet(nn.Module):
    def __init__(self, embedding_dim,
                 hidden_dim,
                 lstm_layers,
                 dropout,
                 vocab_size,
                 init_embedding_weight,
                 bidir=False):
        super(PointerNet, self).__init__()

        self.embedding = PoemEmbedding(vocab_size, embedding_dim, init_embedding_weight)
        self.encoder = CNNEncoder(embedding_dim, output_dim=hidden_dim)
        self.decoder = Decoder(embedding_dim, hidden_dim)

        self.decoder_input0 = Parameter(torch.FloatTensor(embedding_dim), requires_grad=False)

        # Initialize decoder_input0
        nn.init.uniform(self.decoder_input0, -1, 1)

    def forward(self, inputs):
        """
        :param inputs: inputs tokens, [bz, seq_len]
        :return:
        """
        batch_size = inputs.size(0)
        input_length = inputs.size(1)

        # encode
        inputs = inputs.view(batch_size * input_length, -1)
        embedded_inputs = self.embedding(inputs).view(batch_size, input_length, -1)

        # encoder_outputs: [batch, hidden_size]
        encoder_outputs = self.encoder(embedded_inputs)
        # transfer to decoder lstm hidden0
        decoder_hidden0 = (encoder_outputs,
                           encoder_outputs)

        # [bz, embedding_dim]
        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)
        # rint("decoder input 0: ", decoder_input0.size())  # [256, 100]
        (outputs, pointers), decoder_hidden = self.decoder(embedded_inputs,
                                                           decoder_input0,
                                                           decoder_hidden0)

        return outputs, pointers


class CNNEncoder(nn.Module):
    def __init__(self,
                 embedding_dim,
                 num_filters=3,
                 ngram_filter_sizes=(2, 3, 4),
                 conv_layer_activation=nn.ReLU(),
                 output_dim=None):
        super(CNNEncoder, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_filters = num_filters
        self._ngram_filter_sizes = ngram_filter_sizes
        self._activation = conv_layer_activation
        self._output_dim = output_dim

        self._convolution_layers = [
            nn.Conv1d(
                in_channels=self._embedding_dim,
                out_channels=self._num_filters,
                kernel_size=ngram_size) for ngram_size in self._ngram_filter_sizes
        ]
        for i, conv_layer in enumerate(self._convolution_layers):
            self.add_module('conv_layer_%d' % i, conv_layer)

        maxpool_output_dim = self._num_filters * len(self._ngram_filter_sizes)
        if self._output_dim:
            self.projection_layer = nn.Linear(maxpool_output_dim, self._output_dim)
        else:
            self.projection_layer = None
            self._output_dim = maxpool_output_dim

    def get_input_dim(self):
        return self._embedding_dim

    def get_output_dim(self):
        return self._output_dim

    def forward(self, tokens, mask=None):
        """
        Args:
            tokens (:class:`torch.FloatTensor` [batch_size, num_tokens, input_dim]): Sequence
                matrix to encode.
            mask (:class:`torch.FloatTensor`): Broadcastable matrix to `tokens` used as a mask.
        Returns:
            (:class:`torch.FloatTensor` [batch_size, output_dim]): Encoding of sequence.

        """
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).float()

        # Our input is expected to have shape `(batch_size, num_tokens, embedding_dim)`.  The
        # convolution layers expect input of shape `(batch_size, in_channels, sequence_length)`,
        # where the conv layer `in_channels` is our `embedding_dim`.  We thus need to transpose the
        # tensor first.
        tokens = torch.transpose(tokens, 1, 2)
        # Each convolution layer returns output of size `(batch_size, num_filters, pool_length)`,
        # where `pool_length = num_tokens - ngram_size + 1`.  We then do an activation function,
        # then do max pooling over each filter for the whole input sequence.  Because our max
        # pooling is simple, we just use `torch.max`.  The resultant tensor of has shape
        # `(batch_size, num_conv_layers * num_filters)`, which then gets projected using the
        # projection layer, if requested.

        filter_outputs = []
        for i in range(len(self._convolution_layers)):
            convolution_layer = getattr(self, 'conv_layer_{}'.format(i))
            filter_outputs.append(self._activation(convolution_layer(tokens)).max(dim=2)[0])

        # Now we have a list of `num_conv_layers` tensors of shape `(batch_size, num_filters)`.
        # Concatenating them gives us a tensor of shape
        # `(batch_size, num_filters * num_conv_layers)`.
        maxpool_output = torch.cat(
            filter_outputs, dim=1) if len(filter_outputs) > 1 else filter_outputs[0]

        if self.projection_layer:
            result = self.projection_layer(maxpool_output)
        else:
            result = maxpool_output
        return result


class PoemEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embedding_dim, init_weight):
        """
        :param vocab_size: vocab size
        :param embedding_dim: embedding dim
        :param init_weight: embedding init weight
        """
        super(PoemEmbedding, self).__init__(vocab_size, embedding_dim)
        self.init_weight = init_weight
        self._reset_parameters()

    def _reset_parameters(self):
        self.weight.data = torch.Tensor(self.init_weight)
        # self.weight.data.normal_(0, 1)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class Decoder(nn.Module):
    """
    Decoder model for Pointer-Net
    """

    def __init__(self, embedding_dim, hidden_dim):
        """
        Initiate Decoder

        :param int embedding_dim: Number of embeddings in Pointer-Net
        :param int hidden_dim: Number of hidden units for the decoder's RNN
        """

        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.lstmCell = nn.LSTMCell(embedding_dim, hidden_dim)

        self.linear = nn.Linear(in_features=hidden_dim, out_features=9)

    def forward(self, embedded_inputs,
                decoder_input,
                hidden,
                context=None):
        """
        Decoder - Forward-pass

        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net, [bz, seq_len, embedding_dim]
        :param Tensor decoder_input: First decoder's input, [bz, embedding_dim]
        :param Tensor hidden: a tuple, First decoder's hidden states, [1, bz, hidden_size]
        :param Tensor context: Encoder's outputs, [seq_len, bz, hidden_size]
        :return: (Output probabilities, Pointers indices), last hidden state
        """
        batch_size = embedded_inputs.size(0)
        input_length = embedded_inputs.size(1)

        outputs = []
        pointers = []

        # Recurrence loop(generate 5 output for every seq)
        for _ in range(5):
            h_t, c_t = self.lstmCell(decoder_input, hidden)
            hidden = (h_t, c_t)

            prob = self.linear(h_t)  # [bz, 9]
            indices = prob.max(-1)[1]
            pointers.append(indices.unsqueeze(1))  # [bz, 1]
            outputs.append(h_t.unsqueeze(0))  # [1, bz, hidden_size]

            decoder_input = embedded_inputs[np.arange(batch_size), indices.data, :]  # [bz, embedding_size]
            # print("re decoder input size: ", decoder_input.size())
        outputs = torch.cat(outputs).permute(1, 0, 2)   # [bz, seq_len, hidden_size]
        pointers = torch.cat(pointers, 1)   # [bz, seq_len]

        return (outputs, pointers), hidden
