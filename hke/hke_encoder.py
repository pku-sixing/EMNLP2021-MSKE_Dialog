import torch
from torch import nn
from seq2seq import rnn_helper
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class RNNEncoder(nn.Module):

    def __init__(self, embed, hidden_size, embed_size, dropout, n_layers=1, rnn_type='GRU', enc_birnn=True):
        super(RNNEncoder, self).__init__()
        self.embed = embed
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.tag_embed_size = 0
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.bi_directional = enc_birnn
        rnn_fn = rnn_helper.rnn_factory(rnn_type)

        assert self.bi_directional is True
        self.rnn = rnn_fn(embed_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=self.bi_directional)


    def forward(self, src, lengths, last_state=None, knowledge_readouts=None, enforce_sorted=True):

        embedding = self.embed(src)
        if knowledge_readouts is not None:
            embedding = torch.cat([embedding, knowledge_readouts], -1)

        # Lengths data is wrapped inside a Tensor.
        lengths_list = lengths.view(-1).tolist()
        embedding = pack(embedding, lengths_list, enforce_sorted=self.training & enforce_sorted)

        if self.rnn_type == 'lstm':
            # inputs-LSTM:  inputs (seq_len, batch, input_size), h0/c0:(num_layers * num_directions, batch, hidden_size)
            # hidden/cell_state-GRU:  (num_layers * num_directions, batch, hidden_size):
            outputs, last_state = self.rnn(embedding, last_state)
            outputs = unpack(outputs)[0]
            if self.bi_directional:
                # sum bidirectional outputs
                outputs = torch.cat([outputs[:, :, :self.hidden_size], outputs[:, :, self.hidden_size:]], -1)
                last_state = tuple([torch.cat([x[0:x.size(0):2], x[1:x.size(0):2]], -1) for x in last_state])
        else:
            # inputs-GRU:  inputs(seq_len, batch, input_size), h0: (num_layers * num_directions, batch, hidden_size)
            # outputs:  (seq_len, batch, num_directions * hidden_size)
            outputs, last_state = self.rnn(embedding, last_state)
            outputs = unpack(outputs)[0]
            if self.bi_directional:
                # sum bidirectional outputs
                outputs = torch.cat([outputs[:, :, :self.hidden_size], outputs[:, :, self.hidden_size:]], -1)
                last_state = torch.cat([last_state[0:last_state.size(0):2], last_state[1:last_state.size(0):2]], -1)
        return outputs, last_state
