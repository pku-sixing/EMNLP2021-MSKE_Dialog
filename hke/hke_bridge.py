from torch import nn
import torch

from seq2seq import rnn_helper

from seq2seq.attention import GlobalAttention

class LinearBridge(nn.Module):

    def __init__(self, bridge, rnn_type, hidden_size, num_layers, dropout, bi_rnn, fusion_size=None):
        super(LinearBridge, self).__init__()
        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "lstm" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers
        self.dropout = nn.Dropout(dropout)
        self.fusion_size = fusion_size
        # Build a linear layer for each
        self.bridge_mode = bridge
        if bi_rnn:
            self.bi_rnn = 2
        else:
            self.bi_rnn = 1
        if bridge == 'general':
            self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim * self.bi_rnn ,
                                                   self.total_hidden_dim,
                                                   bias=True)
                                         for _ in range(number_of_states)])
        elif bridge == 'fusion':
            assert fusion_size is not None
            self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim * self.bi_rnn + fusion_size,
                                                   self.total_hidden_dim,
                                                   bias=True)
                                         for _ in range(number_of_states)])


    def forward(self, hidden, fusion_knowledge=None, posterior_hidden=None):
        # from OpenNMT: rnn_encoder
        """Forward hidden state through bridge."""
        def bottle_hidden(linear, states, fusion_knowledge=None):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            # => batch_first

            states = states.permute(1, 0, 2).contiguous()
            size = states.shape
            if fusion_knowledge is None:
                if isinstance(linear, tuple):
                    raise NotImplementedError()
                result = linear(self.dropout(states.view(-1, self.bi_rnn * self.total_hidden_dim)))
            else:
                if isinstance(linear, tuple):
                    tmp = linear[0](self.dropout(
                        torch.cat([fusion_knowledge, states.view(-1, self.bi_rnn * self.total_hidden_dim)], dim=-1)))
                    tmp = self.dropout(torch.tanh(tmp))
                    result = linear[1](tmp)
                else:
                    result = linear(self.dropout(
                        torch.cat([fusion_knowledge, states.view(-1, self.bi_rnn * self.total_hidden_dim)], dim=-1)))

            return torch.tanh(result).view([size[0], size[1], size[2] // self.bi_rnn]).permute(1, 0, 2)

        if self.bridge_mode == 'general':
            assert fusion_knowledge is None
        else:
            assert fusion_knowledge is not None



        bridge_fn = self.bridge
        assert posterior_hidden is None

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix], fusion_knowledge)
                          for ix, layer in enumerate(bridge_fn)])
        else:
            outs = bottle_hidden(bridge_fn[0], hidden, fusion_knowledge)

        return outs