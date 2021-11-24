from torch import nn
import torch

from seq2seq import rnn_helper


from seq2seq.attention import GlobalAttention


class AttentionBridge(nn.Module):
    def __init__(self, bridge, hidden_size, dropout, bi_rnn):
        super(AttentionBridge, self).__init__()
        self.bridge_mode = bridge
        if bi_rnn:
            self.bi_rnn = 2
        else:
            self.bi_rnn = 1
        if bridge == 'attention':
            self.bridge_self_query = torch.nn.Parameter(torch.FloatTensor(1, hidden_size))
            self.bridge_context_attn = GlobalAttention(hidden_size, attn_type='dot')
            self.bridge_out_projection = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        elif bridge == 'field_attention':
            self.bridge_query_projection = torch.nn.Linear(hidden_size * self.bi_rnn, hidden_size, bias=False)
            self.bridge_field_attn = GlobalAttention(hidden_size, attn_type='dot')
            self.bridge_out_projection = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        elif bridge == 'fusion_attention':
            self.bridge_self_query = torch.nn.Parameter(torch.FloatTensor(1, hidden_size))
            self.bridge_context_attn = GlobalAttention(hidden_size, attn_type='dot')
            self.bridge_query_projection = torch.nn.Linear(hidden_size * self.bi_rnn, hidden_size, bias=False)
            self.bridge_field_attn = GlobalAttention(hidden_size, attn_type='dot')
            self.bridge_out_projection = torch.nn.Linear(hidden_size * 2, hidden_size, bias=False)

    def forward(self, hidden, encoder_memory, field_memory):
        layers, batch_size, hidden_size = hidden.shape
        new_hidden = torch.zeros([layers, batch_size, hidden_size // self.bi_rnn], device=hidden.device)
        attn_weights = []
        field_attn_weights = []
        for i in range(layers):
            if self.bridge_mode == 'attention':
                context_query = torch.tanh(self.bridge_self_query.repeat([batch_size, 1]))
                tmp, attn_weight = self.bridge_context_attn(context_query, encoder_memory[0],
                                                            memory_lengths=encoder_memory[1]
                                                            , no_concat=True)
                new_hidden[i] = torch.tanh(self.bridge_out_projection(tmp))
                attn_weights.append(attn_weight)
            elif self.bridge_mode == 'field_attention':
                field_query = torch.tanh(self.bridge_query_projection(hidden[i]))
                tmp, attn_weight = self.bridge_field_attn(field_query, field_memory[0],
                                                          memory_lengths=field_memory[1],
                                                          no_concat=True)
                new_hidden[i] = torch.tanh(self.bridge_out_projection(tmp))
                field_attn_weights.append(attn_weight)
            elif self.bridge_mode == 'fusion_attention':
                tmps = []
                context_query = torch.tanh(self.bridge_self_query.repeat([batch_size, 1]))
                tmp, attn_weight = self.bridge_context_attn(context_query, encoder_memory[0],
                                                            memory_lengths=encoder_memory[1],
                                                            no_concat=True)
                tmps.append(tmp)
                attn_weights.append(attn_weight)
                field_query = torch.tanh(self.bridge_query_projection(hidden[i]))
                tmp, attn_weight = self.bridge_field_attn(field_query, field_memory[0], memory_lengths=field_memory[1],
                                                          no_concat=True)
                tmps.append(tmp)
                new_hidden[i] = torch.tanh(self.bridge_out_projection(torch.cat(tmps, -1)))
                field_attn_weights.append(attn_weight)

            else:
                raise NotImplementedError()


        return new_hidden, attn_weights, field_attn_weights


class LinearBridge(nn.Module):

    def __init__(self, bridge, rnn_type, hidden_size, num_layers, dropout, bi_rnn):
        super(LinearBridge, self).__init__()
        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "lstm" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers
        self.dropout = nn.Dropout(dropout)
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
            self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim * self.bi_rnn + hidden_size,
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