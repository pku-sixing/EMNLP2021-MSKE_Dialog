"""
Char Encoder Modules
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from seq2seq import rnn_helper
from utils import model_helper

valid_encoder_types = {'mean','rnn','cnn'}


class T2STokenEncoder(nn.Module):
    def __init__(self, method, dropout, vocab_size, word_embed_size, embed_size, down_scale=1.00):
        super(T2STokenEncoder, self).__init__()
        self.sub_embed = rnn_helper.build_embedding(vocab_size, word_embed_size)
        self.scaled_embed_size = int(embed_size * down_scale)
        if self.scaled_embed_size != word_embed_size:
            self.down_scale_projection = torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(word_embed_size, self.scaled_embed_size),
                torch.nn.Tanh(),
            )
        else:
            self.down_scale_projection = None

        if method == 'mean':
            self.encoder = TokenMeanEncoder(dropout=dropout)
            self.output_projection = torch.nn.Linear( self.scaled_embed_size,
                                                     embed_size)
        elif method == 'cnn':
            self.encoder = TokenCNNEncoder(embedding_size=self.scaled_embed_size,
                                           filters=[(1, 100), (2, 175), (3, 75)], dropout=dropout)
            self.output_projection = torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(350, embed_size),
                torch.nn.Tanh(),
            )


    def encode(cls, sub_chars):
        # Char_Seq: [batch, seq_len] 这里已经全都用的是固定的idx了，不用担心选择问题。
        char_seq, _, dynamic_char_vocab, dynamic_char_length = sub_chars
        # => (batch, sub_seq_len)
        vocab_size, sub_seq_len = dynamic_char_vocab.shape
        dynamic_char_vocab = dynamic_char_vocab.reshape(vocab_size, sub_seq_len)
        dynamic_char_length = dynamic_char_length.reshape(vocab_size)
        dynamic_char_vocab_embed = cls.sub_embed(dynamic_char_vocab)
        if cls.down_scale_projection is not None:
            dynamic_char_vocab_embed = cls.down_scale_projection(dynamic_char_vocab_embed)
        # =》 (batch*seq, dim)
        dynamic_char_vocab_embed = cls(dynamic_char_vocab_embed, dynamic_char_length)
        # => (batch, seq, dim)
        dynamic_char_vocab_embed = dynamic_char_vocab_embed.view(vocab_size, -1)
        batch_size, seq_len = char_seq.shape
        char_embed = torch.index_select(dynamic_char_vocab_embed,0,char_seq.view(-1))
        char_embed = char_embed.view(batch_size, seq_len, -1)
        char_embed = char_embed.permute(1, 0, 2)
        return  cls.output_projection(char_embed)

    def forward(self, input_embedding, token_length):
        return self.encoder(input_embedding, token_length)



class TokenMeanEncoder(nn.Module):

    def __init__(self,dropout, batch_first=True):
        super(TokenMeanEncoder, self).__init__()

        if batch_first is False:
            raise NotImplementedError()

        self.batch_first = batch_first
        self.dropout = torch.nn.Dropout(dropout)



    def forward(self, input_embedding, token_length):

        batch_size = input_embedding.shape[0]
        seq_length = input_embedding.shape[1]
        embed_dim = input_embedding.shape[2]

        # the last token is </w>
        valid_token_length = token_length - 1
        sequence_mask = model_helper.sequence_mask_fn(valid_token_length, maxlen=seq_length, dtype=input_embedding.dtype)
        sequence_mask[:, 0] = 0
        sequence_mask = sequence_mask.unsqueeze(-1)
        sequence_mask = sequence_mask.repeat([1, 1, embed_dim])
        masked_embedding = input_embedding * sequence_mask
        masked_embedding = masked_embedding.sum(dim=1) / (valid_token_length - 1).unsqueeze(-1)

        return masked_embedding


class TokenRNNEncoder(nn.Module):

    def __init__(self, embedding_size, rnn_type, hidden_size, dropout, bidirectional, num_layers=2, output_size=-1):
        super(TokenRNNEncoder, self).__init__()


        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bi_directional = bidirectional
        rnn = rnn_helper.rnn_factory(rnn_type)
        self.rnn = rnn(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                       dropout=dropout, bidirectional=bidirectional)
        if output_size > -1:
            self.output_projection = nn.Sequential(nn.Linear(hidden_size * 2 if bidirectional else hidden_size * num_layers * 1, output_size),
                                               torch.nn.Tanh())
            self.h_c_projection = nn.Sequential(nn.Linear(hidden_size * num_layers * 2 if bidirectional else hidden_size * num_layers * 1, output_size),
                                                torch.nn.Tanh())

    def forward(self, input_embedding, token_length):

        batch_size = input_embedding.shape[0]
        seq_length = input_embedding.shape[1]
        embed_dim = input_embedding.shape[2]
        input_embedding = input_embedding.permute([1, 0, 2])

        # TODO Remove enforce_sorted to Enable ONNX
        packed_emb = pack(input_embedding, token_length, enforce_sorted=False)
        outputs, last_state = self.rnn(packed_emb)
        outputs = unpack(outputs)[0]
        if self.rnn_type == 'lstm':
            # inputs-LSTM:  inputs (seq_len, batch, input_size), h0/c0:(num_layers * num_directions, batch, hidden_size)
            if self.output_size > -1:
                # => (batch, seq_len, output_size)
                outputs = self.output_projection(outputs).transpose(0, 1)
                # => ((batch, output_size), (batch, output_size))
                last_state = tuple([self.h_c_projection(x.transpose(0, 1).reshape(batch_size, -1)) for x in last_state])
            else:
                # => (batch, seq_len, hidden_size * bidirectional)
                outputs = outputs.transpose(0, 1)
                # => (batch, hidden_size * bidirectional * num_layer)
                last_state = tuple([x.transpose(0, 1).reshape(batch_size, -1) for x in last_state])
        else:
            # inputs-GRU:  inputs(seq_len, batch, input_size), h0: (num_layers * num_directions, batch, hidden_size)
            if self.output_size > -1:
                # => (batch, seq_len, output_size)
                outputs = self.output_projection(outputs).transpose(0, 1)
                # => (batch, output_size)
                last_state = self.h_c_projection(last_state.transpose(0, 1).reshape(batch_size, -1))
            else:
                # => (batch, seq_len, hidden_size * bidirectional)
                outputs = outputs.transpose(0, 1)
                # => (batch, hidden_size * bidirectional * num_layer)
                last_state = last_state.transpose(0, 1).reshape(batch_size, -1)
        return outputs, last_state


class TokenCNNEncoder(nn.Module):

    def __init__(self, embedding_size, filters, dropout, batch_first=True):
        super(TokenCNNEncoder, self).__init__()

        if batch_first is False:
            raise NotImplementedError()

        self.batch_first = batch_first
        self.dropout = torch.nn.Dropout(dropout)
        convs = []
        self.max_kernel_size = filters[0][0]

        feats = 0
        for kernel_size, kernel_num in filters:
            feats += kernel_num
            self.max_kernel_size = max(self.max_kernel_size, kernel_size)
            conv = nn.Conv2d(
                in_channels=1,
                out_channels=kernel_num,
                kernel_size=(kernel_size, embedding_size),
                stride=(1, embedding_size)
            )
            convs.append(conv)

        self.convs = torch.nn.ModuleList(convs)
        self.highway_gate = torch.nn.Linear(feats, feats, bias=True)
        self.highway_fn = torch.nn.Linear(feats, feats, bias=True)

    def forward(self, input_embedding, token_length):

        input_embedding = input_embedding.unsqueeze(1)

        conv_outs = []
        for conv_net in self.convs:
            conv_out = conv_net(self.dropout(input_embedding))
            conv_out = conv_out.squeeze(-1)
            # Max Pooling Over Time
            conv_out = conv_out.max(dim=-1)[0]
            conv_outs.append(conv_out)

        encoded = torch.cat(conv_outs, -1)

        # One Layer Highway

        gate_val = torch.sigmoid(self.highway_gate(encoded))
        transformed_val = torch.relu(self.highway_gate(encoded))
        hw_encoded = transformed_val * gate_val + (1.0 - gate_val) * encoded
        return hw_encoded


def get_encoder(encoder_type, embeddings,  hidden_size, dropout, rnn_type=None, output_size=-1):
    if encoder_type == 'rnn':
        return TokenRNNEncoder(embeddings, rnn_type, hidden_size,  dropout, bidirectional=True, output_size=output_size)
    elif encoder_type == 'cnn':
        return TokenCNNEncoder(embeddings, hidden_size,  dropout)
    elif encoder_type == 'mean':
        return TokenMeanEncoder(dropout)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    filters = [[1, 50], [2, 50], [3, 50]]
    encoder = get_encoder('rnn', 20, 10, 0, rnn_type='lstm')
    # (3,3)
    word_seq = torch.rand([7,9,20])
    # field_seq = torch.rand([9,7,12])
    # field_len = torch.tensor([[1,1,3,2,1,2,1,1,0],
    #                           [1,2,1,1,3,2,1,2,1],
    #                           [1,1,2,1,1,0,0,0,0]], dtype=torch.long).transpose(0,1)
    # kv_pos = torch.tensor([[0, 1, 2, 2, 2, 3, 3, 0, 0],
    #                        [0, 1, 1, 2, 3, 3, 3, 4, 4],
    #                        [0, 1, 2, 2, 0, 0, 0, 0, 0]], dtype=torch.long).transpose(0, 1)
    src_len = torch.tensor([8,9,5,7,8,5,2], dtype=torch.long)

    res = encoder(word_seq, src_len)
    print(res)
