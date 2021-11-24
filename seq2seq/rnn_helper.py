import torch
import torch.nn as nn


def build_embedding(token_num, embed_size):
    return nn.Embedding(token_num, embed_size)

def rnn_factory(rnn_type):
    if rnn_type == 'gru':
        return nn.GRU
    elif rnn_type == 'lstm':
        return nn.LSTM