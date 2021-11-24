import torch
from torch import nn

from hke.hke_encoder import RNNEncoder
from seq2seq import rnn_helper
from seq2seq.attention import GlobalAttention
from utils.logger import logger
from utils import model_helper
from utils.network.transformer.position_ffn import PositionalEncoding
from utils.network.transformer.transformer_encoder import TransformerEncoder


class KnowledgeManager(nn.Module):
    def __init__(self):
        super().__init__()

    def represent_knowledge(self, batch):
        pass

    def get_readout_enc_dim(self):
        """
        获得当前用于增强表示的维度
        """
        pass

    def get_readout_enc(self, batch):
        """
        获得当前用于增强表示
        """
        return None

    def get_bridge_dim(self):
        pass

    def get_bridge_out(self):
        pass

    def get_readout_dec_dim(self):
        """
        获得当前用于增强表示的维度
        """
        pass

    def get_readout_dec(self, decoder_state, decoder_context, last_embed):
        pass

    def get_modes(self):
        if self.copy_mode == "default":
            return ['copy'] #['vocab','copy']
        else:
            return []


    def get_name(self):
        return self.name

    def bias_vocab_predict(self, decoder_state, decoder_context, last_embed):
        pass

    def beam_expand(self, beam_width):
        pass

    def create_knowledge_specific_rnn(self, rnn_fn, out_size, hidden_size, n_layers, dropout):
        my_rnn = rnn_fn(out_size - hidden_size, hidden_size, n_layers, dropout=dropout)
        self.my_rnn = my_rnn
        return my_rnn

    def get_init_readout(self, batch, query_summary, posterior=False, detach=False):
        if detach:
            encoded_knowledge = self.encoded_knowledge #.detach()
        else:
            encoded_knowledge = self.encoded_knowledge
        if self.use_dot_attention:
            if not posterior:
                context, attn_weights = self.init_attention(query_summary, encoded_knowledge, self.knowledge_length)
                return context, attn_weights
            else:
                context, attn_weights = self.post_init_attention(query_summary, encoded_knowledge,
                                                            self.knowledge_length)
                return context, attn_weights
        else:
            if not posterior:
                attn_query = self.init_attention_query_projection(query_summary).transpose(1,0)

                context, attn_weights = self.init_attention(
                    attn_query,
                    encoded_knowledge,
                    coverage=None,
                    memory_lengths=self.knowledge_length ,
                    no_concat=True,
                )

                weight = self.init_weight_func(torch.cat([query_summary, context], dim=-1).squeeze(0))
                return weight
            else:
                assert detach is False
                attn_query = self.post_init_attention_query_projection(query_summary).transpose(1, 0)

                context, attn_weights = self.post_init_attention(
                    attn_query,
                    encoded_knowledge,
                    coverage=None,
                    memory_lengths=self.knowledge_length,
                    no_concat=True,
                )
                weight = self.post_init_weight_func(torch.cat([query_summary, context], dim=-1).squeeze(0))
                return weight