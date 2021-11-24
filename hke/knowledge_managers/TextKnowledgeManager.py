import torch
from torch import nn

from hke.hke_encoder import RNNEncoder
from hke.hke_attention import DotAttention
from hke.knowledge_managers.KnowledgeManager import KnowledgeManager
from seq2seq.attention import GlobalAttention
from utils.logger import logger
from utils import model_helper
from utils.network.transformer.position_ffn import PositionalEncoding
from utils.network.transformer.transformer_encoder import TransformerEncoder




class TextKnowledgeManager(KnowledgeManager):
    def __init__(self, text_embed, embed_size, hidden_size, dropout,max_text_token_num , tgt_vocab_size,
                 n_layers=1, model_type='transformer', bi_directional=True, mp_gru_mode=False, transformer_layers=4,
                 share_copy_attn=True, init_selection_mode='none', copy_mode="default", dual_init_gate=False,
                 word_encoder_mode=False):
        super().__init__()
        self.name = 'TextKnowledge'
        logger.info('[KNOWLEDGE] A Text Knowledge Manager is added!')
        self.text_embed = text_embed
        self.max_copy_token_num = max_text_token_num

        self.hidden_size = hidden_size
        self.can_copy = True
        self.encoder_type = model_type
        self.use_dot_attention = False
        self.dropout = dropout
        self.copy_mode = copy_mode
        self.dual_init_gate = dual_init_gate
        self.word_encoder_mode = word_encoder_mode

        raw_embed_size = embed_size
        if word_encoder_mode:
            embed_size *= 2

        self.embed_size = embed_size


        if model_type == 'gru':
            self.encoder = RNNEncoder(text_embed, hidden_size, embed_size, dropout, n_layers=1, rnn_type='gru',
                                      enc_birnn=True)
            self.attention_key_value_projection = torch.nn.Linear(hidden_size * 2 + embed_size, hidden_size, bias=False)
            self.to_equal_word_embedding_projection = torch.nn.Linear(hidden_size * 2 + embed_size, embed_size, bias=False)
        elif model_type == 'transformer':
            self.positional_embedding = PositionalEncoding(dropout, embed_size)
            self.encoder = TransformerEncoder(num_layers=transformer_layers, d_model=embed_size, heads=4,
                                                          d_ff=embed_size, dropout=dropout, attention_dropout=dropout)
            self.attention_key_value_projection = torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(embed_size + embed_size, embed_size, bias=False)
            )
            self.to_equal_word_embedding_projection = torch.nn.Linear(embed_size + embed_size, embed_size,
                                                                      bias=False)
        # input (z_t-1, y_t-1, c_t-1)
        self.attention_key_projection = nn.Linear(hidden_size * 2 + raw_embed_size, embed_size, bias=False)
        self.attention = GlobalAttention(embed_size, attn_type='general')
        if share_copy_attn:
            self.copy_attention = None
        else:
            self.copy_attention = GlobalAttention(embed_size, attn_type='general')

        if init_selection_mode == 'none':
            pass
        elif init_selection_mode == 'prior' or init_selection_mode == 'posterior':
            self.init_attention_query_projection = torch.nn.Linear(hidden_size, embed_size)
            gate_dim = 1
            if self.dual_init_gate:
                gate_dim += 1
            self.init_weight_func = torch.nn.Sequential(
                torch.nn.Linear(hidden_size + embed_size, hidden_size),
                torch.nn.ELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_size, gate_dim)
            )
            self.init_attention = GlobalAttention(embed_size, attn_type='general')
            if init_selection_mode == 'posterior':
                self.post_init_attention_query_projection = torch.nn.Linear(hidden_size, embed_size)
                gate_dim = 1
                if self.dual_init_gate:
                    gate_dim += 1
                self.post_init_weight_func = torch.nn.Sequential(
                    torch.nn.Linear(hidden_size + embed_size, hidden_size),
                    torch.nn.ELU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden_size, gate_dim)
                )
                self.post_init_attention = GlobalAttention(embed_size, attn_type='general')


    def beam_expand(self, beam_width):
        self.encoded_knowledge = model_helper.create_beam(self.encoded_knowledge, beam_width, 0)
        self.knowledge_length = model_helper.create_beam(self.knowledge_length, beam_width, 0)


    def get_equip_embeddings(self, batch):
        return self.equal_embedding, self.raw_word_idx

    def represent_knowledge(self, batch, char_embeddings=None):
        text, text_len = batch.text
        raw_word_idx = text
        text_embed = self.text_embed(text)
        if self.word_encoder_mode:
            text_char = model_helper.select_dynamic_embeddings_from_time_first_idx(
                char_embeddings,
                batch.char_text[0])
            text_embed = torch.cat([text_embed, text_char], -1)
        else:
            pass


        batch_size = batch.batch_size
        if self.encoder_type == 'gru':
            encoder_output, hidden = self.encoder(text, text_len, enforce_sorted=False)
        else:
            pos_text_embed = self.positional_embedding(text_embed)
            emb, encoder_output, lengths, hidden = self.encoder(pos_text_embed, lengths=text_len)

        self.encoded_knowledge = self.attention_key_value_projection(
            torch.cat([encoder_output, text_embed], -1)
        ).transpose(0, 1)
        self.knowledge_length = text_len
        equal_embedding = self.to_equal_word_embedding_projection(torch.cat([text_embed, encoder_output],-1))
        src_copy_padding_embedding = torch.zeros(
            [batch_size, self.max_copy_token_num - text.shape[0], self.embed_size],
            device=encoder_output.device)
        raw_idx_padding = torch.zeros([batch_size, self.max_copy_token_num - text.shape[0]], device=raw_word_idx.device,
                                      dtype=raw_word_idx.dtype)
        equal_embedding = torch.cat([equal_embedding.transpose(0,1), src_copy_padding_embedding], dim=1)
        self.equal_embedding = equal_embedding
        self.raw_word_idx = torch.cat([raw_word_idx.transpose(0, 1), raw_idx_padding], -1)

    def get_readout_enc_dim(self):
        # return self.embed_size
        return 0

    def get_readout_enc(self, batch):
        # return self.text_embed(batch.src[0])
        return None

    def get_readout_dec_dim(self):
        return self.embed_size

    def get_readout_dec(self, decoder_state, decoder_context, last_embed):
        attn_inputs = torch.cat([decoder_state, decoder_context, last_embed], -1)
        attn_query = self.attention_key_projection(attn_inputs)

        context, attn_weights = self.attention(
            attn_query,
            self.encoded_knowledge,
            coverage=None,
            memory_lengths=self.knowledge_length,
            no_concat=True,
        )
        self.context = context
        return context

    def get_copy_predict(self, decoder_state, decoder_context, last_embed, padding=True):
        attn_inputs = torch.cat([decoder_state, decoder_context, last_embed], -1)
        attn_query = self.attention_key_projection(attn_inputs)
        attn_func = self.attention if self.copy_attention is None else self.copy_attention
        context, attn_weights = attn_func(
            attn_query,
            self.encoded_knowledge,
            coverage=None,
            memory_lengths=self.knowledge_length - 1,
            no_concat=True,
            mask_first_token=True,
        )
        attn_weights = attn_weights.squeeze(0)
        # 处理为空的情况
        attn_weights = torch.where(self.knowledge_length.unsqueeze(1) < 3, torch.zeros_like(attn_weights)+1e-20, attn_weights)
        # padding copy
        assert attn_weights.shape[1] <= self.max_copy_token_num
        if self.max_copy_token_num - attn_weights.shape[1] > 0 and padding:
            padding_probs = torch.zeros(
                [attn_weights.shape[0], self.max_copy_token_num - attn_weights.shape[1]],
                device=attn_weights.device)
            attn_weights = torch.cat([attn_weights, padding_probs], -1)
        return attn_weights

    def bias_vocab_predict(self, decoder_state, decoder_context, last_embed):
        attn_inputs = torch.cat([decoder_state, decoder_context, last_embed, self.context], -1)
        tmp = self.dropout(self.out_elu(self.out_l1(attn_inputs)))
        output = self.out_l2(tmp)
        vocab_probs = torch.softmax(output, dim=1)
        return vocab_probs



