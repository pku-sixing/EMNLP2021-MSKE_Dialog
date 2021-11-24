import torch
from torch import nn

from hke.hke_attention import DotAttention
from hke.knowledge_managers.KnowledgeManager import KnowledgeManager
from seq2seq import rnn_helper
from seq2seq.attention import GlobalAttention
from utils.logger import logger
from utils import model_helper
from utils.network.transformer.transformer_encoder import TransformerEncoder




class InfoboxKnowledgeManager(KnowledgeManager):
    def __init__(self, src_embed, field_key_vocab_size, field_word_vocab_size, word_embed_size, pos_embed_size,
                 max_field_intra_word_num, hidden_size, dropout, max_kw_pairs_num,
                 infobox_transformer_field_encoder_heads=4,
                 infobox_transformer_field_encoder_layers=2,
                 word_encoder_mode=False,
                 share_copy_attn=True, init_selection_mode='none'
                 , copy_mode="default", dual_init_gate=False):
        super().__init__()
        self.name = 'InfoboxKnowledge'
        logger.info('[KNOWLEDGE] An Infobox Knowledge Manager is added!')
        self.src_embed = src_embed
        self.field_key_embed_size = word_embed_size
        self.embed_size = word_embed_size
        self.enc_read_out_dim = 100
        self.can_copy = True
        self.max_copy_token_num = max_kw_pairs_num
        self.hidden_size = hidden_size
        self.use_dot_attention = False
        self.copy_mode = copy_mode
        self.dual_init_gate = dual_init_gate
        self.word_encoder_mode = word_encoder_mode

        self.enc_read_out_projection = nn.Linear(word_embed_size, self.enc_read_out_dim)

        if src_embed is None and word_encoder_mode is False:
            entity_in_word_dim = 0
        else:
            entity_in_word_dim = word_embed_size

        # 正反各一个
        self.field_all_pos_embed_size = pos_embed_size * 2
        self.field_key_embed = rnn_helper.build_embedding(field_key_vocab_size, word_embed_size)
        self.field_word_embed = rnn_helper.build_embedding(field_word_vocab_size, word_embed_size)
        self.local_pos_fw_embed = rnn_helper.build_embedding(max_field_intra_word_num,
                                                             pos_embed_size)

        self.local_pos_bw_embed = rnn_helper.build_embedding(max_field_intra_word_num,
                                                             pos_embed_size)
        # Model
        hidden_size = hidden_size
        heads = infobox_transformer_field_encoder_heads
        layers = infobox_transformer_field_encoder_layers
        dropout = dropout

        #[field_key, field_word, field_word_word, pos_bw, pos_fw]
        input_size = word_embed_size * 2 + 2 * pos_embed_size + entity_in_word_dim * 2
        self.hidden_size = hidden_size
        self.input_projection = nn.Linear(input_size, word_embed_size, bias=False)
        self.transformer_encoder = TransformerEncoder(num_layers=layers, d_model=word_embed_size, heads=heads,
                                                      d_ff=word_embed_size, dropout=dropout, attention_dropout=dropout)

        self.attention_key_value_projection = torch.nn.Linear(entity_in_word_dim + word_embed_size, word_embed_size, bias=False)
        self.to_equal_word_embedding_projection = torch.nn.Linear(entity_in_word_dim + word_embed_size, word_embed_size,
                                                                  bias=False)

        # input (z_t-1, y_t-1, c_t-1)
        self.attention_key_projection = nn.Linear(hidden_size * 2 + word_embed_size, word_embed_size, bias=False)
        self.attention = GlobalAttention(word_embed_size, attn_type='general')
        if share_copy_attn:
            self.copy_attention = None
        else:
            self.copy_attention = GlobalAttention(word_embed_size, attn_type='general')
        self.init_selection_mode = init_selection_mode

        if init_selection_mode == 'none':
            pass
        elif init_selection_mode == 'prior' or init_selection_mode == 'posterior':
            self.init_attention_query_projection = torch.nn.Linear(hidden_size, word_embed_size)
            gate_dim = 1
            if self.dual_init_gate:
                gate_dim += 1
            self.init_weight_func = torch.nn.Sequential(
                torch.nn.Linear(hidden_size + word_embed_size, hidden_size),
                torch.nn.ELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_size, gate_dim)
            )
            self.init_attention = GlobalAttention(word_embed_size, attn_type='general')

            if init_selection_mode == 'posterior':
                self.post_init_attention_query_projection = torch.nn.Linear(hidden_size, word_embed_size)
                gate_dim = 1
                if self.dual_init_gate:
                    gate_dim += 1
                self.post_init_weight_func = torch.nn.Sequential(
                    torch.nn.Linear(hidden_size + word_embed_size, hidden_size),
                    torch.nn.ELU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden_size, gate_dim)
                )
                self.post_init_attention = GlobalAttention(word_embed_size, attn_type='general')








    def beam_expand(self, beam_width):
        self.encoded_knowledge = model_helper.create_beam(self.encoded_knowledge, beam_width, 0)
        self.knowledge_length = model_helper.create_beam(self.knowledge_length, beam_width, 0)


    def get_equip_embeddings(self, batch):
        return self.equal_embedding, self.raw_word_idx

    def represent_knowledge(self, batch, char_embeddings=None):
        batch_size = batch.batch_size
        # [field_key, field_word, field_word_word, pos_bw, pos_fw]
        field_key = self.field_key_embed(batch.attribute_key[0])
        field_word = self.field_word_embed(batch.attribute_word[0])
        raw_word_idx = batch.attribute_uni_word[0]
        pos_fw = self.local_pos_fw_embed(batch.attribute_word_local_fw_pos[0])
        pos_bw = self.local_pos_bw_embed(batch.attribute_word_local_bw_pos[0])

        if self.src_embed is not None:
            field_key_word = self.src_embed(batch.attribute_uni_key[0])
            field_word_word = self.src_embed(batch.attribute_uni_word[0])
            input_embed = self.input_projection(torch.cat([
                field_key, field_word, field_word_word, field_key_word, pos_fw, pos_bw
            ], -1))
        elif self.word_encoder_mode:
            field_key_word = model_helper.select_dynamic_embeddings_from_time_first_idx(
                char_embeddings,
                batch.char_attribute_value[0])
            field_word_word = model_helper.select_dynamic_embeddings_from_time_first_idx(
                char_embeddings,
                batch.char_attribute_key[0])
            input_embed = self.input_projection(torch.cat([
                field_key, field_word, field_word_word, field_key_word, pos_fw, pos_bw
            ], -1))
        else:
            field_word_word = None
            field_key_word = None
            input_embed = self.input_projection(torch.cat([
                field_key, field_word, pos_fw, pos_bw
            ], -1))
        _, outputs, length, last_output = self.transformer_encoder(input_embed, batch.attribute_key[1])

        if field_key_word is not None:
            equal_embedding = self.to_equal_word_embedding_projection(torch.cat([field_word_word, outputs], -1))
        else:
            equal_embedding = self.to_equal_word_embedding_projection( outputs)
        src_copy_padding_embedding = torch.zeros(
            [batch_size, self.max_copy_token_num - field_word.shape[0], self.embed_size],
            device=equal_embedding.device)
        raw_word_idx_padding = torch.zeros([batch_size,self.max_copy_token_num - field_word.shape[0]],
                                           device=raw_word_idx.device, dtype=raw_word_idx.dtype)
        equal_embedding = torch.cat([equal_embedding.transpose(0, 1), src_copy_padding_embedding], dim=1)
        self.equal_embedding = equal_embedding

        if field_key_word is not None:
            self.encoded_knowledge = self.attention_key_value_projection(
                torch.cat([outputs, field_word_word], -1)
            ).transpose(0, 1)
        else:
            self.encoded_knowledge = self.attention_key_value_projection(
               outputs
            ).transpose(0, 1)
        self.raw_word_idx = torch.cat([raw_word_idx.transpose(0, 1), raw_word_idx_padding], dim=-1)
        self.knowledge_length = batch.attribute_key[1]

        return outputs, last_output

    def get_readout_enc_dim(self):
        # return self.embed_size
        # return self.enc_read_out_dim
        return 0

    def get_readout_enc(self, batch):
        return None
        # return self.enc_read_out_projection(self.field_word_embedding(batch.src_in_infobox))

    def get_readout_dec_dim(self):
        return self.embed_size

    def get_readout_dec(self, decoder_state, decoder_context, last_embed):
        attn_inputs = torch.cat([decoder_state, decoder_context, last_embed], -1)
        attn_query = self.attention_key_projection(attn_inputs)
        context, attn_weights = self.attention(
            attn_query,
            self.encoded_knowledge,
            coverage=None,
            memory_lengths=self.knowledge_length ,
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

