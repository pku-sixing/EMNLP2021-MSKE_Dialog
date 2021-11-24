import torch
from torch import nn

from hke.knowledge_managers.KnowledgeManager import KnowledgeManager
from seq2seq import rnn_helper
from seq2seq.attention import GlobalAttention
from utils.logger import logger
from utils import model_helper
from hke.hke_attention import DotAttention




class CommonsenseKnoweldgeManager(KnowledgeManager):
    def __init__(self, text_embed, embed_size, hidden_size, dropout, max_copy_token_num , tgt_vocab_size,
                 entity_field, relation_field, share_copy_attn=True, init_selection_mode='none', debug=None
                 , copy_mode="default", dual_init_gate=False, word_encoder_mode=False):
        super().__init__()
        self.name = 'Commonsense'
        logger.info('[KNOWLEDGE] A Text Knowledge Manager is added!')
        self.text_embed = text_embed
        self.max_copy_token_num = max_copy_token_num
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.entity_embed_size = 100
        self.can_copy = True
        self.use_dot_attention = False
        self.debug = debug
        self.copy_mode = copy_mode
        self.dual_init_gate = dual_init_gate
        self.word_encoder_mode = word_encoder_mode

        self.entity_embedding = rnn_helper.build_embedding(len(entity_field.vocab), self.entity_embed_size)
        self.relation_embedding = rnn_helper.build_embedding(len(relation_field.vocab), self.entity_embed_size)

        if text_embed is None and word_encoder_mode is False:
            entity_in_word_dim = 0
        else:
            entity_in_word_dim = self.embed_size
        self.knowledge_encoder = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(entity_in_word_dim * 2 + self.entity_embed_size * 5, hidden_size),
        )
        attention_size = hidden_size
        self.attention_size = attention_size
        self.to_equal_word_embedding_projection = torch.nn.Linear(self.entity_embed_size + entity_in_word_dim + attention_size, embed_size,
                                                                  bias=False)


        # input (z_t-1, y_t-1, c_t-1)
        self.attention_key_projection = nn.Linear(hidden_size * 2 + embed_size, attention_size, bias=False)
        self.attention = GlobalAttention(attention_size, attn_type='general')
        if share_copy_attn:
            self.copy_attention = None
        else:
            self.copy_attention = GlobalAttention(attention_size, attn_type='general')

        if init_selection_mode == 'none':
            pass
        elif init_selection_mode == 'prior' or init_selection_mode == 'posterior':
            self.init_attention_query_projection = lambda x : x
            gate_dim = 1
            if self.dual_init_gate:
                gate_dim = 2
            self.init_weight_func = torch.nn.Sequential(
                torch.nn.Linear(hidden_size+hidden_size, hidden_size),
                torch.nn.ELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_size, gate_dim)
            )
            self.init_attention = GlobalAttention(hidden_size, attn_type='general')

            if init_selection_mode == 'posterior':
                self.post_init_attention_query_projection = lambda x: x
                self.post_init_weight_func = torch.nn.Sequential(
                    torch.nn.Linear(hidden_size + hidden_size, hidden_size),
                    torch.nn.ELU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden_size, gate_dim)
                )
                self.post_init_attention = GlobalAttention(hidden_size, attn_type='general')

    def beam_expand(self, beam_width):
        self.encoded_knowledge = model_helper.create_beam(self.encoded_knowledge, beam_width, 0)
        self.knowledge_length = model_helper.create_beam(self.knowledge_length, beam_width, 0)


    def get_equip_embeddings(self, batch):
        return self.equal_embedding, self.raw_word_idx

    def represent_knowledge(self, batch, char_embeddings=None):
        batch_size = batch.batch_size
        self.knowledge_length = batch.csk_post_word[1]
        post_word = batch.csk_post_word[0]
        post_entity = batch.csk_post_entity[0]
        response_word = batch.csk_response_word[0]
        response_entity = batch.csk_response_entity[0]
        head_entity = batch.csk_head_entity[0]
        tail_entity = batch.csk_tail_entity[0]
        relation = batch.csk_relation[0]

        raw_word_idx = response_word

        if self.text_embed is not None:
            post_word = self.text_embed(post_word)
            response_word = self.text_embed(response_word)
        elif self.word_encoder_mode:
            post_word = model_helper.select_dynamic_embeddings_from_time_first_idx(
                char_embeddings,
                batch.char_csk_post_word[0])
            response_word = model_helper.select_dynamic_embeddings_from_time_first_idx(
                char_embeddings,
                batch.char_csk_post_word[0])
        else:
            post_word = None
            response_word = None
        post_entity = self.entity_embedding(post_entity)
        response_entity = self.entity_embedding(response_entity)
        head_entity = self.entity_embedding(head_entity)
        tail_entity = self.entity_embedding(tail_entity)
        relation = self.relation_embedding(relation)

        if response_word is not None:
            knowledge_inputs = torch.cat([post_word, response_word,
                                          post_entity, response_entity, head_entity, tail_entity, relation], -1)
        else:
            knowledge_inputs = torch.cat([ post_entity, response_entity, head_entity, tail_entity, relation], -1)


        encoded_knowledge = self.knowledge_encoder(knowledge_inputs)
        self.encoded_knowledge = encoded_knowledge.transpose(0, 1)
        if response_word is not None:
            equal_embedding = self.to_equal_word_embedding_projection(torch.cat([response_word, response_entity, encoded_knowledge], -1))
        else:
            equal_embedding = self.to_equal_word_embedding_projection(
                torch.cat([response_entity, encoded_knowledge], -1))
        padding_embedding = torch.zeros(
            [batch_size, self.max_copy_token_num - post_entity.shape[0], self.embed_size],
            device=equal_embedding.device)
        padding_word_idx = torch.zeros([batch_size, self.max_copy_token_num - post_entity.shape[0]], device=post_entity.device,
                                       dtype=raw_word_idx.dtype)
        equal_embedding = torch.cat([equal_embedding.transpose(0, 1), padding_embedding], dim=1)
        self.equal_embedding = equal_embedding
        self.raw_word_idx = torch.cat([raw_word_idx.transpose(0, 1), padding_word_idx], dim=-1)
        pass

    def get_readout_enc_dim(self):
        return 0

    def get_readout_enc(self, batch):
        return None

    def get_readout_dec_dim(self):
        return self.attention_size

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
