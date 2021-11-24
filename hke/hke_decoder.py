import torch
from torch import nn
import torch.nn.functional as F
from seq2seq import rnn_helper
from hke.hke_attention import DotAttention
from utils import model_helper
from utils.logger import logger


class HKEDecoder(nn.Module):

    def __init__(self, hparams, embed, knowledge_managers=[]):
        super(HKEDecoder, self).__init__()

        self.hidden_size = hidden_size = hparams.hidden_size
        self.embed_size = embed_size = hparams.embed_size
        self.n_layers = n_layers = hparams.dec_layers
        self.dropout = dropout = hparams.dropout
        self.vocab_size = hparams.tgt_vocab_size
        self.vocab_size_with_offsets = hparams.tgt_vocab_size_with_offsets
        self.rnn_type = hparams.rnn_type
        self.embed = embed
        self.dropout = nn.Dropout(dropout)
        self.bow_loss = False
        self.knowledge_managers = knowledge_managers
        self.enable_knowledge_source_indicator_gate = hparams.enable_global_mode_gate
        self.global_mode_gate = hparams.global_mode_gate
        self.fast_mode = False
        self.copy_strategy = hparams.copy_strategy
        self.uniform_word_encoding = hparams.uniform_word_encoding
        self.softmax_global_gate_fusion = hparams.softmax_global_gate_fusion
        self.softmax_global_gate_fusion_projection = []
        self.disable_sigmoid_gate = hparams.disable_sigmoid_gate
        self.ablation = hparams.ablation

        # Current State, hidden_size
        attention_access_dec_dims = [hparams.hidden_size]
        raw_attention_access_dec_dims = [hparams.hidden_size]

        # Attention Size
        self.attention = DotAttention(hidden_size + embed_size, hidden_size, hidden_size, hparams.attn_type,
                                      fast_mode=self.fast_mode)
        if self.ablation != 'no_knowledge_readout':
            fused_context_dim = hidden_size + int(hidden_size * len(knowledge_managers) / 3)
            for knowledge in knowledge_managers:
                knowledge_access_dec_dim = knowledge.get_readout_dec_dim()
                attention_access_dec_dims.append(knowledge_access_dec_dim)
                raw_attention_access_dec_dims.append(knowledge_access_dec_dim)
                if self.softmax_global_gate_fusion:
                    self.softmax_global_gate_fusion_projection.append(
                        torch.nn.Sequential(
                            torch.nn.Linear(knowledge_access_dec_dim, fused_context_dim),
                            torch.nn.Tanh()
                        )
                    )
            if self.softmax_global_gate_fusion:
                attention_access_dec_dims = [fused_context_dim]
                tmp = torch.nn.Sequential(
                            torch.nn.Linear(hidden_size, fused_context_dim),
                            torch.nn.Tanh()
                        )
                self.softmax_global_gate_fusion_projection = [tmp] + self.softmax_global_gate_fusion_projection
                self.softmax_global_gate_fusion_projection = torch.nn.ModuleList(self.softmax_global_gate_fusion_projection)
        attention_size = sum(attention_access_dec_dims)
        raw_attention_size = sum(raw_attention_access_dec_dims)
        # out_size: z_t-1, c_t, y_t-1
        out_size = hidden_size + attention_size + embed_size
        vocab_predict_size = hidden_size + attention_size + embed_size
        attention_mode_size = hidden_size + raw_attention_size + embed_size
        self.out_size = out_size

        # Build Query-Copy
        self.mode_num = 1
        self.src_copy = hparams.copy
        if hparams.copy:
            self.max_copy_token_num = hparams.max_copy_token_num
            self.src_copy_attention = DotAttention(hidden_size + embed_size, hidden_size, hidden_size, hparams.attn_type,
                                                   fast_mode=self.fast_mode)
            self.mode_num += 1
        else:
            self.src_copy_attention = None
        # Build Knowledge-copy
        for knowledge in knowledge_managers:
            sub_modes = knowledge.get_modes()
            logger.info('[MODES] %s has modes:%s ' % (knowledge.get_name(), str(sub_modes)))
            self.mode_num += len(sub_modes)

        # Mode Selector
        if self.mode_num > 1:
            if hparams.mode_selector == 'mlp':
                self.mode_selector = nn.Sequential(
                    nn.Linear(vocab_predict_size, self.hidden_size, bias=True),
                    torch.nn.ELU(),
                    nn.Linear(self.hidden_size, self.mode_num, bias=False),
                )
            else:
                self.mode_selector = nn.Linear(vocab_predict_size, self.mode_num, bias=False)

        if self.softmax_global_gate_fusion:
            if self.ablation != 'no_advanced_knowledge_selection':
                if hparams.mode_selector == 'mlp':
                    self.attn_mode_selector = nn.Sequential(
                        nn.Linear(attention_mode_size, self.hidden_size, bias=True),
                        torch.nn.ELU(),
                        nn.Linear(self.hidden_size, len(knowledge_managers) + 1, bias=False),
                    )
                else:
                    self.attn_mode_selector = nn.Linear(attention_mode_size, len(knowledge_managers) + 1, bias=False)

        # Decoder
        rnn_fn = rnn_helper.rnn_factory(hparams.rnn_type)
        self.rnn = rnn_fn(out_size - hidden_size, hidden_size, n_layers, dropout=dropout)

        # Vocabulary Predictor
        if hparams.naive_baseline_mode:
            self.vocab_predictor = nn.Linear(vocab_predict_size, hparams.tgt_vocab_size, bias=False)
        else:
            mid_vocab_predict_size = max(int(vocab_predict_size * 0.75), int(1.5 * hparams.hidden_size))
            self.vocab_predictor = nn.Sequential(
                # nn.Dropout(dropout),
                nn.Linear(vocab_predict_size, mid_vocab_predict_size),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(mid_vocab_predict_size, hparams.tgt_vocab_size, bias=False),
            )

    def forward(self, modes, input, raw_input, last_hidden, encoder_memory_bank, encoder_length,
                copied_token_equip_embedding=None, copy_attention_coverage_vector=None,
                decoder_cache=dict()):
        """
        :param copy_attention_coverage_vector:
        :param copied_token_equip_embedding:
        :param input:[seq_len]
        :param last_hidden:
        :param encoder_memory_bank: (batch, seq_len, dim)
        :param encoder_length:
        :return:
        """
        knowledge_source_indicator, local_mode = modes
        batch_size = input.shape[0]
        global_weights = decoder_cache['global_weights']
        if self.src_copy_attention is not None and self.copy_strategy != 'dynamic_vocab':
            if self.copy_strategy == 'default':
                # Select copied embeddings
                original_word_embedding = self.embed(input)
                batch_size, equip_embedding_len, encoder_hidden_dim = copied_token_equip_embedding.shape
                # Index
                copied_select_index = input - self.vocab_size
                is_copied_tokens = (copied_select_index >= 0)
                zero_index = torch.zeros_like(copied_select_index, device=copied_select_index.device)
                valid_copied_select_index = torch.where(is_copied_tokens, copied_select_index, zero_index)

                # Offsets for flatten selection
                batch_offsets = torch.arange(0, batch_size * equip_embedding_len, equip_embedding_len, dtype=torch.long,
                                             device=copied_select_index.device)
                valid_copied_select_index = valid_copied_select_index + batch_offsets

                # Select
                flatten_equip_embedding = copied_token_equip_embedding.view(-1, encoder_hidden_dim)
                selected_equip_embedding = flatten_equip_embedding.index_select(index=valid_copied_select_index, dim=0)
                selected_equip_embedding = selected_equip_embedding.view(batch_size, encoder_hidden_dim)
                embedded = torch.where(is_copied_tokens.unsqueeze(-1), selected_equip_embedding, original_word_embedding)
                embedded = embedded.unsqueeze(0)  # (1,B,N)
            elif self.copy_strategy == 'fusion':
                # Select copied embeddings
                original_word_embedding = self.embed(raw_input)
                batch_size, equip_embedding_len, encoder_hidden_dim = copied_token_equip_embedding.shape
                # Index
                copied_select_index = input - self.vocab_size
                is_copied_tokens = (copied_select_index >= 0) & (raw_input == 0)
                zero_index = torch.zeros_like(copied_select_index, device=copied_select_index.device)
                valid_copied_select_index = torch.where(is_copied_tokens, copied_select_index, zero_index)
                # Select
                flatten_equip_embedding = copied_token_equip_embedding.view(-1, encoder_hidden_dim)
                selected_equip_embedding = flatten_equip_embedding.index_select(index=valid_copied_select_index, dim=0)
                selected_equip_embedding = selected_equip_embedding.view(batch_size, encoder_hidden_dim)
                embedded = torch.where(is_copied_tokens.unsqueeze(-1), selected_equip_embedding, original_word_embedding)
                embedded = embedded.unsqueeze(0)  # (1,B,N)

            else:
                raise NotImplementedError()
        elif self.copy_strategy == 'dynamic_vocab' and self.uniform_word_encoding == 'char':
            # 先获得所有的标准Embedding
            original_word_embedding = self.embed(input)
            # 获得Dynamic Embedding
            raw_char_embeddings = decoder_cache['raw_char_embeddings']
            copied_token_equip_embedding = raw_char_embeddings
            batch_size, equip_embedding_len, encoder_hidden_dim = copied_token_equip_embedding.shape
            # Index
            copied_select_index = input - self.vocab_size
            is_copied_tokens = (copied_select_index >= 0)
            zero_index = torch.zeros_like(copied_select_index, device=copied_select_index.device)
            valid_copied_select_index = torch.where(is_copied_tokens, copied_select_index, zero_index)
            # Offsets for flatten selection
            batch_offsets = torch.arange(0, batch_size * equip_embedding_len, equip_embedding_len, dtype=torch.long,
                                         device=copied_select_index.device)
            valid_copied_select_index = valid_copied_select_index + batch_offsets

            # Select
            flatten_equip_embedding = copied_token_equip_embedding.view(-1, encoder_hidden_dim)
            selected_equip_embedding = flatten_equip_embedding.index_select(index=valid_copied_select_index, dim=0)
            selected_equip_embedding = selected_equip_embedding.view(batch_size, encoder_hidden_dim)
            embedded = torch.where(is_copied_tokens.unsqueeze(-1), selected_equip_embedding, original_word_embedding)
            embedded = embedded.unsqueeze(0)  # (1,B,N)
        else:
            # Get the embedding of the current input word (last output word)
            embedded = self.embed(input).unsqueeze(0)  # (1,B,N)

        embedded = self.dropout(embedded)
        last_generated_token_embedded = embedded.squeeze(0)

        # Attention
        if self.rnn_type == 'lstm':
            attn_query = last_hidden[-1][0]
        else:
            attn_query = last_hidden[-1]
        attn_query = attn_query.unsqueeze(1)
        attn_query_for_context = torch.cat([embedded, attn_query.transpose(0, 1)], -1)

        context, attn_weights, attention_cached_memory_bank = self.attention(
            attn_query_for_context,
            encoder_memory_bank,
            encoder_length,
            cached_memory_bank=None
        )
        context = context.unsqueeze(0)

        if isinstance(last_hidden, list) or isinstance(last_hidden, tuple):
            last_hidden = tuple([x.contiguous() for x in last_hidden])
        else:
            last_hidden = last_hidden.contiguous()

        # Knowledge Attentions
        knowledge_contexts = []
        if len(self.knowledge_managers) > 0 and self.ablation != 'no_knowledge_readout':
            for idx, knowledge in enumerate(self.knowledge_managers):
                tmp_attn = knowledge.get_readout_dec(
                    last_hidden.permute(1, 0, 2),
                    context.permute(1, 0, 2),
                    embedded.permute(1, 0, 2),
                )
                if global_weights is not None and self.softmax_global_gate_fusion is False and self.disable_sigmoid_gate is False:
                    if self.global_mode_gate == 'extend':
                        tmp_attn = tmp_attn * global_weights[idx + 1].transpose(0,1).unsqueeze(-1)
                    elif self.global_mode_gate == 'dual':
                        tmp_attn = tmp_attn * global_weights[(idx + 1)*2].transpose(0, 1).unsqueeze(-1)
                    else:
                        tmp_attn = tmp_attn * global_weights[idx].transpose(0,1).unsqueeze(-1)
                knowledge_contexts.append(tmp_attn)

        # Combine embedded input word and attended context, run through RNN
        if (self.global_mode_gate == 'extend' or self.global_mode_gate == 'dual') and self.ablation != 'no_knowledge_readout':
            if self.disable_sigmoid_gate is False and global_weights is not None:
                context = context * global_weights[0].transpose(0, 1).unsqueeze(-1)



        # Knowledge Selection Global
        if self.softmax_global_gate_fusion:
            assert self.ablation != 'no_knowledge_readout'
            if self.ablation == 'no_advanced_knowledge_selection':
                tmp_contexts = [context] + knowledge_contexts
                fused_context = None
                for idx in range(len(tmp_contexts)):
                    tmp = tmp_contexts[idx]
                    tmp = self.softmax_global_gate_fusion_projection[idx](tmp)
                    if fused_context is None:
                        fused_context = torch.zeros_like(tmp)
                    fused_context += tmp
            else:
                attn_mode_selection_input = [embedded, context, last_hidden] + knowledge_contexts
                attn_mode_selection_input = torch.cat(attn_mode_selection_input, 2)
                weights = torch.softmax(self.attn_mode_selector(attn_mode_selection_input), -1)
                fused_context = None
                tmp_contexts = [context] + knowledge_contexts
                if global_weights is not None:
                    new_weights = torch.zeros_like(weights)
                    assert len(global_weights) % len(tmp_contexts) == 0
                    if self.global_mode_gate == 'extend':
                        for idx in range(len(tmp_contexts)):
                            new_weights[:,:,idx:idx+1] = weights[:,:, idx:idx+1] * global_weights[idx].unsqueeze(0)
                    elif self.global_mode_gate == 'dual':
                        for idx in range(len(tmp_contexts)):
                            new_weights[:, :, idx:idx + 1] = weights[:, :, idx:idx + 1] * global_weights[idx * 2].unsqueeze(0)
                    else:
                        raise NotImplementedError()
                    weights = new_weights / new_weights.sum(-1, keepdim=True)



                for idx in range(len(tmp_contexts)):
                    tmp = tmp_contexts[idx]
                    tmp = self.softmax_global_gate_fusion_projection[idx](tmp)
                    tmp = tmp * (weights[:,:,idx:1+idx])
                    if fused_context is None:
                        fused_context = torch.zeros_like(tmp)
                    fused_context += tmp
            knowledge_contexts = []
            context = fused_context

        rnn_input_list = [embedded, context] + knowledge_contexts
        rnn_input = torch.cat(rnn_input_list, 2)
        # Run RNN
        output, hidden = self.rnn(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)

        knowledge_contexts = [x.squeeze(0) for x in knowledge_contexts]
        context_output_list = [output, context.squeeze(0), last_generated_token_embedded] + knowledge_contexts
        context_output = torch.cat(context_output_list, 1)

        # Output Probability Distributions
        output_prob_dists = []
        weighted_logits_dists = []
        copy_attn_weights = None

        # Vocab Probs
        output = self.vocab_predictor(context_output)
        vocab_probs = F.softmax(output, dim=1)
        output_prob_dists.append(vocab_probs)

        # SRC-Copied Probs
        if self.src_copy:
            # Copy Attention
            if self.rnn_type == 'lstm':
                copy_attn_query = hidden[-1][0]
            else:
                copy_attn_query = hidden[-1]

            copy_attn_query = copy_attn_query.unsqueeze(1)
            copy_attn_query = torch.cat([embedded, copy_attn_query.transpose(0, 1)], -1)

            src_copy_attention_cached_memory_bank = decoder_cache.get('src_copy_attention_cached_memory_bank', None)
            copy_context, copy_attn_weights, src_copy_attention_cached_memory_bank = self.src_copy_attention(
                copy_attn_query,
                encoder_memory_bank,
                encoder_length - 1,  # mask the last token <sos>
                max(encoder_length),
                cached_memory_bank=src_copy_attention_cached_memory_bank
            )
            copy_probs = copy_attn_weights.squeeze(0)
            decoder_cache['src_copy_attention_cached_memory_bank'] = src_copy_attention_cached_memory_bank

            # padding copy
            assert copy_probs.shape[1] <= self.max_copy_token_num
            if self.max_copy_token_num - copy_probs.shape[1] > 0 and self.copy_strategy != 'dynamic_vocab':
                padding_probs = torch.zeros(
                    [copy_probs.shape[0], self.max_copy_token_num - copy_probs.shape[1]],
                    device=copy_probs.device)
                src_copy_probs = torch.cat([copy_probs, padding_probs], -1)
                output_prob_dists.append(src_copy_probs)
            else:
                output_prob_dists.append(copy_probs)

        for knowledge in self.knowledge_managers:
            if knowledge.copy_mode == 'none':
                continue
            output_prob_dists.append(knowledge.get_copy_predict(
                hidden.permute(1, 0, 2),
                attn_query,
                embedded.permute(1, 0, 2),
                padding= self.copy_strategy != 'dynamic_vocab',
            ).squeeze(0))

        # Mode Fusion
        assert self.mode_num == len(output_prob_dists)
        if self.mode_num > 1:
            selector = self.mode_selector(context_output)
            if self.enable_knowledge_source_indicator_gate:
                if self.src_copy:
                    st = 2
                else:
                    st = 1
                selector = torch.softmax(selector, -1)

                if len(global_weights) > 0 and global_weights is not None:
                    new_selector = torch.zeros_like(selector)
                    if (self.global_mode_gate == 'extend' or self.global_mode_gate == 'dual'):
                        if self.src_copy:
                            st -= 1
                            start_offset = 0
                            assert st >= 1, 'the first prob can not be modified'
                        else:
                            start_offset = 1
                    else:
                        start_offset = 0
                    for idx in range(0, st):
                        new_selector[:, idx] = selector[:, idx]
                    for idx in range(st, self.mode_num):
                        if self.global_mode_gate == 'dual':
                            new_selector[:, idx] = selector[:, idx] * global_weights[(idx - st + start_offset)*2 +1].squeeze(-1)
                        else:
                            new_selector[:, idx] = selector[:, idx] * global_weights[idx-st + start_offset].squeeze(-1)
                    new_selector = new_selector / new_selector.sum(-1, keepdim=True)
                    selector = new_selector

            else:
                selector = torch.softmax(selector, -1)

            for i in range(self.mode_num):
                select_src_probs = selector[:, i:i + 1]
                output_prob_dists[i] = output_prob_dists[i] * select_src_probs + 1e-20
                weighted_logits_dists.append(output_prob_dists[i])

        if self.copy_strategy == 'dynamic_vocab':
            copy_offset_matrix = decoder_cache['copy_offset_matrix']

            copy_probs = weighted_logits_dists[1:]
            assert len(copy_offset_matrix) == len(copy_probs)
            fused_prob = weighted_logits_dists[0]
            fused_prob = torch.cat([fused_prob, torch.zeros(
                [batch_size, copy_offset_matrix[0].shape[-1] - weighted_logits_dists[0].shape[-1]]
                , dtype=fused_prob.dtype, device=fused_prob.device
            )], dim=-1)
            for copy_matrix, copy_prob in zip(copy_offset_matrix, copy_probs):
                copy_prob = copy_prob.unsqueeze(1)
                offset = torch.bmm(copy_prob, copy_matrix).squeeze(1)
                offset[:, 0] = 0.0
                fused_prob += offset
            output = torch.log(fused_prob + 1e-20)


        else:
            output = torch.cat(output_prob_dists, -1)
            output = torch.log(output + 1e-20)

        return output, hidden, attn_weights, copy_attn_weights, weighted_logits_dists
