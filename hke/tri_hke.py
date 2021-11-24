import torch
from torch import nn

from hke import word_encoder
from hke.knowledge_managers.CommonsenseKnowledgeManager import CommonsenseKnoweldgeManager
from hke.knowledge_managers.InfoboxKnowledgeManager import InfoboxKnowledgeManager
from hke.knowledge_managers.TextKnowledgeManager import TextKnowledgeManager
from seq2seq import rnn_helper
from hke.hke_bridge import LinearBridge
from hke.hke_decoder import HKEDecoder
from hke.hke_encoder import RNNEncoder
from seq2seq.attention import GlobalAttention
from utils.logger import logger
from utils import model_helper


class TriHKE(nn.Module):
    def __init__(self, args, data_fields=None):
        super(TriHKE, self).__init__()
        if data_fields is None:
            data_fields = dict()
        logger.info('[MODEL] Preparing the TriHKE model')
        
        self.drop_out = nn.Dropout(args.dropout)
        self.beam_length_penalize = args.beam_length_penalize
        self.hidden_size = args.hidden_size
        self.teach_force_rate = args.teach_force_rate
        self.embed_size = args.embed_size
        self.init_selection_mode = args.init_selection_mode
        self.src_embed = rnn_helper.build_embedding(args.tgt_vocab_size_with_offsets, args.embed_size)
        self.dec_embed = self.src_embed
        enc_embed = self.src_embed
        dec_embed = self.src_embed
        self.copy_strategy = args.copy_strategy
        self.copy_flags = set()

        self.enable_knowledge_source_indicator_gate = args.enable_global_mode_gate
        self.global_mode_gate = args.global_mode_gate
        self.ablation = args.ablation

        if self.ablation == 'no_advanced_knowledge_selection':
            assert args.enable_global_mode_gate is False
            assert args.softmax_global_gate_fusion is True

        if self.enable_knowledge_source_indicator_gate:
            if self.global_mode_gate == 'extend' or self.global_mode_gate == 'dual':
                self.init_attention_query_projection = torch.nn.Linear(self.hidden_size, self.hidden_size)
                gate_dim = 1
                if self.global_mode_gate == 'dual':
                    gate_dim = 2
                self.init_weight_func = torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size),
                    torch.nn.ELU(),
                    torch.nn.Dropout(args.dropout),
                    torch.nn.Linear(self.hidden_size, gate_dim)
                )
                self.init_attention = GlobalAttention(self.hidden_size, attn_type='general')

                if self.init_selection_mode == 'posterior':
                    self.post_init_attention_query_projection = torch.nn.Linear(self.hidden_size, self.hidden_size)
                    gate_dim = 1
                    if self.global_mode_gate == 'dual':
                        gate_dim = 2
                    self.post_init_weight_func = torch.nn.Sequential(
                        torch.nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size),
                        torch.nn.ELU(),
                        torch.nn.Dropout(args.dropout),
                        torch.nn.Linear(self.hidden_size, gate_dim)
                    )
                    self.post_init_attention = GlobalAttention(self.hidden_size, attn_type='general')





        # TODO 特别注意他们的顺序 Text， Infobox， CSK必须要按照顺序
        self.knowledge_modes = list(args.knowledge.split(','))
        assert args.enc_birnn is True
        self.debug_mode = args.debug

        knowledge_managers = []
        self.text_knowledge_managers = None


        # Char Encoders
        if args.word_encoder != 'none':
            if args.word_encoder_type == 'cnn':
                down_scale = 0.2
            else:
                down_scale = 1.00
            self.word_encoder = word_encoder.T2STokenEncoder(args.word_encoder_type,
                                          dropout=args.dropout,
                                          vocab_size=args.char_vocab_size,
                                          word_embed_size=args.embed_size,
                                          down_scale=down_scale,
                                          embed_size=args.embed_size, )
            self.uni_word_embed_dim = args.embed_size
        else:
            self.word_encoder = None
            self.uni_word_embed_dim =0


        # Knowledge Managers
        no_copy_knowledge_modes = args.mask_copy_flags.split(',')
        if args.uniform_word_encoding == 'none':
            text_embed = None
            word_encoder_mode = False
        elif args.uniform_word_encoding == 'word':
            text_embed = enc_embed
            word_encoder_mode = False
        elif args.uniform_word_encoding == 'char':
            text_embed = None
            word_encoder_mode = True
        else:
            raise NotImplementedError()
        knowledge_index = dict()
        for idx, knowledge in enumerate(self.knowledge_modes):
            knowledge_index[knowledge] = idx
            if knowledge == 'context':
                assert idx == 0
            elif knowledge == 'commonsense':
                assert idx == 1
                knowledge_manager = CommonsenseKnoweldgeManager(text_embed=text_embed, embed_size=args.embed_size,
                                                         hidden_size=args.hidden_size, dropout=args.dropout,
                                                         tgt_vocab_size=args.tgt_vocab_size,
                                                         max_copy_token_num=args.max_csk_num,
                                                         entity_field=data_fields['csk_post_entity'],
                                                         relation_field=data_fields['csk_relation'],
                                                         init_selection_mode=args.init_selection_mode,
                                                         dual_init_gate = args.global_mode_gate == 'dual',
                                                         share_copy_attn=args.share_knowledge_copy_attention,
                                                         copy_mode='default' if 'csk' not in no_copy_knowledge_modes else 'none',
                                                         word_encoder_mode = word_encoder_mode,
                                                         debug=args.debug)
                if 'csk' not in no_copy_knowledge_modes:
                    self.copy_flags.add('csk')
                self.csk_knowledge_manager = knowledge_manager
                knowledge_managers.append(knowledge_manager)
            elif knowledge == 'text':
                assert idx > knowledge_index.get('commonsense', 0)
                # Text 必须要直接共享主要的Embedding
                knowledge_manager = TextKnowledgeManager(text_embed=enc_embed, embed_size=args.embed_size,
                                                         hidden_size=args.hidden_size, dropout=args.dropout,
                                                         tgt_vocab_size=args.tgt_vocab_size,
                                                         transformer_layers=args.text_knowledge_trans_layers,
                                                         max_text_token_num=args.max_text_token_num,
                                                         init_selection_mode=args.init_selection_mode,
                                                         dual_init_gate=args.global_mode_gate == 'dual',
                                                         share_copy_attn=args.share_knowledge_copy_attention,
                                                         copy_mode='default' if 'text' not in no_copy_knowledge_modes else 'none',
                                                         word_encoder_mode=word_encoder_mode,
                                                         )
                if 'text' not in no_copy_knowledge_modes:
                    self.copy_flags.add('text')
                self.text_knowledge_managers = knowledge_manager
                knowledge_managers.append(knowledge_manager)
            elif knowledge == 'infobox':
                assert idx > knowledge_index.get('commonsense', 0)
                assert idx > knowledge_index.get('text', 0)
                knowledge_manager = InfoboxKnowledgeManager(src_embed=text_embed,
                                                            field_key_vocab_size=args.field_key_vocab_size,
                                                            field_word_vocab_size=args.field_word_vocab_size,
                                                            word_embed_size=args.embed_size, pos_embed_size=10,
                                                            max_field_intra_word_num=args.max_field_intra_word_num,
                                                            hidden_size=args.hidden_size, dropout=args.dropout,
                                                            infobox_transformer_field_encoder_heads=4,
                                                            max_kw_pairs_num=args.max_kw_pairs_num,
                                                            infobox_transformer_field_encoder_layers=2,
                                                            init_selection_mode=args.init_selection_mode,
                                                            dual_init_gate=args.global_mode_gate == 'dual',
                                                            share_copy_attn=args.share_knowledge_copy_attention,
                                                            word_encoder_mode=word_encoder_mode,
                                                            copy_mode='default' if 'box' not in no_copy_knowledge_modes else 'none',
                                                            )
                if 'box' not in no_copy_knowledge_modes:
                    self.copy_flags.add('box')

                self.infobox_knowledge_managers = knowledge_manager
                knowledge_managers.append(knowledge_manager)
            else:
                raise NotImplementedError()
        self.knowledge_managers = knowledge_managers


        # Compute RNN Inputs
        encoder_knowledge_input_dim = 0
        for knowledge in knowledge_managers:
            encoder_knowledge_input_dim += knowledge.get_readout_enc_dim()

        # 默认会输出 2*Embedding Size的东西

        self.encoder = RNNEncoder(enc_embed, args.hidden_size, args.embed_size + encoder_knowledge_input_dim + self.uni_word_embed_dim,
                                  args.dropout, n_layers=args.enc_layers,
                                  rnn_type=args.rnn_type, enc_birnn=args.enc_birnn)
        if self.init_selection_mode == 'posterior' and self.training:
            self.tgt_encoder = RNNEncoder(enc_embed, args.hidden_size, args.embed_size + encoder_knowledge_input_dim + self.uni_word_embed_dim,
                                      args.dropout, n_layers=args.enc_layers,
                                      rnn_type=args.rnn_type, enc_birnn=args.enc_birnn)

        self.birnn_down_scale = torch.nn.Sequential(
            torch.nn.Linear(args.hidden_size * 2, args.hidden_size, False),
            torch.nn.Tanh()
        )

        # Build Bridge
        self.bridge_mode = args.bridge

        if args.bridge == 'none':
            self.bridge = None
        # elif args.bridge == 'general':
        #     fusion_size = 0
        #     # 现在已经不需要了
        #     self.bridge = LinearBridge(args.bridge, args.rnn_type, args.hidden_size, args.enc_layers,
        #                           0.0, False, fusion_size=fusion_size)
        else:
            raise NotImplementedError()

        # Build Decoder
        decoder = HKEDecoder(hparams=args, embed=dec_embed, knowledge_managers=self.knowledge_managers)
        self.decoder = decoder

        # Build SRC-Copy Attention
        self.src_copy_mode = False
        self.copy_coverage = False
        self.add_src_word_embed_to_state = args.add_src_word_embed_to_state
        if args.copy:
            self.copy_flags.add('src')
            self.src_copy_mode = True
            self.max_copy_token_num = args.max_copy_token_num
            if self.add_src_word_embed_to_state:
                copy_state_input_dim = args.embed_size
            else:
                copy_state_input_dim = 0
            copy_state_input_dim += self.hidden_size
            if True:
                self.copied_state_to_input_embed_projection = nn.Linear(copy_state_input_dim,
                                                                        args.embed_size, bias=False)
            else:
                self.copied_state_to_input_embed_projection = None
        else:
            self.src_copy_mode = False

        # Special Tokens
        self.tgt_sos_idx = data_fields['tgt'].vocab.stoi['<sos>']
        self.tgt_eos_idx = data_fields['tgt'].vocab.stoi['<eos>']
        self.tgt_unk_idx = data_fields['tgt'].vocab.stoi['<unk>']
        assert self.tgt_unk_idx == 0
        self.tgt_pad_idx = data_fields['tgt'].vocab.stoi['<pad>']

    @classmethod
    def build_s2s_model(cls, args, data_fields):
        seq2seq = cls(args, data_fields)
        return seq2seq

    def beam_search_decoding(self, hparams, batch, beam_width):
        max_len = hparams.infer_max_len
        min_len = hparams.infer_min_len
        src, tgt, src_len, tgt_len, batch_size = self.get_dialogue_inputs_from_batch(batch)
        vocab_size = self.decoder.vocab_size_with_offsets
        raw_vocab_size = hparams.tgt_vocab_size
        copy_vocab_size = vocab_size - raw_vocab_size
        device = tgt.device

        disable_multiple_copy = hparams.disable_multiple_copy
        penalize_repeat_tokens_rate = hparams.penalize_repeat_tokens_rate

        with torch.no_grad():
            # Encoding Kowledge
            # Encoding Chars
            if self.word_encoder is not None:
                # Char_Embeddings: [Vocab_size, batch_size, dim]
                self.char_embeddings = self.word_encoder.encode(batch.char_vocab)
                # test = model_helper.select_dynamic_embeddings_from_time_first_idx(char_embeddings, batch.char_src[0])
            else:
                self.char_embeddings = None

            for knowledge_manager in self.knowledge_managers:
                knowledge_manager.represent_knowledge(batch, self.char_embeddings)
            # Encoding Query
            query_summary, encoder_memory_bank = self.encode_query(batch)
            # Bridge
            hidden, prior_hidden, mse_loss = self.bridge_init(batch, query_summary,
                                                              encoder_memory=(encoder_memory_bank, src_len),
                                                              knowledge_managers=self.knowledge_managers)
            # Generate Equivalent Embeddings And Prepare Embeddings
            equivalent_embeddings, copy_mask, coverage_losses, copy_attention_coverage_vector, dynamic_vocab = \
                self.generate_equivalent_embeddings(batch, encoder_memory_bank)

            # Multiply Query
            encoder_memory_bank = model_helper.create_beam(encoder_memory_bank, beam_width, 0)

            # Multiply Equivalent Embeddings
            if equivalent_embeddings is not None:
                equivalent_embeddings = model_helper.create_beam(equivalent_embeddings, beam_width, 0)
            if dynamic_vocab is not None:
                dynamic_vocab = model_helper.create_beam(dynamic_vocab, beam_width, 0)
            if copy_attention_coverage_vector is not None:
                copy_attention_coverage_vector = model_helper.create_beam(copy_attention_coverage_vector, beam_width, 0)

            # Decoding
            if disable_multiple_copy:
                copied_indicator = torch.zeros([batch_size * beam_width, copy_vocab_size], device=device)

            repeat_indicator = torch.ones([batch_size * beam_width, vocab_size], device=device)
            output_probs = torch.zeros(max_len, batch_size * beam_width, vocab_size, device=device)
            last_token = torch.ones([batch_size * beam_width], dtype=torch.long, device=device) * self.tgt_sos_idx
            last_raw_token = torch.ones([batch_size * beam_width], dtype=torch.long, device=device) * self.tgt_sos_idx
            token_outputs = [last_token]
            decoded_seq_len = torch.zeros([batch_size * beam_width], device=device)
            ending_flags = torch.zeros([batch_size * beam_width], dtype=torch.bool, device=device)
            padding_label = torch.ones([batch_size * beam_width], dtype=torch.long, device=device) * self.tgt_pad_idx

            padding_logits = -1e20 * torch.ones([batch_size, beam_width, beam_width], device=device)
            padding_logits[:, :, 0] = 0.0

            diverse_decoding_offsets = torch.arange(0, beam_width, device=device) * hparams.diverse_decoding
            diverse_decoding_offsets = -diverse_decoding_offsets.repeat(batch_size * beam_width).view(batch_size,
                                                                                                      beam_width,
                                                                                                      beam_width)

            # multiply beam
            beam_scores = torch.zeros(max_len, batch_size, beam_width, device=device)
            init_beam_score = torch.zeros(batch_size, beam_width, device=device) + -1e20
            init_beam_score[:, 0] = 0.0
            beam_scores[0] = init_beam_score

            hidden = model_helper.create_beam(hidden, beam_width, 1)
            src_len = model_helper.create_beam(src_len, beam_width, 0)

            knowledge_source_indicator = batch.knowledge_source_indicator
            local_mode = batch.local_mode
            knowledge_source_indicator = model_helper.create_beam(knowledge_source_indicator, beam_width, 1)
            local_mode = model_helper.create_beam(local_mode, beam_width, 1)

            for knowledge in self.knowledge_managers:
                knowledge.beam_expand(beam_width=beam_width)

            copy_matrix_cache = None
            vocab_size_with_offsets = self.decoder.vocab_size_with_offsets
            decoder_cache = dict()

            if self.copy_strategy == 'dynamic_vocab':
                self.generate_dynamic_matrix(batch, decoder_cache, vocab_size_with_offsets)
                for idx in range(len(decoder_cache['copy_offset_matrix'])):
                    decoder_cache['copy_offset_matrix'][idx] = model_helper.create_beam(decoder_cache['copy_offset_matrix'][idx], beam_width, 0)

            decoder_cache['global_weights'] = self.get_global_gates(model_helper.create_beam(query_summary, beam_width, 1)
                                                                    ,encoder_memory_bank, src_len)

            if self.word_encoder is not None:
                # [Vocab_size, batch_size, dim]
                raw_char_embeddings = model_helper.select_dynamic_embeddings_from_time_first_idx(
                    self.char_embeddings,
                    batch.char_dynamic_vocab[0])
                decoder_cache['raw_char_embeddings'] = model_helper.create_beam(raw_char_embeddings.permute(1, 0, 2).contiguous(), beam_width, 0)



            for t in range(1, max_len):
                # print('step', t)
                prob_output, hidden, attn_weights, copy_attn_weights, weighted_logits_dists = self.decoder(
                    (knowledge_source_indicator, local_mode),
                    last_token, last_raw_token, hidden, encoder_memory_bank, src_len,
                    copied_token_equip_embedding=equivalent_embeddings,
                    copy_attention_coverage_vector=copy_attention_coverage_vector,
                    decoder_cache = decoder_cache,
                )

                if self.copy_strategy == 'fusion':
                    prob_output = torch.exp(prob_output)
                    prob_output_with_copy, copy_matrix_cache = model_helper.generate_copy_offset_probs(
                        batch_size * beam_width, prob_output, raw_vocab_size, vocab_size, dynamic_vocab,
                        copy_matrix_cache)
                    prob_output_with_copy = torch.log(prob_output_with_copy)
                    prob_output = prob_output_with_copy

                # Ensure the min len
                if min_len != -1 and t <= min_len:
                    prob_output[:, self.tgt_eos_idx] = -1e20

                # Mask the unk
                if hparams.disable_unk_output:
                    prob_output[:, self.tgt_unk_idx] = -1e20

                prob_output = repeat_indicator * prob_output
                if disable_multiple_copy:
                    copy_probs = prob_output[:, raw_vocab_size:]
                    copy_probs += copied_indicator
                    prob_output[:, raw_vocab_size:] = copy_probs

                # Beam prob_output
                # (batch*beam, vocab_size)=> (batch, beam, vocab_size)
                prob_output = prob_output.view(batch_size, beam_width, vocab_size)

                # Select top-k for each beam => (batch, beam, beam]
                # topk_scores_in_each_beam 每个Beam中的Top-K，每个Beam中Top-K的Index（词）
                topk_scores_in_each_beam, topk_indices_in_each_beam = prob_output.topk(beam_width, dim=-1)

                # add diverse decoding penalize
                if t > 1:
                    topk_scores_in_each_beam = topk_scores_in_each_beam + diverse_decoding_offsets

                # avoid copying a finished beam(batch_size, beam)
                tmp_ending_flags = ending_flags.view(batch_size, beam_width, 1)
                finished_offsets = - topk_scores_in_each_beam * tmp_ending_flags
                finished_offsets = finished_offsets + padding_logits * tmp_ending_flags

                # compute total beam scores
                total_beam_scores = beam_scores.sum(dim=0)
                topk_total_scores_in_each_beam = topk_scores_in_each_beam + total_beam_scores.unsqueeze(-1)
                topk_total_scores_in_each_beam = topk_total_scores_in_each_beam + finished_offsets

                # decoded_seq_len (batch, beam_width, 1)
                tmp_decoded_seq_len = torch.where(ending_flags, decoded_seq_len, decoded_seq_len + 1)
                if self.beam_length_penalize == 'avg':
                    length_factor = tmp_decoded_seq_len.view(batch_size, beam_width, 1)
                    # 限制最大惩罚长度
                    length_factor = torch.min(length_factor, torch.ones_like(length_factor) * 20)
                    topk_total_scores_in_each_beam = topk_total_scores_in_each_beam / length_factor

                # Group beams/tokens to the corresponding group
                # => (batch, beams*beams)
                topkk_scores_in_batch_group = topk_total_scores_in_each_beam.view(batch_size, -1)
                topkk_tokens_in_batch_group = topk_indices_in_each_beam.view(batch_size, -1)

                # current step scores
                topk_current_scores_in_each_beam = topk_scores_in_each_beam + finished_offsets
                topk_current_scores_in_batch_group = topk_current_scores_in_each_beam.view(batch_size, -1)

                # Select top-K beams in each top-k*top-k batch group => (batch, beam)
                # topk_scores_in_batch 每个Batch Group中选择出来的K个分数，topk_indices_in_batch 每个Batch Group对应的 Beam Index
                topk_scores_in_batch, topk_indices_in_batch = topkk_scores_in_batch_group.topk(beam_width, -1)
                # => top_k tokens
                topk_tokens_in_batch = topkk_tokens_in_batch_group.gather(dim=-1, index=topk_indices_in_batch)

                # Current Scores (batch, beam*beam) => (batch, beam) 每个Batch内的Top-K分数
                current_scores_in_batch = topk_current_scores_in_batch_group.gather(dim=-1, index=topk_indices_in_batch)

                # => (batch, beam) the selected beam ids of the new top-k beams
                selected_last_beam_indices_in_batch = topk_indices_in_batch // beam_width

                # => this indices  is used to select batch*beam
                flatten_indices = selected_last_beam_indices_in_batch.view(-1)
                flatten_offsets = torch.arange(0, batch_size * beam_width, beam_width, device=device)
                flatten_offsets = flatten_offsets.repeat_interleave(beam_width, -1)
                flatten_offsets = flatten_offsets.view(-1)
                flatten_indices = flatten_indices + flatten_offsets

                # select hidden, from outputted hidden, select original beam
                # (layer, batch*beam, dim) => (layer, batch*beam, dim)
                if not isinstance(hidden, tuple) and not isinstance(hidden, list):
                    resorted_hidden = hidden.index_select(dim=1, index=flatten_offsets)
                    hidden = resorted_hidden
                else:
                    hidden = tuple([x.index_select(dim=1, index=flatten_offsets) for x in hidden])

                # select beam_scores, from original
                # (max_len, batch_size, beam_width) => (max_len, batch_size*beam_width)
                flatten_beam_scores = beam_scores.view(-1, batch_size * beam_width)
                resorted_beam_scores = flatten_beam_scores.index_select(dim=1, index=flatten_indices).view(
                    beam_scores.shape)
                beam_scores = resorted_beam_scores

                # select token_outputs, from original
                # (max_len, batch_size*beam_width)
                next_token_outputs = []
                for time_id in range(t):
                    current_tensor = token_outputs[time_id]
                    resorted_current_tensor = current_tensor.index_select(dim=0, index=flatten_indices)
                    next_token_outputs.append(resorted_current_tensor)
                token_outputs = next_token_outputs

                # select ending_flags (batch*beam), from original
                resorted_ending_flags = ending_flags.index_select(dim=0, index=flatten_indices)
                ending_flags = resorted_ending_flags

                # select copied_indicator (batch*beam), from original
                if disable_multiple_copy:
                    copied_indicator = copied_indicator.index_select(dim=0, index=flatten_indices)

                repeat_indicator = repeat_indicator.index_select(dim=0, index=flatten_indices)

                # selected sequence_length
                decoded_seq_len = tmp_decoded_seq_len.index_select(dim=0, index=flatten_indices)

                # select copy attention coverage
                if copy_attention_coverage_vector is not None:
                    # From original
                    resorted_copy_attention_coverage_vector = \
                        copy_attention_coverage_vector.index_select(dim=0, index=flatten_indices)
                    copy_attention_coverage_vector = resorted_copy_attention_coverage_vector
                    copy_attn_weights = copy_attn_weights.index_select(dim=0, index=flatten_indices)
                    copy_attention_coverage_vector = copy_attention_coverage_vector + copy_attn_weights.squeeze(0)

                # update beams
                last_token = torch.where(ending_flags, padding_label, topk_tokens_in_batch.view(-1))
                token_outputs.append(last_token)
                beam_scores[t] = current_scores_in_batch * (last_token.view(batch_size, beam_width) != self.tgt_pad_idx)

                #
                if self.copy_strategy == 'fusion':
                    dynamic_word_idx = last_token - raw_vocab_size
                    zero_idx = torch.zeros_like(dynamic_word_idx, device=dynamic_word_idx.device)
                    valid_dynamic_word_idx = dynamic_word_idx >= 0
                    dynamic_word_idx = torch.where(valid_dynamic_word_idx, dynamic_word_idx, zero_idx)
                    # dynamic_raw_idx = torch.index_select(dynamic_vocab, dim=1, index=dynamic_word_idx)
                    flatten_dynamic_vocab = dynamic_vocab.view(-1)
                    dynamic_word_idx_offset = torch.arange(0, batch_size * beam_width, dtype=torch.int64,
                                                           device=dynamic_vocab.device) * dynamic_vocab.shape[1]
                    dynamic_raw_idx = torch.index_select(flatten_dynamic_vocab, dim=0,
                                                         index=dynamic_word_idx + dynamic_word_idx_offset)
                    valid_dynamic_word_idx = valid_dynamic_word_idx.to(torch.int64)
                    last_raw_token = valid_dynamic_word_idx * dynamic_raw_idx + (
                                1 - valid_dynamic_word_idx) * last_token

                # Is finished
                is_finished = last_token == self.tgt_eos_idx
                ending_flags |= is_finished
                if torch.sum(ending_flags) == batch_size * beam_width:
                    break

                # Adding mask
                if disable_multiple_copy:
                    new_copy_indicator = torch.zeros_like(copied_indicator, device=copied_indicator.device)
                    copy_index = last_token - raw_vocab_size
                    is_copied_token = copy_index >= 0
                    copy_index = torch.max(torch.zeros_like(copy_index, device=copy_index.device), copy_index)
                    new_copy_indicator = new_copy_indicator.scatter(-1, copy_index.unsqueeze(-1), -1e10)
                    copied_indicator = copied_indicator + new_copy_indicator

                # Repeat repeat_indicator
                new_repeat_indicator = torch.ones_like(repeat_indicator, device=repeat_indicator.device)
                new_repeat_indicator = new_repeat_indicator.scatter(-1, last_token.unsqueeze(-1),
                                                                    penalize_repeat_tokens_rate)
                repeat_indicator = repeat_indicator * new_repeat_indicator

            return output_probs, token_outputs, beam_scores.view(-1, batch_size * beam_width)

    def greedy_search_decoding(self, hparams, batch):
        # Set variables
        max_len = hparams.infer_max_len
        min_len = hparams.infer_min_len
        src, tgt, src_len, tgt_len, batch_size = self.get_dialogue_inputs_from_batch(batch)
        vocab_size = self.decoder.vocab_size_with_offsets
        raw_vocab_size = self.decoder.vocab_size
        device = tgt.device

        with torch.no_grad():

            # Encoding Chars
            if self.word_encoder is not None:
                # Char_Embeddings: [Vocab_size, batch_size, dim]
                self.char_embeddings = self.word_encoder.encode(batch.char_vocab)
                # test = model_helper.select_dynamic_embeddings_from_time_first_idx(char_embeddings, batch.char_src[0])
            else:
                self.char_embeddings = None

            # Encoding Knowledge
            for knowledge_manager in self.knowledge_managers:
                knowledge_manager.represent_knowledge(batch, self.char_embeddings )
            # Encoding Query
            query_summary, encoder_memory_bank = self.encode_query(batch)
            # Bridge
            hidden, prior_hidden, mse_loss = self.bridge_init(batch, query_summary,
                                                              encoder_memory=(encoder_memory_bank, src_len),
                                                              knowledge_managers= self.knowledge_managers)

            output_probs = torch.zeros(max_len, batch_size, vocab_size, device=device)
            scores = torch.zeros(max_len, batch_size, device=device)
            last_token = torch.ones([batch_size], dtype=torch.long, device=device) * self.tgt_sos_idx
            last_raw_token = torch.ones([batch_size], dtype=torch.long, device=device) * self.tgt_sos_idx
            token_outputs = [last_token]
            ending_flags = torch.zeros([batch_size], dtype=torch.bool, device=device)
            padding_label = torch.ones([batch_size], dtype=torch.long, device=device) * self.tgt_pad_idx
            padding_score = torch.zeros([batch_size], device=device)

            # Generate Equivalent Embeddings And Prepare Embeddings
            equivalent_embeddings, copy_mask, coverage_losses, copy_attention_coverage_vector, dynamic_vocab = \
                self.generate_equivalent_embeddings(batch, encoder_memory_bank)

            copy_matrix_cache = None

            decoder_cache = dict()
            vocab_size_with_offsets = self.decoder.vocab_size_with_offsets
            decoder_cache['global_weights'] = self.get_global_gates(query_summary,encoder_memory_bank, src_len)
            if self.copy_strategy == 'dynamic_vocab':
                self.generate_dynamic_matrix(batch, decoder_cache, vocab_size_with_offsets)

            if self.word_encoder is not None:
                # [Vocab_size, batch_size, dim]
                raw_char_embeddings = model_helper.select_dynamic_embeddings_from_time_first_idx(
                    self.char_embeddings,
                    batch.char_dynamic_vocab[0])
                decoder_cache['raw_char_embeddings'] = raw_char_embeddings.permute(1, 0, 2).contiguous()

            for t in range(1, max_len):
                prob_output, hidden, attn_weights, copy_attn_weights, weighted_logits_dists = self.decoder(
                    (batch.knowledge_source_indicator, batch.local_mode),
                    last_token, last_raw_token, hidden, encoder_memory_bank, src_len,
                    copied_token_equip_embedding=equivalent_embeddings,
                    copy_attention_coverage_vector=copy_attention_coverage_vector,
                    decoder_cache=decoder_cache
                )

                if self.src_copy_mode:
                    copy_attn_weights = copy_attn_weights.squeeze(0)
                    coverage_diff = torch.min(copy_attn_weights, copy_attention_coverage_vector)
                    copy_attention_coverage_vector = copy_attention_coverage_vector + copy_attn_weights

                # Ensure the min len
                if min_len != -1 and t <= min_len:
                    prob_output[:, self.tgt_eos_idx] = -1e20

                # Mask unk
                if hparams.disable_unk_output:
                    prob_output[:, self.tgt_unk_idx] = -1e20

                output_probs[t] = prob_output


                if self.copy_strategy == 'fusion':
                    prob_output = torch.exp(prob_output)
                    prob_output_with_copy, copy_matrix_cache = model_helper.generate_copy_offset_probs(batch_size, prob_output, raw_vocab_size, vocab_size, dynamic_vocab,copy_matrix_cache)
                    prob_output_with_copy = torch.log(prob_output_with_copy)
                    score, last_token = prob_output_with_copy.data.max(1)
                else:
                    score, last_token = prob_output.data.max(1)

                last_token = torch.where(ending_flags, padding_label, last_token)

                if self.copy_strategy == 'fusion':
                    dynamic_word_idx = last_token - raw_vocab_size
                    zero_idx = torch.zeros_like(dynamic_word_idx, device=dynamic_word_idx.device)
                    valid_dynamic_word_idx = dynamic_word_idx >= 0
                    dynamic_word_idx = torch.where(valid_dynamic_word_idx, dynamic_word_idx, zero_idx)
                    # dynamic_raw_idx = torch.index_select(dynamic_vocab, dim=1, index=dynamic_word_idx)
                    flatten_dynamic_vocab = dynamic_vocab.view(-1)
                    dynamic_word_idx_offset = torch.arange(0, batch_size, dtype=torch.int64, device=dynamic_vocab.device) * dynamic_vocab.shape[1]
                    dynamic_raw_idx = torch.index_select(flatten_dynamic_vocab, dim=0, index=dynamic_word_idx + dynamic_word_idx_offset)
                    valid_dynamic_word_idx = valid_dynamic_word_idx.to(torch.int64)
                    last_raw_token = valid_dynamic_word_idx * dynamic_raw_idx + (1 - valid_dynamic_word_idx) * last_token

                token_outputs.append(last_token)
                score = torch.where(ending_flags, padding_score, score)
                scores[t] = score

                # Is finished
                is_finished = last_token == self.tgt_eos_idx
                ending_flags |= is_finished
                if torch.sum(ending_flags) == batch_size:
                    break

            return output_probs, token_outputs, scores

    def get_dialogue_inputs_from_batch(self, batch):
        src = batch.src[0]
        src_len = batch.src[1]
        tgt = batch.tgt[0]
        tgt_len = batch.tgt[1]
        batch_size = tgt.size(1)
        return src, tgt, src_len, tgt_len, batch_size

    def encode_query(self, batch, posterior=False):
        src, tgt, src_len, tgt_len, batch_size = self.get_dialogue_inputs_from_batch(batch)
        knowledge_readouts = []
        for knowledge in self.knowledge_managers:
            readout = knowledge.get_readout_enc(batch)
            if readout is not None:
                knowledge_readouts.append(readout)
        if len(knowledge_readouts) == 0:
            knowledge_readouts = None
        else:
            raise NotImplementedError('目前暂时不支持')
            knowledge_readouts = torch.cat(knowledge_readouts, -1)



        if posterior is False:
            if self.word_encoder is not None:
                dynamic_char_embedding = model_helper.select_dynamic_embeddings_from_time_first_idx(
                    self.char_embeddings,
                    batch.char_src[0])
                if knowledge_readouts is None:
                    knowledge_readouts = dynamic_char_embedding
                else:
                    knowledge_readouts = torch.cat([dynamic_char_embedding, knowledge_readouts], -1)
            encoder_output, hidden = self.encoder(src, src_len, knowledge_readouts=knowledge_readouts)
        else:
            if self.word_encoder is not None:
                dynamic_char_embedding = model_helper.select_dynamic_embeddings_from_time_first_idx(
                    self.char_embeddings,
                    batch.char_tgt[0])
                if knowledge_readouts is None:
                    knowledge_readouts = dynamic_char_embedding
                else:
                    knowledge_readouts = torch.cat([dynamic_char_embedding, knowledge_readouts], -1)
            encoder_output, hidden = self.tgt_encoder(tgt, tgt_len, knowledge_readouts=knowledge_readouts,
                                                      enforce_sorted=False)
        encoder_memory_bank = encoder_output.transpose(0, 1)

        # 降采样
        if self.birnn_down_scale is not None:
            encoder_memory_bank = self.birnn_down_scale(encoder_memory_bank)
            if isinstance(hidden, tuple):
                hidden = [self.birnn_down_scale(x) for x in hidden]
            else:
                hidden = self.birnn_down_scale(hidden)

        return hidden, encoder_memory_bank

    def encode_response(self, batch):
        src, tgt, src_len, tgt_len, batch_size = self.get_dialogue_inputs_from_batch(batch)
        knowledge_readouts = []
        for knowledge in self.knowledge_managers:
            readout = knowledge.get_readout_enc(batch)
            if readout is not None:
                knowledge_readouts.append(readout)
        if len(knowledge_readouts) == 0:
            knowledge_readouts = None
        else:
            knowledge_readouts = torch.cat(knowledge_readouts, -1)
        assert knowledge_readouts is None
        encoder_output, hidden = self.encoder(tgt, tgt_len, knowledge_readouts=knowledge_readouts, enforce_sorted=False)
        encoder_memory_bank = encoder_output.transpose(0, 1)
        if self.birnn_down_scale is not None:
            encoder_memory_bank = torch.tanh(self.birnn_down_scale(encoder_memory_bank))
        return hidden, encoder_memory_bank

    def bridge_init(self, batch, hidden, encoder_memory=None, knowledge_managers=None):
        mse_loss = None
        prior_hidden = None



        # Bridge Part
        if self.bridge_mode == 'general':
            hidden = self.bridge(hidden)
        elif self.bridge_mode == 'fusion' and (self.init_selection_mode == 'prior' or not self.training):
            prior_knowledge_fusion = []
            for knowledge in knowledge_managers:
                prior_knowledge_fusion.append(knowledge.get_init_readout(batch, hidden)[0])
            prior_knowledge_fusion = torch.cat(prior_knowledge_fusion, -1)
            hidden = self.bridge(hidden, fusion_knowledge=prior_knowledge_fusion)
        elif self.bridge_mode == 'fusion' and  self.init_selection_mode == 'posterior' and self.training:

            def kld(p, q):
                tmp = p * torch.log(p / (q + 1e-20) + 1e-20)
                tmp = tmp.sum(-1)
                return tmp

            prior_knowledge_fusion = []
            prior_knowledge_attns = []
            for knowledge in knowledge_managers:
                know_out = knowledge.get_init_readout(batch, hidden)
                prior_knowledge_fusion.append(know_out[0])
                prior_knowledge_attns.append(know_out[1])
            prior_knowledge_fusion = torch.cat(prior_knowledge_fusion, -1)
            prior_hidden = self.bridge(hidden, fusion_knowledge=prior_knowledge_fusion)

            post_hidden, _ = self.encode_response(batch)
            post_knowledge_fusion = []
            post_knowledge_attns = []
            for knowledge in knowledge_managers:
                know_out = knowledge.get_init_readout(batch, post_hidden, posterior=True)
                post_knowledge_fusion.append(know_out[0])
                post_knowledge_attns.append(know_out[1])
            post_knowledge_fusion = torch.cat(post_knowledge_fusion, -1)
            post_hidden = self.bridge(hidden, fusion_knowledge=post_knowledge_fusion)

            mse_loss = 0
            for post, prior in zip(post_knowledge_attns, prior_knowledge_attns):
                mse_loss += kld(post, prior).mean() * 0.5

            hidden = post_hidden
            prior_hidden = prior_hidden


        else:
            # 默认情况 没有Bridge
            assert len(hidden) == self.decoder.n_layers
            hidden = hidden[:self.decoder.n_layers]

        return hidden, prior_hidden, mse_loss

    def generate_equivalent_embeddings(self, batch, encoder_memory_bank):
        src, tgt, src_len, tgt_len, batch_size = self.get_dialogue_inputs_from_batch(batch)
        max_len = tgt.size(0)

        equip_embeddings = []
        copy_mask = None
        coverage_losses = None
        copy_attention_coverage_vector = None
        dynamic_vocab = []
        copy_strategy = self.copy_strategy

        if copy_strategy != 'dynamic_vocab':
            if self.src_copy_mode:
                if self.add_src_word_embed_to_state:
                    copy_state_input = [self.decoder.embed(src).transpose(1, 0)]
                else:
                    copy_state_input = []
                copy_mask = model_helper.sequence_mask_fn(src_len, dtype=torch.float32)
                copy_state_input.append(encoder_memory_bank)
                copy_state_input = torch.cat(copy_state_input, -1)
                if self.copied_state_to_input_embed_projection is not None:
                    copied_token_equip_embedding = self.copied_state_to_input_embed_projection(copy_state_input)
                else:
                    copied_token_equip_embedding = copy_state_input
                max_src_len = src.shape[0]
                src_copy_padding_embedding = torch.zeros(
                    [batch_size, self.max_copy_token_num - max_src_len, self.embed_size],
                    device=copy_state_input.device)
                src_idx_padding = torch.zeros(
                    [batch_size, self.max_copy_token_num - max_src_len], device=copy_state_input.device, dtype=src.dtype
                )
                copied_token_equip_embedding = torch.cat([copied_token_equip_embedding, src_copy_padding_embedding], dim=1)
                padded_src_idx = torch.cat([src.transpose(0, 1), src_idx_padding], dim=-1)
                equip_embeddings.append(copied_token_equip_embedding)
                dynamic_vocab.append(padded_src_idx)
                coverage_losses = torch.zeros(max_len, device=tgt.device)
                copy_attention_coverage_vector = torch.zeros([batch_size, max_src_len],
                                                             dtype=torch.float32, device=tgt.device)
            for knowledge in self.knowledge_managers:
                if knowledge.can_copy:
                    embed, idx = knowledge.get_equip_embeddings(batch)
                    equip_embeddings.append(embed)
                    dynamic_vocab.append(idx)
        else:
            if self.src_copy_mode:
                max_src_len = src.shape[0]
                copy_mask = model_helper.sequence_mask_fn(src_len, dtype=torch.float32)
                coverage_losses = torch.zeros(max_len, device=tgt.device)
                copy_attention_coverage_vector = torch.zeros([batch_size, max_src_len],
                                                             dtype=torch.float32, device=tgt.device)

        if len(equip_embeddings) == 0:
            copied_token_equip_embedding = None
            dynamic_vocab = None
        else:
            copied_token_equip_embedding = torch.cat(equip_embeddings, 1).contiguous()
            dynamic_vocab = torch.cat(dynamic_vocab, -1)
        return copied_token_equip_embedding, copy_mask, coverage_losses, copy_attention_coverage_vector, dynamic_vocab


    def generate_dynamic_matrix(self, batch, decoder_cache, vocab_size_with_offsets):
        copy_offset_matrix = []
        if 'src' in self.copy_flags:
            copy_matrix = model_helper.generate_copy_offset_matrix(vocab_size_with_offsets,
                                                                   batch.dynamic_vocab_src[0],
                                                                   batch.dynamic_vocab_src[1])
            copy_offset_matrix.append(copy_matrix)
        if 'csk' in self.copy_flags:
            copy_matrix = model_helper.generate_copy_offset_matrix(vocab_size_with_offsets,
                                                                   batch.dynamic_vocab_csk[0],
                                                                   batch.dynamic_vocab_csk[1])
            copy_offset_matrix.append(copy_matrix)
        if 'text' in self.copy_flags:
            copy_matrix = model_helper.generate_copy_offset_matrix(vocab_size_with_offsets,
                                                                   batch.dynamic_vocab_text[0],
                                                                   batch.dynamic_vocab_text[1])
            copy_offset_matrix.append(copy_matrix)
        if 'box' in self.copy_flags:
            copy_matrix = model_helper.generate_copy_offset_matrix(vocab_size_with_offsets,
                                                                   batch.dynamic_vocab_attribute[0],
                                                                   batch.dynamic_vocab_attribute[1])
            copy_offset_matrix.append(copy_matrix)
        decoder_cache['copy_offset_matrix'] = copy_offset_matrix


    def get_global_gates(self, query_summary, encoder_memory_bank, src_len, tgt_query_summary=None):
        if query_summary is not None and self.init_selection_mode == 'posterior':
            # 砍掉往前传的梯度，只优化当前的选择器, 且不能出现tgt
            assert tgt_query_summary is None
            # query_summary = query_summary.detach()
            # encoder_memory_bank = encoder_memory_bank.detach()
            detach = True
        else:
            detach = False
        # Global Gate
        if self.enable_knowledge_source_indicator_gate:
            global_weights = []
            if self.global_mode_gate == 'extend' or self.global_mode_gate == 'dual':
                if tgt_query_summary is None:
                    attn_query = self.init_attention_query_projection(query_summary).transpose(1, 0)
                    context, attn_weights = self.init_attention(
                        attn_query,
                        encoder_memory_bank,
                        coverage=None,
                        memory_lengths=src_len,
                        no_concat=True,
                    )
                    logit = self.init_weight_func(torch.cat([query_summary, context], dim=-1).squeeze(0))
                else:
                    attn_query = self.post_init_attention_query_projection(tgt_query_summary).transpose(1, 0)
                    context, attn_weights = self.post_init_attention(
                        attn_query,
                        encoder_memory_bank,
                        coverage=None,
                        memory_lengths=src_len,
                        no_concat=True,
                    )
                    logit = self.post_init_weight_func(torch.cat([tgt_query_summary, context], dim=-1).squeeze(0))

                logit = torch.sigmoid(logit)
                if self.global_mode_gate == 'dual':
                    # 两个不同的Gate, 第一个Gate给Readout用，第二个给概率用
                    global_weights.append(logit[:, 0:1])
                    global_weights.append(logit[:, 1:2])
                else:
                    global_weights.append(logit)
            for knowledge_manager in self.knowledge_managers:
                if tgt_query_summary is not None:
                    logit = knowledge_manager.get_init_readout(None, tgt_query_summary, posterior=True)
                else:
                    logit = knowledge_manager.get_init_readout(None, query_summary, detach=detach)
                logit = torch.sigmoid(logit)
                if self.global_mode_gate == 'dual':
                    # 两个不同的Gate, 第一个Gate给Readout用，第二个给概率用
                    global_weights.append(logit[:, 0:1])
                    global_weights.append(logit[:, 1:2])
                else:
                    global_weights.append(logit)

        else:
            global_weights = None
        return global_weights

    def forward(self, batch, mode='train'):

        src, tgt, src_len, tgt_len, batch_size = self.get_dialogue_inputs_from_batch(batch)
        tgt_raw, _ = batch.tgt_raw
        max_len = tgt.size(0)
        vocab_size_with_offsets = self.decoder.vocab_size_with_offsets

        # Encoding Chars
        if self.word_encoder is not None:
            # Char_Embeddings: [Vocab_size, batch_size, dim]
            self.char_embeddings = self.word_encoder.encode(batch.char_vocab)
            # test = model_helper.select_dynamic_embeddings_from_time_first_idx(char_embeddings, batch.char_src[0])
        else:
            self.char_embeddings = None

        # Representing Knowledge
        for knowledge_manager in self.knowledge_managers:
            knowledge_manager.represent_knowledge(batch, self.char_embeddings)

        # Encoding Query
        query_summary, encoder_memory_bank = self.encode_query(batch)
        if self.init_selection_mode == 'posterior' :
            tgt_query_summary, tgt_encoder_memory_bank = self.encode_query(batch, posterior=True)
        else:
            tgt_query_summary = None
            tgt_encoder_memory_bank = None

        # Bridge
        hidden, prior_hidden, mse_loss = self.bridge_init(batch, query_summary,
                                                          encoder_memory=(encoder_memory_bank, src_len),
                                                          knowledge_managers=self.knowledge_managers)


        # Generate Equivalent Embeddings And Prepare Embeddings
        equivalent_embeddings, copy_mask, coverage_losses, copy_attention_coverage_vector, dynamic_vocab = \
            self.generate_equivalent_embeddings(batch, encoder_memory_bank)

        # Decoder Part

        if self.copy_strategy == 'dynamic_vocab':
            weighted_logits_dists_by_time = []
            outputs = torch.zeros(max_len, batch_size, vocab_size_with_offsets, device=tgt.device)
        else:
            weighted_logits_dists_by_time = None
            outputs = torch.zeros(max_len, batch_size, vocab_size_with_offsets, device=tgt.device)

        # Training and Evaluating
        output = tgt.data[0, :]  # 第一个输入 sos
        raw_input = tgt_raw.data[0, :]
        if mode == 'train':
            token_outputs = []
            teach_force_rate = self.teach_force_rate
            if self.copy_strategy != 'default':
                assert teach_force_rate == 1
        else:
            token_outputs = [output.cpu().numpy()]
            teach_force_rate = 1.0

        if self.decoder.bow_loss and mode == 'train':
            # BOW Representation
            if prior_hidden is not None:
                tmp = prior_hidden.permute(2, 1, 0).reshape(batch_size, -1)
                tmp = self.decoder.dropout(torch.tanh(self.decoder.out_bow_l1(tmp)))
                bow_output = -torch.log(torch.softmax(self.decoder.out_bow_l2(tmp), -1))
                bag_of_words_tgt_index = batch.tgt_bow[0].transpose(1, 0)
                bag_of_words_probs = bow_output.gather(dim=-1, index=bag_of_words_tgt_index)
                valid_index = bag_of_words_tgt_index > 5
                bag_of_words_loss_v2 = torch.where(valid_index, bag_of_words_probs,
                                                   torch.zeros_like(bag_of_words_probs))
                valid_length = valid_index.sum(dim=-1)
                valid_length = torch.max(valid_length, torch.ones_like(valid_length))
                bag_of_words_loss_v2 = bag_of_words_loss_v2.sum(dim=-1) / valid_length
                bag_of_words_loss = bag_of_words_loss_v2.mean()
            else:
                tmp = hidden.permute(2, 1, 0).reshape(batch_size, -1)
                tmp = self.decoder.dropout(torch.tanh(self.decoder.out_bow_l1(tmp)))
                bow_output = -torch.log(torch.softmax(self.decoder.out_bow_l2(tmp), -1))
                bag_of_words_tgt_index = batch.tgt_bow[0].transpose(1, 0)
                bag_of_words_probs = bow_output.gather(dim=-1, index=bag_of_words_tgt_index)
                valid_index = bag_of_words_tgt_index > 5
                bag_of_words_loss = torch.where(valid_index, bag_of_words_probs, torch.zeros_like(bag_of_words_probs))
                valid_length = valid_index.sum(dim=-1)
                valid_length = torch.max(valid_length, torch.ones_like(valid_length))
                bag_of_words_loss = bag_of_words_loss.sum(dim=-1) / valid_length
                bag_of_words_loss = bag_of_words_loss.mean()

        else:
            bag_of_words_loss = None

        # For each batch
        decoder_cache = dict()
        decoder_cache['global_weights'] = self.get_global_gates(query_summary, encoder_memory_bank, src_len, None)
        if self.init_selection_mode == 'posterior' and self.training:
            decoder_cache['post_global_weights'] = self.get_global_gates(None, encoder_memory_bank, src_len, tgt_query_summary)
            decoder_cache['prior_global_weights'] = decoder_cache['global_weights']
            decoder_cache['global_weights'] = decoder_cache['post_global_weights']

        if self.copy_strategy == 'dynamic_vocab':
            self.generate_dynamic_matrix(batch, decoder_cache, vocab_size_with_offsets)

        if self.word_encoder is not None:
            # [Vocab_size, batch_size, dim]
            raw_char_embeddings = model_helper.select_dynamic_embeddings_from_time_first_idx(
                self.char_embeddings,
                batch.char_dynamic_vocab[0])
            decoder_cache['raw_char_embeddings'] = raw_char_embeddings.permute(1, 0, 2).contiguous()

        for t in range(1, max_len):
            output, hidden, attn_weights, copy_attn_weights, weighted_logits_dists = self.decoder(
                (batch.knowledge_source_indicator, batch.local_mode),
                output, raw_input,  hidden, encoder_memory_bank, src_len,
                copied_token_equip_embedding=equivalent_embeddings,
                copy_attention_coverage_vector=copy_attention_coverage_vector,
                decoder_cache=decoder_cache
            )

            if self.copy_strategy == 'dynamic_vocab':
                weighted_logits_dists_by_time.append(weighted_logits_dists)
                outputs[t] = output
            else:
                outputs[t] = output

            _, predict_token = output.data.max(dim=1)
            rand_teach_force_flag = torch.rand([batch_size], device=predict_token.device) < teach_force_rate
            output = torch.where(rand_teach_force_flag, tgt.data[t], predict_token)
            raw_input = tgt_raw[t]

            if mode == 'eval':
                token_outputs.append(output.cpu().numpy())
            if self.src_copy_mode:
                copy_attn_weights = copy_attn_weights.squeeze(0)
                coverage_diff = torch.min(copy_attn_weights, copy_attention_coverage_vector)
                coverage_loss = (coverage_diff * copy_mask.unsqueeze(1)).sum(-1).mean()
                coverage_losses[t] = coverage_loss
                copy_attention_coverage_vector = copy_attention_coverage_vector + copy_attn_weights
            else:
                copy_attention_coverage_vector = None

        decoder_cache['weighted_logits_dists_by_time'] = weighted_logits_dists_by_time
        return outputs, token_outputs, coverage_losses, bag_of_words_loss, None, decoder_cache
