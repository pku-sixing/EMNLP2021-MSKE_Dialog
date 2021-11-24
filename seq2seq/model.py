import torch
from torch import nn
from seq2seq import rnn_helper
from seq2seq.bridge import LinearBridge
from seq2seq.decoder import RNNDecoder
from seq2seq.encoder import RNNEncoder
from utils.logger import logger
from utils import model_helper


class Seq2Seq(nn.Module):
    def __init__(self, args, src_field, tgt_field):
        super(Seq2Seq, self).__init__()
        logger.info('[MODEL] Preparing the Standard Seq2Seq model')
        self.beam_length_penalize = args.beam_length_penalize

        # Build Embedding
        if args.share_embedding:
            self.src_embed = rnn_helper.build_embedding(args.tgt_vocab_size_with_offsets, args.embed_size)
            self.dec_embed = self.src_embed
            assert args.share_vocab
            enc_embed = self.src_embed
            dec_embed = self.src_embed
        else:
            self.src_embed = rnn_helper.build_embedding(args.src_vocab_size, args.embed_size)
            self.dec_embed = rnn_helper.build_embedding(args.tgt_vocab_size_with_offsets, args.embed_size)
            dec_embed = self.dec_embed

        # Build Encoder
        encoder = RNNEncoder(hparams=args, embed=enc_embed)
        self.encoder = encoder

        # Build Bridge
        if args.bridge == 'general':
            bridge = LinearBridge(hparams=args)
        elif args.bridge == 'none':
            bridge = None
        else:
            raise NotImplementedError()
        self.bridge = bridge

        # Build Decoder
        decoder = RNNDecoder(hparams=args, embed=dec_embed)
        self.decoder = decoder

        # Build Copy Attention
        self.copy_mode = False
        self.copy_coverage = False
        if args.copy:
            self.copy_mode = True
            # Coverage
            if args.copy_coverage > 0.0:
                self.copy_coverage = True

        # Special Tokens
        self.tgt_sos_idx = tgt_field.vocab.stoi['<sos>']
        self.tgt_eos_idx = tgt_field.vocab.stoi['<eos>']
        self.tgt_unk_idx = tgt_field.vocab.stoi['<unk>']
        self.tgt_pad_idx = tgt_field.vocab.stoi['<pad>']

    @classmethod
    def build_s2s_model(cls, args, src_field, tgt_field):
        seq2seq = cls(args, src_field, tgt_field)
        return seq2seq


    def beam_search_decoding(self, hparams, src, src_len, beam_width):
        max_len = hparams.infer_max_len
        min_len = hparams.infer_min_len
        batch_size = src.size(1)
        vocab_size = self.decoder.vocab_size_with_offsets

        with torch.no_grad():
            # Encoder Part
            encoder_output, hidden = self.encoder(src, src_len)

            # Bridge Part
            if self.bridge is not None:
                hidden = self.bridge(hidden)

            if self.encoder.rnn_type == 'lstm':
                hidden = [x[:self.decoder.n_layers] for x in hidden]
                device = hidden[0].device
            else:
                hidden = hidden[:self.decoder.n_layers]
                device = hidden.device

            # Decoding
            output_probs = torch.zeros(max_len, batch_size * beam_width, vocab_size, device=device)
            last_token = torch.ones([batch_size * beam_width], dtype=torch.long, device=device) * self.tgt_sos_idx

            token_outputs = [last_token]
            ending_flags = torch.zeros([batch_size * beam_width], dtype=torch.bool, device=device)
            decoded_seq_len = torch.zeros([batch_size * beam_width], device=device)
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
            encoder_output = model_helper.create_beam(encoder_output, beam_width, 1)
            encoder_memory_bank = encoder_output.transpose(0, 1)
            hidden = model_helper.create_beam(hidden, beam_width, 1)
            src_len = model_helper.create_beam(src_len, beam_width, 0)
            src = model_helper.create_beam(src, beam_width, 1)

            # Copy
            copy_attention_coverage_vector = None
            encoder_copy_bank = None
            if self.copy_mode:
                encoder_copy_bank = self.decoder.embed(src).transpose(1, 0).contiguous()
                if self.copy_coverage:
                    max_src_len = src.shape[0]
                    copy_attention_coverage_vector = torch.zeros([batch_size * beam_width, max_src_len],
                                                                 dtype=torch.float32, device=src.device)

            for t in range(1, max_len):
                prob_output, hidden, attn_weights, copy_attn_weights, _selector = self.decoder(
                    last_token, hidden, encoder_memory_bank, src_len,
                    encoder_copy_bank=encoder_copy_bank,
                    copy_attention_coverage_vector=copy_attention_coverage_vector,
                )

                # Ensure the min len
                if min_len != -1 and t <= min_len:
                    prob_output[:, self.tgt_eos_idx] = -1e20

                # Mask the unk
                if hparams.disable_unk_output:
                    prob_output[:, self.tgt_unk_idx] = -1e20

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

                # compute total beam scores =>(batch_size, beam)
                total_beam_scores = beam_scores.sum(dim=0)
                topk_total_scores_in_each_beam = topk_scores_in_each_beam + total_beam_scores.unsqueeze(-1)
                topk_total_scores_in_each_beam = topk_total_scores_in_each_beam + finished_offsets

                # decoded_seq_len (batch, beam_width, 1)
                tmp_decoded_seq_len = torch.where(ending_flags, decoded_seq_len, decoded_seq_len + 1)
                if self.beam_length_penalize == 'avg':
                    length_factor = tmp_decoded_seq_len.view(batch_size, beam_width, 1)
                    # 限制最大惩罚长度
                    length_factor = torch.min(length_factor, torch.ones_like(length_factor) * 10)
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

                # selected sequence_length
                decoded_seq_len = tmp_decoded_seq_len.index_select(dim=0, index=flatten_indices)

                # select copy attention coverage
                if copy_attention_coverage_vector is not None:
                    # From original
                    resorted_copy_attention_coverage_vector = \
                        copy_attention_coverage_vector.index_select(dim=0, index=flatten_indices)
                    copy_attention_coverage_vector = resorted_copy_attention_coverage_vector
                    copy_attn_weights = copy_attn_weights.index_select(dim=1, index=flatten_indices)
                    copy_attention_coverage_vector = copy_attention_coverage_vector + copy_attn_weights.squeeze(0)

                # update beams
                last_token = torch.where(ending_flags, padding_label, topk_tokens_in_batch.view(-1))
                token_outputs.append(last_token)
                beam_scores[t] = current_scores_in_batch * (last_token.view(batch_size, beam_width) != self.tgt_pad_idx)

                # Is finished
                is_finished = last_token == self.tgt_eos_idx
                ending_flags |= is_finished
                if torch.sum(ending_flags) == batch_size * beam_width:
                    break

            return output_probs, token_outputs, beam_scores.view(-1, batch_size * beam_width)

    def greedy_search_decoding(self, hparams, src, src_len):

        # Set variables
        max_len = hparams.infer_max_len
        min_len = hparams.infer_min_len
        batch_size = src.size(1)
        vocab_size = self.decoder.vocab_size_with_offsets
        device = src.device

        with torch.no_grad():
            # Encoder Part
            encoder_output, hidden = self.encoder(src, src_len)

            # Bridge Part
            if self.bridge is not None:
                hidden = self.bridge(hidden)

            # Select Init Decoder States
            if self.encoder.rnn_type == 'lstm':
                hidden = [x[:self.decoder.n_layers] for x in hidden]
            else:
                hidden = hidden[:self.decoder.n_layers]

            # Preparing states
            output_probs = torch.zeros(max_len, batch_size, vocab_size, device=device)
            scores = torch.zeros(max_len, batch_size, device=device)
            last_token = torch.ones([batch_size], dtype=torch.long, device=device) * self.tgt_sos_idx
            token_outputs = [last_token]
            ending_flags = torch.zeros([batch_size], dtype=torch.bool, device=device)
            encoder_memory_bank = encoder_output.transpose(0, 1)
            padding_label = torch.ones([batch_size], dtype=torch.long, device=device) * self.tgt_pad_idx
            padding_score = torch.zeros([batch_size], device=device)
            max_src_len = src.shape[0]

            # Copy Attention coverage
            copy_attention_coverage_vector = torch.zeros([batch_size, max_src_len],
                                                         dtype=torch.float32, device=src.device)

            # Copy
            if self.copy_mode:
                copy_mask = model_helper.sequence_mask_fn(src_len, dtype=torch.float32)
                encoder_copy_bank =  self.decoder.embed(src).transpose(1, 0).contiguous()
            else:
                copy_mask = None
                encoder_copy_bank = None

            for t in range(1, max_len):
                prob_output, hidden, attn_weights, copy_attn_weights, _ = self.decoder(
                    last_token, hidden, encoder_memory_bank, src_len,
                    encoder_copy_bank=encoder_copy_bank,
                    copy_attention_coverage_vector=copy_attention_coverage_vector)

                if self.copy_mode:
                    copy_attn_weights = copy_attn_weights.squeeze(0)
                    coverage_diff = torch.min(copy_attn_weights, copy_attention_coverage_vector)
                    coverage_loss = (coverage_diff * copy_mask.unsqueeze(1)).sum(-1).mean()
                    copy_attention_coverage_vector = copy_attention_coverage_vector + copy_attn_weights

                # Ensure the min len
                if min_len != -1 and t <= min_len:
                    prob_output[:, self.tgt_eos_idx] = -1e20

                # Mask unk
                if hparams.disable_unk_output:
                    prob_output[:, self.tgt_unk_idx] = -1e20

                output_probs[t] = prob_output

                # Mask Ended
                score, last_token = prob_output.data.max(1)
                last_token = torch.where(ending_flags, padding_label, last_token)
                token_outputs.append(last_token)
                score = torch.where(ending_flags, padding_score, score)
                scores[t] = score

                # Is finished
                is_finished = last_token == self.tgt_eos_idx
                ending_flags |= is_finished
                if torch.sum(ending_flags) == batch_size:
                    break

            return output_probs, token_outputs, scores

    def forward(self, src, src_len, tgt, mode='train'):
        batch_size = src.size(1)
        max_len = tgt.size(0)
        vocab_size_with_offsets = self.decoder.vocab_size_with_offsets

        outputs = torch.zeros(max_len, batch_size, vocab_size_with_offsets, device=src.device)
        selector_outputs = torch.zeros(max_len, batch_size, 1, device=src.device)

        # Encoder Part
        encoder_output, hidden = self.encoder(src, src_len)

        # Bridge Part
        if self.bridge is not None:
            hidden = self.bridge(hidden)

        if self.encoder.rnn_type == 'lstm':
            hidden = [x[:self.decoder.n_layers] for x in hidden]
        else:
            hidden = hidden[:self.decoder.n_layers]

        encoder_memory_bank = encoder_output.transpose(0, 1)

        # Copy
        if self.copy_mode:
            copy_mask = model_helper.sequence_mask_fn(src_len, dtype=torch.float32)
            encoder_copy_bank = self.decoder.embed(src).transpose(1, 0).contiguous()
            coverage_losses = torch.zeros(max_len, device=src.device)
            max_src_len = src.shape[0]
            copy_attention_coverage_vector = torch.zeros([batch_size, max_src_len],
                                                         dtype=torch.float32, device=src.device)
        else:
            copy_mask = None
            encoder_copy_bank = None
            coverage_losses = None
            copy_attention_coverage_vector = None

        # Decoder Part
        if mode == 'train':
            # Training
            output = tgt.data[0, :]  # sos
            token_outputs = []
            for t in range(1, max_len):
                output, hidden, attn_weights, copy_attn_weights, selector = self.decoder(
                    output, hidden, encoder_memory_bank, src_len,
                    encoder_copy_bank=encoder_copy_bank,
                    copy_attention_coverage_vector=copy_attention_coverage_vector)
                outputs[t] = output
                if selector is not None:
                    selector_outputs[t] = selector
                output = tgt.data[t]
                if self.copy_mode:
                    copy_attn_weights = copy_attn_weights.squeeze(0)
                    coverage_diff = torch.min(copy_attn_weights, copy_attention_coverage_vector)
                    coverage_loss = (coverage_diff * copy_mask.unsqueeze(1)).sum(-1).mean()
                    coverage_losses[t] = coverage_loss
                    copy_attention_coverage_vector = copy_attention_coverage_vector + copy_attn_weights
                else:
                    copy_attention_coverage_vector = None
            return outputs, token_outputs, coverage_losses, selector_outputs
        elif mode == 'eval':
            # Evaluating
            output = tgt.data[0, :]  # sos
            token_outputs = [output.cpu().numpy()]
            for t in range(1, max_len):
                output, hidden, attn_weights, copy_attn_weights, selector = self.decoder(
                    output, hidden, encoder_memory_bank, src_len,
                    encoder_copy_bank=encoder_copy_bank,
                    copy_attention_coverage_vector=copy_attention_coverage_vector)
                outputs[t] = output
                if selector is not None:
                    selector_outputs[t] = selector
                output = tgt.data[t]
                token_outputs.append(output.cpu().numpy())
                if self.copy_mode:
                    copy_attn_weights = copy_attn_weights.squeeze(0)
                    coverage_diff = torch.min(copy_attn_weights, copy_attention_coverage_vector)
                    coverage_loss = (coverage_diff * copy_mask.unsqueeze(1)).sum(-1).mean()
                    coverage_losses[t] = coverage_loss
                    copy_attention_coverage_vector = copy_attention_coverage_vector + copy_attn_weights
            return outputs, token_outputs
        else:
            raise NotImplementedError()
