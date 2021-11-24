import torch
from torch import nn
import torch.nn.functional as F
from seq2seq import rnn_helper
from seq2seq.attention import GlobalAttention

class RNNDecoder(nn.Module):

    def __init__(self, hparams, embed):
        super(RNNDecoder, self).__init__()
        self.hidden_size = hidden_size = hparams.hidden_size
        self.embed_size = embed_size = hparams.embed_size
        self.n_layers = n_layers = hparams.dec_layers
        self.dropout = dropout = hparams.dropout
        self.vocab_size = hparams.tgt_vocab_size
        self.vocab_size_with_offsets = hparams.tgt_vocab_size_with_offsets
        self.rnn_type = hparams.rnn_type
        self.add_last_generated_token = hparams.add_last_generated_token

        self.embed = embed
        self.dropout = nn.Dropout(dropout)
        self.attention = GlobalAttention(hidden_size,
                                         attn_type=hparams.attn_type,
                                         attn_func=hparams.attn_func)

        self.out_size = hidden_size * 2
        if self.add_last_generated_token:
            self.out_size += embed_size

        self.copy_coverage = False
        if hparams.copy:
            self.max_copy_token_num = hparams.max_copy_token_num
            if hparams.copy_coverage > 0:
                self.copy_coverage = True
            self.copy_attention = GlobalAttention(hidden_size,
                                                  coverage=self.copy_coverage,
                                                  attn_type=hparams.copy_attn_type,
                                                  attn_func=hparams.copy_attn_func)
            # Selector
            self.copy_selector = nn.Linear(self.out_size, 1, bias=False)
        else:
            self.copy_attention = None

        rnn_fn = rnn_helper.rnn_factory(hparams.rnn_type)
        self.rnn = rnn_fn(hidden_size + embed_size, hidden_size,
                          n_layers, dropout=dropout)
        self.out = nn.Linear(self.out_size, hparams.tgt_vocab_size)


    def forward(self, input, last_hidden, encoder_memory_bank, encoder_length,
                encoder_copy_bank=None, copy_attention_coverage_vector=None):
        """
        :param copy_attention_coverage_vector:
        :param encoder_copy_bank:
        :param input:[seq_len]
        :param last_hidden:
        :param encoder_memory_bank: (batch, seq_len, dim)
        :param encoder_length:
        :return:
        """

        if self.copy_attention is not None:
            # Select copied encoder states
            original_word_embedding = self.embed(input)
            batch_size, max_src_len, encoder_hidden_dim = encoder_copy_bank.shape

            # Index
            copied_select_index = input - self.vocab_size
            is_copied_tokens = (copied_select_index >= 0)
            zero_index = torch.zeros_like(copied_select_index, device=copied_select_index.device)
            valid_copied_select_index = torch.where(is_copied_tokens, copied_select_index, zero_index)

            # Offsets for flatten selection
            batch_offsets = torch.arange(0, batch_size * max_src_len, max_src_len, dtype=torch.long,
                                         device=copied_select_index.device)
            valid_copied_select_index = valid_copied_select_index + batch_offsets

            # Select
            flatten_encoder_outputs = encoder_copy_bank.view(-1, encoder_hidden_dim)
            selected_encoder_outputs = flatten_encoder_outputs.index_select(index=valid_copied_select_index, dim=0)
            selected_encoder_outputs = selected_encoder_outputs.view(batch_size, encoder_hidden_dim)

            selected_encoder_outputs = selected_encoder_outputs
            embedded = torch.where(is_copied_tokens.unsqueeze(-1), selected_encoder_outputs, original_word_embedding)
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
        context, attn_weights = self.attention(
            attn_query,
            encoder_memory_bank,
            memory_lengths=encoder_length,
            no_concat=True,
        )

        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        if isinstance(last_hidden, list) or isinstance(last_hidden, tuple):
            last_hidden = tuple([x.contiguous() for x in last_hidden])
        else:
            last_hidden = last_hidden.contiguous()

        # Run RNN
        output, hidden = self.rnn(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)

        if self.add_last_generated_token:
            context_output = torch.cat([output, context, last_generated_token_embedded], 1)
        else:
            context_output = torch.cat([output, context], 1)


        # Copy Mechanism
        if self.copy_attention is None:
            output = self.out(context_output)
            output = F.log_softmax(output, dim=1)
            copy_attn_weights = None
            selector = None
        else:
            selector = torch.sigmoid(self.copy_selector(context_output))
            output = self.out(context_output)
            gen_probs = F.softmax(output, dim=1)

            # Copy Attention
            if self.rnn_type == 'lstm':
                copy_attn_query = last_hidden[-1][0]
            else:
                copy_attn_query = last_hidden[-1]

            copy_attn_query = copy_attn_query.unsqueeze(1)
            copy_context, copy_attn_weights = self.copy_attention(
                copy_attn_query,
                encoder_memory_bank,
                coverage=copy_attention_coverage_vector if self.copy_coverage else None,
                memory_lengths=encoder_length - 1, #mask the last token <sos>
                no_concat=True,
                mask_first_token=True,
            )

            copy_probs = copy_attn_weights.squeeze(0)

            # padding copy probs
            if self.max_copy_token_num - copy_probs.shape[1] > 0:
                padding_probs = torch.zeros(
                    [copy_probs.shape[0], self.max_copy_token_num - copy_probs.shape[1]],
                    device=copy_probs.device)
                output = torch.cat([gen_probs * (1.0 - selector), copy_probs * selector, padding_probs * selector], -1)
            else:
                output = torch.cat([gen_probs * (1.0 - selector), copy_probs * selector], -1)

            output = torch.log(output + 1e-20)

        return output, hidden, attn_weights, copy_attn_weights, selector
