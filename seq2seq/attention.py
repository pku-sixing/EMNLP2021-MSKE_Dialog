"""Global attention modules (Luong / Bahdanau)"""
"""
 Copy from OpenNMT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import aeq, sequence_mask,sparsemax

# This class is mainly used by decoder.py for RNNs but also
# by the CNN / transformer decoder when copy attention is used
# CNN has its own attention mechanism ConvMultiStepAttention
# Transformer has its own MultiHeadedAttention


class GlobalAttention(nn.Module):
    r"""
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.


    .. mermaid::

       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G

    All models compute the output as
    :math:`c = \sum_{j=1}^{\text{SeqLength}} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`\text{score}(H_j,q) = H_j^T q`
       * general: :math:`\text{score}(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`\text{score}(H_j, q) = v_a^T \text{tanh}(W_a q + U_a h_j)`


    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]
       attn_func (str): attention function to use, options [softmax,sparsemax]

    """

    def __init__(self, dim, coverage=False, attn_type="dot",
                 attn_func="softmax", dual_attn=False):
        super(GlobalAttention, self).__init__()

        self.dim = dim
        assert attn_type in ["dot", "general", "mlp"], (
            "Please select a valid attention type (got {:s}).".format(
                attn_type))
        self.attn_type = attn_type
        assert attn_func in ["softmax", "sparsemax"], (
            "Please select a valid attention function.")
        self.attn_func = attn_func

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)
        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

        self.dual_attn = False
        if dual_attn:
            self.dual_attn = True
            if self.attn_type == "general":
                self.linear_in_beta = nn.Linear(dim, dim, bias=False)
            elif self.attn_type == "mlp":
                self.linear_context_beta = nn.Linear(dim, dim, bias=False)
                self.linear_query_beta = nn.Linear(dim, dim, bias=True)
                self.v_beta = nn.Linear(dim, 1, bias=False)

        self.use_coverage = False
        if coverage:
            assert dual_attn is False, "doesn't support this now"
            self.use_coverage = True
            self.linear_cover = nn.Linear(1, dim, bias=False)
            if self.dual_attn:
                self.linear_cover_beta = nn.Linear(1, dim, bias=False)

    def score(self, h_t, h_s, dual_mode=False):
        """
        Args:
          h_t (FloatTensor): sequence of queries ``(batch, tgt_len, dim)``
          h_s (FloatTensor): sequence of sources ``(batch, src_len, dim``
        Returns:
          FloatTensor: raw attention scores (unnormalized) for each src index
            ``(batch, tgt_len, src_len)``
        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)


        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                attn_linear_in = self.linear_in if not dual_mode else self.linear_in_beta
                h_t_ = h_t.reshape(tgt_batch * tgt_len, tgt_dim)
                h_t_ = attn_linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            attn_linear_query = self.linear_query if not dual_mode else self.linear_query_beta
            attn_linear_context = self.linear_context if not dual_mode else self.linear_context_beta
            dim = self.dim
            wq = attn_linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = attn_linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = torch.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)

    def forward(self, source, memory_bank, memory_lengths=None,
                coverage=None, no_concat=False, mask_first_token=False):
        """

        Args:
          source (FloatTensor): query vectors ``(batch, tgt_len, dim)``
          memory_bank (FloatTensor): source vectors ``(batch, src_len, dim)``
          memory_lengths (LongTensor): the source context lengths ``(batch,)``
          coverage (FloatTensor): None (Suppport Now)

        Returns:
          (FloatTensor, FloatTensor):

          * Computed vector ``(tgt_len, batch, dim)``
          * Attention distribtutions for each query
            ``(tgt_len, batch, src_len)``
        """
        if self.use_coverage:
            assert coverage is not None
        assert no_concat is True

        if self.dual_attn:
            assert isinstance(memory_bank, tuple)
            # unpack
            if len(memory_bank) == 2:
                #  'fw_fk'
                source = source
                memory_bank, memory_bank_beta = memory_bank
                memory_bank_value = memory_bank

                x, y, z = memory_bank.size()
                xb, yb, zb = memory_bank_beta.size()
                aeq(x, xb)
                aeq(y, yb)
                aeq(z, zb)
            elif len(memory_bank) == 3:
                # 'fwk_fwv_fk'
                source = source
                memory_bank, memory_bank_value, memory_bank_beta = memory_bank
            else:
                raise NotImplemented()
        else:
            memory_bank_value = memory_bank

        # one step input
        if source.dim() == 2:
            one_step = True
            source = source.unsqueeze(1)
        else:
            one_step = False

        if self.use_coverage:
            assert coverage is not None

        batch, source_l, dim = memory_bank.size()
        batch_, target_l, dim_ = source.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)
        if coverage is not None and self.use_coverage:
            batch_, source_l_ = coverage.size()
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        if coverage is not None and self.use_coverage:
            cover = coverage.view(-1).unsqueeze(1)
            memory_bank = memory_bank + self.linear_cover(cover).view_as(memory_bank)
            memory_bank = torch.tanh(memory_bank)
            if self.dual_attn:
                memory_bank_beta = memory_bank_beta + self.linear_cover_beta(cover).view_as(memory_bank_beta)
                memory_bank_beta = torch.tanh(memory_bank_beta)

        # compute attention scores, as in Luong et al.
        align = self.score(source, memory_bank)
        if self.dual_attn:
            align_beta = self.score(source, memory_bank_beta)

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.masked_fill_(~mask, -float('inf'))
            if self.dual_attn:
                align_beta.masked_fill_(~mask, -float('inf'))

        if mask_first_token:
            align[:,:,0] = -float('inf')
            if self.dual_attn:
                align_beta[:, :, 0] = -float('inf')

        # Softmax or sparsemax to normalize attention weights
        if self.attn_func == "softmax":
            align_vectors = F.softmax(align.view(batch*target_l, source_l), -1)
            if self.dual_attn:
                align_beta_vectors = F.softmax(align_beta.view(batch * target_l, source_l), -1)

        else:
            align_vectors = sparsemax(align.view(batch*target_l, source_l), -1)
            if self.dual_attn:
                align_beta_vectors = sparsemax(align_beta.view(batch*target_l, source_l), -1)

        if not self.dual_attn:
            align_vectors = align_vectors.view(batch, target_l, source_l)
        else:
            align_alpha = align_vectors.view(batch, target_l, source_l)
            align_beta = align_beta_vectors.view(batch, target_l, source_l)

            gamma = (align_alpha + 1e-20) * (align_beta + 1e-20)
            align_vectors = gamma / gamma.sum(dim=-1, keepdim=True)


        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, memory_bank_value)

        # concatenate
        if no_concat:
            attn_h = c
        else:
            concat_c = torch.cat([c, source], 2).view(batch*target_l, dim*2)
            attn_h = self.linear_out(concat_c).view(batch, target_l, dim)
        if self.attn_type in ["general", "dot"]:
            attn_h = torch.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)

            # Check output sizes
            batch_, dim_ = attn_h.size()
            aeq(batch, batch_)
            aeq(dim, dim_)
            batch_, source_l_ = align_vectors.size()
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()
            # Check output sizes
            target_l_, batch_, dim_ = attn_h.size()
            aeq(target_l, target_l_)
            aeq(batch, batch_)
            aeq(dim, dim_)
            target_l_, batch_, source_l_ = align_vectors.size()
            aeq(target_l, target_l_)
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        return attn_h, align_vectors
