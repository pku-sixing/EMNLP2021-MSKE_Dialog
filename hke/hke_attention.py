import torch.nn as nn
import torch

from utils.misc import sequence_mask


class DotAttention(nn.Module):

    def __init__(self, query_dim, value_dim, mid_dim=64, attention_type='dot', fast_mode=False):
        """
        FastMode：
        """
        super().__init__()
        self.query_dim = query_dim
        self.value_dim = value_dim
        self.mid_dim = mid_dim
        assert fast_mode is False

        # 1 Project query to mid_dim
        self.query_projection = nn.Sequential(
            nn.Linear(query_dim, mid_dim),
            nn.Tanh()
        )
        # 2 Project key to mid_dim
        self.key_projection = nn.Sequential(
            nn.Linear(value_dim, mid_dim),
            nn.Tanh()
        )
        if attention_type == 'dot_val':
            self.value_projection = nn.Sequential(
                nn.Linear(value_dim, mid_dim),
                nn.Tanh()
            )
        else:
            self.value_projection = None

        self.fast_mode = fast_mode



    def forward(self, query, memory_bank, memory_length, max_length=None, cached_memory_bank=None):
        """

        :param query: [1,batch, embed]
        :param memory_bank: [batch, num, embed]
        :param memory_length: [batch]
        :return:
        """
        if not self.fast_mode or cached_memory_bank is None:
            assert cached_memory_bank is None
            attention_key = self.key_projection(memory_bank)
            if self.value_projection is not None:
                memory_bank = self.value_projection(memory_bank)

        else:
            raise NotImplementedError()
            assert isinstance(cached_memory_bank, tuple)
            attention_key, memory_bank = cached_memory_bank

        query_mid = self.query_projection(query.transpose(0, 1))

        # => [batch, num, 1]
        logits = torch.matmul(attention_key, query_mid.transpose(1, 2)).squeeze(-1)
        mask = sequence_mask(memory_length, max_len=max_length)
        logits = logits + -1e10 * (1.0 - mask.to(torch.float32))
        probs = torch.softmax(logits, -1)
        readout = torch.matmul(memory_bank.transpose(2, 1), probs.unsqueeze(-1)).squeeze()

        if self.fast_mode:
            return readout, probs, (attention_key, memory_bank)
        else:
            return readout, probs, None





