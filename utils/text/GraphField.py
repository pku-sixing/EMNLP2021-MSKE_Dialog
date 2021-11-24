from collections import Counter

import torch
import six
from torchtext.data import RawField, Pipeline
from torchtext.data.utils import dtype_to_attr, is_tokenizer_serializable, get_tokenizer
from torchtext.vocab import Vocab
import numpy as np


class GraphField(RawField):
    """Defines a datatype together with instructions for converting to Tensor.
    提供(batch, seq, seq) 的
    Field class models common text processing datatypes that can be represented
    by tensors.  It holds a Vocab object that defines the set of possible values
    for elements of the field and their corresponding numerical representations.
    The Field object also holds other parameters relating to how a datatype
    should be numericalized, such as a tokenization method and the kind of
    Tensor that should be produced.

    If a Field is shared between two columns in a dataset (e.g., question and
    answer in a QA dataset), then they will have a shared vocabulary.

    Attributes:
        sequential: Whether the datatype represents sequential data. If False,
            no tokenization is applied. Default: True.
        use_vocab: Whether to use a Vocab object. If False, the data in this
            field should already be numerical. Default: True.
        init_token: A token that will be prepended to every example using this
            field, or None for no initial token. Default: None.
        eos_token: A token that will be appended to every example using this
            field, or None for no end-of-sentence token. Default: None.
        fix_length: A fixed length that all examples using this field will be
            padded to, or None for flexible sequence lengths. Default: None.
        dtype: The torch.dtype class that represents a batch of examples
            of this kind of data. Default: torch.long.
        preprocessing: The Pipeline that will be applied to examples
            using this field after tokenizing but before numericalizing. Many
            Datasets replace this attribute with a custom preprocessor.
            Default: None.
        postprocessing: A Pipeline that will be applied to examples using
            this field after numericalizing but before the numbers are turned
            into a Tensor. The pipeline function takes the batch as a list, and
            the field's Vocab.
            Default: None.
        lower: Whether to lowercase the text in this field. Default: False.
        tokenize: The function used to tokenize strings using this field into
            sequential examples. If "spacy", the SpaCy tokenizer is
            used. If a non-serializable function is passed as an argument,
            the field will not be able to be serialized. Default: string.split.
        tokenizer_language: The language of the tokenizer to be constructed.
            Various languages currently supported only in SpaCy.
        include_lengths: Whether to return a tuple of a padded minibatch and
            a list containing the lengths of each examples, or just a padded
            minibatch. Default: False.
        batch_first: Whether to produce tensors with the batch dimension first.
            Default: False.
        pad_token: The string token used as padding. Default: "<pad>".
        unk_token: The string token used to represent OOV words. Default: "<unk>".
        pad_first: Do the padding of the sequence at the beginning. Default: False.
        truncate_first: Do the truncating of the sequence at the beginning. Default: False
        stop_words: Tokens to discard during the preprocessing step. Default: None
        is_target: Whether this field is a target variable.
            Affects iteration over batches. Default: False
    """

    vocab_cls = Vocab
    # Dictionary mapping PyTorch tensor dtypes to the appropriate Python
    # numeric type.
    dtypes = {
        torch.float32: float,
        torch.float: float,
        torch.float64: float,
        torch.double: float,
        torch.float16: float,
        torch.half: float,

        torch.uint8: int,
        torch.int8: int,
        torch.int16: int,
        torch.short: int,
        torch.int32: int,
        torch.int: int,
        torch.int64: int,
        torch.long: int,
    }

    ignore = ['dtype', 'tokenize']

    def __init__(self, sequential=True, use_vocab=True, init_token=None,
                 eos_token=None, fix_length=None, dtype=torch.long,
                 preprocessing=None, postprocessing=None, lower=False,
                 tokenize=None, tokenizer_language='en', include_lengths=False,
                 batch_first=True, pad_token="<pad>", unk_token="<unk>",
                 pad_first=False, truncate_first=False, stop_words=None,
                 is_target=False, dynamic_vocab=True):

        self.dynamic_vocab = dynamic_vocab
        self.sequential = sequential
        self.use_vocab = use_vocab
        self.init_token = init_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.sub_init_token = '<w>'
        self.sub_eos_token = '</w>'
        # self.sub_init_token = '<w>'
        # self.sub_eos_token = '</w>'
        self.sub_unk_token = unk_token
        self.fix_length = fix_length
        self.dtype = dtype
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.lower = lower
        # store params to construct tokenizer for serialization
        # in case the tokenizer isn't picklable (e.g. spacy)
        self.tokenizer_args = (tokenize, tokenizer_language)
        self.tokenize = tokenize
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_token = pad_token if self.sequential else None
        self.sub_pad_token = '<p>'
        self.pad_first = pad_first
        self.truncate_first = truncate_first
        try:
            self.stop_words = set(stop_words) if stop_words is not None else None
        except TypeError:
            raise ValueError("Stop words must be convertible to a set")
        self.is_target = is_target

    def __getstate__(self):
        str_type = dtype_to_attr(self.dtype)
        if is_tokenizer_serializable(*self.tokenizer_args):
            tokenize = self.tokenize
        else:
            # signal to restore in `__setstate__`
            tokenize = None
        attrs = {k: v for k, v in self.__dict__.items() if k not in self.ignore}
        attrs['dtype'] = str_type
        attrs['tokenize'] = tokenize

        return attrs

    def __setstate__(self, state):
        state['dtype'] = getattr(torch, state['dtype'])
        if not state['tokenize']:
            state['tokenize'] = get_tokenizer(*state['tokenizer_args'])
        self.__dict__.update(state)

    def __hash__(self):
        # we don't expect this to be called often
        return 42

    def __eq__(self, other):
        if not isinstance(other, RawField):
            return False

        return self.__dict__ == other.__dict__

    def preprocess(self, x):
        """Load a single example using this field, tokenizing if necessary.

        If the input is a Python 2 `str`, it will be converted to Unicode
        first. If `sequential=True`, it will be tokenized. Then the input
        will be optionally lowercased and passed to the user-provided
        `preprocessing` Pipeline."""
        if (six.PY2 and isinstance(x, six.string_types)
                and not isinstance(x, six.text_type)):
            x = Pipeline(lambda s: six.text_type(s, encoding='utf-8'))(x)
        if self.sequential and isinstance(x, six.text_type):
            x = self.tokenize(x.rstrip('\n'))
        if self.lower:
            x = Pipeline(six.text_type.lower)(x)
        if self.sequential and self.use_vocab and self.stop_words is not None:
            x = [w for w in x if w not in self.stop_words]
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, device=None):
        """ Process a list of examples to create a torch.Tensor.
        生成一个
        """
        # 采用这种简单方式构建 i*MAX_NODE + j
        # node_type_dict = {
        #     0: 0,# Unknow
        #     1: 1, # G
        #     2: 2, # C
        #     3: 3, # F
        #     11: 4,# G2G
        #     12: 5,# G2C
        #     13: 6,# G2F
        #     21: 7,# C2G
        #     22: 8,# C2C
        #     23: 9,# C2F
        #     31: 10,# F2G
        #     32: 11,# F2C
        #     33: 12,# F2F
        # }
        #padded = self.pad(batch)
        if self.init_token is not None:
            batch = [x + 1 for x in batch]
        if self.eos_token is not None:
            batch = [x + 1 for x in batch]
        max_field_num = max(batch)
        global_node_num = 1
        context_node_num = 1
        field_type_num = global_node_num + context_node_num + 1
        total_node_num = max_field_num + global_node_num + context_node_num
        batch_size = len(batch)
        relation_matrix = list()
        for bid in range(batch_size):
            node_types = [x + 1 for x in range(global_node_num + context_node_num)] + [
                field_type_num] * batch[bid] + [0] * (max_field_num - batch[bid])
            node_types = torch.LongTensor(node_types)
            my_relation_matrix = node_types.view(-1, 1) * field_type_num + node_types.view(1,-1)
            # 对角线除以自己
            dia_coff = torch.eye(total_node_num, dtype=torch.long) * node_types.view(1,-1) * field_type_num
            my_relation_matrix = my_relation_matrix - dia_coff
            my_relation_matrix_mask = (node_types.view(-1, 1) != 0) * (node_types.view(1,-1) != 0)
            my_relation_matrix = my_relation_matrix * my_relation_matrix_mask
            relation_matrix.append(my_relation_matrix.unsqueeze(0))
        relation_matrix = torch.cat(relation_matrix, 0)
        return relation_matrix.to(device), torch.LongTensor(batch).to(device)

    def pad(self, minibatch):
        """Pad a batch of examples using this field.

        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Prepends self.init_token and appends
        self.eos_token if those attributes are not None. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True` and `self.sequential` is `True`, else just
        returns the padded list. If `self.sequential` is `False`, no padding is applied.
        """
        minibatch = list(minibatch)
        if not self.sequential:
            raise NotImplementedError()
        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
            max_sub_len = 1
            for x in minibatch:
                for y in x:
                    max_sub_len = max(max_sub_len, len(y))
        else:
            raise NotImplementedError()
            # max_len = self.fix_length + (
            #     self.init_token, self.eos_token).count(None) - 2

        # 首先Sequence 级别Padding
        padded, lengths = [], []
        for x in minibatch:
            if self.pad_first:
                padded.append(
                    [self.pad_token] * max(0, max_len - len(x))
                    + ([] if self.init_token is None else [self.init_token])
                    + list(x[-max_len:] if self.truncate_first else x[:max_len])
                    + ([] if self.eos_token is None else [self.eos_token]))
            else:
                padded.append(
                    ([] if self.init_token is None else [self.init_token])
                    + list(x[-max_len:] if self.truncate_first else x[:max_len])
                    + ([] if self.eos_token is None else [self.eos_token])
                    + [self.pad_token] * max(0, max_len - len(x)))
            lengths.append(min(3, len(padded[-1]) - max(0, max_len - len(x))))

        # 然后进行SubLevel的Padding

        spec_tokens = set()
        if self.init_token is not None: spec_tokens.add(self.init_token)
        if self.eos_token is not None: spec_tokens.add(self.eos_token)
        if self.pad_token is not None: spec_tokens.add(self.pad_token)

        if self.dynamic_vocab :
            if self.include_lengths:
                return (padded, lengths, None)
            return padded
        else:
            padded_subs = []
            sub_lengths = []
            for seq in padded:
                padded_x_seq = []
                padded_x_len = []
                for x in list(seq):
                    # TODO 可能存在一些空格的空值，需要特别处理
                    if len(x) == 0:
                        x = ['<unk>']
                    if x in spec_tokens or ( x[0] == '<' and x[-1] == '>') :
                        x = [x]
                    else:
                        x = list(x)
                    padded_x_seq.append(
                        ([] if self.sub_init_token is None else [self.sub_init_token])
                        + list(x[-max_sub_len:] if self.truncate_first else x[:max_sub_len])
                        + ([] if self.sub_eos_token is None else [self.sub_eos_token])
                        + [self.sub_pad_token] * max(0, max_sub_len - len(x)))
                    x_len = len(padded_x_seq[-1]) - max(0, max_sub_len - len(x))
                    assert x_len > 0
                    padded_x_len.append(x_len)
                padded_subs.append(padded_x_seq)
                sub_lengths.append(padded_x_len)


            if self.include_lengths:
                return (padded_subs, lengths, sub_lengths)
            return padded_subs


    def numericalize(self, arr, vocab, device=None):
        """Turn a batch of examples that use this field into a Variable.

        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.

        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (str or torch.device): A string or instance of `torch.device`
                specifying which device the Variables are going to be created on.
                If left as default, the tensors will be created on cpu. Default: None.
        """
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths, sub_lengths = arr
            lengths = torch.tensor(lengths, dtype=self.dtype, device=device)


        for bid in range(len(arr)):
            for sid in range(len(arr[bid])):
                    arr[bid][sid]= vocab[arr[bid][sid]]

        if self.postprocessing is not None:
            arr = self.postprocessing(arr, self.vocab)


        var = torch.tensor(arr, dtype=self.dtype, device=device)

        if self.sequential and not self.batch_first:
            var.t_()
        if self.sequential:
            var = var.contiguous()

        if self.include_lengths:
            return var, lengths
        return var

