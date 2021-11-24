import torch
import torchtext
from collections import Counter, OrderedDict
from torchtext.data import Field, BucketIterator, Dataset, Iterator
from torchtext.vocab import Vocab
from utils.logger import logger
import numpy as np
import re

SRC_SUFFIX = 'src'
TGT_SUFFIX = 'tgt'
STOP_COPY_BEFORE_POSITION = 300

def translate(ids, field):
    trans_res = []
    max_time = len(ids)
    batch_num = len(ids[0])
    for batch in range(batch_num):
        res = ' '.join([field.vocab.itos[ids[t][batch]] for t in range(max_time)])
        res = re.sub('<[(sos|pad|eos)]+>', '', res)
        res = re.sub('<[ ]+>', ' ', res)
        trans_res.append(res.strip())
    return trans_res

def _process_pointer_copy_examples(srcs, tgts, has_sos=True, word_freq_dict=None):
    num_total = len(srcs)
    word_types = []
    for idx in range(num_total):
        src = srcs[idx].split()
        tgt = tgts[idx].split()
        src_map = dict()
        offset = 1 if has_sos else 0
        if word_freq_dict is  None:
            for i, word in enumerate(src):
                src_map[word] = '<src_%d>' % (i + offset)
        else:
            for i, word in enumerate(src):
                if word_freq_dict.get(word, STOP_COPY_BEFORE_POSITION + 1) > STOP_COPY_BEFORE_POSITION:
                    src_map[word] = '<src_%d>' % (i + offset)
        word_type = []
        for i, word in enumerate(tgt):
            tgt[i] = src_map.get(word, word)
        tgts[idx] = ' '.join(tgt)
        word_types.append(word_type)
    return word_types

def restore_pointer_copy_example(src, tgt, has_sos=True):
        src = src.split()
        tgt = tgt.split()
        src_map = dict()
        offset = 1 if has_sos else 0
        for i, word in enumerate(src):
            src_map['<src_%d>' % (i + offset)] = word
        for i, word in enumerate(tgt):
            tgt[i] = src_map.get(word, word)
        return ' '.join(tgt)


def load_examples(example_path):
    with open(example_path + '.' + SRC_SUFFIX, 'r', encoding='utf-8') as fin:
        src = fin.readlines()
    logger.info('[DATASET] Loaded %s.src, lines=%d' % (example_path, len(src)))
    with open(example_path + '.' + TGT_SUFFIX, 'r', encoding='utf-8') as fin:
        tgt = fin.readlines()
    logger.info('[DATASET] Loaded %s.tgt, lines=%d' % (example_path, len(tgt)))

    src = [x.strip('\r\n') for x in src]
    tgt = [x.strip('\r\n') for x in tgt]

    return src, tgt

def get_dataset(example_path, fields, max_line=-1, pointer_copy=False, max_src_len=-11, max_tgt_len=-1,
                word_freq_dict=None):
    with open(example_path + '.' + SRC_SUFFIX, 'r', encoding='utf-8') as fin:
        src = fin.readlines()
    logger.info('[DATASET] Loaded %s.src, lines=%d' % (example_path, len(src)))
    with open(example_path + '.' + TGT_SUFFIX, 'r', encoding='utf-8') as fin:
        tgt = fin.readlines()
    logger.info('[DATASET] Loaded %s.tgt, lines=%d' % (example_path, len(tgt)))

    if max_line != -1:
        src = src[:max_line]
        tgt = tgt[:max_line]

    if max_src_len != -1:
        logger.info('[DATASET] Maximum source sequence length is set to %d' % (max_src_len))
        src = [' '.join(x.strip('\r\n').split()[0:max_src_len]) for x in src]
    if max_tgt_len != -1:
        logger.info('[DATASET] Maximum target sequence length is set to %d' % (max_tgt_len))
        tgt = [' '.join(x.strip('\r\n').split()[0:max_tgt_len]) for x in tgt]

    src_max_len = max([len(x.split()) for x in src])

    if pointer_copy:
        logger.info('[DATASET] Aligning tgts to srcs for Pointer-Copy ')
        assert word_freq_dict is not None
        word_types = _process_pointer_copy_examples(src, tgt, word_freq_dict=word_freq_dict)

    lines = [[x.strip('\r\n').split(), y.strip('\r\n').split()] for x, y in zip(src, tgt)]
    examples = []
    for line in lines:
        example = torchtext.data.Example.fromlist(data=line, fields=fields)
        examples.append(example)
    dataset = Dataset(examples=examples, fields=fields)
    # dataset.sort_key = lambda x: len(x.src)
    return dataset, src_max_len


def load_vocab(vocab_path, field, special_tokens=[], pointer_copy_tokens=0):
    counter = Counter()
    with open(vocab_path, 'r+', encoding='utf-8') as fin:
        vocabs = [x.strip('\r\n') for x in fin.readlines()]
        if pointer_copy_tokens > 0:
            vocabs += ['<src_%d>' % x for x in range(pointer_copy_tokens)]
        vocab_size = len(vocabs)
        for idx, token in enumerate(vocabs):
            counter[token] = vocab_size - idx
    specials = list(OrderedDict.fromkeys(
        tok for tok in [field.unk_token, field.pad_token, field.init_token,
                        field.eos_token] + special_tokens
        if tok is not None))
    field.vocab = Vocab(counter, specials=specials)


def load_dataset(hparams, is_eval=False, test_data_path=None):
    batch_size = hparams.batch_size
    max_copy_token_num = hparams.max_copy_token_num
    pointer_copy_tokens = hparams.max_copy_token_num if hparams.copy else 0

    def tokenize(text):
        return text.strip('\r').split

    src_field = Field(tokenize=tokenize, include_lengths=True,
                      init_token='<ssos>', eos_token='<seos>')
    tgt_field = Field(tokenize=tokenize, include_lengths=True,
                      init_token='<sos>', eos_token='<eos>')
    fields = [('src', src_field), ('tgt', tgt_field)]

    if not hparams.share_vocab:
        logger.info('[VOCAB] Constructing two vocabs for the src and tgt')
        logger.info('[VOCAB] Loading src vocab from: %s' % hparams.src_vocab_path)
        load_vocab(hparams.src_vocab_path, src_field)
        logger.info('[VOCAB] src vocab size: %d' % len(src_field.vocab.itos))
        logger.info('[VOCAB] Loading tgt vocab from: %s' % hparams.tgt_vocab_path)
        load_vocab(hparams.tgt_vocab_path, tgt_field, pointer_copy_tokens=pointer_copy_tokens)
        logger.info('[VOCAB] tgt vocab size: %d' % len(tgt_field.vocab.itos))
    else:
        logger.info('[VOCAB] Constructing a sharing vocab for the src and tgt')
        logger.info('[VOCAB] Loading src&tgt vocab from: %s' % hparams.src_vocab_path)
        load_vocab(hparams.vocab_path, src_field,
                   pointer_copy_tokens=pointer_copy_tokens,
                   special_tokens=[tgt_field.unk_token, tgt_field.pad_token, tgt_field.init_token, tgt_field.eos_token])
        tgt_field.vocab = src_field.vocab
        logger.info('[VOCAB] src vocab size: %d' % len(src_field.vocab.itos))
        logger.info('[VOCAB] tgt vocab size: %d' % len(tgt_field.vocab.itos))

    def sort_key(x):
        return len(x.tgt)+len(x.src)*100

    device = 'cuda' if hparams.cuda else 'cpu'

    val, max_val_len = get_dataset(hparams.val_data_path_prefix, fields=fields,
                                   max_src_len=hparams.max_src_len, max_tgt_len=hparams.max_tgt_len,
                                   pointer_copy=hparams.copy, word_freq_dict=src_field.vocab.stoi)
    test, max_test_len = get_dataset(hparams.test_data_path_prefix if not test_data_path else test_data_path,
                       fields=fields, pointer_copy=hparams.copy,  max_src_len=hparams.max_src_len,
                                     max_tgt_len=hparams.max_tgt_len, word_freq_dict=src_field.vocab.stoi)

    if hparams.copy:
        assert max_val_len + 1 < max_copy_token_num, max_val_len
        assert max_test_len + 1 < max_copy_token_num, max_test_len

    if not is_eval:
        logger.info('[DATASET] Training Mode' )
        train, max_train_len = get_dataset(hparams.train_data_path_prefix, fields=fields,pointer_copy=hparams.copy,
                                           max_src_len=hparams.max_src_len, max_tgt_len=hparams.max_tgt_len,
                                           word_freq_dict=src_field.vocab.stoi)
        if hparams.copy:
            assert max_train_len + 1 < max_copy_token_num, max_train_len
        train_iter = BucketIterator(train, batch_size=batch_size, repeat=False, shuffle=True,
                                    sort_key=sort_key, sort=False, train=True, sort_within_batch=True,
                                    device=device)
        val_iter = BucketIterator(val, batch_size=batch_size, repeat=False, shuffle=True,
                                  sort_key=sort_key, sort=False, train=False, sort_within_batch=True,
                                    device=device)
        test_iter = Iterator(test, batch_size=batch_size, repeat=False, shuffle=False,
                                   sort_key=sort_key, sort=False, train=False, sort_within_batch=False,
                                    device=device)
        return train_iter, val_iter, test_iter, src_field, tgt_field
    else:
        logger.info('[DATASET] Eval/Inference Mode')
        val_iter = Iterator(val, batch_size=batch_size, repeat=False, shuffle=False,
                                   sort_key=sort_key, sort=False, train=False, sort_within_batch=False,
                                    device=device)
        test_iter = Iterator(test, batch_size=batch_size, repeat=False, shuffle=False,
                                   sort_key=sort_key, sort=False, train=False, sort_within_batch=False,
                                    device=device)
        return None, val_iter, test_iter, src_field, tgt_field


def load_pretrain_embeddings(embedding, filed, embedding_path, dim=200, char2word=None):

    total_word = len(filed.vocab)
    loaded_vecs = dict()
    flag = True
    logger.info('[PRE-EMBED] Loading for %s, Char2Word: %s' % (str(filed), str(char2word)))

    token_set = set()
    for word in filed.vocab.stoi:
        token_set.add(word)
        if char2word is not None:
            for char in word:
                token_set.add(char)

    with open(embedding_path, 'r+', encoding='utf-8') as fin:
        line = fin.readline()
        while len(line) > 0:
            items = line.strip('\r\n').split()
            if len(items) != dim+1:
                line = fin.readline()
                continue
            word = items[0]
            weights = items[1:]
            if word in token_set:
                weights = np.array([float(x) for x in weights])
                loaded_vecs[word] = weights
                flag = True
            if len(loaded_vecs) % 1000 == 0 and flag:
                logger.info('[PRE-EMBED] Loading: %d/%d' % (len(loaded_vecs), total_word))
                flag = False
            line = fin.readline()
    logger.info('[PRE-EMBED] Loaded Token/Total: %d/%d' % (len(loaded_vecs), total_word))
    pretrained_weight = np.zeros([total_word, dim])
    weights = embedding.weight.data.cpu().numpy()

    load_count = 0
    generate_count = 0
    for i in range(total_word):
        word = filed.vocab.itos[i]
        if word in loaded_vecs:
            load_count += 1
            pretrained_weight[i] = loaded_vecs[word]
        else:
            if char2word is None:
                pretrained_weight[i] = weights[i]
            elif char2word == 'avg':
                tmp = np.zeros([dim])
                tmp_flag = False
                for char in word:
                    if char in loaded_vecs:
                        tmp += loaded_vecs[char]
                        tmp_flag = True
                    else:
                        tmp += weights[i]

                if tmp_flag:
                    generate_count += 1
                    tmp /= len(word)
                    pretrained_weight[i] = tmp
                else:
                    pretrained_weight[i] = weights[i]
            else:
                raise NotImplementedError()


    logger.info('[PRE-EMBED] Loaded/Generated/Word/Total: %d/%d/%d' % (load_count, generate_count, total_word))
    is_cuda = embedding.weight.device
    embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
    embedding.weight.to(is_cuda)


