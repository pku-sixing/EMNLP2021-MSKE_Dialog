import copy
import os

import torchtext
from collections import Counter, OrderedDict, defaultdict
from torchtext.data import Field, BucketIterator, Dataset, Iterator, RawField

from utils.text.GraphField import GraphField
from utils.text.SubTextField import SubTextField
from torchtext.vocab import Vocab
from utils.logger import logger
from utils.text.tokenization import get_tokenizer
import re
import torch
import random
import numpy as np
import pickle

SRC_SUFFIX = 'src'
BOX_SUFFIX = 'box'
TGT_SUFFIX = 'tgt'
TEXT_SUFFIX = 'text'
CSK_SUFFIX = 'naf_fact'
CSK_GOLDEN_SUFFIX = 'naf_golden_facts'
STOP_COPY_BEFORE_POSITION = 300


def load_pretrain_embeddings(embedding, filed, embedding_path, dim=200, char2word=None, suffix=None):
    total_word = len(filed.vocab)
    flag = True
    loaded_vecs = None
    if suffix is not None:
        logger.info('[PRE-EMBED] Try to load embedding for %s from the cache %s.embed' % (str(filed), suffix))
        try:
            loaded_vecs = pickle.load(open('embed_cache/%s.embed' % suffix, 'rb'))
            logger.info('[PRE-EMBED] Successfully loaded embedding for %s from the cache .%s' % (str(filed), suffix))
        except FileNotFoundError:
            loaded_vecs = None
            logger.info('[PRE-EMBED] Failed to load embedding for %s from the cache .%s' % (str(filed), suffix))

    if loaded_vecs is None:
        loaded_vecs = dict()
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
                if len(items) != dim + 1:
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
        if suffix is not None:
            pickle.dump(loaded_vecs, open('embed_cache/%s.embed' % suffix, 'wb'), )
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


def load_pretrain_trans_embeddings(embedding, embedding_path, dim=100, offset=0):
    total_word = embedding.num_embeddings

    flag = True
    loaded_vecs = []
    with open(embedding_path, 'r+', encoding='utf-8') as fin:
        line = fin.readline()
        while len(line) > 0:
            items = line.strip('\r\n').split()
            assert len(items) == dim
            weights = items
            weights = np.array([float(x) for x in weights])
            loaded_vecs.append(weights)
            line = fin.readline()
    logger.info('[PRE-EMBED] Loaded Token/Total: %d/%d' % (len(loaded_vecs), total_word))
    assert total_word == len(loaded_vecs) + offset
    pretrained_weight = np.zeros([total_word, dim])
    weights = embedding.weight.data.cpu().numpy()

    load_count = 0
    generate_count = 0
    for i in range(total_word):
        if i < offset:
            pretrained_weight[i] = weights[i]
        else:
            pretrained_weight[i] = loaded_vecs[i - offset]

    logger.info('[PRE-EMBED] Loaded/Generated/Word/Total: %d/%d/%d' % (load_count, generate_count, total_word))
    is_cuda = embedding.weight.device
    embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
    embedding.weight.to(is_cuda)


def translate(ids, field):
    if ids is None or field is None:
        return None
    trans_res = []
    max_time = len(ids)
    batch_num = len(ids[0])
    for batch in range(batch_num):
        res = ' '.join([field.vocab.itos[ids[t][batch]] for t in range(max_time)])
        res = re.sub('<[(sos|pad|eos)]+>', '', res)
        res = re.sub('<[ ]+>', ' ', res)
        trans_res.append(res.strip())
    return trans_res


def get_dynamic_vocab(fixed_vocab_set, modes, dynamic_inputs):
    dynamic_vocab = {'<vocab>': 0}
    for mode, inputs in zip(modes, dynamic_inputs):
        if mode == 'box':
            fields = inputs.split(' ')
            for field in fields:
                word = field.split(':')[-1]
                if word not in dynamic_vocab and word not in fixed_vocab_set:
                    dynamic_vocab[word] = len(dynamic_vocab)
        else:
            for word in inputs:
                if word not in dynamic_vocab and word not in fixed_vocab_set:
                    dynamic_vocab[word] = len(dynamic_vocab)
    dynamic_vocab_itos = {}
    for key in dynamic_vocab.keys():
        dynamic_id = dynamic_vocab[key]
        dynamic_id = '<dmc_%d>' % dynamic_id
        dynamic_vocab_itos[dynamic_id] = key
    return dynamic_vocab_itos



def _process_dynamic_vocab_copy_examples(copy_modes, copy_inputs, has_sos=True, word_freq_dict=None, query_first=False,
                           dropout=0.0, mask_copy_flags='', copy_strategy='default', fixed_vocab_path=''):
    """
     对齐的顺序取决于Field顺序，所以Copy时会有所影响
    :param copy_modes:
    :param copy_inputs:
    :param has_sos:
    :param word_freq_dict:
    :param query_first:
    :param dropout:
    :return:
    """
    assert copy_strategy == 'dynamic_vocab'
    with open(fixed_vocab_path, 'r+', encoding='utf-8') as fin:
        lines = fin.readlines()
        words = [line.strip('\r\n') for line in lines]
        logger.info('[DynamicVocab] Loaded %s fixed vocab' % len(words))
        fixed_vocab_set = set(words)
        assert len(words) == len(fixed_vocab_set)

    num_total = len(copy_inputs['tgt'])
    mask_copy_flags = set(mask_copy_flags.split(','))
    word_types = []
    tmp_copy_modes = []
    for copy_mode in copy_modes:
        if copy_mode not in mask_copy_flags:
            tmp_copy_modes.append(copy_mode)
    copy_modes = tmp_copy_modes
    for idx in range(num_total):
        tgt = copy_inputs['tgt'][idx].split()
        local_copy_inputs = []
        for copy_mode in copy_modes:
            if copy_mode == 'src':
                local_copy_inputs.append(copy_inputs['src'][idx].split())
                if 'dynamic_vocab_src' not in copy_inputs:
                    copy_inputs['dynamic_vocab_src'] = []
                copy_inputs['dynamic_vocab_src'].append('')
            elif copy_mode == 'box':
                local_copy_inputs.append(copy_inputs['box'][idx])
                if 'dynamic_vocab_attribute' not in copy_inputs:
                    copy_inputs['dynamic_vocab_attribute'] = []
                copy_inputs['dynamic_vocab_attribute'].append('')
            elif copy_mode == 'text':
                local_copy_inputs.append(copy_inputs['text'][idx].split())
                if 'dynamic_vocab_text' not in copy_inputs:
                    copy_inputs['dynamic_vocab_text'] = []
                copy_inputs['dynamic_vocab_text'].append('')
            elif copy_mode == 'csk':
                local_copy_inputs.append(copy_inputs['csk_response_word'][idx])
                if 'dynamic_vocab_csk' not in copy_inputs:
                    copy_inputs['dynamic_vocab_csk'] = []
                copy_inputs['dynamic_vocab_csk'].append('')

        if 'dynamic_vocab' not in copy_inputs:
            copy_inputs['dynamic_vocab'] = []
            copy_inputs['dynamic_vocab_raw'] = []
        copy_inputs['dynamic_vocab'].append('<vocab>')
        copy_inputs['dynamic_vocab_raw'].append('<vocab>')

        dynamic_vocab = {'<vocab>': 0}

        for copy_mode, copy_input in zip(copy_modes, local_copy_inputs):
            if copy_mode == 'src' or copy_mode == 'text':
                # 首先增加Copy的占位符，随后再增加Table的
                rand_flags = np.random.random([len(copy_input)])
                if word_freq_dict is None:
                    for i, word in enumerate(copy_input):
                        if rand_flags[i] < dropout:
                            continue
                        if word not in dynamic_vocab and word not in fixed_vocab_set:
                            dynamic_vocab[word] = len(dynamic_vocab)
                else:
                    for i, word in enumerate(copy_input):
                        if rand_flags[i] < dropout:
                            continue
                        if word_freq_dict.get(word, STOP_COPY_BEFORE_POSITION + 1) > STOP_COPY_BEFORE_POSITION:
                            if word not in dynamic_vocab and word not in fixed_vocab_set:
                                dynamic_vocab[word] = len(dynamic_vocab)

            if copy_mode == 'csk':
                rand_flags = np.random.random([len(copy_input)])
                for i, word in enumerate(copy_input):
                    if rand_flags[i] < dropout:
                        continue
                    if word not in dynamic_vocab and word not in fixed_vocab_set:
                        dynamic_vocab[word] = len(dynamic_vocab)

            if copy_mode == 'box':
                # 增加Table的
                box = copy_input
                field_values = []
                for field in box.split(' '):
                    items = field.split(':')
                    field_value = items[1]
                    field_values.append(field_value)
                rand_flags = np.random.random([len(field_values)])
                if word_freq_dict is None:
                    for i, word in enumerate(field_values):
                        if rand_flags[i] < dropout:
                            continue
                        if word not in dynamic_vocab and word not in fixed_vocab_set:
                            dynamic_vocab[word] = len(dynamic_vocab)
                else:
                    for i, word in enumerate(field_values):
                        if rand_flags[i] < dropout:
                            continue
                        if word_freq_dict.get(word, STOP_COPY_BEFORE_POSITION + 1) > STOP_COPY_BEFORE_POSITION:
                            if word not in dynamic_vocab and word not in fixed_vocab_set:
                                dynamic_vocab[word] = len(dynamic_vocab)

        word_type = []

        for i, word in enumerate(tgt):
            dynamic_id = dynamic_vocab.get(word, -1)
            if dynamic_id > -1:
                assert word not in fixed_vocab_set and dynamic_id != 0
                tgt[i] = '<dmc_%d>' % dynamic_id
        copy_inputs['tgt'][idx] = ' '.join(tgt)

        if 'src' in copy_modes:
            # dynamic_vocab_src
            src = copy_inputs['src'][idx].split()
            for i, word in enumerate(src):
                if word not in fixed_vocab_set:
                    dynamic_id = dynamic_vocab.get(word, 0)
                    assert  dynamic_id != 0
                    src[i] = '<dmc_%d>' % dynamic_id
            copy_inputs['dynamic_vocab_src'][idx] =' '.join(src)
            del src
        if 'csk' in copy_modes:
            csk = copy.deepcopy(copy_inputs['csk_response_word'][idx])
            for i, word in enumerate(csk):
                if word not in fixed_vocab_set:
                    dynamic_id = dynamic_vocab.get(word, 0)
                    assert dynamic_id != 0
                    csk[i] = '<dmc_%d>' % dynamic_id
            copy_inputs['dynamic_vocab_csk'][idx] = ' '.join(csk)
            del csk
        if 'text' in copy_modes:
            text = copy_inputs['text'][idx].split()
            for i, word in enumerate(text):
                if word not in fixed_vocab_set:
                    dynamic_id = dynamic_vocab.get(word, 0)
                    assert dynamic_id != 0
                    text[i] = '<dmc_%d>' % dynamic_id
            copy_inputs['dynamic_vocab_text'][idx] = ' '.join(text)
            del text
        if 'box' in copy_modes:
            box = copy_inputs['box'][idx]
            attribute = []
            for field in box.split(' '):
                items = field.split(':')
                field_value = items[1]
                if field_value not in fixed_vocab_set:
                    dynamic_id = dynamic_vocab.get(field_value, 0)
                    assert dynamic_id != 0
                    attribute.append('<dmc_%d>' % dynamic_id)
                else:
                    attribute.append(field_value)
            copy_inputs['dynamic_vocab_attribute'][idx] = ' '.join(attribute)
            del attribute

        dynamic_vocab_list = [None] * len(dynamic_vocab)
        for word in dynamic_vocab:
            dynamic_vocab_list[dynamic_vocab[word]] = word

        assert len(dynamic_vocab_list) < 200, 'dynamic vocab size should less than 200'
        copy_inputs['dynamic_vocab'][idx] = ' '.join(dynamic_vocab_list)
        copy_inputs['dynamic_vocab_raw'][idx] = ' '.join(dynamic_vocab_list)
        get_dynamic_vocab(fixed_vocab_set, copy_modes, local_copy_inputs)
        word_types.append(word_type)



    return word_types




def _process_copy_examples(copy_modes, copy_inputs, has_sos=True, word_freq_dict=None, query_first=False,
                           dropout=0.0, mask_copy_flags='', copy_strategy='default', fixed_vocab_path=None):
    """
     对齐的顺序取决于Field顺序，所以Copy时会有所影响
    :param copy_modes:
    :param copy_inputs:
    :param has_sos:
    :param word_freq_dict:
    :param query_first:
    :param dropout:
    :return:
    """
    if copy_strategy == 'dynamic_vocab':
        return _process_dynamic_vocab_copy_examples(copy_modes, copy_inputs, has_sos, word_freq_dict, query_first, dropout,
                                            mask_copy_flags, copy_strategy, fixed_vocab_path)
    num_total = len(copy_inputs['tgt'])
    mask_copy_flags = set(mask_copy_flags.split(','))
    word_types = []
    tmp_copy_modes = []
    for copy_mode in copy_modes:
        if copy_mode not in mask_copy_flags:
            tmp_copy_modes.append(copy_mode)
    copy_modes = tmp_copy_modes
    for idx in range(num_total):
        tgt = copy_inputs['tgt'][idx].split()
        local_copy_inputs = []
        for copy_mode in copy_modes:
            if copy_mode == 'src':
                local_copy_inputs.append(copy_inputs['src'][idx].split())
            elif copy_mode == 'box':
                local_copy_inputs.append(copy_inputs['box'][idx])
            elif copy_mode == 'text':
                local_copy_inputs.append(copy_inputs['text'][idx].split())
            elif copy_mode == 'csk':
                local_copy_inputs.append(copy_inputs['csk_response_word'][idx])

        src_map = dict()
        offset = 1 if has_sos else 0
        table_offset = 0

        for copy_mode, copy_input in zip(copy_modes, local_copy_inputs):
            if copy_mode == 'src' or copy_mode == 'text':
                # 首先增加Copy的占位符，随后再增加Table的
                rand_flags = np.random.random([len(copy_input)])
                if word_freq_dict is None:
                    for i, word in enumerate(copy_input):
                        if rand_flags[i] < dropout:
                            continue
                        src_map[word] = '<%s_%d>' % (copy_mode, i + offset)
                else:
                    for i, word in enumerate(copy_input):
                        if rand_flags[i] < dropout:
                            continue
                        if word_freq_dict.get(word, STOP_COPY_BEFORE_POSITION + 1) > STOP_COPY_BEFORE_POSITION:
                            src_map[word] = '<%s_%d>' % (copy_mode, i + offset)

            if copy_mode == 'csk':
                # 首先增加Copy的占位符，随后再增加Table的
                rand_flags = np.random.random([len(copy_input)])
                for i, word in enumerate(copy_input):
                    if rand_flags[i] < dropout:
                        continue
                    src_map[word] = '<%s_%d>' % (copy_mode, i)

            if copy_mode == 'box':
                # 增加Table的
                box = copy_input
                field_values = []
                for field in box.split(' '):
                    items = field.split(':')
                    field_value = items[1]
                    field_values.append(field_value)
                rand_flags = np.random.random([len(field_values)])
                if word_freq_dict is None:
                    for i, word in enumerate(field_values):
                        if rand_flags[i] < dropout:
                            continue
                        src_map[word] = '<field_%d>' % (i + table_offset)
                else:
                    for i, word in enumerate(field_values):
                        if rand_flags[i] < dropout:
                            continue
                        if word_freq_dict.get(word, STOP_COPY_BEFORE_POSITION + 1) > STOP_COPY_BEFORE_POSITION:
                            src_map[word] = '<field_%d>' % (i + table_offset)

        word_type = []
        for i, word in enumerate(tgt):
            tgt[i] = src_map.get(word, word)
        copy_inputs['tgt'][idx] = ' '.join(tgt)
        word_types.append(word_type)
    return word_types


def restore_field_copy_example(infobox, text, csk, src, tgt, has_sos=True, dynamic_vocab=[]):
    if infobox is not None:
        field_values = infobox.split()
    else:
        field_values = None
    if text is not None:
        if isinstance(text, str):
            text = text.split()
        assert isinstance(text, list)
    if csk is not None:
        if isinstance(csk, str):
            csk = csk.split()
        assert isinstance(csk, list)
    src = src.split()
    tgt = tgt.split()
    src_map = dict()
    offset = 1 if has_sos else 0
    for i, word in enumerate(src):
        src_map['<src_%d>' % (i + offset)] = word
    if csk is not None:
        for i, word in enumerate(csk):
            src_map['<csk_%d>' % (i)] = word
    if text is not None:
        for i, word in enumerate(text):
            src_map['<text_%d>' % (i + offset)] = word
    if field_values is not None:
        table_offset = 0
        for i, word in enumerate(field_values):
            src_map['<field_%d>' % (i + table_offset)] = word
    if dynamic_vocab is not None and len(dynamic_vocab) > 0:
        for i, word in enumerate(dynamic_vocab):
            src_map['<dmc_%d>' % i] = word

    for i, word in enumerate(tgt):
        tgt[i] = src_map.get(word, word)
    return ' '.join(tgt)


def load_examples(example_path, fact_dict_path):
    with open(example_path + '.' + SRC_SUFFIX, 'r', encoding='utf-8') as fin:
        src = fin.readlines()
    logger.info('[DATASET] Loaded %s.src, lines=%d' % (example_path, len(src)))
    with open(example_path + '.' + TGT_SUFFIX, 'r', encoding='utf-8') as fin:
        tgt = fin.readlines()
    logger.info('[DATASET] Loaded %s.tgt, lines=%d' % (example_path, len(tgt)))

    src = [x.strip('\r\n') for x in src]
    tgt = [x.strip('\r\n') for x in tgt]
    try:
        with open(example_path + '.' + BOX_SUFFIX, 'r', encoding='utf-8') as fin:
            boxes = fin.readlines()
        logger.info('[DATASET] Loaded %s.box, lines=%d' % (example_path, len(boxes)))
        fields = []
        for box in boxes:
            box = box.strip('\r\n')
            field_values = []
            for field in box.split(' '):
                items = field.split(':')
                field_value = items[1]
                field_values.append(field_value)
            fields.append(' '.join(field_values))
    except Exception as e:
        logger.info('[DATASET] Field to Load %s.box' % (example_path))
        fields = None

    try:
        with open(example_path + '.' + TEXT_SUFFIX, 'r', encoding='utf-8') as fin:
            text = fin.readlines()
            logger.info('[DATASET] Loaded %s.text, lines=%d' % (example_path, len(text)))
            text = [x.strip('\r\n').split() for x in text]

    except Exception as e:
        logger.info('[DATASET] Field to Load %s.text' % (example_path))
        text = None

    try:
        with open(fact_dict_path, 'r', encoding='utf-8') as fin:
            facts = [x.strip('\r\n').split()[1] for x in fin.readlines()]
        with open(example_path + '.' + CSK_SUFFIX, 'r', encoding='utf-8') as fin:
            golden_csks = fin.readlines()
            logger.info('[DATASET] Loaded %s.CSK_SUFFIX, lines=%d' % (example_path, len(golden_csks)))
            facts = [[facts[int(y)] for y in x.strip('\r\n').split()] for x in golden_csks]

    except Exception as e:
        logger.info('[DATASET] Field to Load %s.golden_csks' % (example_path))
        facts = None

    return src, tgt, fields, text, facts


def get_dataset(hparams, example_path, field_orders, field_dict, copy_dropout=0.0, random_order=False):
    max_line = hparams.max_line
    max_src_len = hparams.max_src_len
    max_tgt_len = hparams.max_tgt_len
    pointer_copy = hparams.copy
    max_csk_num = hparams.max_csk_num

    knowledge_sources = set(hparams.knowledge.split(','))
    no_copy_flags = set(hparams.mask_copy_flags.split(','))

    field_use = 'infobox' in knowledge_sources
    text_use = 'text' in knowledge_sources
    csk_use = 'commonsense' in knowledge_sources

    field_copy = ('infobox' in knowledge_sources) and ('box' not in no_copy_flags)
    text_copy = ('text' in knowledge_sources) and ('text' not in no_copy_flags)
    csk_copy = ('commonsense' in knowledge_sources) and ('csk' not in no_copy_flags)

    max_kw_pairs_num = hparams.max_kw_pairs_num
    max_text_token_num = hparams.max_text_token_num
    # 1 是单纯的词表
    vocab_source_num = 1
    # 1 是单纯的Attention
    knowledge_source_num = 1

    if pointer_copy:
        assert 'src' in field_dict
        vocab_source_num += 1
    if field_use:
        assert 'attribute_key' in field_dict
        vocab_source_num += 1
        knowledge_source_num += 1
    if text_use:
        assert 'text' in field_dict
        vocab_source_num += 1
        knowledge_source_num += 1
    if csk_use:
        assert 'csk_golden_idx' in field_dict
        vocab_source_num += 1
        knowledge_source_num += 1

    # store
    my_examples = dict()

    # Commonsense
    if 'csk_golden_idx' in field_dict:
        with open(example_path + '.' + CSK_SUFFIX, 'r', encoding='utf-8') as fin:
            csk_idx = [[int(y) for y in x.split()] for x in fin.readlines()]
        with open(example_path + '.' + CSK_GOLDEN_SUFFIX, 'r', encoding='utf-8') as fin:
            csk_golden_idx = [[int(y) for y in x.split()] for x in fin.readlines()]
        for x in csk_idx:
            assert len(x) < hparams.max_csk_num
        logger.info('[DATASET] Loaded %s.naf_fact, lines=%d' % (example_path, len(csk_idx)))
        my_examples['csk_golden_idx'] = csk_golden_idx
        with open(hparams.fact_dict_path, encoding='utf-8') as fin:
            lines = fin.readlines()
            print('Total Entity-Fact : %d' % len(lines))
            post_entities = []
            post_words = []
            response_entities = []
            response_words = []
            head_entities = []
            head_words = []
            tail_entities = []
            tail_words = []
            relation_words = []
            for line in lines:
                line = line.strip('\n')
                items = line.split()
                assert len(items) == 5
                post_entities.append(items[0])
                post_words.append(items[0])
                response_entities.append(items[1])
                response_words.append(items[1])
                head_entities.append(items[2])
                head_words.append(items[2])
                tail_entities.append(items[4])
                tail_words.append(items[4])
                relation_words.append(items[3])
            my_examples['csk_post_word'] = [[post_entities[y] for y in x[0:max_csk_num]] for x in csk_idx]
            my_examples['csk_post_entity'] = [[post_words[y] for y in x[0:max_csk_num]] for x in csk_idx]
            my_examples['csk_response_word'] = [[response_entities[y] for y in x[0:max_csk_num]] for x in csk_idx]
            my_examples['csk_response_entity'] = [[response_words[y] for y in x[0:max_csk_num]] for x in csk_idx]
            my_examples['csk_head_word'] = [[head_entities[y] for y in x[0:max_csk_num]] for x in csk_idx]
            my_examples['csk_head_entity'] = [[head_words[y] for y in x[0:max_csk_num]] for x in csk_idx]
            my_examples['csk_tail_word'] = [[tail_entities[y] for y in x[0:max_csk_num]] for x in csk_idx]
            my_examples['csk_tail_entity'] = [[tail_words[y] for y in x[0:max_csk_num]] for x in csk_idx]
            my_examples['csk_relation'] = [[relation_words[y] for y in x[0:max_csk_num]] for x in csk_idx]

            # TEXT
    if 'text' in field_dict:
        with open(example_path + '.' + TEXT_SUFFIX, 'r', encoding='utf-8') as fin:
            text = fin.readlines()
        logger.info('[DATASET] Loaded %s.text, lines=%d' % (example_path, len(text)))
        my_examples['text'] = text

    # SRC
    if 'src' in field_dict:
        with open(example_path + '.' + SRC_SUFFIX, 'r', encoding='utf-8') as fin:
            src = [x.strip('\n') for x in fin.readlines()]
        logger.info('[DATASET] Loaded %s.src, lines=%d' % (example_path, len(src)))
        my_examples['src'] = src

    # Infobox
    if 'attribute_key' in field_dict:
        with open(example_path + '.' + BOX_SUFFIX, 'r', encoding='utf-8') as fin:
            tmp = fin.readlines()
            box = []
            for item in tmp:
                items = ' '.join(item.strip('\r\n').split(' ')[0:max_kw_pairs_num - 1])
                box.append(items)
        logger.info('[DATASET] Loaded %s.box, lines=%d' % (example_path, len(box)))
        my_examples['box'] = box

    # Global Modes
    text = my_examples.get('text', [None] * len(src))
    box = my_examples.get('box', [None] * len(src))
    csk = my_examples.get('csk_post_word', [None] * len(src))
    global_modes = []
    for b, t, c in zip(box, text, csk):
        global_mode = [1]
        # The first is \n
        if c is not None:
            if len(c) > 1:
                global_mode.append(1)
            else:
                global_mode.append(0)
        if t is not None:
            if len(t) > 1:
                global_mode.append(1)
            else:
                global_mode.append(0)
        if b is not None:
            if len(b) > 21:
                global_mode.append(1)
            else:
                global_mode.append(0)
        assert len(global_mode) == knowledge_source_num, '%s    %s' % (global_mode, knowledge_source_num)
        global_modes.append(global_mode)
    # global_modes = np.array(global_modes, dtype=np.int64)
    my_examples['global_mode'] = global_modes
    my_examples['local_mode'] = global_modes

    # TGT
    with open(example_path + '.' + TGT_SUFFIX, 'r', encoding='utf-8') as fin:
        tgt = fin.readlines()
        logger.info('[DATASET] Loaded %s.tgt, lines=%d' % (example_path, len(tgt)))
        my_examples['tgt'] = tgt
        tgt = None

    # TGT_RAW
    with open(example_path + '.' + TGT_SUFFIX, 'r', encoding='utf-8') as fin:
        tgt_bow = fin.readlines()
        logger.info('[DATASET-RAW] Loaded %s.tgt, lines=%d' % (example_path, len(tgt_bow)))
        my_examples['tgt_raw'] = tgt_bow
        tgt_bow = None

    # TGT_BOW
    with open(example_path + '.' + TGT_SUFFIX, 'r', encoding='utf-8') as fin:
        tgt_bow = fin.readlines()
        logger.info('[DATASET-BOW] Loaded %s.tgt, lines=%d' % (example_path, len(tgt_bow)))
        my_examples['tgt_bow'] = tgt_bow
        tgt_bow = None

    # 裁剪一部分数据
    if max_line != -1:
        for idx in my_examples.keys():
            my_examples[idx] = my_examples[idx][0:max_line]

    if max_src_len != -1 and 'src' in field_dict:
        logger.info('[DATASET] Maximum source sequence length is set to %d' % (max_src_len))
        my_examples['src'] = [' '.join(x.strip('\r\n').split()[0:max_src_len]) for x in my_examples['src']]
        src_max_len = max([len(x.split()) for x in my_examples['src']])
    else:
        src_max_len = -1

    if max_tgt_len != -1:
        logger.info('[DATASET] Maximum target sequence length is set to %d' % (max_tgt_len))
        my_examples['tgt'] = [' '.join(x.strip('\r\n').split()[0:max_tgt_len]) for x in my_examples['tgt']]
        my_examples['tgt_raw'] = [' '.join(x.strip('\r\n').split()[0:max_tgt_len]) for x in my_examples['tgt_raw']]
        my_examples['tgt_bow'] = [' '.join(x.strip('\r\n').split()[0:max_tgt_len]) for x in my_examples['tgt_bow']]

    if 'text' in knowledge_sources and max_text_token_num != -1:
        logger.info('[DATASET] Maximum text sequence length is set to %d' % (max_text_token_num))
        my_examples['text'] = [' '.join(x.strip('\r\n').split()[0:max_text_token_num - 2]) for x in my_examples['text']]

    copy_modes = []

    if pointer_copy:
        copy_modes.append('src')
    if text_copy:
        copy_modes.append('text')
    if field_copy:
        copy_modes.append('box')
    if csk_copy:
        copy_modes.append('csk')


    if len(copy_modes) > 0:
        copy_inputs = '-'.join(copy_modes)
        logger.info('[DATASET] Aligning %s -> tgt for Pointer-Copy ' % copy_inputs)
        word_freq_dict = field_dict['tgt'].vocab.stoi
        word_types = _process_copy_examples(copy_modes, my_examples,
                                            word_freq_dict=word_freq_dict, dropout=copy_dropout,
                                            mask_copy_flags=hparams.mask_copy_flags,
                                            copy_strategy=hparams.copy_strategy,
                                            fixed_vocab_path=hparams.vocab_path)


    if hparams.word_encoder != 'none':
        my_examples['char_vocab'] = [[]] * len(my_examples['tgt'])
        if 'src' in my_examples:
            my_examples['char_src']= [[]] * len(my_examples['tgt'])
        if 'tgt' in my_examples:
            my_examples['char_tgt']= [[]] * len(my_examples['tgt'])
        if 'text' in my_examples:
            my_examples['char_text']=  [[]] * len(my_examples['tgt'])
        if 'box' in my_examples:
            my_examples['char_attribute_key'] =  [[]] * len(my_examples['tgt'])
            my_examples['char_attribute_value'] =  [[]] * len(my_examples['tgt'])
        if 'csk_post_word' in my_examples:
            my_examples['char_csk_post_word'] =  [[]] * len(my_examples['tgt'])
            my_examples['char_csk_response_word'] =  [[]] * len(my_examples['tgt'])
        if 'dynamic_vocab' in my_examples:
            my_examples['char_dynamic_vocab'] = [[]] * len(my_examples['dynamic_vocab'])
        for idx in range(len(my_examples['tgt'])):
            local_words = set()
            if 'src' in my_examples:
                local_words = local_words | set(my_examples['src'][idx].strip('\r\n').split())
            if 'tgt' in my_examples:
                local_words = local_words | set(my_examples['tgt_raw'][idx].strip('\r\n').split())
            if 'text' in my_examples:
                local_words = local_words | set(my_examples['text'][idx].strip('\r\n').split())
            if 'box' in my_examples:
                box = my_examples['box'][idx].strip('\r\n')
                for field in box.split():
                    field_value = field.split(':')[1]
                    field_key = field.split("_")[0]
                    local_words.add(field_value)
                    local_words.add(field_key)
            if 'csk_post_word' in my_examples:
                local_words = local_words | set(my_examples['csk_post_word'][idx])
                local_words = local_words | set(my_examples['csk_response_word'][idx])

            if 'dynamic_vocab' in my_examples:
                local_words = local_words | set(my_examples['dynamic_vocab'][idx].strip('\r\n').split())
            # to list
            local_words = ['<NotAWord>'] + list(local_words)
            my_examples['char_vocab'][idx] = local_words
            word_to_id = dict()

            for word in local_words:
                word_to_id[word] = len(word_to_id)

            if 'src' in my_examples:
                my_examples['char_src'][idx] = [word_to_id[word] for word in my_examples['src'][idx].strip('\r\n').split()]
            if 'tgt' in my_examples:
                my_examples['char_tgt'][idx] = [word_to_id[word] for word in
                                                my_examples['tgt_raw'][idx].strip('\r\n').split()]
            if 'text' in my_examples:
                my_examples['char_text'][idx] = [word_to_id[word] for word in
                                                my_examples['text'][idx].strip('\r\n').split()]
            if 'box' in my_examples:

                box = my_examples['box'][idx].strip('\r\n')
                keys = []
                values = []
                for field in box.split():
                    field_value = field.split(':')[1]
                    field_key = field.split("_")[0]
                    keys.append(word_to_id[field_key])
                    values.append(word_to_id[field_value])
                my_examples['char_attribute_key'][idx] = keys
                my_examples['char_attribute_value'][idx] = values
            if 'csk_post_word' in my_examples:
                my_examples['char_csk_post_word'][idx] = [word_to_id[word] for word in my_examples['csk_post_word'][idx]]
                my_examples['char_csk_response_word'][idx] = [word_to_id[word] for word in my_examples['csk_response_word'][idx]]

            if 'dynamic_vocab' in my_examples:
                my_examples['char_dynamic_vocab'][idx] = [word_to_id[word] for word in
                                                my_examples['dynamic_vocab'][idx].strip('\r\n').split()]





    def _box_split(z, sub_key=False, sub_word=False, random_order=False, keep_tags=False):
        z = z.strip('\r\n')
        field_keys = []
        sub_field_keys = []
        pos_sts = []
        pos_eds = []
        pos_kv = []
        pos_kw = []
        tags = []
        field_values = []
        sub_field_values = []

        key_position = 0
        previous_start = -1
        previous_key = '<none>'
        kv_list = [key_position]
        field_items = z.split(' ')
        if random_order:
            if len(field_items) == 1:
                pass
            else:
                field_groups = []
                tmp_group = []
                for field in field_items[1:]:
                    items = field.split(':')
                    field_key, pos_st, pos_ed, tag = items[0].split('_')
                    pos_st = int(pos_st)
                    pos_ed = int(pos_ed)
                    tmp_group.append(field)
                    if pos_ed == 1:
                        field_groups.append(tmp_group)
                        tmp_group = []
                field_groups = random.sample(field_groups, len(field_groups))
                new_items = [field_items[0]]
                for group in field_groups:
                    new_items += group
                field_items = new_items

        for field in field_items:
            items = field.split(':')
            if len(items[0].split('_')) != 4:
                print(field)
            field_key, pos_st, pos_ed, tag = items[0].split('_')
            pos_st = int(pos_st)
            pos_ed = int(pos_ed)

            if pos_st != previous_start + 1 or previous_key != field_key:
                key_position += 1
                kv_list.append(key_position)
            previous_start = pos_st
            previous_key = field_key
            pos_kv.append(key_position)
            pos_kw.append(len(pos_kw))

            field_value = items[1]
            field_keys.append(field_key)
            sub_field_keys.append(field_key + '')
            pos_sts.append(pos_st)
            pos_eds.append(pos_ed)
            tags.append(tag)
            field_values.append(field_value)
            sub_field_values.append(field_value + '')

        graph_node_number = len(kv_list)
        field_key_tags = field_keys
        # 3 keys
        res = [graph_node_number, field_keys]
        if sub_key:
            res += [sub_field_keys]
        if keep_tags is False:
            res += [field_keys, field_values]
        else:
            res += [field_keys, field_key_tags, field_values]
        if sub_word:
            res += [sub_field_values]
        if keep_tags is False:
            res += [field_values, pos_kv, pos_kw, pos_sts, pos_eds]
        else:
            res += [field_values, tags, pos_kv, pos_kw, pos_sts, pos_eds]
        return res

    # 创造初始的输入
    lines = []
    raw_num = len(my_examples['tgt'])
    for idx in range(raw_num):
        line = []
        if 'global_mode' in my_examples:
            line += [my_examples['global_mode'][idx]]
        if 'local_mode' in my_examples:
            line += [my_examples['local_mode'][idx]]
        if 'src' in my_examples:
            line += [my_examples['src'][idx].strip('\r\n')]
        if 'sub_src' in field_dict:
            line += [my_examples['src'][idx].strip('\r\n')]
        line += [my_examples['tgt'][idx].strip('\r\n'), my_examples['tgt_raw'][idx].strip('\r\n'),
                 my_examples['tgt_bow'][idx].strip('\r\n')]
        if 'csk_golden_idx' in my_examples:
            line += [my_examples['csk_golden_idx'][idx],
                     my_examples['csk_post_word'][idx], my_examples['csk_post_entity'][idx],
                     my_examples['csk_response_word'][idx], my_examples['csk_response_entity'][idx],
                     my_examples['csk_head_word'][idx], my_examples['csk_head_entity'][idx],
                     my_examples['csk_tail_word'][idx], my_examples['csk_tail_entity'][idx],
                     my_examples['csk_relation'][idx],
                     ]
        if 'text' in my_examples:
            line += [my_examples['text'][idx].strip('\r\n')]
        if 'box' in my_examples:
            line += _box_split(my_examples['box'][idx], 'sub_attribute_key' in field_dict,
                               'sub_attribute_word' in field_dict, random_order=random_order)

        dynamic_flag = False
        if 'dynamic_vocab_src' in my_examples:
            dynamic_flag = True
            line += [my_examples['dynamic_vocab_src'][idx].split()]
        if 'dynamic_vocab_csk' in my_examples:
            dynamic_flag = True
            line += [my_examples['dynamic_vocab_csk'][idx].split()]
        if 'dynamic_vocab_text' in my_examples:
            dynamic_flag = True
            line += [my_examples['dynamic_vocab_text'][idx].split()]
        if 'dynamic_vocab_attribute' in my_examples:
            dynamic_flag = True
            line += [my_examples['dynamic_vocab_attribute'][idx].split()]
        if dynamic_flag:
            line += [my_examples['dynamic_vocab'][idx].split()]
            line += [my_examples['dynamic_vocab_raw'][idx].split()]

        if hparams.word_encoder != 'none':
            if 'char_src' in my_examples:
                line += [my_examples['char_src'][idx]]
            if 'char_tgt' in my_examples:
                line += [my_examples['char_tgt'][idx]]
            if 'char_csk_post_word' in my_examples:
                line += [my_examples['char_csk_post_word'][idx]]
            if 'char_csk_response_word' in my_examples:
                line += [my_examples['char_csk_response_word'][idx]]
            if 'char_text' in my_examples:
                line += [my_examples['char_text'][idx]]
            if 'char_attribute_key' in my_examples:
                line += [my_examples['char_attribute_key'][idx]]
            if 'char_attribute_value' in my_examples:
                line += [my_examples['char_attribute_value'][idx]]
            if 'char_dynamic_vocab' in my_examples:
                line += [my_examples['char_dynamic_vocab'][idx]]
            line += [my_examples['char_vocab'][idx]]




        lines.append(line)
        assert len(line) == len(field_orders), '%s-%s' % (len(line), len(field_orders))

    examples = []
    for line in lines:
        example = torchtext.data.Example.fromlist(data=line, fields=field_orders)
        examples.append(example)
    dataset = Dataset(examples=examples, fields=field_orders)
    return dataset, src_max_len


def load_vocab(vocab_path, field, special_tokens=[], pointer_copy_tokens=0, text_copy_tokens=0,
               field_copy_tokens=0, ccm_copy_tokens=0, dynamic_tokens=0, vanilla_mode=False):
    counter = Counter()
    if dynamic_tokens > 0:
        assert dynamic_tokens > 1
        assert pointer_copy_tokens + text_copy_tokens + field_copy_tokens + ccm_copy_tokens == 0
        assert vocab_path is not None
        with open(vocab_path, 'r+', encoding='utf-8') as fin:
            vocabs = [x.strip('\r\n') for x in fin.readlines()]
    else:
        assert vocab_path is not None
        with open(vocab_path, 'r+', encoding='utf-8') as fin:
            vocabs = [x.strip('\r\n') for x in fin.readlines()]


    if pointer_copy_tokens > 0:
        vocabs += ['<src_%d>' % x for x in range(pointer_copy_tokens)]
    if ccm_copy_tokens > 0:
        vocabs += ['<csk_%d>' % x for x in range(ccm_copy_tokens)]
    if text_copy_tokens > 0:
        vocabs += ['<text_%d>' % x for x in range(text_copy_tokens)]
    if field_copy_tokens > 0:
        vocabs += ['<field_%d>' % x for x in range(field_copy_tokens)]
    if dynamic_tokens > 0:
        vocabs += ['<vocab>'] + ['<dmc_%d>' % x for x in range(dynamic_tokens-1)]
    vocab_size = len(vocabs)
    for idx, token in enumerate(vocabs):
        counter[token] = vocab_size - idx


    specials = list(OrderedDict.fromkeys(
        tok for tok in [field.unk_token, field.pad_token, field.init_token,
                        field.eos_token] + special_tokens
        if tok is not None))
    if vanilla_mode:
        assert len(counter) == len(vocabs)
        specials = []
    field.vocab = Vocab(counter, specials=specials)


def load_dataset(hparams, is_eval=False, test_data_path=None):
    # 开始加载数据集
    batch_size = hparams.batch_size
    max_copy_token_num = hparams.max_copy_token_num
    pointer_copy_tokens = hparams.max_copy_token_num if hparams.copy else 0
    knowledge_sources = set(hparams.knowledge.split(','))

    no_copy_flags = set(hparams.mask_copy_flags.split(','))
    field_copy_tokens = hparams.max_kw_pairs_num if 'infobox' in knowledge_sources and 'box' not in no_copy_flags else 0
    text_copy_tokens = hparams.max_text_token_num if 'text' in knowledge_sources and 'text' not in no_copy_flags else 0
    ccm_copy_tokens = hparams.max_csk_num if 'commonsense' in knowledge_sources and 'csk' not in no_copy_flags else 0
    tokenize = get_tokenizer('default')

    # Mode Indicator
    # 标识哪些知识有用
    knowledge_source_indicator = Field(tokenize=None, include_lengths=False, use_vocab=False,
                                       init_token=None, eos_token=None, pad_token=None, unk_token=None)
    # 标识每一步使用的知识是谁
    local_mode_indicator_field = Field(tokenize=None, include_lengths=False, use_vocab=False,
                                       init_token=None, eos_token=0, pad_token=0, unk_token=0)
    # Dialogue
    src_field = Field(tokenize=tokenize, include_lengths=True,
                      init_token='<ssos>', eos_token='<seos>')
    # sub_src_field = SubTextField(tokenize=tokenize, include_lengths=True,
    #                              init_token='<ssos>', eos_token='<seos>')
    tgt_field = Field(tokenize=tokenize, include_lengths=True,
                      init_token='<sos>', eos_token='<eos>')
    tgt_raw_text_field = Field(tokenize=tokenize, include_lengths=True,
                               init_token='<sos>', eos_token='<eos>')
    tgt_bow_field = Field(tokenize=tokenize, include_lengths=True,
                          init_token='<sos>', eos_token='<eos>')

    # Commonsense knowledge
    csk_golden_idx_field = Field(tokenize=None, include_lengths=True, use_vocab=False,
                                 init_token=None, eos_token=0, pad_token=0, unk_token=0)
    csk_post_word_field = Field(tokenize=None, include_lengths=True, init_token=None, eos_token=None)
    csk_response_word_field = Field(tokenize=None, include_lengths=True, init_token=None, eos_token=None)
    csk_post_entity_field = Field(tokenize=None, include_lengths=True, init_token=None, eos_token=None,
                                  unk_token='#N', pad_token='#N')
    csk_response_entity_field = Field(tokenize=None, include_lengths=True, init_token=None, eos_token=None,
                                      unk_token='#N', pad_token='#N')
    csk_head_word_field = Field(tokenize=None, include_lengths=True, init_token=None, eos_token=None)
    csk_tail_word_field = Field(tokenize=None, include_lengths=True, init_token=None, eos_token=None)
    csk_head_entity_field = Field(tokenize=None, include_lengths=True, init_token=None, eos_token=None,
                                  unk_token='#NH', pad_token='#NH')
    csk_tail_entity_field = Field(tokenize=None, include_lengths=True, init_token=None, eos_token=None,
                                  unk_token='#NT', pad_token='#NT')
    csk_relation_field = Field(tokenize=None, include_lengths=True, init_token=None, eos_token=None,
                               unk_token='#NF', pad_token='#NF')

    # Text
    text_field = Field(tokenize=tokenize, include_lengths=True,
                       init_token='<ssos>', eos_token='<seos>')

    # Infobox
    attribute_graph_field = GraphField(tokenize=None, include_lengths=True,
                                       init_token=None, eos_token='<seos>')
    attribute_key_field = Field(tokenize=None, include_lengths=True,
                                init_token=None, eos_token='<seos>')
    sub_attribute_key_field = SubTextField(tokenize=None, include_lengths=True,
                                           init_token=None, eos_token='<seos>')
    attribute_uni_key_field = Field(tokenize=None, include_lengths=True,
                                    init_token=None, eos_token='<seos>')
    attribute_key_tag_field = Field(tokenize=None, include_lengths=True,
                                    init_token=None, eos_token='<seos>', )
    attribute_word_field = Field(tokenize=None, include_lengths=True,
                                 init_token=None, eos_token='<seos>', )
    sub_attribute_word_field = SubTextField(tokenize=None, include_lengths=True,
                                            init_token=None, eos_token='<seos>', )
    attribute_uni_word_field = Field(tokenize=None, include_lengths=True,
                                     init_token=None, eos_token='<seos>', )
    attribute_word_tag_field = Field(tokenize=None, include_lengths=True,
                                     init_token=None, eos_token='<seos>', )
    attribute_kv_pos_field = Field(tokenize=None, include_lengths=True, use_vocab=False,
                                   init_token=None, eos_token=0, pad_token=0, unk_token=0)
    attribute_kw_pos_field = Field(tokenize=None, include_lengths=True, use_vocab=False,
                                   init_token=None, eos_token=0, pad_token=0, unk_token=0)

    attribute_word_local_fw_pos_field = Field(tokenize=None, include_lengths=True, use_vocab=False,
                                              init_token=None, eos_token=0, pad_token=0, unk_token=0)
    attribute_word_local_bw_pos_field = Field(tokenize=None, include_lengths=True, use_vocab=False,
                                              init_token=None, eos_token=0, pad_token=0, unk_token=0)

    # Copy 注意和之前保持一致
    dynamic_vocab_src_field = Field(tokenize=tokenize, include_lengths=True,
                               init_token='<dmc_0>', eos_token='<dmc_0>', unk_token='<dmc_0>', pad_token='<dmc_0>')
    dynamic_vocab_csk_field = Field(tokenize=tokenize, include_lengths=True,
                               init_token=None, eos_token=None, unk_token='<dmc_0>', pad_token='<dmc_0>')
    dynamic_vocab_attribute_field = Field(tokenize=tokenize, include_lengths=True,
                                     init_token=None, eos_token='<dmc_0>', unk_token='<dmc_0>', pad_token='<dmc_0>')
    dynamic_vocab_text_field = Field(tokenize=tokenize, include_lengths=True,
                                init_token='<dmc_0>', eos_token='<dmc_0>', unk_token='<dmc_0>', pad_token='<dmc_0>')
    dynamic_vocab_field = RawField()
    dynamic_vocab_raw_field = RawField()

    if hparams.word_encoder != 'none':
        char_src_field = Field(tokenize=tokenize, include_lengths=True, use_vocab=False,
                                        init_token=0, eos_token=0, unk_token=0,
                                        pad_token=0)
        char_tgt_field = Field(tokenize=tokenize, include_lengths=True, use_vocab=False,
                          init_token=0, eos_token=0, unk_token=0, pad_token=0)
        char_csk_post_word_field = Field(tokenize=tokenize, include_lengths=True,use_vocab=False,
                                        init_token=None, eos_token=None, unk_token=0, pad_token=0)
        char_csk_response_word_field = Field(tokenize=tokenize, include_lengths=True,use_vocab=False,
                                        init_token=None, eos_token=None, unk_token=0, pad_token=0)
        char_attribute_key_field = Field(tokenize=tokenize, include_lengths=True, use_vocab=False,
                                              init_token=None, eos_token=0, unk_token=0,
                                              pad_token=0)
        char_attribute_value_field = Field(tokenize=tokenize, include_lengths=True, use_vocab=False,
                                              init_token=None, eos_token=0, unk_token=0,
                                              pad_token=0)
        char_text_field = Field(tokenize=tokenize, include_lengths=True, use_vocab=False,
                                         init_token=0, eos_token=0, unk_token=0,
                                         pad_token=0)
        max_char_seq_len = 5 if hparams.word_encoder_type == 'cnn' else -1
        char_vocab_field = SubTextField(tokenize=None, include_lengths=True, max_char_seq_len=max_char_seq_len,
                                           init_token=None, eos_token=None,pad_token='<p>', unk_token='<u>')
        char_dynamic_vocab_field = Field(tokenize=tokenize, include_lengths=True, use_vocab=False,
                                         init_token=None, eos_token=None, unk_token=0,
                                         pad_token=0)

    fields = []
    copy_fields = []
    char_fields = []
    fields += [('knowledge_source_indicator', knowledge_source_indicator)]
    fields += [('local_mode', knowledge_source_indicator)]
    fields += [('src', src_field)]
    if hparams.word_encoder != 'none':
        char_fields +=  [('char_src', char_src_field)]
    fields += [('tgt', tgt_field), ('tgt_raw', tgt_raw_text_field), ('tgt_bow', tgt_bow_field)]
    if hparams.word_encoder != 'none':
        char_fields +=  [('char_tgt', char_tgt_field)]

    if hparams.copy and hparams.copy_strategy == 'dynamic_vocab':
        copy_fields.append(('dynamic_vocab_src', dynamic_vocab_src_field))

    if 'commonsense' in knowledge_sources:
        fields += [('csk_golden_idx', csk_golden_idx_field)]
        fields += [('csk_post_word', csk_post_word_field)]
        fields += [('csk_post_entity', csk_post_entity_field)]
        fields += [('csk_response_word', csk_response_word_field)]
        fields += [('csk_response_entity', csk_response_entity_field)]
        fields += [('csk_head_word', csk_head_word_field)]
        fields += [('csk_head_entity', csk_head_entity_field)]
        fields += [('csk_tail_word', csk_tail_word_field)]
        fields += [('csk_tail_entity', csk_tail_entity_field)]
        fields += [('csk_relation', csk_relation_field)]
        if 'csk' not in no_copy_flags and hparams.copy_strategy == 'dynamic_vocab':
            copy_fields.append(('dynamic_vocab_csk', dynamic_vocab_csk_field))
        if hparams.word_encoder != 'none':
            char_fields += [('char_csk_post_word', char_csk_post_word_field)]
            char_fields += [('char_csk_response_word', char_csk_response_word_field)]
    if 'text' in knowledge_sources:
        fields += [('text', text_field)]
        if 'text' not in no_copy_flags and hparams.copy_strategy == 'dynamic_vocab':
            copy_fields.append(('dynamic_vocab_text', dynamic_vocab_text_field))
        if hparams.word_encoder != 'none':
            char_fields += [('char_text', char_text_field)]

    if 'infobox' in knowledge_sources:
        fields += [('attribute_graph', attribute_graph_field)]
        fields += [('attribute_key', attribute_key_field)]
        fields += [('attribute_uni_key', attribute_uni_key_field),
                   ('attribute_key_tag', attribute_key_tag_field)]
        fields += [('attribute_word', attribute_word_field)]
        fields += [('attribute_uni_word', attribute_uni_word_field),
                   ('attribute_word_tag', attribute_word_tag_field)]
        fields += [('attribute_kv_pos', attribute_kv_pos_field), ('attribute_kw_pos', attribute_kw_pos_field)]
        fields += [('attribute_word_local_fw_pos', attribute_word_local_fw_pos_field),
                   ('attribute_word_local_bw_pos', attribute_word_local_bw_pos_field)]
        if 'box' not in no_copy_flags and hparams.copy_strategy == 'dynamic_vocab':
            copy_fields.append(('dynamic_vocab_attribute', dynamic_vocab_attribute_field))
        if hparams.word_encoder != 'none':
            char_fields += [('char_attribute_key', char_attribute_key_field)]
            char_fields += [('char_attribute_value', char_attribute_value_field)]

    if len(copy_fields) > 0:
        copy_fields.append(('dynamic_vocab', dynamic_vocab_field))
        copy_fields.append(('dynamic_vocab_raw', dynamic_vocab_raw_field))

    if len(char_fields) > 0:
        char_fields += [('char_dynamic_vocab', char_dynamic_vocab_field)]
        char_fields += [('char_vocab', char_vocab_field)]

    # Copy Fields
    fields = fields + copy_fields + char_fields

    # To Field Dict
    field_dict = {}
    for x in fields:
        field_dict[x[0]] = x[1]

    # Dialogue Vocabs
    if not hparams.share_vocab:
        logger.info('[VOCAB] Constructing two vocabs for the src and tgt')
        if 'src' in field_dict:
            logger.info('[VOCAB] Loading src vocab from: %s' % hparams.src_vocab_path)
            load_vocab(hparams.src_vocab_path, src_field)
            logger.info('[VOCAB] src vocab size: %d' % len(src_field.vocab.itos))
        else:
            logger.info('[VOCAB] Ignored src vocab')
        logger.info('[VOCAB] Loading tgt vocab from: %s' % hparams.tgt_vocab_path)
        load_vocab(hparams.tgt_vocab_path, tgt_field, pointer_copy_tokens=pointer_copy_tokens,
                   field_copy_tokens=field_copy_tokens, text_copy_tokens=text_copy_tokens,
                   ccm_copy_tokens=ccm_copy_tokens)
        logger.info('[VOCAB] tgt vocab size: %d' % len(tgt_field.vocab.itos))

    else:
        logger.info('[VOCAB] Constructing a sharing vocab for the src and tgt')
        logger.info('[VOCAB] Loading src&tgt vocab from: %s' % hparams.vocab_path)
        if hparams.copy_strategy == 'dynamic_vocab':
            load_vocab(hparams.vocab_path, tgt_field,
                       pointer_copy_tokens=0,
                       field_copy_tokens=0,
                       text_copy_tokens=0,
                       ccm_copy_tokens=0,
                       dynamic_tokens=hparams.max_dynamic_vocab_size,
                       special_tokens=[src_field.unk_token, src_field.pad_token, src_field.init_token, src_field.eos_token])
        else:
            load_vocab(hparams.vocab_path, tgt_field,
                       pointer_copy_tokens=pointer_copy_tokens,
                       field_copy_tokens=field_copy_tokens,
                       text_copy_tokens=text_copy_tokens,
                       ccm_copy_tokens=ccm_copy_tokens,
                       special_tokens=[src_field.unk_token, src_field.pad_token, src_field.init_token, src_field.eos_token])
        if 'src' in field_dict:
            src_field.vocab = tgt_field.vocab
            logger.info('[VOCAB] src vocab size: %d' % len(src_field.vocab.itos))
        else:
            logger.info('[VOCAB] src vocab size: -1')
        logger.info('[VOCAB] tgt vocab size: %d' % len(tgt_field.vocab.itos))

    tgt_bow_field.vocab = tgt_field.vocab
    tgt_raw_text_field.vocab = tgt_field.vocab
    if 'text' in field_dict:
        if 'src' in field_dict:
            text_field.vocab = src_field.vocab
        else:
            text_field.vocab = tgt_field.vocab

    if 'csk_golden_idx' in field_dict:
        csk_head_word_field.vocab = src_field.vocab
        csk_tail_word_field.vocab = src_field.vocab
        csk_post_word_field.vocab = src_field.vocab
        csk_response_word_field.vocab = src_field.vocab
        load_vocab(hparams.csk_entity_vocab_path, csk_head_entity_field,
                   vanilla_mode=True)
        csk_tail_entity_field.vocab = csk_head_entity_field.vocab
        csk_post_entity_field.vocab = csk_head_entity_field.vocab
        csk_response_entity_field.vocab = csk_head_entity_field.vocab
        load_vocab(hparams.csk_relation_vocab_path, csk_relation_field,
                   vanilla_mode=True)

    if 'attribute_uni_key' in field_dict:
        if 'src' in field_dict:
            vocab = src_field.vocab
        else:
            vocab = tgt_field.vocab
        attribute_uni_word_field.vocab = vocab
        attribute_uni_key_field.vocab = vocab

        logger.info('[FIELD_KEY_VOCAB] Constructing field vocab from %s' % hparams.field_key_vocab_path)
        load_vocab(hparams.field_key_vocab_path, attribute_key_field)
        logger.info('[FIELD_KEY_VOCAB]  vocab size: %d' % len(attribute_key_field.vocab.itos))

        logger.info('[FIELD_WORD_VOCAB] Constructing field  word vocab from %s' % hparams.field_word_vocab_path)
        load_vocab(hparams.field_word_vocab_path, attribute_word_field)
        logger.info('[FIELD_WORD_VOCAB]  vocab size: %d' % len(attribute_word_field.vocab.itos))

        if hparams.field_word_tag_path != 'none':
            logger.info('[FIELD_WORD_TAG_VOCAB] Constructing tag word vocab from %s' % hparams.field_word_tag_path)
            load_vocab(hparams.field_word_tag_path, attribute_word_tag_field)
            logger.info('[FIELD_WORD_TAG_VOCAB]  vocab size: %d' % len(attribute_word_tag_field.vocab.itos))
        else:
            assert hparams.field_tag_usage == 'none', 'please add a field_tag vocab'
            field_dict['attribute_word_tag'] = None

        if hparams.field_key_tag_path != 'none':
            logger.info('[FIELD_KEY_TAG_VOCAB] Constructing tag key vocab from %s' % hparams.field_key_tag_path)
            load_vocab(hparams.field_key_tag_path, attribute_key_tag_field)
            logger.info('[FIELD_KEY_TAG_VOCAB]  vocab size: %d' % len(attribute_key_tag_field.vocab.itos))
        else:
            assert hparams.field_tag_usage in ['none', 'general'], 'please add a field_tag vocab'
            field_dict['attribute_key_tag'] = None

    if hparams.copy_strategy == 'dynamic_vocab':
        # dynamic_vocab_field.vocab = tgt_field.vocab
        # dynamic_vocab_raw_field.vocab = tgt_field.vocab
        dynamic_vocab_src_field.vocab = tgt_field.vocab
        dynamic_vocab_csk_field.vocab = tgt_field.vocab
        dynamic_vocab_text_field.vocab = tgt_field.vocab
        dynamic_vocab_attribute_field.vocab = tgt_field.vocab

    if hparams.word_encoder != 'none':
        logger.info('[CHAR_VOCAB] Constructing tag char vocab from %s' % hparams.char_vocab_path)
        load_vocab(hparams.char_vocab_path, char_vocab_field,  special_tokens=[char_vocab_field.unk_token,
                                                                               char_vocab_field.pad_token,
                                                                               char_vocab_field.init_token,
                                                                               char_vocab_field.eos_token])





    sort_key = lambda x: len(x.tgt) + len(x.src) * (hparams.max_src_len + 5)
    device = 'cuda' if hparams.cuda else 'cpu'

    # Update Fields:
    field_orders = []
    for field in fields:
        if field_dict[field[0]] is None:
            logger.info('[FIELD] Removing the invalid field %s:' % field[0])
            del field_dict[field[0]]
            assert field[0] not in field_dict
        else:
            logger.info('[FIELD] Keep the valid field %s:' % field[0])
            field_orders.append(field)

    val, max_val_len = get_dataset(hparams, hparams.val_data_path_prefix,
                                   field_orders=field_orders, field_dict=field_dict)

    test, max_test_len = get_dataset(hparams, hparams.test_data_path_prefix if not test_data_path else test_data_path,
                                     field_orders=field_orders, field_dict=field_dict)

    if hparams.copy:
        assert max_val_len + 1 < max_copy_token_num, max_val_len
        assert max_test_len + 1 < max_copy_token_num, max_test_len

    if not is_eval:
        logger.info('[DATASET] Training Mode')
        train, max_train_len = get_dataset(hparams, hparams.train_data_path_prefix,
                                           field_orders=field_orders, field_dict=field_dict,
                                           copy_dropout=hparams.align_dropout)
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
        return train_iter, val_iter, test_iter, field_dict
    else:
        logger.info('[DATASET] Eval/Inference Mode')
        val_iter = Iterator(val, batch_size=batch_size, repeat=False, shuffle=False,
                            sort_key=sort_key, sort=False, train=False, sort_within_batch=False,
                            device=device)
        test_iter = Iterator(test, batch_size=batch_size, repeat=False, shuffle=False,
                             sort_key=sort_key, sort=False, train=False, sort_within_batch=False,
                             device=device)
        return None, val_iter, test_iter, field_dict
