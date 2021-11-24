# This code is forked/revised from Tensorflow-NMT

"""Utility for evaluating various tasks, e.g., translation & summarization."""
import codecs
import os
import re
import subprocess
import sys
import math
import re

import numpy as np

from utils.evaluation.scripts import bleu
from utils.evaluation.scripts import rouge
from utils.evaluation.scripts import embed
from utils.evaluation.scripts import tokens2wordlevel

from rouge import Rouge


from collections import Counter, defaultdict

__all__ = ["evaluate"]


def evaluate(ref_tgt_file, ref_src_file, generated_file, embed_file, metric, dim=200, vocab_size=None, word_option=None,
             beam_width=10):
    """
    :param ref_tgt_file: Ground-truth reference tgt file
    :param ref_src_file: Ground-truth reference src file
    :param generated_file: : The generated tgt file
    :param embed_file:
    :param metric:
    :param dim:
    :param vocab_size:
    :param word_option:
    :param beam_width:
    :return:
    """
    if metric.lower() == "embed":
        evaluation_score = embed.eval(ref_src_file, ref_tgt_file, generated_file, embed_file, dim,
                                      word_option)
    elif metric.lower() == "unk":
        evaluation_score = unk(generated_file, -1, word_option=word_option)
    elif metric.lower() == 'unk_sentence':
        evaluation_score = unk_sentence(generated_file, -1, word_option=word_option)
    elif len(metric.lower()) > 4 and metric.lower()[0:4] == 'bleu':
        max_order = int(metric.lower()[5:])
        evaluation_score = _bleu(ref_tgt_file, generated_file, max_order=max_order,
                                 word_option=word_option)
        reverse_evaluation_score = _bleu(ref_src_file, generated_file, max_order=max_order,
                                 word_option=word_option)
        for key in reverse_evaluation_score.keys():
            evaluation_score['SRC_'+key] = reverse_evaluation_score[key]
            
    elif metric.lower() == "rouge":
        evaluation_score = _rouge(ref_tgt_file, generated_file,
                                  word_option=word_option)
    elif metric.lower()[0:len('distinct_c')] == 'distinct_c':
        max_order = int(metric.lower()[len('distinct_c') + 1:])
        evaluation_score = _distinct_c(generated_file, max_order, word_option=word_option)
    elif metric.lower()[0:len('distinct')] == 'distinct':
        max_order = int(metric.lower()[len('distinct') + 1:])
        evaluation_score = _distinct(generated_file, max_order, word_option=word_option)
    elif metric.lower()[0:len('intra_distinct')] == 'intra_distinct':
        max_order = int(metric.lower()[len('intra_distinct') + 1:])
        evaluation_score = _intra_distinct(generated_file, max_order, beam_width, word_option=word_option)
    elif metric.lower()[0:len('len')] == 'len':
        evaluation_score = seq_len(generated_file, -1, word_option=word_option)
    elif metric.lower() == 'entropy':
        dict_files = [ref_tgt_file, ref_src_file]
        evaluation_score = _entropy_nrg(dict_files, generated_file, word_option=word_option, vocab_size=vocab_size)
    elif metric.lower() == 'std_ent':
        evaluation_score = _std_ent_n(generated_file, word_option=word_option)
    else:
        raise ValueError("Unknown metric %s" % metric)

    return evaluation_score


def _clean(sentence, word_option):
    sentence = tokens2wordlevel.revert_from_sentence(sentence, word_option)
    return sentence


# 验证没问题OK https://github.com/mgalley/DSTC7-End-to-End-Conversation-Modeling/blob/master/evaluation/src/metrics.py
def _distinct(trans_file, max_order=1, word_option=None):
    """Compute Distinct Score"""
    translations = []
    with open(trans_file, 'r', encoding='utf-8') as fh:
        for line in fh.readlines():
            line = _clean(line, word_option=word_option)
            translations.append(line.split(" "))

    num_tokens = 0
    unique_tokens = set()
    scores = []
    for words in translations:
        local_unique_tokens = set()
        local_count = 0.0
        # print(items)
        for i in range(0, len(words) - max_order + 1):
            valid_words = []
            for x in words[i:i + max_order]:
                if x != '<unk>':
                    valid_words.append(x)
            tmp = ' '.join(valid_words)
            unique_tokens.add(tmp)
            num_tokens += 1
            local_unique_tokens.add(tmp)
            local_count += 1
        if local_count == 0:
            scores.append(0)
        else:
            scores.append(100 * len(local_unique_tokens) / local_count)
    if num_tokens == 0:
        ratio = 0
    else:
        ratio = len(unique_tokens) / num_tokens

    result_dict = {
      'DIST_%d' % max_order: ratio * 100,
      'DIST_%d_Scores' % max_order: scores,
    }
    return result_dict


def _intra_distinct(trans_file, max_order, beam_width, word_option=None):
    """Compute Distinct Score"""
    if beam_width == 0:
        beam_width = 1
    translations = []
    with open(trans_file, 'r', encoding='utf-8') as fh:
        for line in fh.readlines():
            line = _clean(line, word_option=word_option)
            translations.append(line.split(" "))

    assert len(translations) % beam_width == 0

    all_ratios = []
    for i in range(0, len(translations), beam_width):
        num_tokens = 0
        unique_tokens = set()
        for items in translations[i:i + beam_width]:
            # print(items)
            for i in range(0, len(items) - max_order + 1):
                valid_words = []
                for x in items[i:i + max_order]:
                    if x != '<unk>':
                        valid_words.append(x)
                tmp = ' '.join(valid_words)
                unique_tokens.add(tmp)
                num_tokens += 1
        if num_tokens == 0:
            ratio = 0
        else:
            ratio = len(unique_tokens) / num_tokens
        all_ratios.append(ratio*100)

    result_dict = {
      'DIST_Intra_%d' % max_order: sum(all_ratios) / len(all_ratios),
      'DIST_Intra_%d_Scores' % max_order: all_ratios,
    }
    return result_dict


def seq_len(trans_file, max_order=1, word_option=None):
    """Compute Length Score"""
    sum = 0.0
    total = 0
    scores = []
    with open(trans_file, 'r', encoding='utf-8') as fh:
        for line in fh.readlines():
            line = _clean(line, word_option=word_option)
            word_len = len(line.split())
            sum += word_len
            scores.append(word_len)
            total += 1
    result_dict = {
      'Len': sum / total,
      'Len_Scores': scores,
    }
    return result_dict


def unk(trans_file, max_order=1, word_option=None):
    """Compute UNK Number"""
    unk = 0.0
    total = 0.0
    scores = []
    with open(trans_file, 'r', encoding='utf-8') as fh:
        for line in fh.readlines():
            line = _clean(line, word_option=word_option).split()
            total += len(line)
            my_unk = 0.0
            for word in line:
                if word.find('<unk>') > -1:
                    unk += 1
                    my_unk += 1
            if len(line) == 0:
                scores.append(1.0)
            else:
                scores.append(1.0 - (my_unk / len(line)))

    result_dict = {
      'NonUNKRate': unk / total,
      'NonUNKRate_Scores': scores,
    }
    return result_dict


def unk_sentence(trans_file, max_order=1, word_option=None):
    """Compute UNK Number"""
    unk = 0.0
    total = 0.0
    scores = []
    with open(trans_file, 'r', encoding='utf-8') as fh:
        for line in fh.readlines():
            line = _clean(line, word_option=word_option).split()
            total += 1
            my_unk = 0.0
            for word in line:
                if word == '<unk>':
                    unk += 1
                    my_unk += 1
                    break
            if len(line) == 0:
                scores.append(0)
            else:
                scores.append(1.0 - my_unk)
    result_dict = {
      'NonUNKSent': unk / total,
      'NonUNKSent_Scores': scores,
    }
    return result_dict


def _distinct_c(trans_file, max_order=1, word_option=None):
    """Compute Distinct Score"""

    translations = []
    with open(trans_file, 'r', encoding='utf-8') as fh:
        for line in fh.readlines():
            line = _clean(line, word_option=word_option)
            translations.append(line.split(" "))

    unique_tokens = set()
    local_counts = []
    for items in translations:
        # print(items)
        local_unique_tokens = set()
        for i in range(0, len(items) - max_order + 1):
            valid_words = []
            for x in items[i:i+max_order]:
                if x != '<unk>':
                    valid_words.append(x)
            tmp = ' '.join(valid_words)
            unique_tokens.add(tmp)
            local_unique_tokens.add(tmp)
        local_counts.append(len(local_unique_tokens))
    ratio = len(unique_tokens)
    result_dict = {
      'DIST_Count_%d' % max_order: ratio,
      'DIST_Count_%d_Scores' % max_order: local_counts,
    }
    return result_dict


def _std_ent_n(trans_file, word_option=None):
    # based on Yizhe Zhang's code https://github.com/mgalley/DSTC7-End-to-End-Conversation-Modeling/blob/master/evaluation/src/metrics.py
    ngram_scores = [0, 0, 0, 0]
    counter = [defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)]
    i = 0
    for line in open(trans_file, encoding='utf-8'):
        i += 1
        words = _clean(line, word_option=word_option).split()
        for n in range(4):
            for idx in range(len(words) - n):
                ngram = ' '.join(words[idx:idx + n + 1])
                counter[n][ngram] += 1

    for n in range(4):
        total = sum(counter[n].values())
        for v in counter[n].values():
            ngram_scores[n] += - v / total * (np.log(v) - np.log(total))

    result_dict = {
        'ent1': ngram_scores[0],
        'ent2': ngram_scores[1],
        'ent3': ngram_scores[2],
        'ent4': ngram_scores[3],
    }

    return result_dict

def _entropy_nrg(dict_files, trans_file, vocab_size, word_option=None):
    """Compute Entropy Score"""
    counter = Counter()
    num_tokens = 0.0
    print("entropy reference files:", str(dict_files))
    for dict_file in dict_files:
        # with codecs.getreader("utf-8")(tf.gfile.GFile(dict_file)) as fh:
        with open(dict_file, encoding='utf-8') as fh:
            for line in fh.readlines():
                line = _clean(line, word_option=word_option)
                tokens = line.split(" ")
                num_tokens += len(tokens)
                counter.update(tokens)
    entropy = 0
    vocab_counts = counter.most_common(vocab_size)
    num_vocab_count = sum([x[1] for x in vocab_counts])
    unk_count = num_tokens - num_vocab_count
    num_infer_tokens = 0

    scores1 = []
    scores2 = []
    scores3 = []
    scores4 = []
    # Word_Level with UNK
    with open(trans_file, encoding='utf-8') as fh:
        for line in fh.readlines():
            line = _clean(line, word_option=word_option)
            tokens = line.split(" ")
            local_scores = []
            for item in tokens:
                if item.find('unk') > 0:
                    fre = unk_count
                else:
                    fre = max(1, counter[item])
                p = fre / num_tokens
                tmp = -math.log(p, 2)
                local_scores += [tmp]
                entropy += tmp
                num_infer_tokens += 1.0
            scores1.append(sum(local_scores) / len(local_scores))
    score1 = entropy / num_infer_tokens
    
     # Word_Level without UNK
    entropy = 0
    num_infer_tokens = 0
    with open(trans_file, encoding='utf-8') as fh:
        for line in fh.readlines():
            line = _clean(line, word_option=word_option)
            tokens = line.split(" ")
            local_scores = []
            for item in tokens:
                if item.find('unk') > 0:
                    continue
                else:
                    fre = max(1, counter[item])
                p = fre / num_tokens
                tmp = -math.log(p, 2)
                local_scores += [tmp]
                entropy += tmp
                num_infer_tokens += 1.0
            scores2.append(sum(local_scores) / max(1, len(local_scores)))
    score2 = entropy / num_infer_tokens
    
    # Utterance_Level with UNK
    entropy = 0
    num_infer_lines = 0
    with open(trans_file, encoding='utf-8') as fh:
        for line in fh.readlines():
            line = _clean(line, word_option=word_option)
            tokens = line.split(" ")
            local_scores = []
            for item in tokens:
                if item.find('unk') > 0:
                    fre = unk_count
                else:
                    fre = max(1, counter[item])
                p = fre / num_tokens
                tmp = -math.log(p, 2)
                local_scores += [tmp]
                entropy += tmp
            scores3.append(sum(local_scores))
            num_infer_lines += 1.0

    score3 = entropy / num_infer_lines

    entropy = 0
    num_infer_lines = 0
    with open(trans_file, encoding='utf-8') as fh:
        for line in fh.readlines():
            line = _clean(line, word_option=word_option)
            tokens = line.split(" ")
            local_scores = []
            for item in tokens:
                if item.find('unk') > 0:
                    continue
                else:
                    fre = max(1, counter[item])
                p = fre / num_tokens
                tmp = -math.log(p, 2)
                local_scores += [tmp]
                entropy += tmp
            num_infer_lines += 1.0
            scores4.append(sum(local_scores))
    score4 = entropy / num_infer_lines
    
            
    result_dict = {
        'Entropy' :score1,
        'Entropy_Scores' : scores1,
        
        'Entropy_Known' :score2,
        'Entropy_Known_Scores' : scores2,
        
        
        'EntropyU' :score3,
        'EntropyU_Scores' : scores3,
        
        'EntropyU_Known' :score4,
        'EntropyU_Known_Scores' : scores4,
    }
    return result_dict




# Follow //transconsole/localization/machine_translation/metrics/bleu_calc.py
def _bleu(ref_file, trans_file, max_order=4, word_option=None):
    """Compute BLEU scores and handling BPE."""
    smooth = False
    ref_files = [ref_file]
    reference_text = []
    for reference_filename in ref_files:
        with open(reference_filename, 'r+', encoding='utf-8') as fh:
            reference_text.append(fh.readlines())

    per_segment_references = []
    for references in zip(*reference_text):
        reference_list = []
        for reference in references:
            reference = _clean(reference, word_option)
            reference_list.append(reference.split(" "))
        per_segment_references.append(reference_list)

#     print(per_segment_references[0:3])

    translations = []
    with open(trans_file, 'r+', encoding='utf-8') as fh:
        for line in fh.readlines():
            line = _clean(line, word_option=word_option)
            translations.append(line.split(" "))
#     print(translations[0:3])

    # bleu_score, precisions, bp, ratio, translation_length, reference_length
    bleu_score, _, _, _, _, _ = bleu.compute_bleu(
        per_segment_references, translations, max_order, smooth)

    blue_scores = []
    for ref, trans in zip(per_segment_references, translations):
        tmp_bleu_score, _, _, _, _, _ = bleu.compute_bleu(
            [ref], [trans], max_order, smooth)
        blue_scores.append(tmp_bleu_score * 100)
    
    result_dict = {
        'BLEU-%d' % max_order: 100 * bleu_score,
        'BLEU-%d_Scores' % max_order: blue_scores,
    }
    return result_dict 



def _rouge(ref_file, summarization_file, word_option=None):
    """Compute ROUGE scores and handling BPE."""

    references = []
    with open(ref_file, 'r', encoding='utf-8') as fh:
        for line in fh.readlines():
            references.append(_clean(line, word_option))

    hypotheses = []
    with open(summarization_file, 'r', encoding='utf-8') as fh:
        for line in fh.readlines():
            hypotheses.append(_clean(line, word_option=word_option))

    rouge_score_map = rouge.rouge(hypotheses, references)

    rouge_l_scores = []
    rouge_1_scores = []
    rouge_2_scores = []
    for hypo, ref in zip(hypotheses, references):
        tmp_rouge_score_map = rouge.rouge([hypo], [ref])
        rouge_l_scores.append(100 * tmp_rouge_score_map["rouge_l/f_score"])
        rouge_1_scores.append(100 * tmp_rouge_score_map["rouge_1/f_score"])
        rouge_2_scores.append(100 * tmp_rouge_score_map["rouge_2/f_score"])

    result_dict = {
        'ROUGE_L': 100 * rouge_score_map["rouge_l/f_score"],
        'ROUGE_L_Scores': rouge_l_scores,
        'ROUGE_1': 100 * rouge_score_map["rouge_1/f_score"],
        'ROUGE_1_Scores': rouge_1_scores,
        'ROUGE_2': 100 * rouge_score_map["rouge_2/f_score"],
        'ROUGE_2_Scores': rouge_2_scores,
    }
    return result_dict 



def _moses_bleu(multi_bleu_script, tgt_test, trans_file, word_option=None):
    """Compute BLEU scores using Moses multi-bleu.perl script."""

    # TODO(thangluong): perform rewrite using python
    # BPE
    if word_option == "bpe":
        debpe_tgt_test = tgt_test + ".debpe"
        if not os.path.exists(debpe_tgt_test):
            # TODO(thangluong): not use shell=True, can be a security hazard
            subprocess.call("cp %s %s" % (tgt_test, debpe_tgt_test), shell=True)
            subprocess.call("sed s/@@ //g %s" % (debpe_tgt_test),
                            shell=True)
        tgt_test = debpe_tgt_test
    elif word_option == "spm":
        despm_tgt_test = tgt_test + ".despm"
        if not os.path.exists(despm_tgt_test):
            subprocess.call("cp %s %s" % (tgt_test, despm_tgt_test))
            subprocess.call("sed s/ //g %s" % (despm_tgt_test))
            subprocess.call(u"sed s/^\u2581/g %s" % (despm_tgt_test))
            subprocess.call(u"sed s/\u2581/ /g %s" % (despm_tgt_test))
        tgt_test = despm_tgt_test
    cmd = "%s %s < %s" % (multi_bleu_script, tgt_test, trans_file)

    # subprocess
    # TODO(thangluong): not use shell=True, can be a security hazard
    bleu_output = subprocess.check_output(cmd, shell=True)

    # extract BLEU score
    m = re.search("BLEU = (.+?),", bleu_output)
    bleu_score = float(m.group(1))

    return bleu_score


if __name__ == "__main__":
    sys.path.append(os.path.dirname(sys.path[0]))
    model_id = ''
    ref_file = r"debug_model2/test_out.txt"
    # ref_file = r"D:\nmt\ref\char2_dev.response"
    trans_folder = r"debug_model2/test_out.txt"
    files = os.listdir(trans_folder)
    fout = open(sys.argv[4], 'w+')
    for file_name in files:
        if file_name[-3:] != 'tgt':
            continue
        items = file_name.split('.')
        model_id = items[0]
        beam_width = items[1]
        trans_file = trans_folder + file_name
        out_path = sys.argv[4]
        metrics = sys.argv[5].split(',')
        subword = None

        if file_name.find('bpe') > -1:
            print('bpe')

        res = ''
        scores = []
        for metric in metrics:
            if file_name.find('bpe') > -1:
                metric = metric + '@bpe'
            score = evaluate(ref_file, trans_file, metric, word_option=subword)
            res += ('%s\t%f\n') % (metric, score)
            scores.append(str(score))
        print(res)
        fout.write('%s\t%s\t%s\t%s\n' % (file_name, model_id, beam_width, '\t'.join(scores)))