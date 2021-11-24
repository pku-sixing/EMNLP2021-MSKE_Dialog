import time
import argparse
from collections import defaultdict

from seq2seq import data_utils
from utils.evaluation import evaluation_agent

from utils.evaluation.evaluation_helper import ScoreManager


def std_eval_with_args(args, file_prefix, mode='test'):
    if mode == 'test':
        ref_src_file = args.test_data_path_prefix + '.' + data_utils.SRC_SUFFIX
        ref_tgt_file = args.test_data_path_prefix + '.' + data_utils.TGT_SUFFIX
    elif mode == 'val':
        ref_src_file = args.val_data_path_prefix + '.' + data_utils.SRC_SUFFIX
        ref_tgt_file = args.val_data_path_prefix + '.' + data_utils.TGT_SUFFIX
    else:
        raise NotImplementedError()

    train_src_file = args.train_data_path_prefix + '.' + data_utils.SRC_SUFFIX
    train_tgt_file = args.train_data_path_prefix + '.' + data_utils.TGT_SUFFIX
    top1_out_file_path = file_prefix + '.top1.txt'
    topk_out_file_path = file_prefix + '.topk.txt'
    return std_eval(ref_src_file, ref_tgt_file, train_src_file, train_tgt_file,
                    top1_out_file_path, topk_out_file_path,
                    pre_embed_file=args.pre_embed_file, pre_embed_dim=args.pre_embed_dim,
                    subword=args.subword, vocab_size=args.tgt_vocab_size, beam_width=args.beam_width)


def std_eval(test_src_path, top1_out_file_path, fact_dict_path, text_base_path, infobox_base_path,
             beam_width, stop_words_path):
    # StopWords
    with open(stop_words_path, 'r+', encoding='utf-8') as fin:
        stop_words = [x.strip('\r\n') for x in fin.readlines()]
        stop_words = set(stop_words)

    # Results
    result_dict = {}
    with open(test_src_path, 'r+', encoding='utf-8') as fin:
        srcs = [x.strip('\r\n').split() for x in fin.readlines()]
    # Results
    with open(top1_out_file_path, 'r+', encoding='utf-8') as fin:
        tgts = [x.strip('\r\n').split() for x in fin.readlines()]
    # Entity Scores
    with open(fact_dict_path, 'r+', encoding='utf-8') as fin:
        facts = [x.strip('\r\n').split() for x in fin.readlines()]
        head_entities = defaultdict(set)
        for fact in facts:
            head_entity = fact[0]
            tail_entity = fact[1]
            head_entities[head_entity].add(tail_entity)
        two_hop_entities = defaultdict(set)
        for head in head_entities:
            for tail in head_entities[head]:
                two_hop_entities[head].add(tail)
                if tail in head_entities:
                    two_hop_entities[head] |= head_entities[tail]
    one_hop_entity_scores = []
    two_hop_entity_scores = []
    for src, line in zip(srcs, tgts):
        one_hop_entity_set = set()
        two_hop_entity_set = set()
        my_head = set(src)
        my_tail = set(line) - stop_words
        for word in my_head:
            if word in head_entities:
                one_hop_entity_set = one_hop_entity_set | head_entities[word]
        two_hop_entity_set = two_hop_entity_set | one_hop_entity_set
        for word in my_head:
            if word in two_hop_entities:
                two_hop_entity_set = two_hop_entity_set | two_hop_entities[word]
        one_hop_entity_score = len(one_hop_entity_set & my_tail)
        one_hop_entity_scores.append(one_hop_entity_score)
        two_hop_entity_score = len(two_hop_entity_set & my_tail - my_head)
        two_hop_entity_scores.append(two_hop_entity_score)
    one_hop_entity_score = sum(one_hop_entity_scores) / len(one_hop_entity_scores)
    two_hop_entity_score = sum(two_hop_entity_scores) / len(two_hop_entity_scores)
    print(one_hop_entity_score, two_hop_entity_score)
    result_dict['ONE_Entity'] = one_hop_entity_score
    result_dict['ONE_Entity_Scores'] = one_hop_entity_scores
    result_dict['TWO_Entity'] = two_hop_entity_score
    result_dict['TWO_Entity_Scores'] = two_hop_entity_scores
    # Text
    # Results
    with open(text_base_path, 'r+', encoding='utf-8') as fin:
        texts = [x.strip('\r\n').split() for x in fin.readlines()]
    text_scores = []
    for src, tgt, text in zip(srcs, tgts, texts):
        text_score = len((set(tgt) & set(text)) - stop_words )
        text_scores.append(text_score)
    text_score = sum(text_scores) / len(text_scores)
    print(text_score)
    result_dict['TEXT_GRAM'] = text_score
    result_dict['TEXT_GRAM_Scores'] = text_scores
    scores_suffix = '_Scores'
    scores_suffix_len = len(scores_suffix)
    scores_len = -1
    # for metric in ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4', 'rouge']:
    #     old_scores = evaluation_agent.evaluate(text_base_path, text_base_path, top1_out_file_path,
    #                                          None, metric, dim=0, word_option=None)
    #     scores = dict()
    #     for key in old_scores:
    #         scores['TEXT2_'+key] = old_scores[key]
    #     for key in scores:
    #         assert key not in result_dict
    #         result_dict[key] = scores[key]
    #         if key[-scores_suffix_len:] != scores_suffix:
    #             print('%s->%s' % (key, scores[key]))
    #         else:
    #             if scores_len == -1:
    #                 scores_len = len(scores[key])
    #             else:
    #                 assert scores_len == len(scores[key]), metric + '/' + key
    #Infobox
    with open(infobox_base_path, 'r+', encoding='utf-8') as fin:
         boxes = [set([y.split(':')[1] for y in x.strip('\r\n').split()][1:]) for x in fin.readlines()]
         infobox_scores = []
         for box, tgt, src in zip(boxes, tgts, srcs):
            infobox_score = len((box & set(tgt)) - stop_words)
            infobox_scores.append(infobox_score)
         infobox_score = sum(infobox_scores) / len(infobox_scores)
         print(infobox_score, len(infobox_scores))
    result_dict['INF_ENTITY'] = infobox_score
    result_dict['INF_ENTITY_Scores'] = infobox_scores

    knowledge_scores = []
    for txt,csk,ibt in zip(text_scores, one_hop_entity_scores, infobox_scores):
        knowledge_scores.append((txt+csk+ibt)/3.0)

    knowledge_score = sum(knowledge_scores) / len(knowledge_scores)
    result_dict['KNOWLEDGE'] = knowledge_score
    result_dict['KNOWLEDGE_Scores'] = knowledge_scores
    print('knowledge_score', knowledge_score)





    return result_dict


def main(args):
    score_manager = ScoreManager(args.res_path, 'evaluation_score')
    top1_out_file_path = args.test_file
    fact_vocab_path = args.fact_vocab_path
    test_src = args.test_src
    text_base_path = args.text_base_path
    infobox_base_path = args.infobox_base_path
    group_name = args.group_name
    beam_width = args.beam_width
    stop_words = args.stop_words
    res_dict = std_eval(test_src, top1_out_file_path, fact_vocab_path, text_base_path, infobox_base_path, beam_width,
                        stop_words)
    score_manager.update_group(group_name, res_dict)


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_json", type=str, default="/Users/mebiuw/Downloads/evaluation_score.json", help="enable binary selector")
    parser.add_argument("--test_src", type=str, default="/Users/mebiuw/Downloads/test.src", help="enable binary selector")
    parser.add_argument("--test_file", type=str, default="/Users/mebiuw/Downloads/test_B0_D0.00_masked.top1.txt", help="enable binary selector")
    parser.add_argument("--group_name", type=str, default="test_B0_D0.00_masked.top1", help="enable binary selector")
    parser.add_argument("--fact_vocab_path", type=str, default="/Users/mebiuw/PycharmProjects/torch_seq2seq/hke/data/naf_fact_vocab.txt", help="enable binary selector")
    parser.add_argument("--text_base_path", type=str, default="/Users/mebiuw/Downloads/test.text", help="enable binary selector")
    parser.add_argument("--infobox_base_path", type=str, default="/Users/mebiuw/Downloads/test.box", help="enable binary selector")
    parser.add_argument("--res_path", type=str, default="/Users/mebiuw/Downloads/demo", help="enable binary selector")
    parser.add_argument("--vocab_path", type=str, default="/home/sixing/dataset/MultiSource/HKE/vocab.txt", help="enable binary selector")
    # parser.add_argument("--vocab_path", type=str, default="/Users/mebiuw/Downloads/demo", help="enable binary selector")
    parser.add_argument("--beam_width", type=int, default=0, help="/Users/mebiuw/Downloads/vocab.txt")
    # parser.add_argument("--stop_words", type=str, default="/home/sixing/dataset/MultiSource/HKE/vocab.txt")
    parser.add_argument("--stop_words", type=str, default="Datasets/HeterogeneousDataset/stopwords.txt")
    args = parser.parse_args()

    main(args)
    print('Evaluation Time Consuming : %d' % (time.time() - start_time))
