import time
import argparse

from seq2seq import data_utils
from utils.evaluation import evaluation_agent
from multiprocessing import Pool

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


def std_eval(ref_src_file, ref_tgt_file, train_src_file, train_tgt_file,
             top1_out_file_path, topk_out_file_path,
             pre_embed_file, pre_embed_dim, subword, vocab_size,beam_width):
    metrics = 'unk,unk_sentence,rouge,bleu-1,bleu-2,bleu-3,bleu-4,' \
              'intra_distinct-1,intra_distinct-2,distinct-1,distinct-2,distinct_c-1,distinct_c-2,' \
              'len,entropy,std_ent'.split(',')
    thread_pool = Pool(8)
    jobs = []
    for metric in metrics:
        if metric[:len('intra_distinct')] == 'intra_distinct':
            job = thread_pool.apply_async(evaluation_agent.evaluate, (
                ref_tgt_file, ref_src_file, topk_out_file_path,
                pre_embed_file, metric, pre_embed_dim, None, subword, beam_width))
        elif metric == 'entropy':
            job = thread_pool.apply_async(evaluation_agent.evaluate, (
                train_tgt_file, train_src_file, top1_out_file_path, pre_embed_file, 'entropy',
                pre_embed_dim, vocab_size, subword))
        else:
            job = thread_pool.apply_async(evaluation_agent.evaluate, (
                ref_tgt_file, ref_src_file, top1_out_file_path,
                pre_embed_file, metric, pre_embed_dim, None, subword, beam_width))
        jobs.append(job)

    thread_pool.close()
    thread_pool.join()

    result_dict = {}
    embed_scores = evaluation_agent.evaluate(ref_tgt_file, ref_src_file, top1_out_file_path,
                                             pre_embed_file, 'embed', dim=pre_embed_dim, word_option=subword)

    scores_suffix = '_Scores'
    scores_suffix_len = len(scores_suffix)
    scores_len = -1

    for key in embed_scores:
        assert key not in result_dict
        result_dict[key] = embed_scores[key]
        if key[-scores_suffix_len:] != scores_suffix:
            print('%s->%s' % (key, embed_scores[key]))
        else:
            if scores_len == -1:
                scores_len = len(embed_scores[key])
            else:
                assert scores_len == len(embed_scores[key]), 'embed/'+key

    for job, metric in zip(jobs, metrics):
        scores = job.get()
        for key in scores:
            assert key not in result_dict
            result_dict[key] = scores[key]
            if key[-scores_suffix_len:] != scores_suffix:
                print('%s->%s' % (key, scores[key]))
            else:
                if scores_len == -1:
                    scores_len = len(scores[key])
                else:
                    assert scores_len == len(scores[key]), metric + '/' + key
    return result_dict


def main(args):
    score_manager = ScoreManager(args.res_path, 'evaluation_score')

    ref_src_file = args.ref_src_file
    ref_tgt_file = args.ref_tgt_file
    train_src_file = args.train_src_file
    train_tgt_file = args.train_tgt_file
    top1_out_file_path = args.test_file
    topk_out_file_path = args.test_file_topk
    pre_embed_file = args.pre_embed_file
    pre_embed_dim = args.pre_embed_dim
    vocab_size = args.vocab_size
    subword = args.subword

    res_dict = std_eval(ref_src_file, ref_tgt_file, train_src_file, train_tgt_file,
             top1_out_file_path, topk_out_file_path,
             pre_embed_file, pre_embed_dim, subword, vocab_size, args.beam_width)

    score_manager.update_group('test_B%d_D0.00_std' % args.beam_width, res_dict)


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_path", type=str, default="",
                        help="enable binary selector")
    parser.add_argument("--pre_embed_file", type=str, default="/home/sixing/dataset/embed/tencent.txt",
                        help="enable binary selector")
    parser.add_argument("--test_file", type=str, default="", help="enable binary selector")
    parser.add_argument("--test_file_topk", type=str, default="", help="enable binary selector")
    parser.add_argument("--ref_src_file", type=str,
                        default="/home/sixing/dataset/DialogwithSenFunAbnormal/word_level/test.src",
                        help="training src file")
    parser.add_argument("--ref_tgt_file", type=str,
                        default="/home/sixing/dataset/DialogwithSenFunAbnormal/word_level/test.tgt",
                        help="enable binary selector")
    parser.add_argument("--train_src_file", type=str,
                        default="/home/sixing/dataset/DialogwithSenFunAbnormal/word_level/train.src",
                        help="enable binary selector")
    parser.add_argument("--train_tgt_file", type=str,
                        default="/home/sixing/dataset/DialogwithSenFunAbnormal/word_level/train.tgt",
                        help="enable binary selector")
    parser.add_argument("--fact_vocab", type=str, default="", help="enable binary selector")
    parser.add_argument("--test_fact_path", type=str, default="", help="enable binary selector")
    parser.add_argument("--pre_embed_dim", type=int, default=200, help="enable binary selector")
    parser.add_argument("--vocab_size", type=int, default=50000, help="enable binary selector")
    parser.add_argument("--thread", type=int, default=16, help="thread")
    parser.add_argument("--beam_width", type=int, default=0, help="beam_width")
    parser.add_argument("--subword", type=str, default=None)
    args = parser.parse_args()

    main(args)
    print('Evaluation Time Consuming : %d' % (time.time() - start_time))
