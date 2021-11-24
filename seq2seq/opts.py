import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')

    # Global
    p.add_argument('-mode', type=str, default='train',
                   choices=['train', 'eval', 'infer'],
                   help='running mode')

    # Train
    p.add_argument('-report_steps', type=int, default=200,
                   help='number of steps for reporting results')
    p.add_argument('-model_path', type=str, default='model')
    p.add_argument('-unk_learning', type=str, default="none",
                   choices=["none", "skip", "penalize"])
    p.add_argument('-cuda', type=str2bool, default=False,
                   help='use cuda or not')
    p.add_argument('-epochs', type=int, default=200,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=64,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0001,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=5.0,
                   help='in case of gradient explosion')

    # Dataset
    p.add_argument('-train_data_path_prefix', type=str, default='data/test',
                   help='the data path and prefix for the training files')
    p.add_argument('-val_data_path_prefix', type=str, default='data/test',
                   help='the data path and prefix for the validation files')
    p.add_argument('-test_data_path_prefix', type=str, default='data/test',
                   help='the data path and prefix for the test files')
    p.add_argument('-vocab_path', type=str, default='data/vocab.txt',
                   help='vocab_path, available if share_vocab')
    p.add_argument('-init_word_vecs', type=str2bool, default=False,
                   help='Use the pre-trained word embedding ')
    p.add_argument('-src_vocab_path', type=str, default='data/vocab.txt',
                   help='src_vocab_path, available if not share_vocab')
    p.add_argument('-tgt_vocab_path', type=str, default='data/vocab.txt',
                   help='tgt_vocab_path, available if not share_vocab')
    p.add_argument('-share_vocab', type=str2bool, default=True,
                   help='src and tgt share a vocab')
    p.add_argument('-share_embedding', type=str2bool, default=True,
                   help='src and tgt share an embedding')
    p.add_argument('-max_src_len', type=int, default=25,
                   help='max_src_len')
    p.add_argument('-max_tgt_len', type=int, default=25,
                   help='max_tgt_len')


    # Infer
    p.add_argument('-infer_max_len', type=int, default=25,
                   help='infer_max_len')
    p.add_argument('-infer_min_len', type=int, default=3,
                   help='infer_min_len')
    p.add_argument('-disable_unk_output', type=str2bool, default=False,
                   help='mask the output of unks')
    p.add_argument('-use_best_model', type=str2bool, default=True,
                   help='mask the output of unks')
    p.add_argument('-skip_infer', type=str2bool, default=False,
                   help='only evaluating the result')
    p.add_argument('-beam_width', type=int, default=10,
                   help='beam search width')
    p.add_argument('-diverse_decoding', type=float, default=0.0,
                   help='mask the output of unks')
    p.add_argument("--pre_embed_file", type=str, default="/home/sixing/dataset/embed/tencent.txt",
                        help="enable binary selector")
    p.add_argument("--pre_embed_dim", type=int, default=200, help="enable binary selector")
    p.add_argument("--subword", type=str, default=None)
    p.add_argument('-beam_length_penalize', type=str, default='none',
                   choices=['none', 'avg'], help='beam_length_penalize')

    # Model
    p.add_argument('-dropout', type=float, default=0.5,
                   help='dropout value')
    p.add_argument('-hidden_size', type=int, default=512,
                   help='size of a hidden layer')
    p.add_argument('-embed_size', type=int, default=300,
                   help='size of word embeddings')
    p.add_argument('-rnn_type', type=str, default='gru',
                   choices=['lstm', 'gru'],
                   help='size of word embeddings')
    p.add_argument('-init', type=str, default='xavier',
                   choices=['xavier', 'xavier_normal', 'uniform', 'kaiming', 'kaiming_normal'],
                   help='size of word embeddings')

    # Encoder
    p.add_argument('-enc_layers', type=int, default=2,
                   help='number of encoder layers')
    p.add_argument('-enc_birnn', type=str2bool, default=True,
                   help='use cuda or not')

    # Bridge
    p.add_argument('-bridge', type=str, default='none',
                   choices=['none', 'general'],
                   help='size of word embeddings')

    # Decoder

    p.add_argument('-attn_type',
              type=str, default='general',
              choices=['dot', 'general', 'mlp', 'none'],
              help="The attention type to use: "
                   "dotprod or general (Luong) or MLP (Bahdanau)")
    p.add_argument( '-attn_func',
              type=str, default="softmax", choices=["softmax", "sparsemax"])
    p.add_argument('-dec_layers', type=int, default=2,
                   help='number of decoder layers')
    p.add_argument('-add_last_generated_token', type=str2bool, default=False,
                   help='add_last_generated_token')
    p.add_argument('-copy', type=str2bool, default=False,
                   help='enable pointer-copy')
    p.add_argument('-field_copy', type=str2bool, default=False,
                   help='placeholder only')
    p.add_argument('-max_copy_token_num', type=int, default=30,
                   help='it should be greater than maximum src len')
    p.add_argument('-copy_coverage', type=float, default=-1,
                   help='copy coverage loss rate, <= 0.0 disabled')
    p.add_argument('-mode_loss', type=float, default=-1,
                   help='copy coverage loss rate, <= 0.0 disabled')
    p.add_argument('-copy_attn_type',
              type=str, default='mlp',
              choices=['dot', 'general', 'mlp', 'none'],
              help="The copy attention type to use: "
                   "dotprod or general (Luong) or MLP (Bahdanau)")
    p.add_argument( '-copy_attn_func',
              type=str, default="softmax", choices=["softmax", "sparsemax"])


    return p.parse_args()

