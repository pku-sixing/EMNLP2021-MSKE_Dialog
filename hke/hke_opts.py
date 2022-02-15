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
    p.add_argument('-dataset_version', type=str, default='hke',
                   help='running mode')
    p.add_argument('-debug', type=str, default='False',
                   help='debug_mode')
    p.add_argument('-ablation', type=str, default='none',
                   help='debug_mode')
    p.add_argument('-wait_gpu', type=int, default=-1,
                   help='number of epochs for train')
    p.add_argument('-seed', type=int, default=12345,
                   help='number of epochs for train')
    p.add_argument('-naive_baseline_mode', type=str2bool, default=False,
                   help='softmax mode if false, else sigmoid mode')


    # Knowledge
    p.add_argument('-knowledge', type=str, default='context')
    p.add_argument('-word_encoder', type=str, default='none')
    p.add_argument('-uniform_word_encoding', type=str, default='word')
    p.add_argument('-word_encoder_type', type=str, default='mean')
    p.add_argument('-mask_copy_flags', type=str, default='') # box,text,csk
    p.add_argument('-share_knowledge_copy_attention', type=str2bool, default=True,
                   help='share_the_copy_attention')
    p.add_argument('-add_src_word_embed_to_state', type=str2bool, default=True,
                   help='share_the_copy_attention')
    p.add_argument('-init_selection_mode', type=str, default="none",
                   choices=["none", "prior", "posterior"])
    p.add_argument('-enable_global_mode_gate', type=str2bool, default=False,
                   help='set knowledge to 0 if no knowledge has been provided')
    p.add_argument('-disable_sigmoid_gate', type=str2bool, default=False,
                   help='set knowledge to 0 if no knowledge has been provided')
    p.add_argument('-softmax_global_gate_fusion', type=str2bool, default=False,
                   help='softmax mode if false, else sigmoid mode')

    p.add_argument('-global_mode_gate', type=str, default='default',
                   help='set knowledge to 0 if no knowledge has been provided')


    # Commonsense
    p.add_argument('-fact_dict_path', type=str, default='data/naf_fact_vocab_demo.txt',
                   help='field word vocab_path')
    p.add_argument('-csk_relation_vocab_path', type=str, default='data/relation_vocab.txt',
                   help='field word vocab_path')
    p.add_argument('-csk_entity_vocab_path', type=str, default='data/entity_vocab.txt',
                   help='field word vocab_path')
    p.add_argument('-csk_entity_embed_path', type=str, default='data/entity2vec.vec',
                   help='field word vocab_path')
    p.add_argument('-csk_relation_embed_path', type=str, default='data/relation2vec.vec',
                   help='field word vocab_path')
    p.add_argument('-max_csk_num', type=int, default=202,
                   help='number of epochs for train')

    # Text
    p.add_argument('-text_knowledge_trans_layers', type=int, default=2,
                   help='number of epochs for train')
    p.add_argument('-max_text_token_num', type=int, default=102,
                   help='number of epochs for train')


    # Infobox
    p.add_argument('-max_kw_pairs_num', type=int, default=70,
                   help='number of epochs for train')
    p.add_argument('-max_field_intra_word_num', type=int, default=200,
                   help='number of epochs for train')
    p.add_argument('-field_key_vocab_path', type=str, default='data/field_vocab.txt',
                   help='field_key_vocab_path')
    p.add_argument('-field_word_vocab_path', type=str, default='data/field_word.txt',
                   help='field word vocab_path')
    p.add_argument('-field_key_tag_path', type=str, default='none',
                   help='field word pos tag vocab_path')
    p.add_argument('-field_word_tag_path', type=str, default='none',
                   help='field word pos tag vocab_path')
    p.add_argument('-field_tag_usage',
              type=str, default='none',
              choices=['none'],
              help="")

    # Train
    p.add_argument('-init_word_vecs', type=str2bool, default=False,
                   help='Use the pre-trained word embedding ')
    p.add_argument('-report_steps', type=int, default=500,
                   help='number of steps for reporting results')
    p.add_argument('-model_path', type=str, default='model')
    p.add_argument('-unk_learning', type=str, default="none",
                   choices=["none", "skip", "penalize"])
    p.add_argument('-cuda', type=str2bool, default=False,
                   help='use cuda or not')
    p.add_argument('-epochs', type=int, default=20,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=32,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0001,
                   help='standard learning rate')
    p.add_argument('-start_to_adjust_lr', type=int, default=2,
                   help='standard learning rate')
    p.add_argument('-init_lr', type=float, default=-1.0,
                   help='initial learning rate, -1.0 is disable')
    p.add_argument('-init_lr_decay_epoch', type=int, default=6,
                   help='number of epochs for train')

    p.add_argument('-lr_decay_rate', type=float, default=0.5,
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
    p.add_argument('-char_vocab_path', type=str, default='data/vocab.txt',
                   help='vocab_path, available if share_vocab')
    p.add_argument('-src_vocab_path', type=str, default='data/vocab.txt',
                   help='src_vocab_path, available if not share_vocab')
    p.add_argument('-tgt_vocab_path', type=str, default='data/vocab.txt',
                   help='tgt_vocab_path, available if not share_vocab')
    p.add_argument('-share_vocab', type=str2bool, default=True,
                   help='src and tgt share a vocab')
    p.add_argument('-share_embedding', type=str2bool, default=True,
                   help='src and tgt share an embedding')
    p.add_argument('-max_src_len', type=int, default=35,
                   help='max_src_len')
    p.add_argument('-max_tgt_len', type=int, default=35,
                   help='max_tgt_len')
    p.add_argument('-max_line', type=int, default=-1,
                   help='max loaded line number, -1 means do not restrict')


    # Infer
    p.add_argument('-repeat_index', type=int, default=0,
                   help='repeat_number')
    p.add_argument('-infer_max_len', type=int, default=35,
                   help='infer_max_len')
    p.add_argument('-infer_min_len', type=int, default=3,
                   help='infer_min_len')
    p.add_argument('-penalize_repeat_tokens_rate', type=float, default=1.0,
                   help='')
    p.add_argument('-mlp_vocab_mode', type=str2bool, default=False,
                   help='')
    p.add_argument('-disable_multiple_copy', type=str2bool, default=False,
                   help='')
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
    p.add_argument("--pre_embed_file", type=str, default="dataset/embed/tencent.txt",
                        help="enable binary selector")
    p.add_argument("--pre_embed_dim", type=int, default=200, help="enable binary selector")
    p.add_argument("--subword", type=str, default=None)
    p.add_argument('-beam_length_penalize', type=str, default='none',
                   choices=['none', 'avg'], help='beam_length_penalize')
    # Model
    p.add_argument('-bow_loss', type=float, default=0.0,
                   help='bow_loss rate')
    p.add_argument('-posterior_prior_loss', type=float, default=1.0,
                   help='bow_loss rate')
    p.add_argument('-dropout', type=float, default=0.5,
                   help='dropout value')
    p.add_argument('-align_dropout', type=float, default=0.0,
                   help='dropout value')
    p.add_argument('-hidden_size', type=int, default=512,
                   help='size of a hidden layer')
    p.add_argument('-embed_size', type=int, default=200, help='size of word embeddings')
    p.add_argument('-rnn_type', type=str, default='gru',
                   choices=['lstm', 'gru'],
                   help='size of word embeddings')
    p.add_argument('-init', type=str, default='xavier',
                   choices=['xavier', 'xavier_normal', 'uniform', 'kaiming', 'kaiming_normal'],
                   help='size of word embeddings')

    # Encoder
    p.add_argument('-enc_layers', type=int, default=1,
                   help='number of encoder layers')
    p.add_argument('-enc_birnn', type=str2bool, default=True,
                   help='use cuda or not')


    # Bridge
    p.add_argument('-bridge', type=str, default='none',
                   choices=['none', 'general','fusion'],
                   help='size of word embeddings')


    # Decoder
    p.add_argument('-mode_selector',
              type=str, default='general',
              choices=['general', 'mlp'],
              help="Mode Selector")
    p.add_argument('-attn_type',
              type=str, default='dot',
              choices=['dot', 'dot_val'],
              help="The attention type to use: "
                   "dotprod or general (Luong) or MLP (Bahdanau)")
    p.add_argument( '-attn_func',
              type=str, default="softmax", choices=["softmax", "sparsemax"])
    p.add_argument('-dec_layers', type=int, default=1,
                   help='number of decoder layers')

    p.add_argument('-teach_force_rate', type=float, default=1.0,
                   help='teach_force_rate, 1.0 means use the ground truth, 0.0 means use the predicted tokens')
    p.add_argument('-copy', type=str2bool, default=True,
                   help='enable pointer-copy')
    p.add_argument('-copy_strategy', type=str, default='default',
                   help='how to copy', choices=["default", "fusion", "dynamic_vocab"])
    p.add_argument('-max_copy_token_num', type=int, default=40,
                   help='it should be greater than maximum src len')
    p.add_argument('-max_dynamic_vocab_size', type=int, default=300,
                   help='it should be greater than maximum src len')
    p.add_argument('-copy_coverage', type=float, default=-1,
                   help='copy coverage loss rate, 0.0 = disabled')
    p.add_argument('-copy_attn_type',
              type=str, default='mlp',
              choices=['dot', 'general', 'mlp', 'none'],
              help="The copy attention type to use: "
                   "dotprod or general (Luong) or MLP (Bahdanau)")
    p.add_argument( '-copy_attn_func',
              type=str, default="softmax", choices=["softmax", "sparsemax"])


    return p.parse_args()

