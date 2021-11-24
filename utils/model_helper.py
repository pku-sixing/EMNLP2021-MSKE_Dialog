from collections import defaultdict
import torch.nn as nn
import torch
from hke import  hke_data_utils
from utils.logger import logger
import os


# 权重初始化，默认xavier


def adjust_learning_rate(optimizer, rate=0.5, min_value=None):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    assert len(optimizer.param_groups) == 1
    for param_group in optimizer.param_groups:
        new_lr = param_group['lr'] * rate
        if min_value is not None:
            new_lr = max(new_lr, min_value)
        logger.info('[LEARNING RATE] adjusting %f to %f ' % (param_group['lr'], new_lr))
        param_group['lr'] = new_lr
        return param_group['lr']


def name_a_generation(args, mode):
    test_file_name = args.test_data_path_prefix
    if test_file_name.find('/') > -1:
        mode = test_file_name.split('/')[-1]
    else:
        mode = test_file_name.split('\\')[-1]
    print('mode name => ', mode)

    if args.disable_unk_output:
        mask_unk = 'masked'
    else:
        mask_unk = 'std'
    try:
        if args.random_test_field_order:
            mask_unk += '_roder'
        if args.beam_length_penalize == 'avg':
            mask_unk += '_avg'
        if args.repeat_index > 0:
            mask_unk += '_%d' % args.repeat_index

    except:
        pass
    return '%s_B%d_D%.2f_%s' % (mode, args.beam_width, args.diverse_decoding, mask_unk)


def reset_learning_rate(optimizer, lr_rate):
    assert len(optimizer.param_groups) == 1
    for param_group in optimizer.param_groups:
        new_lr = lr_rate
        logger.info('[LEARNING RATE] adjusting %f to %f ' % (param_group['lr'], new_lr))
        param_group['lr'] = new_lr
        return param_group['lr']


def show_parameters(model):
    trainable_param_counter = defaultdict(float)
    logger.info('Trainable Parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            prefix = name.split('.')[0]
            trainable_param_counter[prefix] += param.nelement()
            logger.info('{}-{}-{}-{}'.format(name, param.shape, param.dtype, param.device))
    logger.info('-------------')
    trainable_sum = 0
    for key in trainable_param_counter.keys():
        logger.info('[PARAMS-COUNTING] #%s:%.2fM' % (key, trainable_param_counter[key] / 1e6))
        trainable_sum += trainable_param_counter[key]
    logger.info('[PARAMS-SUM] #%s:%.2fM' % ('Trainable', trainable_sum / 1e6))

    non_trainable_param_counter = defaultdict(float)
    logger.info('###########')
    logger.info('Non-Trainable Parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad is False:
            prefix = name.split('.')[0]
            non_trainable_param_counter[prefix] += param.nelement()
            logger.info('{}-{}-{}-{}'.format(name, param.shape, param.dtype, param.device))
    logger.info('-------------')
    non_trainable_sum = 0
    for key in non_trainable_param_counter.keys():
        logger.info('[PARAMS-COUNTING] #%s:%.2fM' % (key, non_trainable_param_counter[key] / 1e6))
        non_trainable_sum += non_trainable_param_counter[key]
    logger.info('[PARAMS-SUM] #%s:%.2fM' % ('Non-Trainable', non_trainable_sum / 1e6))
    logger.info('-------------')
    logger.info('[PARAMS-SUM] #%s:%.2fM' % ('Total', (trainable_sum + non_trainable_sum) / 1e6))


def try_restore_model(model_path, model, optimizer, states, best_model):
    if best_model:
        model_path = os.path.join(model_path, 'best_model')
    else:
        model_path = os.path.join(model_path, 'check_point')
    if not os.path.isdir(model_path):
        logger.info('[CHECKPOINT] No checkpoint is found! Dir is not existed :%s' % model_path)
        return False
    files = os.listdir(model_path)
    files = sorted(files, reverse=False)
    for file in files:
        if file[-3:] == '.pt':
            model_name = '%s/%s' % (model_path, file)
            checkpoint = torch.load(model_name)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(checkpoint['model'])
            for key in checkpoint['states']:
                states[key] = checkpoint['states'][key]
            logger.info('[CHECKPOINT] Loaded params from  :%s' % model_name)
            return True
    logger.info('[CHECKPOINT] No checkpoint is found in :%s' % model_path)
    return False


def save_model(model_path, epoch, val_loss, model, optimizer, arguments, states, best_model=True, clear_history=True):
    if best_model:
        model_path = os.path.join(model_path, 'best_model')
    else:
        model_path = os.path.join(model_path, 'check_point')
    if not os.path.isdir(model_path):
        logger.info('[CHECKPOINT] Creating model file:%s' % model_path)
        os.makedirs(model_path)
    model_name = '%s/seq2seq_%d_%d.pt' % (model_path, epoch, int(val_loss * 100))

    arguments_dict = {}
    for key, value in vars(arguments).items():
        arguments_dict[key] = value

    model_state = {
        'arguments': arguments_dict,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'states': states
    }
    torch.save(model_state, model_name)

    logger.info('[CHECKPOINT] Model has been saved to :%s' % model_name)
    if clear_history:
        logger.info('[CHECKPOINT] Removing old checkpoints')
        files = os.listdir(model_path)
        for file in files:
            file_name = '%s/%s' % (model_path, file)
            if file_name != model_name:
                logger.info('[CHECKPOINT] Removing %s' % file_name)
                try:
                    os.remove(file_name)
                except Exception as e:
                    print(e)


def write_results(args, file_prefix, output_queries, output_responses, output_scores, output_generations, raw_fields=None):
    if raw_fields is None:
        raw_fields = [None] * len(output_queries)
    with open(file_prefix + '.top1.txt', 'w+', encoding='utf-8') as ftop1:
        if args.copy or args.field_copy:
            for field, query, generations in zip(raw_fields, output_queries, output_generations):
                rst_generation = data_utils.restore_field_copy_example(field, query, generations[0])
                ftop1.write('%s\n' % rst_generation)
        else:
            for query, generations in zip(output_queries, output_generations):
                if len(generations) < 1:
                    continue
                ftop1.write('%s\n' % generations[0])

    with open(file_prefix + '.topk.txt', 'w+', encoding='utf-8') as ftopk:
        if args.copy or args.field_copy:
            for field, query, generations in zip(raw_fields, output_queries, output_generations):
                for generation in generations:
                    rst_generation = data_utils.restore_field_copy_example(field, query, generation)
                    ftopk.write('%s\n' % rst_generation)
        else:
            for generations in output_generations:
                for generation in generations:
                    ftopk.write('%s\n' % generation)

    with open(file_prefix + '.detail.txt', 'w+', encoding='utf-8') as fout:
        idx = 0
        for field, query, generation_beam, response, score_beam in zip(raw_fields, output_queries, output_generations,
                                                                output_responses, output_scores):
            if args.copy or args.field_copy:
                fout.write('#%d\n' % idx)
                fout.write('Query:\t%s\n' % query)
                fout.write('Response:\t%s\n' % response)
                for generation, score in zip(generation_beam, score_beam):
                    fout.write('Generation:\t%.4f\t%s (%s)\n' % (
                        score, data_utils.restore_field_copy_example(field, query, generation), generation))
                idx += 1
            else:
                fout.write('#%d\n' % idx)
                fout.write('Query:\t%s\n' % query)
                fout.write('Response:\t%s\n' % response)
                for generation, score in zip(generation_beam, score_beam):
                    fout.write('Generation:\t%.4f\t%s\n' % (score, generation))
                idx += 1


def write_results_hke(args, file_prefix, output_queries, output_responses, output_scores, output_generations,
                      raw_fields=None,raw_texts=None, raw_csks=None, dynamic_vocabs=None):
    if raw_fields is None:
        raw_fields = [None] * len(output_queries)
    if raw_texts is None:
        raw_texts = [None] * len(output_queries)
    if raw_csks is None:
        raw_csks = [None] * len(output_queries)
    if dynamic_vocabs is None:
        dynamic_vocabs = [None] * len(output_queries)
    with open(file_prefix + '.top1.txt', 'w+', encoding='utf-8') as ftop1:
        for dynamic_vocab, csk, field, text, query, generations in zip(dynamic_vocabs,raw_csks, raw_fields, raw_texts, output_queries, output_generations):
            # csk = [x.split()[1] if x is not None else 'None' for x in csk]
            rst_generation = hke_data_utils.restore_field_copy_example(field, text, csk, query, generations[0], dynamic_vocab=dynamic_vocab)
            ftop1.write('%s\n' % rst_generation)

    with open(file_prefix + '.topk.txt', 'w+', encoding='utf-8') as ftopk:
        for dynamic_vocab, csk, field, text, query, generations in zip(dynamic_vocabs, raw_csks, raw_fields, raw_texts, output_queries, output_generations):
            for generation in generations:
                # csk = [x.split()[1] if x is not None else 'None' for x in csk]
                rst_generation = hke_data_utils.restore_field_copy_example(field, text, csk, query, generation, dynamic_vocab=dynamic_vocab)
                ftopk.write('%s\n' % rst_generation)


    with open(file_prefix + '.detail.txt', 'w+', encoding='utf-8') as fout:
        idx = 0
        for dynamic_vocab, csk, field, text, query, generation_beam, response, score_beam in zip(dynamic_vocabs,raw_csks, raw_fields, raw_texts, output_queries, output_generations,
                                                                output_responses, output_scores):

            fout.write('#%d\n' % idx)
            fout.write('Query:\t%s\n' % query)
            fout.write('Response:\t%s\n' % response)
            fout.write('Infobox:\t%s\n' % field)
            fout.write('CSK:\t%s\n' % csk)
            fout.write('Text:\t%s\n' % text)
            fout.write('DynamicVocabs:\t%s\n' % ' '.join(dynamic_vocab))
            # csk = [x.split()[1] if x is not None else 'None' for x in csk]
            for generation, score in zip(generation_beam, score_beam):
                fout.write('Generation:\t%.4f\t%s (%s)\n' % (
                    score, hke_data_utils.restore_field_copy_example(field, text, csk, query, generation, dynamic_vocab=dynamic_vocab), generation))
            idx += 1



def init_network(model, method='xavier'):
    """
    :param model:
    :param method:
    :param seed:
    :return:
    """
    logger.info('[INIT] Initializing parameters: %s' % method)
    for name, w in model.named_parameters():
        if 'bias' in name:
            nn.init.constant_(w, 0)
        else:
            if method == 'uniform' or w.dim() == 1:
                nn.init.uniform_(w, -0.1, 0.1)
            elif method == 'xavier':
                nn.init.xavier_uniform_(w)
            elif method == 'xavier_normal':
                nn.init.xavier_normal_(w)
            elif method == 'kaiming':
                nn.init.kaiming_uniform_(w)
            elif method == 'kaiming_normal':
                nn.init.kaiming_normal_(w)
            else:
                nn.init.normal_(w)


def sequence_mask_fn(lengths, maxlen=None, dtype=torch.bool, mask_first=False):
    """

    :param lengths: [seq_len]
    :param maxlen: [seq_len,max_len]
    :param dtype:
    :return:
    """
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1, device=lengths.device, requires_grad=False)
    matrix = torch.unsqueeze(lengths, dim=-1)
    # mask = row_vector < matrix
    mask = row_vector.lt(matrix)
    if mask_first:
        mask[:, 0:1] = False
    mask = mask.type(dtype)
    return mask

def select_time_first_sequence_embedding(inputs, index):
    """

    :param inputs: (src_len, batch_size, dim)
    :param dim0's index: (batch_size)
    :return:
    """
    src_len, batch_size, dim = inputs.shape
    inputs = inputs.view(src_len * batch_size, dim)
    index_with_offset = index * batch_size + torch.arange(0, batch_size, dtype=torch.long, device=index.device)
    outputs = inputs.index_select(dim=0, index=index_with_offset)
    return outputs


def create_beam(tensor, beam_width, batch_dim):
    """

    :param tensor:
    :param beam_width:
    :param batch_dim:
    :return:
    """
    if isinstance(tensor, tuple) or isinstance(tensor, list):
        return tuple([torch.repeat_interleave(x, beam_width, batch_dim) for x in tensor])
    else:
        if tensor is None:
            return tensor
        return torch.repeat_interleave(tensor, beam_width, batch_dim)



def generate_copy_offset_matrix(target_vocab_size, dynamic_vocab, dynamic_vocab_len):
    """

    :param target_vocab_size: 目标词的大小
    :param dynamic_vocab: [Seq_len, batch]
    :param dynamic_vocab_len: [Seq_len]
    :return:
    """
    # Dynamic 每个Batch一样
    dynamic_vocab_size = dynamic_vocab.shape[0]
    batch_size = dynamic_vocab.shape[1]
    if dynamic_vocab.device.type.find('cpu') > -1:
        float_precision = torch.float32
    else:
        float_precision = torch.float32
    # [vocab_size]
    copy_matrix = torch.arange(0, target_vocab_size, device=dynamic_vocab.device, dtype=torch.int32)
    # [1,1,vocab_size]
    copy_matrix = copy_matrix.view(1, 1, -1)
    # [batch, dynamic_vocab, vocab_size]
    copy_matrix = copy_matrix.repeat([batch_size, dynamic_vocab_size, 1])
    # 不拷贝超过100的
    valid_dynamic_vocab = torch.where(dynamic_vocab > 100, dynamic_vocab,
                                      torch.zeros_like(dynamic_vocab))
    #  [batch, Seq_len, 1]
    valid_dynamic_vocab = valid_dynamic_vocab.transpose(0, 1).unsqueeze(-1).to(torch.int32)
    copy_matrix = copy_matrix == valid_dynamic_vocab
    # [batch, seq_len, vocab_size]
    copy_matrix = copy_matrix.to(float_precision)
    return copy_matrix


def generate_copy_offset_probs(batch_size, prob_output, raw_vocab_size, std_vocab_size, dynamic_vocab, copy_matrix_cache=None):
    """
    """
    # if True:
    if dynamic_vocab.device.type.find('cpu') > -1:
        float_precision = torch.float32
    else:
        float_precision = torch.float32
    if copy_matrix_cache is None or copy_matrix_cache['batch_size'] != batch_size:
        # Dynamic 每个Batch一样
        copy_matrix = torch.arange(0, raw_vocab_size, device=dynamic_vocab.device, dtype=torch.int32)
        copy_matrix = copy_matrix.view(1, 1, -1)
        # [batch, dynamic_vocab, std_vocab] 每个Dynamic_Vocab所对应的真实词表的映射关系
        copy_matrix = copy_matrix.repeat([batch_size, std_vocab_size - raw_vocab_size, 1])
        valid_dynamic_vocab = torch.where(dynamic_vocab > 100, dynamic_vocab,
                                          torch.zeros_like(dynamic_vocab))
        valid_dynamic_vocab = valid_dynamic_vocab.unsqueeze(-1).to(torch.int32)
        copy_matrix = copy_matrix == valid_dynamic_vocab
        copy_matrix = copy_matrix.to(float_precision)
        copy_matrix_cache = {
            'batch_size': batch_size,
            'copy_matrix': copy_matrix,
        }
    else:
        copy_matrix = copy_matrix_cache['copy_matrix']

    copied_probs = prob_output[:, raw_vocab_size:].unsqueeze(1)
    copied_probs = copied_probs.to(float_precision)
    # [batch, 1, dynamic_vocab] * [batch, dynamic_vocab, std_vocab] => [batch, 1, std_vocab]
    probs_with_offset = torch.bmm(copied_probs, copy_matrix)
    probs_with_offset = probs_with_offset.squeeze(1)
    probs_with_offset[:, 0:100] = 0.0
    prob_output_with_copy = torch.cat(
        [prob_output[:, 0: raw_vocab_size] + probs_with_offset,
         prob_output[:, raw_vocab_size:]], -1)
    return prob_output_with_copy, copy_matrix_cache


def select_dynamic_embeddings_from_time_first_idx(char_embeddings, char_idx):
    """

    :param char_embeddings: [dynamic_vocab_size, batch_size, dim]
    :param char_idx: [seq_len, batch_size]
    :return: [seq_len, batch_size, dim]
    """
    # => [batch_size, vocab_size, dim]
    char_embeddings = char_embeddings.permute(1, 0, 2)
    batch_size, vocab_size, dim = char_embeddings.shape
    # idx [seq_len, batch_size]
    char_idx = char_idx
    # => [batch_size, seq_len]
    char_idx = char_idx.permute(1, 0)
    seq_len = char_idx.shape[1]
    # offset [batch,size, seq_lem]
    offset = torch.arange(0, batch_size * vocab_size, vocab_size, device=char_idx.device).unsqueeze(-1)
    char_idx_with_offset = char_idx + offset
    flatten_char_idx_with_offset = char_idx_with_offset.contiguous().view(-1)
    # [batch_size  * vocab_size, dim]
    flatten_embeddings = char_embeddings.contiguous().view(-1, dim)
    # selected embedding: [batch_size*seq_len, dim]
    flatten_selected_char_embedding = torch.index_select(flatten_embeddings, 0, flatten_char_idx_with_offset)
    selected_char_embedding = flatten_selected_char_embedding.view(batch_size, seq_len, -1)
    return selected_char_embedding.permute(1, 0, 2).contiguous()