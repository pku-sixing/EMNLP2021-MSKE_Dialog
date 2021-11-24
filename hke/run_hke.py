import os
import math

from tensorboardX import SummaryWriter

import numpy as np
import torch
import time
import random
from torch import optim
from torch.nn import functional as F
from hke.hke_opts import parse_arguments
from hke.tri_hke import TriHKE
from hke import hke_data_utils as data_utils
from utils.logger import logger, init_logger
from utils.evaluation.evaluation_helper import ScoreManager
from utils.evaluation.eval import std_eval_with_args
from utils import model_helper

def set_seed(seed=12345):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def evaluate(hparams, model, val_iter, vocab_size, fields, std_vocab_size):
    model.eval()
    src_field = fields['src']
    tgt_field = fields['tgt']
    text_field = fields.get('text', None)
    csk_field = fields.get('csk_response_entity', None)
    attribute_uni_key_field = fields.get('attribute_uni_key', None)
    attribute_uni_word_field = fields.get('attribute_uni_word', None)
    pad = tgt_field.vocab.stoi['<pad>']
    total_loss = 0
    total_ppl = 0
    num_cases = 0
    unk = tgt_field.vocab.stoi['<unk>']
    if hparams.global_mode_gate == 'extend' or hparams.global_mode_gate == 'dual':
        knowledge_modes = hparams.knowledge.split(',')
    elif hparams.global_mode_gate == 'default':
        knowledge_modes = hparams.knowledge.split(',')[1:]
    else:
        raise NotImplementedError()
    if hparams.global_mode_gate == 'dual':
        knowledge_global_weights = np.zeros([len(knowledge_modes) * 2])
    else:
        knowledge_global_weights = np.zeros([len(knowledge_modes)])
    with torch.no_grad():
        for b, batch in enumerate(val_iter):
            src, len_src = batch.src
            tgt, len_tgt = batch.tgt
            tgt_raw, _ = batch.tgt_raw
            num_cases += tgt.size()[1]
            output, token_output, _, _, _, decoder_cache = model(batch,  mode='eval')
            if decoder_cache['global_weights'] is not None:
                assert len(decoder_cache['global_weights']) == len(knowledge_global_weights), '%s-%s' % (len(decoder_cache['global_weights']) ,len(knowledge_global_weights) )
                for idx, mode in enumerate(knowledge_global_weights):
                     knowledge_global_weights[idx] += decoder_cache['global_weights'][idx].sum()

            if hparams.copy_strategy == 'fusion':
                assert hparams.unk_learning == 'none'
                logits = output[1:].view(-1, vocab_size)
                # First Step Compute Loss for TGT_TAW
                raw_tgt, _ = batch.tgt_raw
                raw_labels = raw_tgt[1:].contiguous().view(-1)
                labels = tgt[1:].contiguous().view(-1)
                raw_loss = F.nll_loss(logits, raw_labels, ignore_index=pad, reduction="none")
                std_loss = F.nll_loss(logits, labels, ignore_index=pad, reduction="none")
                is_copied_tokens = labels >= std_vocab_size
                is_copied_raw_tokens = (raw_labels != unk) & is_copied_tokens
                # Final Loss

                std_prob = torch.exp(-std_loss)
                raw_prob = torch.exp(-raw_loss)
                fused_prob = std_prob + raw_prob
                fused_loss = - torch.log(fused_prob)
                is_copied_raw_tokens = is_copied_raw_tokens.to(torch.float32)

                loss = std_loss * (1.0 - is_copied_raw_tokens) + is_copied_raw_tokens * fused_loss
                loss = loss.view(tgt[1:].size())
                loss = loss.sum(dim=0) / (len_tgt - 1)

            elif hparams.unk_learning == 'none':
                loss = F.nll_loss(output[1:].view(-1, vocab_size),
                                  tgt[1:].contiguous().view(-1),
                                  ignore_index=pad, reduction='none')
                loss = loss.view(tgt[1:].size())
                loss = loss.sum(dim=0) / (len_tgt - 1)
            elif hparams.unk_learning == 'penalize':
                logits = output[1:].view(-1, vocab_size)
                labels = tgt[1:].contiguous().view(-1)
                loss = F.nll_loss(logits, labels, ignore_index=pad, reduction="none")
                is_unk = labels == unk
                loss = torch.where(is_unk, loss / is_unk.sum(), loss).sum() / (labels != pad).sum() * tgt.size()[1]
            elif hparams.unk_learning == 'skip':
                logits = output[1:].view(-1, vocab_size)
                labels = tgt[1:].contiguous().view(-1)
                loss = F.nll_loss(logits, labels, ignore_index=pad, reduction="none")
                is_unk = labels == unk
                loss = torch.where(is_unk, loss * 1e-20, loss).sum() / ((labels != pad).sum() - is_unk.sum()) * tgt.size()[1]

            total_loss += loss.sum().item()
            total_ppl += torch.exp(loss).sum().item()
            if b < 1:
                #
                output, token_output, scores = model.greedy_search_decoding(hparams, batch)
                # output, token_output, scores = model.beam_search_decoding(hparams, src, len_src, beam_width=5)
                queries = data_utils.translate(src, src_field)
                generations = data_utils.translate(token_output, tgt_field)
                try:
                    dynamic_vocabs = batch.dynamic_vocab
                except Exception as e:
                    dynamic_vocabs = [['<None>']] * len(generations)

                if csk_field is not None:
                    csks = data_utils.translate(batch.csk_response_entity[0], csk_field)
                else:
                    csks = [None] * len(generations)
                if text_field is not None:
                    texts = data_utils.translate(batch.text[0], text_field)
                else:
                    texts = [None] * len(generations)
                if queries is None:
                    queries = ['None'] * len(generations)
                responses = data_utils.translate(tgt, tgt_field)
                if attribute_uni_key_field is not None:
                    field_words = data_utils.translate(batch.attribute_uni_word[0], attribute_uni_key_field)
                else:
                    field_words = [None] * len(tgt)
                idx = 0
                scores = scores.sum(dim=0)
                assert len(generations) == len(scores)
                beam_width = len(generations) // len(responses)
                beam_generations = []
                beam_scores = []
                for beam_id in range(len(generations)):
                    beam_generations.append(generations[beam_id * beam_width: beam_id * beam_width + beam_width])
                    beam_scores.append(scores[beam_id * beam_width: beam_id * beam_width + beam_width])
                for dynamic_vocab, csk, field_word, text, query, generation_beam, response, score_beam in zip(dynamic_vocabs,
                                                                        csks, field_words, texts, queries, beam_generations, responses,
                                                                        beam_scores):
                    logger.info('#%d' % idx)

                    logger.info('Query:%s' % query)
                    logger.info('CSK_Words:%s' % csk)
                    logger.info('Text:%s' % text)
                    logger.info('Fields Words:%s' % field_word)
                    logger.info('Dynamic Vocabs: %s' % ' '.join(dynamic_vocab))
                    tmp = []
                    tmp_weights = []
                    rst_response = data_utils.restore_field_copy_example(field_word, text, csk, query, response,dynamic_vocab=dynamic_vocab)
                    logger.info('Response:%s (%s)' % (rst_response, response))
                    for generation, score in zip(generation_beam, score_beam):
                        rst_generation = data_utils.restore_field_copy_example(field_word, text, csk, query, generation, dynamic_vocab=dynamic_vocab)
                        logger.info('Generation-RST %.4f :%s (%s)' % (score, rst_generation, generation))
                    if idx >= 5:
                        break
                    idx += 1
                print('###########')

        tmp = []
        if hparams.global_mode_gate == 'dual':
            for mode in knowledge_modes:
                tmp.append(mode+'_read')
                tmp.append(mode+'_gen')
            knowledge_modes = tmp
        return total_loss / num_cases, total_ppl / num_cases


def inference(hparams, model, val_iter, vocab_size, src_field, tgt_field):
    model.eval()
    output_queries = []
    output_responses = []
    output_scores = []
    output_generations = []
    output_dynamic_vocabs = []
    total_loss = 0
    total_ppl = 0
    num_cases = 0
    pad = tgt_field.vocab.stoi['<pad>']
    start_time = time.time()
    with torch.no_grad():
        for b, batch in enumerate(val_iter):
            if b % 100 == 0 and b > 0:
                logger.info('Inference Decoding %s/%s' % (b, len(val_iter)))
                print('Consumed Time mins:', (time.time() - start_time) / 60,
                            'Time per batch secs:', (time.time() - start_time) / b,
                            'Remain mins:', (time.time() - start_time) / b / 60 * (len(val_iter) - b) )
            src, len_src = batch.src
            tgt, len_tgt = batch.tgt
            num_cases += tgt.size()[1]
            if hparams.beam_width == 0:
                output, token_output, scores = model.greedy_search_decoding(hparams, batch)
                loss = scores[1:,:]
                loss = -loss.sum(dim=0) / (len_tgt - 1)
                total_loss += loss.sum().item()
                total_ppl += torch.exp(loss).sum().item()
            else:
                output, token_output, scores = model.beam_search_decoding(hparams, batch, beam_width=hparams.beam_width)

            queries = data_utils.translate(src, src_field)
            generations = data_utils.translate(token_output, tgt_field)
            if queries is None:
                queries = ['None'] * len(generations)
            responses = data_utils.translate(tgt, tgt_field)
            scores = scores.sum(dim=0)
            assert len(generations) == len(scores)
            beam_width = len(generations) // len(responses)
            beam_generations = []
            beam_scores = []
            for beam_id in range(len(queries)):
                beam_generations.append(generations[beam_id * beam_width: beam_id * beam_width + beam_width])
                beam_scores.append(scores[beam_id * beam_width: beam_id * beam_width + beam_width])

            output_queries += queries
            output_responses += responses
            output_scores += beam_scores
            output_generations += beam_generations
            try:
                dynamic_vocabs = batch.dynamic_vocab
            except Exception as e:
                dynamic_vocabs = [['<None>']] * len(queries)
            output_dynamic_vocabs += dynamic_vocabs

        return output_queries, output_responses, output_scores, output_generations, output_dynamic_vocabs, total_loss / num_cases, total_ppl / num_cases



def step_report(epoch, step, total_step, loss, coverage_loss, total_bow_loss, total_post_loss, time_diff, optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lrs.append(str(param_group['lr']))
    lrs = ','.join(lrs)
    time_per_step = time_diff / step
    remaining_time = time_per_step * (total_step - step)
    # 显示最准确的Loss
    loss = loss - coverage_loss - total_bow_loss - coverage_loss-total_post_loss
    logger.info(
        "[Epoch=%d/Step=%d-%d][lrs:%s][loss:%5.2f,cvg:%.3f,bow:%.3f,pos:%.4f][ppl:%5.2f][step_time:%.2fs][remain:%.2fs]" %
        (epoch, step, total_step, lrs, loss, coverage_loss, total_bow_loss, total_post_loss, math.exp(loss), time_per_step, remaining_time))


def train_epoch(hparams, epoch, model, optimizer, train_iter, vocab_size, src_field, tgt_field, tf_recorder=None, std_vocab_size=-1):
    model.train()
    total_loss = 0
    total_post_loss = 0
    total_coverage_loss = 0
    total_bow_loss = 0
    start_time = time.time()
    pad = tgt_field.vocab.stoi['<pad>']
    unk = tgt_field.vocab.stoi['<unk>']
    for b, batch in enumerate(train_iter):

        tgt, len_tgt = batch.tgt
        optimizer.zero_grad()
        output, _, coverage_loss, bow_loss, post_loss, decoder_cache = model(batch)
        assert post_loss is None

        if hparams.copy_strategy == 'fusion':
            logits = output[1:].view(-1, vocab_size)
            labels = tgt[1:].contiguous().view(-1)
            assert hparams.unk_learning == 'none'
            raw_tgt, _ = batch.tgt_raw
            raw_labels = raw_tgt[1:].contiguous().view(-1)
            raw_loss = F.nll_loss(logits, raw_labels, ignore_index=pad, reduction="none")
            std_loss = F.nll_loss(logits, labels, ignore_index=pad, reduction="none")
            is_copied_tokens = labels >= std_vocab_size
            is_copied_raw_tokens = (raw_labels != unk) & is_copied_tokens
            # Final Loss

            std_prob = torch.exp(-std_loss)
            raw_prob = torch.exp(-raw_loss)
            fused_prob = std_prob + raw_prob
            fused_loss = - torch.log(fused_prob)
            is_copied_raw_tokens = is_copied_raw_tokens.to(torch.float32)

            loss = std_loss * (1.0- is_copied_raw_tokens) + is_copied_raw_tokens * fused_loss
            loss = loss.sum() / (labels != pad).sum()
        elif hparams.unk_learning == 'none':
            logits = output[1:].view(-1, vocab_size)
            labels = tgt[1:].contiguous().view(-1)
            loss = F.nll_loss(logits, labels, ignore_index=pad)
        elif hparams.unk_learning == 'penalize':
            logits = output[1:].view(-1, vocab_size)
            labels = tgt[1:].contiguous().view(-1)
            loss = F.nll_loss(logits, labels, ignore_index=pad, reduction="none")
            is_unk = labels == unk
            loss = torch.where(is_unk, loss / is_unk.sum(), loss).sum() / (labels != pad).sum()
        elif hparams.unk_learning == 'skip':
            logits = output[1:].view(-1, vocab_size)
            labels = tgt[1:].contiguous().view(-1)
            loss = F.nll_loss(logits, labels, ignore_index=pad, reduction="none")
            is_unk = labels == unk
            loss = torch.where(is_unk, loss * 1e-20, loss).sum() / ((labels != pad).sum()- is_unk.sum())
        else:
            raise NotImplementedError()

        if hparams.copy_coverage > 0.0 and hparams.copy:
            loss_for_bp = coverage_loss.mean() * hparams.copy_coverage + loss
        else:
            loss_for_bp = loss

        if bow_loss is not None and hparams.bow_loss > 0.0:
            loss_for_bp += bow_loss * hparams.bow_loss
            total_bow_loss += bow_loss.item() * hparams.bow_loss

        if coverage_loss is not None and hparams.copy_coverage > 0.0 :
            total_coverage_loss += coverage_loss.mean().item()
            total_post_loss += coverage_loss.mean().item()

        if 'prior_global_weights' in decoder_cache:

            prior_weights = decoder_cache['prior_global_weights']
            post_weights = decoder_cache['global_weights']

            def safe_log(x):
                # print(x)
                # x = torch.clip(x,1e-20, 1.0-1e-20)
                x = torch.min(torch.ones_like(x) - 1e-20, x)
                x = torch.max(torch.zeros_like(x) + 1e-20, x)
                return torch.log(x)

            def get_pos_loss(post_weights, prior_weights):
                post_loss = 0.0
                def kld(p, q):
                    tmp = p * safe_log(p / (q + 1e-20) + 1e-20)
                    tmp = tmp.sum(-1)
                    return tmp
                post_distributions = torch.cat(post_weights, -1)
                prior_distributions = torch.cat(prior_weights, -1)

                post_distributions = post_distributions/post_distributions.sum(-1, keepdim=True).detach()
                prior_distributions = prior_distributions/prior_distributions.sum(-1, keepdim=True)

                kld_loss = kld(post_distributions, prior_distributions)
                # 保证分布一致
                post_loss += kld_loss.mean()

                def entropy(p):
                    tmp = p * safe_log(p+1e-20)
                    tmp = -tmp.sum(-1).mean()
                    return tmp

                def self_entropy(p):
                    tmp = p * safe_log(p+1e-20) + (1.0-p) * safe_log(1-p+1e-20)
                    tmp = -tmp.sum(-1).mean()
                    return tmp
                post_loss += ( entropy(post_distributions) + entropy(prior_distributions)) * 0.5

                for prior, post in zip(prior_weights, post_weights):
                    post = post.detach()
                    tmp_loss = 0
                    tmp_loss += torch.pow(prior - post, 2).mean()
                    tmp_loss += (self_entropy(prior) * 0.5+ self_entropy(post) * 0.5)
                    post_loss += tmp_loss
                return post_loss

            if hparams.global_mode_gate == 'dual':
                post_loss = get_pos_loss(post_weights[0::2], prior_weights[0::2])
                post_loss += get_pos_loss(post_weights[1::2], prior_weights[1::2])
            else:
                post_loss = get_pos_loss(post_weights, prior_weights)
            loss_for_bp += post_loss * hparams.posterior_prior_loss
            total_post_loss += post_loss.item()


        loss_for_bp.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip)
        #
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(name)
        # print('------')

        optimizer.step()
        optimizer.zero_grad()


        total_loss += loss.item()

        if tf_recorder is not None:
            if decoder_cache['global_weights'] is not None:
                knowledge_modes = hparams.knowledge.split(',')
                if len(decoder_cache['global_weights']) ==  len(knowledge_modes) - 1:
                    # 不包含Context
                    knowledge_modes = knowledge_modes[1:]
                for idx, mode in enumerate(knowledge_modes):
                    if args.global_mode_gate == 'dual':
                        tf_recorder.add_histogram('train/global_weight/global_gate_' + mode + '_read',
                                                  decoder_cache['global_weights'][idx * 2].detach().cpu().numpy(),
                                                  (epoch - 1) * len(train_iter) + b)
                        tf_recorder.add_histogram('train/global_weight/global_gate_' + mode + '_gen',
                                                  decoder_cache['global_weights'][idx * 2+1].detach().cpu().numpy(),
                                                  (epoch - 1) * len(train_iter) + b)
                    else:
                        tf_recorder.add_histogram('train/global_weight/global_gate_'+mode, decoder_cache['global_weights'][idx].detach().cpu().numpy(),  (epoch-1) * len(train_iter) + b )
            tf_recorder.add_scalar('train/loss', loss.item(), (epoch-1) * len(train_iter) + b)
            if 'prior_global_weights' in decoder_cache:
                tf_recorder.add_scalar('train/post_loss', post_loss.item(), (epoch - 1) * len(train_iter) + b)
                knowledge_modes = hparams.knowledge.split(',')
                if len(decoder_cache['prior_global_weights']) == len(knowledge_modes) - 1:
                    # 不包含Context
                    knowledge_modes = knowledge_modes[1:]
                for idx, mode in enumerate(knowledge_modes):
                    if args.global_mode_gate == 'dual':
                        tf_recorder.add_histogram('train/prior_global_weights/global_gate_' + mode + '_read',
                                                  decoder_cache['prior_global_weights'][idx * 2].detach().cpu().numpy(),
                                                  (epoch - 1) * len(train_iter) + b)
                        tf_recorder.add_histogram('train/prior_global_weights/global_gate_' + mode + '_gen',
                                                  decoder_cache['prior_global_weights'][
                                                      idx * 2 + 1].detach().cpu().numpy(),
                                                  (epoch - 1) * len(train_iter) + b)
                    else:
                        tf_recorder.add_histogram('train/prior_global_weights/global_gate_' + mode,
                                                  decoder_cache['prior_global_weights'][idx].detach().cpu().numpy(),
                                                  (epoch - 1) * len(train_iter) + b)

            tf_recorder.add_scalar('train/ppl', math.exp(loss.item()), (epoch-1) * len(train_iter) + b)
            tf_recorder.add_scalar('train/lr', optimizer.param_groups[0]['lr'], (epoch-1) * len(train_iter) + b)

        if b % hparams.report_steps == 0 and b != 0:
            total_post_loss = total_post_loss / hparams.report_steps
            total_loss = total_loss / hparams.report_steps
            total_coverage_loss = total_coverage_loss / hparams.report_steps
            total_bow_loss = total_bow_loss / hparams.report_steps
            time_diff = time.time() - start_time
            step_report(epoch, b, len(train_iter), total_loss, total_coverage_loss, total_bow_loss, total_post_loss,
                        time_diff, optimizer)
            total_loss = 0
            total_coverage_loss = 0
            total_bow_loss = 0
            total_post_loss = 0

def set_dynamic_vocabs(args, fields):
    if 'char_vocab' in fields:
        args.__setattr__('char_vocab_size', len(fields['char_vocab'].vocab))
    if 'sub_attribute_word' in fields:
        args.__setattr__('sub_field_word_vocab_size', len(fields['sub_attribute_word'].vocab))
    if 'sub_src' in fields:
        args.__setattr__('sub_src_vocab_size', len(fields['sub_src'].vocab))
    if 'attribute_key' in fields:
        args.__setattr__('field_key_vocab_size', len(fields['attribute_key'].vocab))
    if 'attribute_word' in fields:
        args.__setattr__('field_word_vocab_size', len(fields['attribute_word'].vocab))


def get_vocab_offset(args):
    logger.info("[PARAM] Setting vocab sizes")
    masked_knowledges = set(args.mask_copy_flags.split(','))
    if args.copy_strategy != 'dynamic_vocab':
        if args.copy:
            vocab_offset = args.max_copy_token_num
        else:
            vocab_offset = 0
        if 'infobox' in args.knowledge.split(',') and 'box' not in masked_knowledges:
            vocab_offset += args.max_kw_pairs_num
        if 'text' in args.knowledge.split(',')  and 'text' not in masked_knowledges:
            vocab_offset += args.max_text_token_num
        if 'commonsense' in args.knowledge.split(',') and  'csk' not in masked_knowledges:
            vocab_offset += args.max_csk_num
    else:
        vocab_offset = args.max_dynamic_vocab_size
    return vocab_offset


def train(args):
    states = {
        'val_loss': [],
        'val_ppl': [],
        'lr': args.lr,
        'epoch': 0,
        'best_epoch': 0,
        'best_val_loss': -1,
    }
    if args.cuda:
        logger.info('[PARAM] Enabling CUDA')
        assert torch.cuda.is_available()

    logger.info("[DATASET] Preparing dataset...")
    train_iter, val_iter, test_iter, fields = data_utils.load_dataset(args)
    if 'src' in fields:
        src_vocab_size, tgt_vocab_size = len(fields.get('src', None).vocab), len(fields['tgt'].vocab)
    else:
        tgt_vocab_size =  len(fields['tgt'].vocab)
    logger.info("[TRAIN]: #Batches=%d (#Cases:%d))" % (len(train_iter), len(train_iter.dataset)))
    logger.info("[VALIDATION]: #Batches=%d (#Cases:%d))" % (len(val_iter), len(val_iter.dataset)))
    logger.info("[TEST]: #Batches=%d (#Cases:%d))" % (len(test_iter), len(test_iter.dataset)))

    logger.info("[PARAM] Setting vocab sizes")

    vocab_offset = get_vocab_offset(args)


    args.__setattr__('tgt_vocab_size', tgt_vocab_size - vocab_offset)
    if 'attribute_key' in fields:
        args.__setattr__('field_vocab_size', len(fields['attribute_key'].vocab))
    if 'src' in fields:
        args.__setattr__('src_vocab_size', src_vocab_size - vocab_offset)
    if 'attribute_word' in fields:
        args.__setattr__('field_word_vocab_size', len(fields['attribute_word'].vocab))
    if 'attribute_word' in fields and args.field_tag_usage != 'none':
        args.__setattr__('field_pos_tag_vocab_size', len(fields['attribute_word_tag'].vocab))
    args.__setattr__('tgt_vocab_size_with_offsets', tgt_vocab_size)
    set_dynamic_vocabs(args, fields)
    logger.info('[VOCAB] tgt_vocab_size_with_offsets = ' + str(tgt_vocab_size))


    logger.info("[MODEL] Preparing model...")
    seq2seq = TriHKE.build_s2s_model(args, fields)

    if args.cuda:
        seq2seq.to('cuda')

    model_helper.show_parameters(seq2seq)
    if args.init_lr > 0:
        logger.info("[LR] Using init LR %f" % args.init_lr)
        optimizer = optim.Adam(seq2seq.parameters(), lr=args.init_lr)
        states['lr'] = args.init_lr
    else:
        optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
        states['lr'] = args.lr
    logger.info(seq2seq)

    # Load model
    is_loaded = model_helper.try_restore_model(args.model_path, seq2seq, optimizer, states, best_model=False)
    if not is_loaded:
        logger.info("[PARAM] Using fresh params")
        os.system('mkdir %s' % args.model_path)
        os.system('rm %s/source_code.zip' % args.model_path)
        os.system('zip -r %s/source_code.zip hke/*.py hke/*/*.py' % args.model_path)
        print('zip -r %s/source_code.zip hke/*' % args.model_path)
        model_helper.init_network(seq2seq, args.init)
        if args.init_word_vecs is True:
            assert args.pre_embed_dim == args.embed_size
            logger.info("[EMBED] Loading the pre-trained word_embeddings")
            # 使用预训练的Embedding
            if 'csk_golden_idx' in fields:
                data_utils.load_pretrain_trans_embeddings(seq2seq.csk_knowledge_manager.entity_embedding,
                                                    args.csk_entity_embed_path, 100, offset=5)
                data_utils.load_pretrain_trans_embeddings(seq2seq.csk_knowledge_manager.relation_embedding,
                                                    args.csk_relation_embed_path, 100, offset=3)
            if 'attribute_key' in fields:
                data_utils.load_pretrain_embeddings(seq2seq.infobox_knowledge_managers.field_key_embed, fields['attribute_key'],
                                                    args.pre_embed_file, args.embed_size, char2word='avg',
                                                    suffix=args.dataset_version + 'attribute_key')
                if args.field_word_vocab_path != 'none':
                    data_utils.load_pretrain_embeddings(seq2seq.infobox_knowledge_managers.field_word_embed,
                                                        fields['attribute_word'],
                                                        args.pre_embed_file, args.embed_size, char2word='avg',
                                                        suffix=args.dataset_version + 'attribute_word')
            if 'src' in fields:
                data_utils.load_pretrain_embeddings(seq2seq.src_embed, fields.get('src', None),
                                                    args.pre_embed_file, args.embed_size,
                                                    suffix=args.dataset_version+'src')

            if 'char_vocab' in fields:
                data_utils.load_pretrain_embeddings(seq2seq.word_encoder.sub_embed, fields.get('char_vocab', None),
                                                    args.pre_embed_file, args.embed_size,
                                                    suffix=args.dataset_version+'char_vocab')
            if not args.share_embedding or 'src' not in fields:
                data_utils.load_pretrain_embeddings(seq2seq.dec_embed, fields['tgt'], args.pre_embed_file,
                                                    args.embed_size, suffix=args.dataset_version+'tgt')


    start_epoch = states['epoch']
    record_writer = SummaryWriter(os.path.join(args.model_path, 'tensorboard'))
    for epoch in range(start_epoch + 1, args.epochs + 1):
        if epoch != start_epoch + 1 and (args.align_dropout > 0.0 and (args.copy or args.field_copy)):
            logger.info("[Dataset] Reloading & Aligning the dataset")
            train_iter, val_iter, test_iter, fields = data_utils.load_dataset(args)
        if epoch - states['best_epoch'] > 2:
            logger.info('[STOP] Early Stopped !')
            break
        start_time = time.time()
        num_batches = len(train_iter)
        logger.info('[NEW EPOCH] %d/%d, num of batches : %d' % (epoch, args.epochs, num_batches))
        train_epoch(args, epoch, seq2seq, optimizer, train_iter,
                    tgt_vocab_size, fields.get('src', None), fields['tgt'], tf_recorder=record_writer,
                    std_vocab_size=tgt_vocab_size - vocab_offset)
        val_loss, val_ppl_macro = evaluate(args, seq2seq, val_iter, tgt_vocab_size, fields, std_vocab_size=tgt_vocab_size - vocab_offset)
        val_ppl = math.exp(val_loss)
        test_loss, test_ppl_macro = evaluate(args, seq2seq, test_iter, tgt_vocab_size, fields, std_vocab_size=tgt_vocab_size - vocab_offset)
        test_ppl = math.exp(test_loss)

        record_writer.add_scalar('test/ppl', test_ppl, epoch)
        record_writer.add_scalar('test/loss', test_loss, epoch)
        record_writer.add_scalar('val/ppl', val_ppl, epoch)
        record_writer.add_scalar('val/loss', val_loss, epoch)

        logger.info("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2f/%5.2f | test_loss:%5.3f | test_pp:%5.2f/%5.2f"
                    % (epoch, val_loss, val_ppl, val_ppl_macro, test_loss, test_ppl, test_ppl_macro))
        time_diff = (time.time() - start_time) / 3600.0
        logger.info("[Epoch:%d] epoch time:%.2fH, est. remaining  time:%.2fH" %
                    (epoch, time_diff, time_diff * (args.epochs - epoch)))

        # Adjusting learning rate
        states['val_ppl'].append(val_ppl)
        states['val_loss'].append(val_loss)
        states['epoch'] = epoch
        if len(states['val_ppl']) >= 2:
            logger.info('[TRAINING] last->now valid ppl : %.3f->%.3f' % (states['val_ppl'][-2], states['val_ppl'][-1]))

        if len(states['val_ppl']) >= args.start_to_adjust_lr and states['val_ppl'][-1] >= states['val_ppl'][-2]:
            logger.info('[TRAINING] Adjusting learning rate due to the increment of val_loss')
            new_lr = model_helper.adjust_learning_rate(optimizer, rate=args.lr_decay_rate)
            states['lr'] = new_lr
        else:
            if args.init_lr > 0 and epoch >= args.init_lr_decay_epoch - 1 and states['lr'] > args.lr:
                logger.info('[TRAINING] Try to decay the init decay, from epoch %d, and the next epoch is %d' %
                            (args.init_lr_decay_epoch, epoch+1))
                new_lr = model_helper.adjust_learning_rate(optimizer, rate=args.lr_decay_rate, min_value=args.lr)
                states['lr'] = new_lr

        # Save the model if the validation loss is the best we've seen so far.
        if states['best_val_loss'] == -1 or val_ppl < states['best_val_loss']:
            logger.info('[CHECKPOINT] New best valid ppl : %.3f->%.2f' % (states['best_val_loss'], val_ppl))
            model_helper.save_model(args.model_path, epoch, val_ppl, seq2seq, optimizer, args, states, best_model=True,
                       clear_history=True)
            states['best_val_loss'] = val_ppl
            states['best_epoch'] = epoch

        # Saving standard model
        model_helper.save_model(args.model_path, epoch, val_ppl, seq2seq, optimizer, args, states, best_model=False,
                   clear_history=True)


def eval(args):
    states = {}
    score_manager = ScoreManager(args.model_path, 'evaluation_score')
    if args.cuda:
        logger.info('[PARAM] Enabling CUDA')
        assert torch.cuda.is_available()

    logger.info("[DATASET] Preparing dataset...")
    train_iter, val_iter, test_iter, fields= data_utils.load_dataset(args, is_eval=True)
    if 'src' in fields:
        src_vocab_size, tgt_vocab_size = len(fields.get('src', None).vocab), len(fields['tgt'].vocab)
    else:
        tgt_vocab_size = len(fields['tgt'].vocab)
    logger.info("[VALIDATION]: #Batches=%d (#Cases:%d))" % (len(val_iter), len(val_iter.dataset)))
    logger.info("[TEST]: #Batches=%d (#Cases:%d))" % (len(test_iter), len(test_iter.dataset)))



    vocab_offset = get_vocab_offset(args)

    if 'src' in fields:
        args.__setattr__('src_vocab_size', src_vocab_size - vocab_offset)
    args.__setattr__('tgt_vocab_size', tgt_vocab_size - vocab_offset)
    args.__setattr__('tgt_vocab_size_with_offsets', tgt_vocab_size)
    if 'attribute_key' in fields:
        args.__setattr__('field_vocab_size', len(fields['attribute_key'].vocab))
        args.__setattr__('field_word_vocab_size', len(fields['attribute_word'].vocab))
        if args.field_tag_usage != 'none':
            args.__setattr__('field_pos_tag_vocab_size', len(fields['attribute_word_tag'].vocab))
    set_dynamic_vocabs(args, fields)
    logger.info("[MODEL] Preparing model...")
    seq2seq = TriHKE.build_s2s_model(args, fields)
    model_helper.show_parameters(seq2seq)
    if args.cuda:
        seq2seq = seq2seq.to('cuda')
    logger.info(seq2seq)

    # Load model
    is_loaded = model_helper.try_restore_model(args.model_path, seq2seq, None, states, best_model=args.use_best_model)
    if not is_loaded:
        logger.info("[PARAM] Could not load a trained model！")
        return
    else:
        val_loss, val_ppl_macro = evaluate(args, seq2seq, val_iter, tgt_vocab_size, fields,std_vocab_size=tgt_vocab_size - vocab_offset)
        val_ppl = math.exp(val_loss)
        test_loss, test_ppl_macro = evaluate(args, seq2seq, test_iter, tgt_vocab_size, fields, std_vocab_size=tgt_vocab_size - vocab_offset)
        test_ppl = math.exp(test_loss)
        logger.info("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2f/%5.2f | test_loss:%5.3f | test_pp:%5.2f/%5.2f"
                    % (states['epoch'], val_loss, val_ppl, val_ppl_macro, test_loss, test_ppl, test_ppl_macro))

        score_manager.update('eval_val_loss', val_loss)
        score_manager.update('eval_val_ppl', val_ppl)
        score_manager.update('eval_val_macro_ppl', val_ppl_macro)
        score_manager.update('eval_test_loss', test_loss)
        score_manager.update('eval_test_ppl', test_ppl)
        score_manager.update('eval_test_macro_ppl', test_ppl_macro)


def infer(args):
    states = {}
    score_manager = ScoreManager(args.model_path, 'evaluation_score')

    if args.cuda:
        logger.info('[PARAM] Enabling CUDA')
        assert torch.cuda.is_available()

    logger.info("[DATASET] Preparing dataset...")
    train_iter, val_iter, test_iter, fields = data_utils.load_dataset(args, is_eval=True)
    if 'src' in fields:
        src_vocab_size, tgt_vocab_size = len(fields.get('src', None).vocab), len(fields['tgt'].vocab)
    else:
        tgt_vocab_size = len(fields['tgt'].vocab)
    logger.info("[VALIDATION]: #Batches=%d (#Cases:%d))" % (len(val_iter), len(val_iter.dataset)))
    logger.info("[TEST]: #Batches=%d (#Cases:%d))" % (len(test_iter), len(test_iter.dataset)))

    logger.info("[PARAM] Setting vocab sizes")

    vocab_offset = get_vocab_offset(args)

    if 'src' in fields:
        args.__setattr__('src_vocab_size', src_vocab_size - vocab_offset)
    args.__setattr__('tgt_vocab_size', tgt_vocab_size - vocab_offset)
    args.__setattr__('tgt_vocab_size_with_offsets', tgt_vocab_size)
    if 'attribute_key' in fields:
        args.__setattr__('field_vocab_size', len(fields['attribute_key'].vocab))
        args.__setattr__('field_word_vocab_size', len(fields['attribute_word'].vocab))
        if args.field_tag_usage != 'none':
            args.__setattr__('field_pos_tag_vocab_size', len(fields['attribute_word_tag'].vocab))
    set_dynamic_vocabs(args, fields)

    generation_name = model_helper.name_a_generation(args, 'test')
    file_prefix = os.path.join(args.model_path, generation_name)

    if not args.skip_infer:
        logger.info("[MODEL] Preparing model...")
        seq2seq = TriHKE.build_s2s_model(args, fields)
        model_helper.show_parameters(seq2seq)
        if args.cuda:
            seq2seq = seq2seq.to('cuda')
        logger.info(seq2seq)

        # Load model
        is_loaded = model_helper.try_restore_model(args.model_path, seq2seq, None, states, best_model=args.use_best_model)

        if not is_loaded:
            logger.info("[PARAM] Could not load a trained model！")
            return

        # Test Set
        raw_queries, raw_responses, raw_fields, raw_texts, raw_csks = data_utils.load_examples(args.test_data_path_prefix,
                                                                                     args.fact_dict_path)
        output_queries, output_responses, output_scores, output_generations, output_dynamic_vocabs, test_loss, test_ppl_macro  = \
            inference(args, seq2seq, test_iter, tgt_vocab_size, fields.get('src', None), fields['tgt'])
        model_helper.write_results_hke(args, file_prefix, raw_queries, raw_responses, output_scores, output_generations,
                                   raw_fields=raw_fields, raw_texts=raw_texts, raw_csks=raw_csks, dynamic_vocabs=output_dynamic_vocabs)
        del seq2seq


    logger.info('[STD Evaluation] Evaluating the test generations...')
    res_dict = std_eval_with_args(args, file_prefix, 'test')
    score_manager.update_group(generation_name, res_dict)

    if args.beam_width == 0 and args.skip_infer is False:
        test_ppl = math.exp(test_loss)
        score_manager.update('infer_b%d_test_loss' % args.beam_width, test_loss)
        score_manager.update('infer_b%d_test_ppl' % args.beam_width, test_ppl)
        score_manager.update('infer_b%d_test_macro_ppl' % args.beam_width, test_ppl_macro)


if __name__ == "__main__":
    try:
        init_logger()
        args = parse_arguments()
        random_seed = args.seed
        set_seed(random_seed)
        first_try = True
        if args.wait_gpu != -1:
            while True:
                command = os.popen("nvidia-smi -q -d PIDS | grep Processes")
                lines = command.read().split("\n")
                free_gpu = []
                busy_gpu = []
                for i in range(len(lines)):
                    print(i, lines[i])
                    if len(lines[i]) < 1:
                        continue
                    if "None" in lines[i]:
                        free_gpu.append(i)
                    else:
                        busy_gpu.append(i)
                logger.info('busy_gpu ' + str(busy_gpu))
                logger.info('free_gpu ' + str(free_gpu))
                if args.wait_gpu in free_gpu:
                    logger.info(str(args.wait_gpu) + ' is available now')
                    if not first_try:
                        logger.info(str(args.wait_gpu) + ' is available now, run in 30 mins')
                        time.sleep(60 * 30)
                    break
                else:
                    logger.info(str(args.wait_gpu) + ' is busy now')
                    time.sleep(60*30)
                    first_try = False


        if args.mode == 'train':
            train(args)
            eval(args)
        elif args.mode == 'eval':
            eval(args)
        elif args.mode == 'infer':
            infer(args)
    except KeyboardInterrupt as e:
        print("[STOP]", e)
