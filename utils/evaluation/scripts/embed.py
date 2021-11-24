import numpy as np
from utils.evaluation.scripts import tokens2wordlevel
from multiprocessing import Pool
import os


# 中文的Embedding https://github.com/SixingWu/Chinese-Word-Vectors Shen Li, Zhe Zhao, Renfen Hu, Wensi Li, Tao Liu, Xiaoyong Du, Analogical Reasoning on Chinese Morphological and Semantic Relations, ACL 2018.
# 英文的 https://github.com/SixingWu/fastText

def load_vocab(path, lower=True):
    vocab = set()
    with open(path, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        for line in lines:
            word = line.strip('\n')
            if lower:
                word = word.lower()
            vocab.add(word)
    print('vocab size: %d' % len(vocab))
    return vocab


def load_embed_from_file(path, vocab=None, dim=300):
    """
    each line:
    word [float]*dim
    :param path:
    :return:
    """
    embeddings = dict()
    with open(path, 'r', encoding='utf-8') as fin:
        # fout = open(path+'.small', 'w', encoding='utf-8')
        line = fin.readline()
        while len(line) > 0:
            try:
                items = line.strip('\n').split()
                if vocab is None or items[0] in vocab:
                    embeddings[items[0]] = np.array([float(x) for x in items[1:dim + 1]])
            except:
                pass
                # print(line)
            finally:
                line = fin.readline()

    vocab_size = -1 if vocab is None else len(vocab)
    print('vocab size: %d, embedding_vocab size: %d' % (vocab_size, len(embeddings)))
    return embeddings


def unk_embed(dim):
    # return (np.random.rand(dim) - 0.5)
    return np.zeros([dim])


def unk_embed_with_char(word, embeddings, dim):
    if len(word) == 1:
        word = [word]
    else:
        word = word.split()

    embedding = np.zeros([dim])
    for char in word:
        if char not in embeddings:
            embeddings[char] = unk_embed(dim)
        embedding += embeddings[char]
    word = ''.join(word)
    embeddings[word] = embedding / max(1, len(word))
    return embeddings[word]


def sentence_2_embedding(embeddings, sentence, dim=300, unk='<unk>'):
    words = sentence.strip('\r\n').split(' ')
    embedding = np.zeros([dim])
    counter = 0
    for word in words:
        if word in embeddings:
            embedding += embeddings[word]
            counter += 1
        elif unk in word:
            embedding += unk_embed(dim)
            counter += 1
        else:
            # add a new random_value
            new_embedding = unk_embed_with_char(word, embeddings, dim)
            embeddings[word] = new_embedding
            embedding += embeddings[word]
            counter += 1
    if counter > 0:
        embedding /= counter
    return embedding, counter


def cosine_sim(a, b):
    if np.sum(np.abs(a)) == 0 or np.sum(np.abs(b)) == 0:
        if np.sum(np.abs(a)) == 0 and np.sum(np.abs(b)) == 0:
            return 1
        return 0
    return sum(a * b) / (np.sqrt(sum(a * a)) * np.sqrt(sum(b * b)))


def evaluate_a_bulk(inputs, refs, embedding, dim, unk):
    sims = 0
    counts = 0
    final_res = []

    # Embedding Average
    assert len(inputs) == len(refs)
    avg_scores = []
    for input, ref in zip(inputs, refs):
        a, tmp = sentence_2_embedding(embedding, input, dim=dim)
        if tmp == 0:
            continue
        b, tmp = sentence_2_embedding(embedding, ref, dim=dim)
        if tmp == 0:
            continue
        avg_score = cosine_sim(a, b)
        avg_scores.append(avg_score)
        sims += avg_score
        counts += 1
    assert len(inputs) == len(avg_scores)
    final_res.append((sims, counts, avg_scores))

    # Embedding Greedy
    greedy_scores = []
    cache = dict()
    assert len(inputs) == len(refs)
    for tuple in [(inputs, refs), (refs, inputs)]:
        for seq1, seq2 in zip(tuple[0], tuple[1]):
            seq1 = seq1.strip('\n').split(' ')
            seq2 = seq2.strip('\n').split(' ')
            local_counter = 0
            local_score = 0
            for a in seq1:  # 以Reference为主
                score = -1 # argmax(a,b \in seq2)
                local_counter += 1
                for b in seq2:
                    key = a + '\t' + b
                    reverse_key = b + '\t' + a
                    if key in cache:
                        sim = cache[key]
                    elif reverse_key in cache:
                        sim = cache[reverse_key]
                    else:
                        if a in embedding:
                            embed_a = embedding[a]
                        elif a == unk:
                            embed_a = unk_embed(dim)
                        else:
                            embed_a = unk_embed_with_char(a, embedding, dim)
                            embedding[a] = embed_a

                        if b in embedding:
                            embed_b = embedding[b]
                        elif b == unk:
                            embed_b = unk_embed(dim)
                        else:
                            embed_b = unk_embed_with_char(b, embedding, dim)
                            embedding[b] = embed_b
                        sim = cosine_sim(embed_a, embed_b)
                        cache[key] = sim
                        cache[reverse_key] = sim
                    score = max(score, sim)
                local_score += score
            local_counter = max(local_counter, 1)
            local_score /= local_counter
            greedy_scores.append(local_score)
        
#     print(len(greedy_scores), len(inputs))
    assert len(greedy_scores) % 2 == 0
    first_greedy = greedy_scores[0:len(greedy_scores) // 2]
    second_greedy = greedy_scores[len(greedy_scores) // 2:]
    assert len(first_greedy) == len(second_greedy)
    greedy_scores = [(x + y) / 2.0 for x, y in zip(first_greedy, second_greedy)]
    assert len(inputs) == len(greedy_scores)
    final_res.append((sum(greedy_scores), len(greedy_scores), greedy_scores))

    # Embedding extrema
    def create_extrema_vector(inputs, embedding):
        embeddings = []
        for word in inputs.strip('\r\n').split(' '):
            if word in embedding:
                embeddings.append(embedding[word])
            elif unk in word:
                embeddings.append(unk_embed(dim))
            else:
                # add a new random_value
                new_embedding = unk_embed_with_char(word, embedding, dim)
                embedding[word] = new_embedding
                embeddings.append(embedding[word])
        # 取各维度的极值 [seq,embed_dim]
        embeddings = np.array(embeddings)
        abs_embeddings = np.abs(embeddings)
        second_indices = np.arange(np.shape(embeddings)[1])
        first_indices = np.argmax(abs_embeddings, 0)
        extrema_vector = embeddings[first_indices, second_indices]
        return extrema_vector

    scores = []
    for ref, input in zip(refs, inputs):
        if len(ref) == 0 or len(input) == 0:
            continue
        vector_ref = create_extrema_vector(ref, embedding)
        vector_input = create_extrema_vector(input, embedding)
        score = cosine_sim(vector_ref, vector_input)
        scores.append(score)
    assert len(scores) == len(inputs)
    final_res.append((sum(scores), len(scores), scores))
    return final_res


def evaluate_embedding_relevance(ref_file, input_file, embedding, revert_func, dim=200, unk='<unk>'):
    # Reading inputs
    with open(input_file, 'r', encoding='utf-8') as fin:
        inputs = fin.readlines()
        if revert_func is not None:
#             print("Original ref samples")
#             print(inputs[0:3])
            inputs = [revert_func(x) for x in inputs]
#             print("Reverted ref samples")
#             print(inputs[0:3])
    # Reading references
    with open(ref_file, 'r', encoding='utf-8')  as fin:
        refs = fin.readlines()
        if revert_func is not None:
#             print("Original ref samples")
#             print(refs[0:3])
            refs = [revert_func(x) for x in refs]
#             print("Reverted ref samples")
#             print(refs[0:3])

    pool = Pool(16)
    bulk_size = len(inputs) // 8 + 1
    jobs = []
    for i in range(0, len(inputs), bulk_size):
        job = pool.apply_async(evaluate_a_bulk, (inputs[i:i + bulk_size], refs[i:i + bulk_size], embedding, dim, unk))
        jobs.append(job)
    pool.close()
    pool.join()

    avg_score = 0.0
    avg_count = 0.0
    grd_score = 0.0
    grd_count = 0.0
    ext_score = 0.0
    ext_count = 0.0
    avg_scores = []
    grd_scores = []
    ext_scores = []
    for job in jobs:
        avg, grd, ext = job.get()
        avg_score += avg[0]
        avg_count += avg[1]
        avg_scores += avg[2]

        grd_score += grd[0]
        grd_count += grd[1]
        grd_scores += grd[2]

        ext_score += ext[0]
        ext_count += ext[1]
        ext_scores += ext[2]

    return avg_score / avg_count, grd_score / grd_count, ext_score / ext_count, avg_scores, grd_scores, ext_scores


def eval(ref_src_file, ref_tgt_file, input_file, embed_file, dim, word_option):
    vocab = set()
    print([ref_src_file,ref_tgt_file, input_file])
    for file in [ref_src_file, ref_tgt_file, input_file]:
        with open(file, 'r', encoding='utf-8') as fin:
            for line in fin.readlines():
                for word in tokens2wordlevel.revert_from_sentence(line.strip('\n'), word_option).split():
                    vocab.add(word)
                    if len(word) > 1:
                        for char in word:
                            vocab.add(char)
    print('#tokens in two files: %d' % len(vocab))
    embeddings = load_embed_from_file(embed_file, vocab)
    revert_func = lambda x: tokens2wordlevel.revert_from_sentence(x, word_option)

    result_dict = {}
    # (Ref_TGT, Generated)
    avg_sim, greedy_sim, extrema_sim, avg_scores, greedy_scores, extrema_scores = \
        evaluate_embedding_relevance(ref_tgt_file, input_file, embeddings, revert_func, dim=dim)
    result_dict['EmbedA'] = avg_sim
    result_dict['EmbedA_Scores'] = avg_scores
    result_dict['EmbedG'] = greedy_sim
    result_dict['EmbedG_Scores'] = greedy_scores
    result_dict['EmbedX'] = extrema_sim
    result_dict['EmbedX_Scores'] = extrema_scores
    # (Ref_SRC, Generated)
    avg_sim, greedy_sim, extrema_sim, avg_scores, greedy_scores, extrema_scores = \
        evaluate_embedding_relevance(ref_src_file, input_file, embeddings, revert_func, dim=dim)
    result_dict['SRC_EmbedA'] = avg_sim
    result_dict['SRC_EmbedA_Scores'] = avg_scores
    result_dict['SRC_EmbedG'] = greedy_sim
    result_dict['SRC_EmbedG_Scores'] = greedy_scores
    result_dict['SRC_EmbedX'] = extrema_sim
    result_dict['SRC_EmbedX_Scores'] = extrema_scores

    return result_dict


def main():
    ref_file = '/Users/mebiuw/Downloads/test_20000.tgt'
    input_file = '/Users/mebiuw/Downloads/res_10.txt'
    embed_file = '/Users/mebiuw/Downloads/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.txt'
    avg, greedy = eval(ref_file, input_file, embed_file, word_option='wpm')
    print(avg)
    print(greedy)


if __name__ == '__main__':
    main()
