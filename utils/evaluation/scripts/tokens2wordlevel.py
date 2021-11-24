import re
import jieba
"""
将每一个种不同的Subword Encoding解析为原本的Word-Level模式
"""


def word_cleaner(x):
    return x.strip('\r\n')


def revert_from_sentence(sentence, subword_option):
    assert subword_option is None
    sentence = word_cleaner(sentence)
    return sentence

