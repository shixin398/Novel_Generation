# -*- coding: utf-8 -*-

"""
Created on 2020-08-20 11:21
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""

import numpy as np
import sys

from uitls import CorpusReader
from bidireactionalGRU import BidirectionalGRU


def sample(preds, temperature=1.0):
    """
    将原先的分布加入temperature后，转换为新分布，temperature越大则新概率分布越均匀，随机性也就越大，越容易生成一些意想不到的词
    :param preds:
    :param temperature:
    :return:
    """
    if not isinstance(temperature, float) and not isinstance(temperature, int):
        print('\n\n', "temperature must be a number")
        raise TypeError

    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def write(model, temperature, word_num, max_length, token_dict, all_words, begin_sentence):
    start = begin_sentence[:30]
    print(''.join(start), end='|||')  # 打印起始句子
    for _ in range(word_num):
        sampled = np.zeros((1, max_length))
        for t, char in enumerate(start):
            sampled[0, t] = token_dict[char]

        preds = model.predict(sampled, verbose=0)[0]
        if temperature is None:  # 较为死板
            next_word = all_words[np.argmax(preds)]
        else:  # 随temperature越高越随机
            next_index = sample(preds, temperature)
            next_word = all_words[next_index]

        start.append(next_word)
        start = start[1:]
        sys.stdout.write(next_word)
        sys.stdout.flush()


if __name__ == '__main__':
    file_path = '../data/白夜行.txt'
    data_reader = CorpusReader(file_path)
    all_words = data_reader.get_all_words()  # 获得文章全量分词结果
    token_dict = data_reader.build_words_dict()  # 构建后获得全量tokens的索引字典
    max_length = 300  # 每个预测片段的长度
    train_x, train_y = data_reader.get_trainingSet(max_length)

    biGRU = BidirectionalGRU(all_words, token_dict, max_length)
    generator = biGRU.model()

    generator.fit(train_x, train_y, epochs=100, batch_size=32)

    begin_sentence = all_words[50000:50100]
    print('-----原始句子-----')
    print('//'.join(begin_sentence))
    print('-----原始句子-----')

    print('-----生成死板句子-----')
    write(generator, None, max_length, begin_sentence)
    print('-----生成死板句子-----')

    print('-----生成创意句子-----')
    write(generator, 0.5, max_length, begin_sentence)
    print('-----生成创意句子-----')




