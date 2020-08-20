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


def write(model, temperature, word_num, max_length, token_dict, all_words,begin_sentence):
    gg = begin_sentence[:30]
    print(''.join(gg), end='/// ')
    for _ in range(word_num):
        sampled = np.zeros((1, max_length))
        for t, char in enumerate(gg):
            sampled[0, t] = token_dict[char]

        preds = model.predict(sampled, verbose=0)[0]
        if temperature is None:
            next_word = all_words[np.argmax(preds)]
        else:
            next_index = sample(preds, temperature)
            next_word = all_words[next_index]

        gg.append(next_word)
        gg = gg[1:]
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

    for _ in range(100):
        generator.fit(train_x, train_y, epochs=1, batch_size=32)

