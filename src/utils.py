# -*- coding: utf-8 -*-

"""
Created on 2020-08-19 15:14
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""


import jieba


class CorpusReader(object):
    def __init__(self, file_path):
        self.file_path = file_path

        self.token_dict = self.build_words_dict()

    def read_txt(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
        data = [sample.strip('\n').strip() for sample in data]
        return data

    def get_all_words(self):
        data = ''.join(self.read_txt())
        words = list(jieba.cut(data))
        return words

    def build_words_dict(self):
        words = self.get_all_words()
        token_dict = dict()
        for word in words:
            if word not in token_dict:
                token_dict.setdefault(word, len(token_dict))
        print("共有 {} 个tokens!".format(str(len(token_dict))))
        return token_dict

    def build_nsp(self, max_length):
        words = self.get_all_words()
        print(len(words))

        sentence = []
        next_sentence = []

        for idx in range(len(words)-max_length):
            sentence.append(words[idx: idx+max_length])
            print(words[idx: idx+max_length])
            next_sentence.append(words[idx+max_length])
            print(words[idx+max_length])
            print('-'*30)
        print("共提取句子总数：", len(sentence))
        return sentence, next_sentence

    def splited2token(self, splited):
        result = []
        for word in splited:
            if word in self.token_dict:
                result.append(self.token_dict.get(word))
        return result

    def get_trainingSet(self, max_length):
        current, next = self.build_nsp(max_length)
        current_result = []
        next_result = []
        for sentence in current:
            current_result.append(self.splited2token(sentence))
        for word in next:
            next_result.append(self.token_dict.get(word))
        return current_result,next_result


if __name__ == '__main__':
    test = CorpusReader('../data/白夜行.txt')
    # print(test.build_nsp(5))
    print(test.get_trainingSet(5))