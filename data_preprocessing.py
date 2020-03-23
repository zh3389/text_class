# -*- encoding:utf-8 -*-
"""
@作者：Mr.zhang
@文件名：data_preprocessing.py
@时间：20-3-18  上午9:36
@文档说明:
"""
import os
from tqdm import tqdm
import jieba
import numpy as np
import pandas as pd
from keras_preprocessing import sequence, text
from data_equilibrium import Data_equalization_initialization
import config


class Data_preprocessing():
    def __init__(self, data, column_names):
        '''config path:'''
        self.config = config.data_preprocessing_config()
        self.tokenizer_path = self.config.tokenizer_path
        self.pad_train_data_path = self.config.pad_train_data_path
        self.pad_train_label_path = self.config.pad_train_label_path
        self.stopwordpath = self.config.stopwordpath
        self.embedded_matrix_size = config.embedded_matrix_size
        self.input_max_seq_len = config.input_max_seq_len
        '''data param init'''
        self.data_column_name = column_names[0]
        self.class_column_name = column_names[1]
        self.all_data = data[[self.data_column_name, self.class_column_name]]
        self.data = self.all_data[self.data_column_name]
        self.label = self.all_data[self.class_column_name]
        #
        if not os.path.exists(self.pad_train_data_path) or not os.path.exists(self.pad_train_label_path) or not os.path.exists(self.tokenizer_path):
            if os.path.exists(self.pad_train_data_path):
                os.remove(self.pad_train_data_path)
            if os.path.exists(self.pad_train_label_path):
                os.remove(self.pad_train_label_path)
            if os.path.exists(self.tokenizer_path):
                os.remove(self.tokenizer_path)
            self.remove_stop_word_list = self.get_remove_stop_word()
        self.tokenizer = self.get_tokenizer()

    def remove_stop_words(self, sentence, cut_all=True):
        '''
        :param sentence: 传入需要分词的字符串列表
        :param cut_all: 分词模式
        :return: 返回分词后的列表
        '''
        stopwords = [line.strip() for line in open(self.stopwordpath, 'r', encoding='utf-8').readlines()]
        processed_docs_train = []
        # jieba.add_word("添加jieba不能分的词")
        for word_list in tqdm(sentence):
            outstr = []
            word = jieba.cut(word_list.strip().rstrip(), cut_all=cut_all)
            for i in word:
                if i not in stopwords:
                    if i != '\t' and i != ' ':
                        outstr.append(i)
            processed_docs_train.append(" ".join(outstr))
        return processed_docs_train

    def get_tokenizer(self):
        '''
        :param embedded_matrix_size: 嵌入矩阵大小
        :return: tokenizer
        '''
        if not os.path.exists(self.tokenizer_path):
            tokenizer = text.Tokenizer(num_words=self.embedded_matrix_size, lower=False, char_level=False)
            tokenizer.fit_on_texts(self.remove_stop_word_list)
            tokenizer_json = tokenizer.to_json()
            with open(self.tokenizer_path, "w") as f:
                f.write(tokenizer_json)
                print("save tokenizer_json success as '{}'".format(self.tokenizer_path))
            return tokenizer
        else:
            print("更换数据集需手动删除{}此文件,并重新运行代码后会自动生成tokenizer.".format(self.tokenizer_path))
            with open(self.tokenizer_path, "r") as f:
                tokenizer_json = f.read()
            tokenizer = text.tokenizer_from_json(tokenizer_json)
            print("load tokenizer_json success as '{}'".format(self.tokenizer_path))
            return tokenizer

    def get_remove_stop_word(self):
        train_data = self.data.tolist()
        train_data = self.remove_stop_words(train_data)
        return train_data

    def read_train_data(self):
        if not os.path.exists(self.pad_train_data_path) or not os.path.exists(self.pad_train_label_path):
            # transformer train data and pad sequences
            self.pad_data = self.tokenizer.texts_to_sequences(self.remove_stop_word_list)
            self.pad_data = sequence.pad_sequences(self.pad_data, maxlen=self.input_max_seq_len)
            np.save(self.pad_train_data_path, self.pad_data)
            np.save(self.pad_train_label_path, self.label)
            print("save remove stop word success as '{}'".format(self.pad_train_data_path))
            print("save remove stop word success as '{}'".format(self.pad_train_label_path))
            return self.pad_data, self.label
        else:
            train_data = np.load(self.pad_train_data_path, allow_pickle=True)
            train_label = np.load(self.pad_train_label_path, allow_pickle=True)
            print("load remove stop word success as '{}'".format(self.pad_train_data_path))
            print("load remove stop word success as '{}'".format(self.pad_train_label_path))
            return train_data, train_label


if __name__ == '__main__':
    data = pd.read_csv(config.train_data_path, sep=",", header=0)
    # data expan
    equalization = Data_equalization_initialization(data, config.column_names)
    train_data = equalization.group_equalization()
    # shuffle
    train_data.sample(frac=1)
    # data prepro
    preprocessing = Data_preprocessing(train_data, config.column_names)
    data, label = preprocessing.read_train_data()
    print(data[:3])
    print(label[:3])
