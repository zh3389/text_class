# -*- encoding:utf-8 -*-
"""
@作者：Mr.zhang
@文件名：net.py
@时间：20-3-18  上午10:16
@文档说明:
"""
import os
import numpy as np
import config
import codecs
from keras import Sequential, regularizers
from keras_preprocessing import text
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dropout, Dense


class Init_embedded_matrix():
    def __init__(self):
        self.embedded_matrix_size = config.embedded_matrix_size
        self.tokenizer = self.get_tokenizer()
        self.word_embedding = self.loading_Word_Embedding()

    def get_tokenizer(self):
        self.tokenizer_path = config.data_preprocessing_config().tokenizer_path
        if not os.path.exists(self.tokenizer_path):
            print("please run data_preprocessing generate '{}'".format(self.tokenizer_path))
            exit()
        else:
            with open(self.tokenizer_path, "r") as f:
                tokenizer_json = f.read()
            tokenizer = text.tokenizer_from_json(tokenizer_json)
            return tokenizer

    def loading_Word_Embedding(self):
        '''
        :param path: wiki.zh.vec
        :return: embeddings_index
        '''
        from tqdm import tqdm
        print('loading word embeddings...')  # 加载字嵌入
        embeddings_index = {}  # Word vector dict
        with codecs.open(config.wiki_zh_vec_path, encoding='utf-8') as f:  # open the file
            for line in tqdm(f):
                values = line.rstrip().rsplit(' ')  # 删除 string 字符串末尾的指定字符（默认为空格） rsplit(以 ‘ ’对字符串进行分割)
                word = values[0]  # word
                coefs = np.asarray(values[1:], dtype='float32')  # Word vector to np.array
                embeddings_index[word] = coefs
        print('found %s word vectors' % len(embeddings_index))  # print len(dict) (111052 word vectors)
        return embeddings_index

    def get_embedding_matrix(self, embed_dim=300):
        '''
        :param embed_dim: embed dim=300
        :return: word_matrix
        word_index: {word1:1,
                     word2:2,
                     word3:3,
                     ...}
        word_embedding: {word1:[300,],
                         word2:[300,],
                         word3:[300,],
                         ...}
        word_matrix: {[300,],  # 对应词的词向量放置在对应的行(array)
                      [300,],
                      [300,],
                      ...}
        '''
        # We can now prepare our embedding matrix limiting to a max number of words:
        print('preparing embedding matrix...')
        words_not_found = []  # not found word
        nb_words = min(self.embedded_matrix_size, len(self.tokenizer.word_index))  # 确保设置的词频大小和实际统计的词频大小取最小
        word_matrix = np.zeros((nb_words, embed_dim))  # create zeros embedding matrix shape=(100000, 300)
        for word, i in self.tokenizer.word_index.items():  # ergodic word_index.items()
            if i >= nb_words:  # nb_words = 100000
                continue
            embedding_vector = self.word_embedding.get(
                word)  # embedding_vector = wiki.sample.vec.get(word) = values = (300, )
            if (embedding_vector is not None) and len(embedding_vector) > 0:
                # words not found in embedding index will be all-zeros.
                word_matrix[i] = embedding_vector
            else:
                words_not_found.append(word)
        print('number of null word embeddings: %d' % np.sum(np.sum(word_matrix, axis=1) == 0))  # 求出每行和为零的总共行数
        # It's interesting to look at the words not found in the embeddings
        print("sample words not found: ", np.random.choice(words_not_found, 100))
        return word_matrix, nb_words, embed_dim


class Model_init_parameter_config():
    def __init__(self, data, column_names):
        self.data = data
        self.data_column_name = column_names[0]
        self.class_column_name = column_names[1]
        self.num_class = self.get_num_class()
        self.input_max_seq_len = config.input_max_seq_len

    def get_num_class(self):
        # label_dict = one_hot(label_names)
        label_names = self.data[self.class_column_name].unique()
        num_classes = len(label_names)
        print("label_names:", label_names)
        return num_classes


class Init_model():
    def __init__(self):
        self.embedded_matrix, self.nb_words, self.embed_dim = Init_embedded_matrix().get_embedding_matrix()

    def Model(self,
              num_classes,
              input_max_seq_len,
              num_filters=512,
              weight_decay=1e-4):
        self.model = Sequential()
        self.model.add(Embedding(self.nb_words, self.embed_dim, weights=[self.embedded_matrix], input_length=input_max_seq_len, trainable=False,
                            name="input"))
        self.model.add(Dropout(0.2))
        self.model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
        self.model.add(GlobalMaxPooling1D())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(16, activation="relu", kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Dense(num_classes, activation="sigmoid", name="output"))
        self.model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
        return self.model


if __name__ == '__main__':
    import pandas as pd
    data = pd.read_csv(config.train_data_path, sep=",", header=0)
    model_parame = Model_init_parameter_config(data, config.column_names)
    print(model_parame.get_num_class())
    model = Init_model().Model(model_parame.num_class, model_parame.input_max_seq_len)
    model.summary()
