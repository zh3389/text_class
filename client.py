import json
import os
import jieba
import numpy as np
import requests
from keras_preprocessing import sequence, text


new_word_list_path = "./data/find_new_word/new_word_list.txt"  # 非必须
stopwordpath = "./data/stopwordlist/"   # 必须
tokenizer_path = "./data/tokenizer.json"  # 必须
post_url = 'http://localhost:8501/v1/models/docker_test:predict'  # 必须
class_dict = {0: "phone", 1: "bank", 2: "country"}  # 必须
input_max_len = 512  # 必须


class Data_preprocessing():
    def __init__(self):
        self.input_max_len = input_max_len
        self.stopwordpath = stopwordpath
        self.new_word_list = new_word_list_path
        # jieba.add_word("添加jieba不能分的词")
        if os.path.exists(self.new_word_list):
            self.add_jieba_word()
        self.get_stopwords_list()
        self.tokenizer = self.load_tokenizer()

    def load_tokenizer(self):
        with open(tokenizer_path, "r") as f:
            tokenizer_json = f.read()
        tokenizer = text.tokenizer_from_json(tokenizer_json)
        return tokenizer

    def get_stopwords_list(self):
        '''
        :return: get stopwordpath all stop word
        '''
        print("self.stopwordpath:", self.stopwordpath)
        self.stopwords = []
        for item in os.listdir(self.stopwordpath):
            self.stopwords += list(set([line.strip() for line in open(self.stopwordpath + item, 'r', encoding='utf-8').readlines()]))
        return self.stopwords

    def add_jieba_word(self):
        '''
        :return: add new word to jieba
        '''
        with open(self.new_word_list, "r") as file:
            print("add jieba word...")
            for word in file.readlines():
                word = word.strip()
                jieba.add_word(word)
        return None

    def remove_stop_words(self, sentence):
        processed_docs_train = []
        for word_list in sentence:
            outstr = []
            word = jieba.cut(word_list.strip().rstrip(), cut_all=True)
            for i in word:
                if i not in self.stopwords:
                    if i != '\t' and i != ' ':
                        outstr.append(i)
            processed_docs_train.append(" ".join(outstr))
        return processed_docs_train

    def precess_data(self, input_data):
        stopword_data = self.remove_stop_words(input_data)
        sequences_data = self.tokenizer.texts_to_sequences(stopword_data)
        pad_data = sequence.pad_sequences(sequences_data, maxlen=self.input_max_len)
        return pad_data


class Predictor():
    def __init__(self):
        self.post_url = post_url
        self.class_dict = class_dict
        self.data_preprocessing = Data_preprocessing()

    def predict(self, string):
        input_data = self.data_preprocessing.precess_data([string])
        payload = {"instances": [{'input': input_data.tolist()[0]}]}
        r = requests.post(post_url, json=payload)
        output = json.loads(r.content.decode('utf-8'))["predictions"]
        text_class = int(np.argmax(output, axis=1))
        predict_score = max(output[0])
        print(self.class_dict[text_class])
        print(predict_score)
        return self.class_dict[text_class]


if __name__ == '__main__':
    pre = Predictor()
    pre.predict("""
            需要预测的文本
            """)
