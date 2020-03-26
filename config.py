# -*- encoding:utf-8 -*-
"""
@作者：Mr.zhang
@文件名：config.py
@时间：20-1-16  上午10:26
@文档说明:
"""
import os
import numpy as np
import pandas as pd
train_data_path = "./data/train_data.csv"  # 训练数据的路径
column_names = ["data", "class"]
wiki_zh_vec_path = "./data/wiki.zh.vec"  # 词嵌入向量的路径
embedded_matrix_size = 10240  # 根据词频保留的词数,用于构建嵌入矩阵,并确定嵌入矩阵的大小
new_word_list_path = "./data/find_new_word/new_word_list.txt"  # 保存发现的新词列表的路径


class Compute_input_max_seq_len():
    # 计算模型输入长度
    def __init__(self, data):
        self.data = data
        self.input_len_threshold = 256
        self.default_inputmax_seq_len = 512
        self.input_max_seq_len = self.get_input_max_seq_len()
        print("模型输入长度为:{}".format(self.input_max_seq_len))

    def compute_input_max_seq_len(self):
        self.data['doc_len'] = self.data[column_names[0]].apply(lambda words: len(words.split(" ")))  # 添加一列用于存储每条数据的长度
        input_max_seq_len = np.round(self.data['doc_len'].mean() + self.data['doc_len'].std()).astype(int)  # 模型输入长度
        return input_max_seq_len

    def get_input_max_seq_len(self):
        input_max_seq_len = self.compute_input_max_seq_len()
        if input_max_seq_len > self.input_len_threshold:
            return input_max_seq_len
        else:
            return self.default_inputmax_seq_len


data = pd.read_csv(train_data_path)
compute_seq_len = Compute_input_max_seq_len(data)
input_max_seq_len = compute_seq_len.input_max_seq_len


class data_preprocessing_config():
    def __init__(self):
        self.tokenizer_path = "./data/tokenizer.json"  # 根据现有数据和tf-idf逆文件频率生成的词频文件
        self.pad_train_data_path = "./data/pad_train_data.npy"  # 用于存储去停用词后的data npy文件
        self.pad_train_label_path = "./data/pad_train_label.npy"  # 用于存储去去停用词后的label npy文件
        self.stopwordpath = "./data/stopwordlist/"  # 停用词列表路径


class Model_train_parameter_config():
    def __init__(self):
        self.validation_ratio = 0.1  # 测试集占总数据集的比例
        self.epochs = 512
        self.batch_size = 2
        self.learning_rate = 0.01
        self.learning_rate_decay = 0.95


# 创建部署文件路径
export_pb_path = "./save_model/deploy"
if not os.path.isdir(export_pb_path):
    os.makedirs(export_pb_path)
# 创建保存模型文件路径
save_h5_path = "./save_model/save/"
if not os.path.isdir(save_h5_path):
    os.makedirs(save_h5_path)
# 创建保存模型文件路径
train_logs = "./save_model/logs"
if not os.path.isdir(train_logs):
    os.makedirs(train_logs)


if __name__ == '__main__':
    pass