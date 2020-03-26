# -*- encoding:utf-8 -*-
"""
@作者：Mr.zhang
@文件名：save_new_word.py
@时间：20-3-25  下午4:32
@文档说明:
"""
from data.find_new_word import find_new_word as find_word
import pandas as pd
import config


class Find_new_word():
    def __init__(self, data, new_word_path=config.new_word_list_path):
        self.data = data
        self.write_path = new_word_path
        print("如更换数据集需手动删除{}此文件用于生成新词列表...".format(new_word_path))

    def find(self):
        from tqdm import tqdm
        for _, item in tqdm(self.data.iteritems()):
            word_list = list(find_word(item).keys())
            self.write_local_txt(word_list)

    def write_local_txt(self, word_list):
        with open(self.write_path, "a+") as file:
            for word in word_list:
                file.write(str(word) + "\n")


if __name__ == '__main__':
    data = pd.read_csv("train_data.csv")["data"]
    find_new_word = Find_new_word(data)
    find_new_word.find()