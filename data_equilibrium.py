# -*- encoding:utf-8 -*-
"""
@作者：Mr.zhang
@文件名：data_equilibrium.py
@时间：20-3-18  上午10:54
@文档说明:
"""
import pandas as pd


class Data_equalization_initialization:
    def __init__(self, data, column_names):
        self.data_column_name = column_names[0]
        self.class_column_name = column_names[1]
        self.data = data[[self.data_column_name, self.class_column_name]]
        self.supplementary_dict = self.get_replenishment_quantity()
        self.final_data = pd.DataFrame()

    def group_equalization(self):
        '''
        :return: Balance data by group
        '''
        grouped = self.data.groupby(self.class_column_name)
        for class_name, dataframe in grouped:
            temp_copy_data = self.data_copy(dataframe, self.supplementary_dict[class_name])
            self.final_data = pd.concat([self.final_data, temp_copy_data])
        return self.final_data

    def get_replenishment_quantity(self):
        '''
        :return: Return the multiples and remainders of supplementary data required for each category
        '''
        data_class_count = dict(self.data[self.class_column_name].value_counts())
        max_num = max(data_class_count.values())
        supplementary_dict = {}
        for item in data_class_count.items():
            temp_dict = {}
            temp_dict["multiple"] = max_num // item[1] - 1
            temp_dict["remainder"] = max_num % item[1]
            supplementary_dict[item[0]] = temp_dict
        return supplementary_dict

    def data_copy(self, dataframe, sup_dict):
        '''
        :param dataframe: data
        :param sup_dict: copy data is dict
        :return: final data
        '''
        temp_df = dataframe
        for i in range(sup_dict["multiple"]):
            dataframe = pd.concat([dataframe, temp_df])
        dataframe = pd.concat([dataframe, dataframe.sample(sup_dict["remainder"])])
        return dataframe


if __name__ == '__main__':
    data = pd.read_csv("./data/train_data.csv")
    column_names = ["data", "class"]
    print("数据均衡前:")
    print(data[column_names[1]].value_counts())
    equalization = Data_equalization_initialization(data, column_names)
    data = equalization.group_equalization()
    print("数据均衡后:")
    print(data[column_names[1]].value_counts())
