# -*- encoding:utf-8 -*-
"""
@作者：Mr.zhang
@文件名：train.py
@时间：20-3-18  上午10:17
@文档说明:
"""
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler, EarlyStopping
import config
import numpy as np
import pandas as pd
import net
import data_preprocessing
########################################################################################################################
# export model code
########################################################################################################################
import keras
import os
import tensorflow as tf
from tensorflow.python.util import compat
from keras import backend as K
import config


def export_savedmodel(model, save_path=config.export_pb_path):
    model_path = save_path  # 模型保存的路径
    model_version = 0  # 模型保存的版本
    # 从网络的输入输出创建预测的签名
    model_signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'input': model.input}, outputs={'output': model.output})
    # 使用utf-8编码将 字节或Unicode 转换为字节
    export_path = os.path.join(compat.as_bytes(model_path), compat.as_bytes(str(model_version)))  # 将保存路径和版本号join
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)  # 生成"savedmodel"协议缓冲区并保存变量和模型
    builder.add_meta_graph_and_variables(  # 将当前元图添加到savedmodel并保存变量
        sess=K.get_session(),  # 返回一个 session 默认返回tf的sess,否则返回keras的sess,两者都没有将创建一个全新的sess返回
        tags=[tf.saved_model.tag_constants.SERVING],  # 导出模型tag为SERVING(其他可选TRAINING,EVAL,GPU,TPU)
        clear_devices=True,  # 清除设备信息
        signature_def_map={  # 签名定义映射
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:  # 默认服务签名定义密钥
                model_signature  # 网络的输入输出策创建预测的签名
        })
    builder.save()  # 将"savedmodel"协议缓冲区写入磁盘.
    print("save model pb success ...")
########################################################################################################################
# export model code
########################################################################################################################


class Trainer():
    def __init__(self, data, column_names):
        train_config = config.Model_train_parameter_config()
        self.data = data
        self.column_names = column_names
        # train parameter
        self.learning_rate = train_config.learning_rate
        self.learning_rate_decay = train_config.learning_rate_decay
        self.epochs = train_config.epochs
        self.validation_ratio = train_config.validation_ratio
        self.batch_size = train_config.batch_size

        self.read_data()
        self.build_model()
        self.monitor_tools()

    def build_model(self):
        self.model_parame = net.Model_init_parameter_config(self.data, self.column_names)
        self.model = net.Init_model().Model(self.model_parame.num_class, self.model_parame.input_max_seq_len)
        self.model.summary()
        return self.model

    def label_transfor_one_hot(self, label):
        class_list = self.data[self.column_names[1]].unique()
        label_dict = {}
        one_hot_array = np.eye(len(class_list), dtype=int)  # generate 对角矩阵
        for i in range(len(class_list)):
            label_dict[class_list[i]] = one_hot_array[i]
        train_label = np.array(list(map(lambda x: label_dict[x], label)))
        print("label_dict:", label_dict)
        return train_label

    def read_data(self):
        # data expan
        equalization = data_preprocessing.Data_equalization_initialization(self.data, self.column_names)
        train_data = equalization.group_equalization()
        # shuffle
        train_data.sample(frac=1)
        # data prepro
        preprocessing = data_preprocessing.Data_preprocessing(train_data, self.column_names)
        self.train_data, self.train_label = preprocessing.read_train_data()
        self.train_label = self.label_transfor_one_hot(self.train_label)
        return self.train_data, self.train_label

    def monitor_tools(self):
        lr_schedule = lambda epoch: self.learning_rate * self.learning_rate_decay ** epoch
        learning_rate = np.array([lr_schedule(i) for i in range(self.epochs)])
        self.changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
        print("学习率前五:", learning_rate[:5])
        print("学习率后五:", learning_rate[-5:])
        self.checkpointer = ModelCheckpoint(filepath=config.save_h5_path + "model.h5", monitor='val_accuracy', verbose=1,
                                            save_best_only=True, mode='max')
        self.tensorboard = TensorBoard(log_dir=config.train_logs, histogram_freq=0, write_graph=True, write_images=True)
        self.earlystopping = EarlyStopping(monitor='loss', min_delta=0.005, patience=5, verbose=0, mode='min',
                                           baseline=None, restore_best_weights=False)

    def train(self):
        self.model.fit(self.train_data, self.train_label,
                       validation_split=self.validation_ratio,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       shuffle=True,
                       verbose=1,
                       callbacks=[self.checkpointer,
                                  self.tensorboard,
                                  self.changelr,
                                  self.earlystopping
                                  ])
        self.model.save(config.save_h5_path + "final_model.h5")


if __name__ == '__main__':
    data = pd.read_csv("./data/train_data.csv", sep=",", header=0)
    column_names = ["data", "class"]
    train_model = Trainer(data, column_names)
    train_model.train()
    # export model
    model = keras.models.load_model(config.save_h5_path + "model.h5")
    export_savedmodel(model, save_path=config.export_pb_path)