# -*- coding: utf-8 -*-
"""
@author: Fei Liang

@Reference: github: https://github.com/JasonZhang156/Sound-Recognition-Tutorial

tensorboard --logdir="./log/foldone_second"

"""

from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
import keras.backend as K
import esc10_input
import numpy as np
import models
import os

import matplotlib.pyplot as plt

def plot_learningCurve(history, epochs):
    # Plot training & validation accuracy values
    plt.figure()
    epoch_range = range(1, epochs+1)
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
    plt.savefig("accuracy.png")

    # Plot training & validation loss values
    plt.figure()
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
    plt.savefig("loss.png")


def use_gpu():
    """Configuration for GPU"""
    # from keras.backend.tensorflow_backend import set_session
    # from tf.compat.v1.keras.backend import set_session
    # import tensorflow 
    from tensorflow.compat.v1.keras.backend import set_session

    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)   # 使用第一台GPU
    # config = tf.ConfigProto()
    config = tf.compat.v1.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5  # GPU使用率为50%
    config.gpu_options.allow_growth = True    # 允许容量增长
    set_session(tf.compat.v1.InteractiveSession(config=config))


def CNN_train(test_fold, feat):
    """
    Training CNN using extracted feature
    :param test_fold: test fold of 5-fold cross validation
    :param feat: which feature to use
    """
    # 学习率衰减策略，每20个epoch衰减一次，变为0.1倍。
    def scheduler(epoch):
        if epoch in [20, 40]:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * 0.1)
            print("lr changed to {}".format(lr * 0.1))
        return K.get_value(model.optimizer.lr)

    # 读取特征数据
    # train_features, train_labels, test_features, test_labels = esc10_input.get_data(test_fold, feat)
    ob_folder = '/home/ascc/LF_Workspace/Motion-Trigered-Activity/Sound-Recognition-Tutorial/data/ascc_activity_1second/feature/ascc_logmel_total.npz'
    train_features, train_labels, test_features, test_labels = esc10_input.get_data_all(ob_folder, feat)

    print(train_features.shape)
    # 一些超参的配置
    epoch = 5
    num_class = 33
    batch_size = 128
    input_shape = (64, 138, 1)

    # 构建CNN模型
    model = models.CNN(input_shape, num_class)

    # 回调函数
    reduce_lr = LearningRateScheduler(scheduler)   # 学习率衰减
    logs = TensorBoard(log_dir='./log/fold{}/'.format(test_fold))   # 保存模型训练日志
    checkpoint = ModelCheckpoint('./saved_model/cnn_{}_fold{}_best.h5'.format(feat, test_fold),  # 保存在验证集上性能最好的模型
                                 monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
    # 训练模型
    # model.fit(train_features, train_labels, batch_size=batch_size, nb_epoch=epoch, verbose=1, validation_split=0.1,
    #           callbacks=[checkpoint, reduce_lr, logs])
    history = model.fit(train_features, train_labels, batch_size=batch_size, epochs=epoch, verbose=1, validation_split=0.1,
              callbacks=[checkpoint, reduce_lr, logs])

    plot_learningCurve(history, epoch)

    # 保存模型
    model.save('./saved_model/cnn_{}_fold{}.h5'.format(feat, test_fold))

    # 输出训练好的模型在测试集上的表现
    score = model.evaluate(test_features, test_labels)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    return score[1]


if __name__ == '__main__':
    use_gpu()  # 使用GPU
    dict_acc = {}  # results of each fold
    # 5-fold cross validation
    print('### [Start] Test model for ESC10 dataset #####')
    # for fold in [1, 2, 3, 4, 5]:
    #     print("## Start test fold{} model #####".format(fold))
    #     acc = CNN_train(fold, 'logmel')
    #     dict_acc['fold{}'.format(fold)] = acc
    #     print("## Finish test fold{} model #####".format(fold))
    # dict_acc['mean'] = np.mean(list(dict_acc.values()))
    # print(dict_acc)
    # print('### [Finish] Test model finished for ESC10 dataset #####')

    fold = 'one_second'
    acc = CNN_train(fold, 'logmel')
    print('acc: ',acc)
