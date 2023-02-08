# -*- coding: utf-8 -*-
"""
@author: Fei Liang

@Reference: github: https://github.com/JasonZhang156/Sound-Recognition-Tutorial

tensorboard --logdir="./log/foldone_second"

"""

from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
import keras.backend as K
from keras import optimizers

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
    
    
    # labels = ['door_open_closed', 'eating', 'keyboard', 'pouring_water_into_glass', 'vacuum', 'drinking', 'flush_toilet', 'quiet', 'tv_news', 'washing_hand']
    labels = ['door_open_closed', 'eating', 'keyboard', 'pouring_water_into_glass', 'toothbrushing', 'vacuum', 'drinking', 'flush_toilet', 'microwave', 'quiet', 'tv_news', 'washing_hand']
    num_class = 12


    # 读取特征数据
    # total sample:  2324
    # train_feats: (1855, 64, 138)
    # test_feat: (469, 64, 138)
    # test_labes: (469,)
    # train_features, train_labels, test_features, test_labels = esc10_input.get_data(test_fold, feat)
    # ob_folder = '/home/ascc/LF_Workspace/Motion-Trigered-Activity/Sound-Recognition-Tutorial/data/ascc_activity_1second/feature/ascc_logmel_total.npz' # acc:0.9396
    # ob_folder = '/home/ascc/LF_Workspace/Bayes_model/ADL_HMM_BAYES_V2/ADL_HMM_BAYES/room_sound/sound_dataset/ascc_activity_1second/feature/ascc_logmel_total.npz'
    ob_folder = '/home/ascc/LF_Workspace/Bayes_model/Product_ADL/ADL_HMM_BAYES/room_sound/sound_dataset/ascc_activity_1second/feature/ascc_logmel_total.npz'
    train_features, train_labels, test_features, test_labels = esc10_input.get_data_all(ob_folder, feat, number_class=num_class)

    print('train_labels: ',train_labels.shape)

    train_features_subset = train_features[0:int(train_features.shape[0]/5)]
    train_labels_subset = train_labels[0:int(train_features.shape[0]/5)]

    print("convert feature mode:", train_features_subset.shape)
    print("convert lables mode:", train_labels_subset.shape)


    # exit(0)
    # # 一些超参的配置
    # epoch = 70
    # batch_size = 128
    # input_shape = (64, 138, 1)

    # # 构建CNN模型
    # model = models.CNN(input_shape, num_class)

    # # 回调函数
    # reduce_lr = LearningRateScheduler(scheduler)   # 学习率衰减
    # logs = TensorBoard(log_dir='./log/fold{}/'.format(test_fold))   # 保存模型训练日志
    # checkpoint = ModelCheckpoint('./saved_model/cnn_{}_fold{}_best.h5'.format(feat, test_fold),  # 保存在验证集上性能最好的模型
    #                              monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
    # # 训练模型
    # # model.fit(train_features, train_labels, batch_size=batch_size, nb_epoch=epoch, verbose=1, validation_split=0.1,
    # #           callbacks=[checkpoint, reduce_lr, logs])
    # history = model.fit(train_features, train_labels, batch_size=batch_size, epochs=epoch, verbose=1, validation_split=0.1,
    #           callbacks=[checkpoint, reduce_lr, logs])

    # plot_learningCurve(history, epoch)


    # # confusion matrix
    # from mlxtend.plotting import plot_confusion_matrix
    # from sklearn.metrics import confusion_matrix

    # # y_pred = model.predict_classes(X_test)

    # # print('X_test:', X_test)

    # predict_x=model.predict(test_features) 
    # y_pred=np.argmax(predict_x,axis=1)
    # print('y_pred len:', len(y_pred))
    # print('y_pred:', y_pred)
    # print('test_labels:', len(test_labels))

    # print('test_labels:', test_labels)
    # test_rounded_labels=np.argmax(test_labels, axis=1)
    # print('test_rounded_labels:', test_rounded_labels)

    # plt.figure()
    # mat = confusion_matrix(test_rounded_labels, y_pred)
    # cm = plot_confusion_matrix(conf_mat=mat, show_normed=True, figsize=(7,7))
    
    # # import seaborn as sns
    # # ax = plt.subplot()
    # # sns.set(font_scale=3.0) # Adjust to fit
    # # sns.heatmap(cm, annot=True, ax=ax, cmap="Blues", fmt="g");  

    # plt.show()
    # plt.savefig("cm_sound.png")

    # plt.figure()
    # mat = confusion_matrix(test_rounded_labels, y_pred)
    # cm = plot_confusion_matrix(conf_mat=mat, class_names=labels, show_normed=True, figsize=(7,7))
    
    # # import seaborn as sns
    # # ax = plt.subplot()
    # # sns.set(font_scale=3.0) # Adjust to fit
    # # sns.heatmap(cm, annot=True, ax=ax, cmap="Blues", fmt="g");  

    # plt.show()
    # plt.savefig("cm_sound1.png")


    # 保存模型
    # model.save('./saved_model/cnn_{}_fold{}.h5'.format(feat, test_fold))
    # MODEL_SAVED_PATH = 'sound-saved-model'
    # # model.save_weights('model.h5')
    # model.save(MODEL_SAVED_PATH)

    # 输出训练好的模型在测试集上的表现
    # score = model.evaluate(test_features, test_labels)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])

    ###############################################################################33
    # Quantization aware training example
    import tensorflow_model_optimization as tfmot


    from keras.models import load_model
    # MODEL_SAVED_PATH = 'saved-model_onlinedataset'

    # ml = load_model(MODEL_SAVED_PATH)

    model = load_model('./saved_model/cnn_logmel_foldone_second.h5')
    print("Load model successfully")

    quantize_model = tfmot.quantization.keras.quantize_model

    # q_aware stands for for quantization aware.
    q_aware_model = quantize_model(model)


    sgd = optimizers.gradient_descent_v2.SGD(lr=0.01, momentum=0.9, nesterov=True)  # 优化器为SGD

    # `quantize_model` requires a recompile.
    q_aware_model.compile(optimizer=sgd,
                loss="categorical_crossentropy",
                metrics=['accuracy'])

    q_aware_model.summary()


    train_features_subset = train_features[0:int(train_features.shape[0]/1)]
    train_labels_subset = train_labels[0:int(train_features.shape[0]/1)]

    q_aware_model.fit(train_features_subset, train_labels_subset,
                    batch_size=128, epochs=20, validation_split=0.1)


    converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    quantized_tflite_model = converter.convert()

    # Save the model.
    res_model = 'sound_default_model.tflite'
    with open(res_model, 'wb') as f:
        f.write(quantized_tflite_model)


    return 0


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