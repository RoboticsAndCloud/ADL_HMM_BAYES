# Reference https://www.tensorflow.org/lite/models/modify/model_maker/image_classification

import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt

image_path = '/home/ascc/LF_Workspace/Bayes_model/Product_ADL/ADL_HMM_BAYES/TinyML/room_classifier/room_ascc_dataset/labelled/'
data = DataLoader.from_folder(image_path)
train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)

plt.figure(figsize=(10,10))
for i, (image, label) in enumerate(data.gen_dataset().unbatch().take(25)):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(image.numpy(), cmap=plt.cm.gray)
  plt.xlabel(data.index_to_label[label.numpy()])
plt.show()


# model = image_classifier.create(train_data, validation_data = validation_data, epochs = 20)  # default EfficientNet-Lite0.
model = image_classifier.create(train_data, model_spec=model_spec.get('mobilenet_v2'), validation_data=validation_data, epochs = 10)


model.summary()


loss, accuracy = model.evaluate(test_data)

#model.export(export_dir='.')
model.export(export_dir='.', tflite_filename='ascc_mobilev2_model_default.tflite')



model.evaluate_tflite('ascc_mobilev2_model_default.tflite', test_data)
