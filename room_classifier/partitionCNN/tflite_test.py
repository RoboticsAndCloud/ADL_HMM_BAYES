# Import TensorFlow
import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np

# Refer: https://groups.google.com/a/tensorflow.org/g/tflite/c/IgfpNtbU8bo/m/O8Qzzb5DAgAJ?pli=1

MODEL_SAVED_PATH = 'watch-saved-model-alex'
# Initial Model
#base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3))
base_model = load_model(MODEL_SAVED_PATH)
base_model.summary()

# The names of layers whose outputs we are interested in.
output_layer_names = [
    'Part_1',
    'Part_Final',
]
# Helper function: Print all model layer names: print([layer.name for layer in base_model.layers])
# Helper function: Visualize model with layer names: tf.keras.utils.plot_model(base_model, show_shapes=True, show_layer_names=True)
# Helper function: Save model using 'base_model.save('mobilenetv2.h5')' and visualize it using Netron (https://lutzroeder.github.io/netron/)

# Get outputs tensors
for layer in base_model.layers:
    print(layer.name)
output_layers = list(filter(lambda l: l.name in output_layer_names, base_model.layers))
output_tensors = list(map(lambda l: l.output, output_layers))

print(output_layers)

# Final Model
model_multiple_outputs = tf.keras.Model(inputs=base_model.input, outputs=output_tensors)

model = model_multiple_outputs

print('executing prediction')
img_size = 299
channel = 3
test_img = './Images_test' + '/' + 'hunter_room.jpg'
#test_img = './Images_test' + '/' + 'bedroom.jpg'
test_img = './watch_data/Images_test' + '/' + 'kitchen.jpg'
test_img = '/home/ascc/LF_Workspace/Bayes_model/IROS23/ADL_HMM_BAYES/room_classifier/partitionCNN/test_img/20230425115416.jpg'
#img_array = cv2.imread(test_img,cv2.IMREAD_GRAYSCALE)
img_array = cv2.imread(test_img,cv2.IMREAD_COLOR)
new_array = cv2.resize(img_array, (img_size, img_size))
new_array=new_array.reshape(-1,img_size, img_size,channel)
prediction = model.predict([new_array])
print(type(prediction[0][0]))
print(prediction[0][0].shape)
print(prediction[1])
print(len(prediction))
p_inputs = prediction[0]

prediction = prediction[1][0]
categories = ['bathroom','bedroom', 'kitchen','livingroom', 'hallway', 'door']
print(categories[np.argmax(prediction)])



#functions for partition call
def mobile_call(model, inputs, layer_end):
  for layer in model.layers[:layer_end]:
    inputs = layer(inputs)
  return inputs


def server_call(model, inputs, layer_start):
  for layer in model.layers[layer_start:]:
    inputs = layer(inputs)
  return inputs

layer_end = 8
res = server_call(model, p_inputs, layer_end)
print('================')
print('partion from Tensor Flow model res:', res)

res_model = 'watch-saved-model-alex_multioutput.tflite'
# Convert and save the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
# Save the model.
with open(res_model, 'wb') as f:
  f.write(tflite_model)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=res_model)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(output_details)



