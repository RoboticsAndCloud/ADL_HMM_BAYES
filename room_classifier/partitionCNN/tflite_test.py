# Import TensorFlow
import tensorflow as tf
import cv2

# Initial Model
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3))

# The names of layers whose outputs we are interested in.
output_layer_names = [
    'predictions',
    'block_16_depthwise_relu',
    'out_relu',
    'Conv_1_bn',
    'Conv1',
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
img_size = 224
channel = 3
test_img = './Images_test' + '/' + 'hunter_room.jpg'
#test_img = './Images_test' + '/' + 'bedroom.jpg'
test_img = './watch_data/Images_test' + '/' + 'kitchen.jpg'
#img_array = cv2.imread(test_img,cv2.IMREAD_GRAYSCALE)
img_array = cv2.imread(test_img,cv2.IMREAD_COLOR)
new_array = cv2.resize(img_array, (img_size, img_size))
new_array=new_array.reshape(-1,img_size, img_size,channel)
prediction = model.predict([new_array])
print(prediction)
#print(categories[np.argmax(prediction)])
