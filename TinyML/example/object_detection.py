import numpy as np
from tflite_runtime.interpreter import Interpreter
import time


# Load the TFLite model and allocate tensors.
interpreter = Interpreter(model_path="home_model.tflite")
#interpreter = Interpreter(model_path="dynamic_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
#print('input_details:',input_details)

output_details = interpreter.get_output_details()
output_details


from PIL import Image

image_path = './hometest/'

#im = Image.open("cats.jpg")
# im = Image.open("umbrella.jpg")
# im = Image.open("safari.jpg")
#im = Image.open(image_path + "bathroom2.jpg")
im = Image.open(image_path + "livingroom_test.jpg")
#im = Image.open(image_path + "livingroom_test.jpg").convert('RGB').resize((299, 299))


print(im.size)
im


res_im = im.resize((299, 299))
print('res_im:', res_im)

#np_res_im = im
np_res_im = np.expand_dims(res_im, axis=0).astype('float32')
#print('np_res_im:',np_res_im)

input_mean = 0.
input_std = 255.
np_res_im = (np.float32(np_res_im) - input_mean) / input_std

#np_res_im = np.array(res_im) # np.asarray(img1) which will show the values [0, 255]:
#np_res_im = (np_res_im).astype('float32')
#print('np_res_im:',np_res_im)


input_details[0]['shape']

#np_res_im.shape

if len(np_res_im.shape) == 3:
    np_res_im = np.expand_dims(np_res_im, 0)
# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np_res_im

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

start = time.time()
print('start:')
interpreter.set_tensor(input_details['index'], input_data)
#print('input_details2:',input_details)
end = time.time()
print('Time cost:', np.round(end-start, 3))

interpreter.invoke()
end = time.time()
print('Time cost:', np.round(end-start, 3))

output = interpreter.get_tensor(output_details["index"])[0]
# np.argmax()
print('output:',output)
prediction_classes = np.argmax(output)
print('prediction_classes:', prediction_classes)

output_details = interpreter.get_output_details()
#print("output details:", output_details)
output_details = interpreter.get_output_details()[0]
#print("output2 details:", output_details)
#output = np.squeeze(interpreter.get_tensor(output_details['index']))
output = interpreter.get_tensor(output_details['index'])
print('output1:', output)

# todo:
# np.argmax()
prediction_classes = np.argmax(output)
print('prediction_classes:', prediction_classes)

exit(0)

scale, zero_point = output_details['quantization']
output = scale * (output - zero_point)
print('output:', output)

ordered = np.argpartition(-output, 1)
#return [(i, output[i]) for i in ordered[:top_k]][0]


# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
outputLocations = interpreter.get_tensor(output_details[0]['index'])
print(outputLocations.shape)

end = time.time()
print('Time cost:', np.round(end-start, 3))

outputClasses = interpreter.get_tensor(output_details[1]['index'])
print(outputClasses.shape)
outputScores = interpreter.get_tensor(output_details[2]['index'])
print(outputScores.shape)
numDetections = interpreter.get_tensor(output_details[3]['index'])
print(numDetections.shape)
end = time.time()
print('Time cost:', np.round(end-start, 3))

label_names = [line.rstrip('\n') for line in open("labelmap.txt")]
label_names = np.array(label_names)

numDetectionsOutput = int(np.minimum(numDetections[0],10))
numDetectionsOutput

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

#for i in range(numDetectionsOutput):
    # Create figure and axes
    #fig, ax = plt.subplots()

    # Display the image
    #ax.imshow(res_im)

    # Create a Rectangle patch
#    inputSize = 300
#    left = outputLocations[0][i][1] * inputSize
#    top = outputLocations[0][i][0] * inputSize
#    right = outputLocations[0][i][3] * inputSize
#    bottom = outputLocations[0][i][2] * inputSize
#    class_name = label_names[int(outputClasses[0][i])]
#    print("Output class: "+class_name+" | Confidence: "+ str(outputScores[0][i]))
#    rect = patches.Rectangle((left, bottom), right-left, top-bottom, linewidth=1, edgecolor='r', facecolor='none', label=class_name)

    # Add the patch to the Axes
    #ax.add_patch(rect)

    #plt.show()


# Create figure and axes
fig, ax = plt.subplots()

# Display the image
ax.imshow(res_im)

for i in range(numDetectionsOutput):

    # check the confidence
    confidence = outputScores[0][i]
    if confidence < 0.35:
        continue
    # Create a Rectangle patch
    inputSize = 300
    left = outputLocations[0][i][1] * inputSize
    top = outputLocations[0][i][0] * inputSize
    right = outputLocations[0][i][3] * inputSize
    bottom = outputLocations[0][i][2] * inputSize
    class_name = label_names[int(outputClasses[0][i])]
    print("Output class: "+class_name+" | Confidence: "+ str(outputScores[0][i]))
    rect = patches.Rectangle((left, bottom), right-left, top-bottom, linewidth=1, edgecolor='r', facecolor='none', label='class_name')

    # Add the patch to the Axes
    ax.add_patch(rect)
    rx, ry = rect.get_xy()
    cx = rx + rect.get_width()/2.0
    cy = ry + rect.get_height()/1.0
    ax.annotate(class_name, (cx, cy), color='green', weight='bold', fontsize=10, ha='center', va='center')

plt.savefig('res.png')
#plt.show()	

