"""

Reference:https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter
https://tensorflow.google.cn/lite/performance/post_training_integer_quant?hl=zh-cn
https://www.tensorflow.org/lite/performance/post_training_integer_quant
"""

import tensorflow as tf

def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
    yield [input_value]

# Convert the model
saved_model_dir = './saved-model'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory

converter.optimizations = [tf.lite.Optimize.DEFAULT]


#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

#converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
#converter.allow_custom_ops=True
#converter.inference_type = tf.uint8    #tf.lite.constants.QUANTIZED_UINT8
#input_arrays = converter.get_input_arrays()
#converter.quantized_input_stats = {input_arrays[0]: (127.5, 127.5)} # mean, std_dev
#converter.default_ranges_stats = (0, 255)

#tflite_uint8_model = converter.convert()

#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
#converter.experimental_new_converter = True
#converter.experimental_new_quantizer = True
#converter.representative_dataset = representative_data_gen
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.inference_input_type = tf.uint8
#converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

# Save the model.
with open('int8_model.tflite', 'wb') as f:
  f.write(tflite_model)
