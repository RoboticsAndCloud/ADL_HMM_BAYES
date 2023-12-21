import pickle
import socket
from keras.models import load_model
import tensorflow as tf

def server_call(model, inputs, layer_start):
  for layer in model.layers[layer_start:]:
    inputs = layer(inputs)
  return inputs

MODEL_SAVED_PATH = 'watch-saved-model-alex'
model = load_model(MODEL_SAVED_PATH)
model.summary()
for layer in model.layers:
    print('Layer:', layer)


TCP_IP1 ='10.227.97.123'
TCP_PORT1 = 5006
BUFFER_SIZE = 4096



s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP1, TCP_PORT1))
print('Waiting for partition point')
s.listen(2)


print('waiting for partial output')

conn, addr = s.accept()
data=[]
print ('Raspberry Device:',addr)
while 1:
 tensor = conn.recv(BUFFER_SIZE)
 if not tensor: break
 data.append(tensor)

part_outputs=pickle.loads(b"".join(data))

 #conn.send(data)
conn.close()


partition_outputs = tf.convert_to_tensor(part_outputs)
print(partition_outputs)

part_pnt = 8
final_outputs = server_call(model, partition_outputs, part_pnt)
final_output_send =final_outputs.numpy()
print('server output')
print(final_output_send)





