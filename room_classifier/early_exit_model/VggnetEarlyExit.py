import tensorflow as tf

# library for download dataset
import tensorflow_datasets as tfds

# utils
import pickle
import time 

# visualization:
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# define parameters for the plotting
params_dict = {"axes.titlesize" : 24, "axes.labelsize" : 20,
               "lines.linewidth" : 4, "lines.markersize" : 10,
               "xtick.labelsize" : 16,"ytick.labelsize" : 16}

# here I set a random seed for reproducibility
tf.keras.utils.set_random_seed(1)

# Now we can import the data divided in train and validation
# dataset splitted as 80%-20% of the total percentage of inputs.
(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:70%]', 'train[70%:85%]', 'train[85%:]'],
    with_info=True,
    as_supervised=True,
)

# Using the metadata we can obtain the number of classes 
num_classes = metadata.features['label'].num_classes  # 5
# also, a mapping function that takes as input a label (int) 
# and return the associated type of flower (string).
get_label_name = metadata.features['label'].int2str


print(f"Number of training sample: {train_ds.cardinality()}")
print(f"Number of validation sample: {val_ds.cardinality()}")
print(f"Number of test sample: {test_ds.cardinality()}")

# We can define a preprocessing function with takes as input
# the pair image, label and act a resizing operation on the spatial dimension
# and rescaling the value normalizing them. 
def preprocess(image, label):
    # cast the image to float32
    image = tf.cast(image, tf.float32)
    # resize image to shape (96, 96)
    image = tf.image.resize(image, (96, 96))
    # normalize the RGB values in [0,1]
    image = image / 255.
    return image, label

# we can define also a augment function in which each image
# is randomly flip left or right
def augment(image, label):
  image = tf.image.random_flip_left_right(image)
  return image, label

# Apply the preprocess and augment function on the right dataset (validation and test doesn't have data augmentation)
# the batch size is 32 and applied only on training data for later usage. Validation batching is taken care in fit function.
batch_size = 32

train_ds_p = train_ds.map(preprocess).map(augment).shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_ds_p = val_ds.map(preprocess).batch(1).prefetch(tf.data.AUTOTUNE)
test_ds_p = val_ds.map(preprocess).batch(1).prefetch(tf.data.AUTOTUNE)


# Inspect the first batch of data
for x,y in train_ds_p.take(1):
  break

# Plotting the images
plt.figure(figsize=(20, 10))
plt.suptitle('Visualization of a batch of data', fontsize = 20)
for i, (image, label) in enumerate(zip(x,y)):
  # loop for plot
  ax = plt.subplot(4, 8, i + 1)
  plt.imshow(image)
  plt.title(f'Label: {get_label_name(label)}')
  plt.axis("off")

# check class distribution on the training data
datasets = ['train', 'validation', 'test']
fig, axs = plt.subplots(1,3, figsize = (20,5))
for i, dataset in enumerate([train_ds, val_ds, test_ds]):
    _, y = tuple(zip(*dataset))
    val = tf.unique_with_counts(y)
    _ = axs[i].bar(tf.cast(val.y, tf.float32), val.count, label = f"{datasets[i]} dataset")
    axs[i].set_xlabel("class label", fontsize = 15)
    axs[i].set_ylabel("frequency", fontsize = 15)
    axs[i].legend(fontsize = 15);

def add_ConvBlock(input_shape, n_conv_layers=1, n_filters=32, name = None):
    # Function that creates a Convolutional Block to be added to the model.
    # We need to specify the number of consecutive convolutional layer, 
    # than we perform a batch normalization after each conv, 
    # than use ReLU as activation functions
    # in the end perform a MaxPooling operation.
    inp = tf.keras.layers.Input(shape=input_shape)
    
    x = inp
    for i in range(n_conv_layers):
        x = tf.keras.layers.Conv2D(n_filters, 3, padding='same', 
                                   kernel_initializer = tf.keras.initializers.he_uniform)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        
    out = tf.keras.layers.MaxPool2D()(x)
    # return the Model
    return tf.keras.Model(inputs = [inp], outputs = [out], name=f"{name}") 


class VGGnet11(tf.keras.Model):
    "Class for the model VGGNET11 implemented using Keras Subclassing API"
    def __init__(self, num_classes, input_shape=(96, 96, 3)):
        super(VGGnet11, self).__init__()

        # initialize the layer of the architecture
        self.conv_blocks = [
                add_ConvBlock(input_shape, n_conv_layers = 1, n_filters = 64, name="conv_block1"),
                add_ConvBlock((48, 48, 64), n_conv_layers = 1, n_filters = 128, name="conv_block2"),
                add_ConvBlock((24, 24, 128), n_conv_layers = 2, n_filters = 256, name="conv_block3"),
                add_ConvBlock((12, 12, 256), n_conv_layers = 2, n_filters = 512, name="conv_block4"),
                add_ConvBlock((6, 6, 512), n_conv_layers = 2, n_filters = 512, name="conv_block5")
              ]
                   
        self.global_pool = tf.keras.layers.GlobalAvgPool2D()
        self.fc1 = tf.keras.layers.Dense(4096, activation='relu', kernel_initializer = tf.keras.initializers.he_uniform)  
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.fc2 = tf.keras.layers.Dense(4096, activation='relu', kernel_initializer = tf.keras.initializers.he_uniform) 
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.classification = tf.keras.layers.Dense(num_classes)

    def call(self, x, training=False):
        
        for conv_block in self.conv_blocks:
            x = conv_block(x, training=training)
           
        x = self.global_pool(x)
        x = self.dropout1(self.fc1(x), training = training)
        x = self.dropout2(self.fc2(x), training = training)
        return self.classification(x)   

    # override summary printing
    def summary(self):
        x = tf.keras.layers.Input(shape=(96,96,3))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x), name = "VGGNET11")
        return model.summary()    


# define compile object such metrics, loss and optimizer
cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
optimizer = tf.keras.optimizers.SGD(learning_rate = 1e-4, momentum = 0.9)

def create_model():
    # clean the model creation
    tf.keras.backend.clear_session()
    # instantiate the model and compile it
    model = VGGnet11(num_classes)
    model.compile(loss=cross_entropy, optimizer=optimizer, metrics=[accuracy])
    return model


# ----------- MODEL -------------------
vggnet11 = create_model()
vggnet11.summary()


# define callback: Early Stopping 
early_stopping_callback  = tf.keras.callbacks.EarlyStopping(monitor = 'val_sparse_categorical_accuracy', 
                                                            patience = 10,
                                                            restore_best_weights = True, 
                                                            verbose = 1)

# ---------- TRAINING --------------------
tf.keras.utils.set_random_seed(1)
with tf.device('/GPU:0'):
    history = vggnet11.fit(train_ds_p, validation_data=val_ds_p,
                        validation_batch_size = 32,
                        epochs=70, 
                        callbacks = [early_stopping_callback]) 


# ---------- SAVE history dict -------------
history = history.history
with open('history_vggnet11_base.pkl', 'wb') as f:
    pickle.dump(history, f)
     

# load the history 
with open('history_vggnet11_base.pkl', 'rb') as f:
    history = pickle.load(f)

# Learning curves
with plt.rc_context(params_dict):
    fig, ax = plt.subplots(1, 2, figsize = (20, 7))
    
    fig.suptitle("Learning curves of the model", fontsize = 25)
    ax[0].plot(history['loss'], label = "train", color = "blue")
    ax[0].plot(history['val_loss'], label = "val", color = "red")
    ax[0].legend(fontsize = 20)
    ax[0].set_xlabel("Epochs"); ax[0].set_ylabel("Loss")

    ax[1].plot(history['sparse_categorical_accuracy'], label = "train", color = "blue")
    ax[1].plot(history['val_sparse_categorical_accuracy'], label = "val", color = "red")
    ax[1].legend(fontsize = 20)
    ax[1].set_xlabel("Epochs"); ax[1].set_ylabel("Accuracy")
    plt.show()

y_pred = tf.math.argmax(tf.nn.softmax(logits = vggnet11.predict(test_ds_p, verbose = 0)), axis = -1)
y_true = tf.cast(tf.concat([y for x,y in iter(test_ds_p)], axis = 0), tf.int64)
acc = tf.reduce_mean(tf.cast(y_true == y_pred, tf.float32))

fig, ax = plt.subplots(figsize = (7,5))
cm = tf.math.confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='g', ax=ax); 

# labels, title and ticks
ax.set_xlabel(f'Predicted label\naccuracy={acc:.3f}; misclass={1-acc:.3f}', fontdict = {'fontsize': 15}, labelpad = 15);
ax.set_ylabel('True labels', fontdict = {'fontsize': 15}, labelpad = 15)
ax.set_title('Confusion Matrix', fontdict = {'fontsize': 20})
ax.set_yticklabels(['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses'], fontdict = {'fontsize': 12}) 
ax.set_xticklabels(['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses'], fontdict = {'fontsize': 12});


# inspect some figure
plt.figure(figsize=(20, 11))
plt.suptitle('Flower classification: Real Labels vs. Predictions', fontsize = 20)
for i, (image, label) in enumerate(test_ds_p.take(32)):
  y_pred = vggnet11(image)
  y_pred = tf.math.argmax(tf.nn.softmax(logits = y_pred), axis = 1)
  # loop for plot
  ax = plt.subplot(4, 8, i + 1)
  plt.imshow(image[0])
  plt.title(f'Label: {get_label_name(label[0])}\nPrediction: {get_label_name(y_pred[0])}')
  plt.axis("off")


# class ExitBranch(tf.keras.layers.Layer):
#     "Exit classifier branch"
#     def __init__(self, num_classes):
#       super(ExitBranch, self).__init__()
#       self.classifier = tf.keras.Sequential([
#         # convolution
#         tf.keras.layers.Conv2D(128, 3, padding='same', kernel_initializer = tf.keras.initializers.he_uniform,
#                               activation = "relu"),
#         # flattening (global pooling)
#         tf.keras.layers.GlobalAvgPool2D(),
#         # classifier
#         tf.keras.layers.Dense(num_classes)
#       ])
      
#     def call(self, x, training=False):
#       logits = self.classifier(x, training = training)
#       return logits

# class JointCategoricalCrossentropy(tf.keras.losses.Loss):
#     """
#     Class definition of the Joint Loss for the CrossEntropy:
#           L = CE(y, y_hat) + sum_e CE(y, y_e)
#     """
#     def __init__(self, lambda_exit):
#       super().__init__()
#       self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True) 
#       self.lambda_exit = lambda_exit
#       self.num_exit = len(self.lambda_exit)

#     def call(self, y_true, y_pred):
      
#       loss = self.loss_fn(y_true, y_pred[0])
#       for lambda_e, i in zip(self.lambda_exit, range(1, self.num_exit+1)):
#           loss += lambda_e*self.loss_fn(y_true, y_pred[i])
          
#       return loss

# def entropy(logits):
#     # Function to compute the Entropy H (and optionally the confidence p_e)
    
#     probs = tf.nn.softmax(logits)
#     # compute the entropy (normalized in [0,1])
#     H = -tf.reduce_mean(probs * tf.math.log(probs), axis = -1)
#     # get the confidence of exit e (as option for the exit condition)
#     #p_e = tf.reduce_max(probs, axis = -1)
#     return H


# # instantiate the losses
# ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True) 
# # instantiate accuracy and loss trackers
# num_exits = 6
# accuracy_trackers = [tf.keras.metrics.SparseCategoricalAccuracy(name = f'accuracy_exit{i}') for i in range(1, num_exits+1)]
# loss_tracker = tf.keras.metrics.Mean(name="loss")  

# class ExitVGGnet11(tf.keras.Model):
#     "Class for the model VGGNET11 with ExitBranch implemented using Keras Subclassing API"
#     def __init__(self, num_classes, input_shape=(96, 96, 3), exit_threshold = 0.):
#         super(ExitVGGnet11, self).__init__()

#         # set the exit threshold for inference
#         self.exit_threshold = exit_threshold

#         # architecture layers
#         self.conv_blocks = [
#                 add_ConvBlock(input_shape, n_conv_layers = 1, n_filters = 64, name="conv_block1"),
#                 add_ConvBlock((48, 48, 64), n_conv_layers = 1, n_filters = 128, name="conv_block2"),
#                 add_ConvBlock((24, 24, 128), n_conv_layers = 2, n_filters = 256, name="conv_block3"),
#                 add_ConvBlock((12, 12, 256), n_conv_layers = 2, n_filters = 512, name="conv_block4"),
#                 add_ConvBlock((6, 6, 512), n_conv_layers = 2, n_filters = 512, name="conv_block5")
#               ]
        
#         # add exit at the end of every convolution
#         self.exit_layers = [ExitBranch(num_classes) for _ in range(len(self.conv_blocks))]

#         self.global_pool = tf.keras.layers.GlobalAvgPool2D()
#         self.fc1 = tf.keras.layers.Dense(4096, activation='relu', kernel_initializer = tf.keras.initializers.he_uniform)  
#         self.dropout1 = tf.keras.layers.Dropout(0.5)
#         self.fc2 = tf.keras.layers.Dense(4096, activation='relu', kernel_initializer = tf.keras.initializers.he_uniform) 
#         self.dropout2 = tf.keras.layers.Dropout(0.5)
#         self.classification = tf.keras.layers.Dense(num_classes)

#     def call(self, x, training=False):
#         out = [] # list for storing the outputs {y_e}_e=1^{E+1}
#         exit = 0 # initialize exit which are from 1 to E+1
#         for conv_block, exit_e in zip(self.conv_blocks, self.exit_layers):
#             exit += 1 # update the exit value
#             x = conv_block(x, training=training)
#             logits_e = exit_e(x, training=training)
#             out.append(logits_e) # append the current exit prediction
#             # compute stats
#             H = entropy(logits_e)
#             if not training and tf.math.less(H,self.exit_threshold):
#                 return logits_e, exit, H

#         exit += 1 # update the exit value
#         x = self.global_pool(x)
#         x = self.dropout1(self.fc1(x), training = training)
#         x = self.dropout2(self.fc2(x), training = training)
#         logits = self.classification(x)
#         out.append(logits)

#         H = entropy(logits)
#         if not training: # usual exit
#             return logits, exit, H
#         return out

#     def train_step(self, data):
#         x, y = data
#         with tf.GradientTape() as tape:
#             logits = self(x, training=True) 
#             loss = joint_cross_entropy(y, logits)

#         # Compute gradients
#         trainable_vars = self.trainable_variables
#         gradients = tape.gradient(loss, trainable_vars)
        
#         # Update weights
#         self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
#         # Compute our own metrics (and losses)
#         # during the training I have and accuracy for every exit
#         loss_tracker.update_state(loss)
#         [tracker.update_state(y, logits[i]) for i, tracker in enumerate(accuracy_trackers)]

#         trackers = {'loss': loss_tracker.result()}
#         trackers.update({f"accuracy_exit{i+1}": tracker.result() for i,tracker in enumerate(accuracy_trackers)})
#         return trackers

#     def test_step(self, data):
#         x, y = data
#         logits, exit, _ = self(x, training=False)
        
#         # Update the metrics.
#         accuracy_trackers[0].update_state(y, logits)

#         return {"accuracy": accuracy_trackers[0].result()}
        
#     @property
#     def metrics(self):
#         return [loss_tracker, *accuracy_trackers]


# # define the optimizer: low learning rate for a smooth convergence
# optimizer = tf.keras.optimizers.SGD(learning_rate = 1e-4, momentum = 0.9)

# def create_model(threshold = None):
#     # clean the model creation
#     tf.keras.backend.clear_session()
#     # instantiate the model and compile it
#     model = ExitVGGnet11(num_classes, exit_threshold = threshold)
#     model.compile(optimizer=optimizer, run_eagerly = True)
#     return model



# # define callback: Early Stopping 
# early_stopping_callback  = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', 
#                                                             patience = 10,
#                                                             restore_best_weights = True, 
#                                                             verbose = 1)

# # Instantiate the model
# joint_cross_entropy = JointCategoricalCrossentropy(lambda_exit = [1., 1., 1., 1., 1. ])
# model1 = create_model(threshold = 0.)
# # ---------- TRAINING --------------------
# tf.keras.utils.set_random_seed(1)
# with tf.device('/GPU:0'):
#     history = model1.fit(train_ds_p, validation_data=val_ds_p,
#                         epochs=70, 
#                         callbacks = [early_stopping_callback]) 
# # ---------- SAVE -------------------------
# with open(f'history_exitVggnet11_training1.pkl', 'wb') as f:
#     pickle.dump(history.history, f)


# joint_cross_entropy = JointCategoricalCrossentropy(lambda_exit = [0.3, 0.3, 0.3, 0.3, 0.3])
# model2 = create_model(threshold = 0.)

# # ---------- TRAINING --------------------
# tf.keras.utils.set_random_seed(1)
# with tf.device('/GPU:0'):
#     history = model2.fit(train_ds_p, validation_data=val_ds_p,
#                         epochs=70, 
#                         callbacks = [early_stopping_callback]) 
# # ---------- SAVE -------------------------
# with open(f'history_exitVggnet11_training2.pkl', 'wb') as f:
#     pickle.dump(history.history, f)


# def expected_savings(npreds_by_exit, num_exits):
#     # Function to compute the Expected Saving as 
#     # https://arxiv.org/pdf/2004.12993.pdf
#     Ni = tf.cast(npreds_by_exit, tf.int32)
#     n = num_exits

#     num = tf.map_fn(
#         lambda i: i * Ni[i-1], 
#         tf.range(1, n+1, dtype=tf.int32), 
#         fn_output_signature=(tf.int32))
  
#     den = tf.map_fn(
#         lambda i: n * Ni[i-1], 
#         tf.range(1, n+1, dtype=tf.int32), 
#         fn_output_signature=(tf.int32))
#     return 1 - tf.reduce_sum(num)/tf.reduce_sum(den)

# def nan_to_zero(tensor):
#     # Function to convert NaN value to 0 (for visualization puourpose)
#     return tf.where(tf.math.is_nan(tensor), tf.zeros_like(tensor), tensor)

# def inference_function(model, y_true, threshold_seq, name = None):
#     # Function to perform inference step and return some statistic about the model
#     # 1. Test Accuracy
#     # 2. Accuracy by exit, Number of prediction by exit
#     # 3. Expected Savings
#     # return a dictionary with this object
#     for threshold in threshold_seq:
#         model.exit_threshold = threshold
#         t_history = {}
#         # predict over the test set and store inference time
#         y_preds, exits, entropies = model.predict(test_ds_p, verbose = 0)
#         # compute test accuracies
#         is_correct = tf.cast(tf.argmax(y_preds, -1) == y_true, tf.float32)
#         test_accuracy = tf.reduce_mean(is_correct)
#         acc_by_exit = tf.map_fn(
#             lambda i: tf.experimental.numpy.nanmean(is_correct[exits == i]), 
#             tf.range(1, num_exits+1, dtype=tf.int64), 
#             fn_output_signature=(tf.float32))
#         npreds_by_exit = tf.map_fn(
#             lambda i: tf.experimental.numpy.nansum(tf.cast(exits == i, tf.float32)), 
#             tf.range(1, num_exits+1, dtype=tf.int64), 
#             fn_output_signature=(tf.float32))
#         # expected saving
#         expected_saving = expected_savings(npreds_by_exit, num_exits)
        
#         ## save the history file
#         t_history['test_accuracy'] = test_accuracy
#         t_history['expected_savings'] = expected_saving
#         t_history['acc_by_exit'] = nan_to_zero(acc_by_exit)
#         t_history['npreds_by_exit'] = nan_to_zero(npreds_by_exit)
#         with open(f'history_exitVggnet11_{name}_{threshold}.pkl', 'wb') as f:
#             pickle.dump(t_history, f)



# # retrieve true label
# y_true = tf.cast(tf.concat([y for x,y in iter(val_ds_p)], axis = 0), tf.int64)
# # define threshold sequence
# threshold_seq = [0., 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.]
# # compute the inference over the two different models
# inference_function(model1, y_true, threshold_seq, name = 1)
# inference_function(model2, y_true, threshold_seq, name = 2)


# # !zip -r /content/training_histories.zip /content/
# from google.colab import files
# files.download("/content/training_histories.zip")

# with open(f"history_exitVggnet11_1_0.0.pkl", "rb") as handle:
#     history1 = pickle.load(handle)

# with open(f"history_exitVggnet11_2_0.0.pkl", "rb") as handle:
#     history2 = pickle.load(handle)

# no_ee_accuracy1 = history1['test_accuracy']
# no_ee_accuracy2 = history2['test_accuracy']

# print(f"The test accuracy of the EE model using lambda_e = 1 is: {no_ee_accuracy1*100:0.01f}%")
# print(f"The test accuracy of the EE model using lambda_e = 0.3 is: {no_ee_accuracy2*100:0.01f}%")

# with open(f"history_exitVggnet11_training1.pkl", "rb") as handle:
#     history_train1 = pickle.load(handle)
# with open(f"history_exitVggnet11_training2.pkl", "rb") as handle:
#     history_train2 = pickle.load(handle)

# names = ['accuracy_exit1', 'accuracy_exit4', 'accuracy_exit6', 'val_accuracy']
# with plt.rc_context(params_dict):
#     fig, axs = plt.subplots(1,2, figsize = (15,5))
#     for name in names:
#         axs[0].plot(history_train1[name], label = f"{name}")
#         axs[1].plot(history_train2[name], label = f"{name}")
#     axs[0].set_title("Model 1 (lambda = 1.0)"); axs[1].set_title("Model 2 (lambda = 0.3)")
#     axs[0].set_xlabel("epoch"); axs[1].set_xlabel("epoch")
#     axs[0].set_ylabel("Accuracy")
#     axs[0].legend(fontsize = 15)
#     axs[1].legend(fontsize = 15)

# def plotting_function(threshold_seq = [0.05, 0.2, 0.3], path = None, no_ee_accuracy = None):
#     # Function for plotting visualization

#     # setting some parameters for the visualization pourpose 
#     colors = ["red", "orange", "green"]
#     props = dict(boxstyle='round', facecolor='white', alpha=0.3)
#     x_axis = tf.range(1,7, dtype = tf.float32)

#     with plt.rc_context(params_dict):
#         fig, axs = plt.subplots(1,3, figsize = (17,5))
#         axs = axs.flatten()
#         plt.suptitle('Number of output samples by layer at vary the threshold', fontsize = 20, y = 1.05)

#         for i,t in enumerate(threshold_seq):

#             # load the dictionary with the saved information
#             with open(f"{path}_{t}.pkl", "rb") as handle:
#                 history = pickle.load(handle) 

#             # store the object of interest
#             normalized_npreds_by_exit = history['npreds_by_exit'] / tf.reduce_sum(history['npreds_by_exit'])
#             # plot the average number of prediction at each exit
#             axs[i].bar(x_axis, normalized_npreds_by_exit, 
#                   align='center', edgecolor = "black", linewidth = 3, width=0.35, label = "fraction of input",
#                   color = colors[i])
            
#             # add accuracy per exits over each bin
#             accuracy_per_exits = history['acc_by_exit']
#             rects = axs[i].patches
#             for rect, label in zip(rects, accuracy_per_exits):
#                 height = rect.get_height()
#                 axs[i].text(
#                     rect.get_x() + rect.get_width() / 2, height + 0.02, f"{label*100:.1f}%", ha="center", va="bottom", 
#                     fontsize = 13, fontweight = 'bold'
#                 )

#             accuracy = history['test_accuracy']
#             expected_saving = history['expected_savings']
#             # add a box and naming axis
#             text = f"Threshold = {t}\nAccDrop = {(no_ee_accuracy-accuracy)/no_ee_accuracy *100:.1f}%\nSavings = {(expected_saving *100):.1f}%"
#             axs[i].text(3.95, 1.05, text,  fontsize=15, verticalalignment='top', bbox=props)
#             axs[i].set_xlabel('Exit Layer')
#             axs[i].set_ylim([0,1.1])

#         axs[0].set_ylabel('Fraction of dataset')
#         fig.tight_layout();

# plotting_function(path = "history_exitVggnet11_1", no_ee_accuracy = no_ee_accuracy1)

# plotting_function(path = "history_exitVggnet11_2", no_ee_accuracy = no_ee_accuracy2)


















