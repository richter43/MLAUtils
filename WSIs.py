#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[ ]:


# !apt update && apt install -y openslide-tools
# !pip install openslide-python


# In[ ]:


# !rm -r "/content/drive/MyDrive/Teaching&Thesis/Teaching_dataset/teaching-MLinAPP"
# !git clone https://github.com/frpnz/teaching-MLinAPP.git "/content/drive/MyDrive/Teaching&Thesis/Teaching_dataset/teaching-MLinAPP"


# In[ ]:


import os
import sys
import openslide
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
import matplotlib.pyplot as plt
import openslide.deepzoom as dz
from sklearn import model_selection
rootdir_wsi = "/space/ponzio/CRC_ROIs_4_classes/"
rootdir_src = "/space/ponzio/teaching-MLinAPP/src/"
sys.path.append(rootdir_src)
from resnet import ResNet
from dataset_wsi import DatasetWSI
# ----------------------
tile_size = 300
overlap = 6
epochs = 100
learning_rate = 0.001
batch_size = 64
class_dict = {
    "AC": 0,
    "H": 1,
    "AD": 2
}
checkpoint_filepath = './models_crc/checkpoint_crc_3_cls_224'                                                                                                                                                             
# ----------------------
num_classes = len(class_dict.keys())
wsi_file_paths = glob(os.path.join(rootdir_wsi, '*.svs'))
df = pd.DataFrame([os.path.basename(slide).split('.')[0].split('_') for slide in wsi_file_paths], columns=["Patient",
                                                                                                           "Type",
                                                                                                           "Sub-type",
                                                                                                           "Dysplasia",
                                                                                                           "#-Annotation"])
df['Path'] = wsi_file_paths
splitter = model_selection.GroupShuffleSplit(test_size=.35, n_splits=1, random_state=7)
split = splitter.split(df, groups=df['Patient'])
train_inds, test_inds = next(split)
df_train = df.iloc[train_inds]
df_test = df.iloc[test_inds]
wsi_file_paths_test = df_test['Path']
wsi_labels_test = df_test['Type']
wsi_file_paths_train = df_train['Path']
wsi_labels_train = df_train['Type']


# In[ ]:


print("Common patients between train and test: {}".format(len(set(df['Patient'].iloc[train_inds]).intersection(set(df['Patient'].iloc[test_inds])))))


# In[ ]:


print("Train")
dataset_train = DatasetWSI(wsi_file_paths_train,
                           wsi_labels_train,
                           class_dict,
                           batch_size=batch_size,
                           tile_size=tile_size,
                           overlap=6).make_dataset()
print("Test")
dataset_test = DatasetWSI(wsi_file_paths_test,
                          wsi_labels_test,
                          class_dict,
                          batch_size=batch_size,
                          tile_size=tile_size,
                          overlap=6).make_dataset()


# In[ ]:


inv_class_dict = {v: k for k, v in class_dict.items()}
for batch_x, batch_y in dataset_train.take(2):
    fig, ax = plt.subplots(5, 5, figsize=(18, 18))
    ax = ax.ravel()
    j = 0
    for image, label in zip(batch_x[:25], batch_y[:25]):
        label = label.numpy()
        img = image.numpy()
        input_shape = img.shape
        ax[j].imshow(img)
        ax[j].axis('off')
        ax[j].set_title("Class: {}".format(inv_class_dict[int(np.argmax(label))]))
        j += 1


# In[ ]:


inv_class_dict = {v: k for k, v in class_dict.items()}
for batch_x, batch_y in dataset_test.take(5):
    fig, ax = plt.subplots(5, 5, figsize=(18, 18))
    ax = ax.ravel()
    j = 0
    for image, label in zip(batch_x[:25], batch_y[:25]):
        label = label.numpy()
        img = image.numpy()
        input_shape = img.shape
        ax[j].imshow(img)
        ax[j].axis('off')
        ax[j].set_title("Class: {}".format(inv_class_dict[int(np.argmax(label))]))
        j += 1


# In[ ]:


augmentation_block = [
    tf.keras.layers.RandomCrop(int(tile_size-tile_size*0.1), 
                               int(tile_size-tile_size*0.1)),
    tf.keras.layers.Resizing(224, 224),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomZoom(
        height_factor=(-0.05, -0.15),
        width_factor=(-0.05, -0.15)),
    tf.keras.layers.RandomRotation(0.3),
]


# In[ ]:


inputs = tf.keras.Input(input_shape)
x =  tf.keras.applications.resnet50.preprocess_input(inputs)
for layer in augmentation_block:
    x = layer(x)
base_model = tf.keras.applications.ResNet50(include_top=False, weights="imagenet")
for j, layer in enumerate(base_model.layers[:100]):
    print(layer.name, j)
    layer.trainable = False
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.models.Model(inputs=inputs, outputs=x)
# model = ResNet((input_shape[0], input_shape[1]), 
#                num_classes=num_classes, 
#                augment=True)


# In[ ]:


model.summary()


# In[ ]:


checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(                                                                                                                                          
    filepath=checkpoint_filepath,                                                                                                                                                                        
    save_weights_only=True,                                                                                                                                                                              
    monitor='accuracy',                                                                                                                                                                              
    mode='max',                                                                                                                                                                                          
    save_best_only=True)

lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='accuracy',
    factor=0.1,
    patience=7,
    verbose=0,
    mode='auto',
    min_delta=0.0001,
    cooldown=0,
    min_lr=0,
)

early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="accuracy",
    min_delta=0.001,
    patience=10,
    verbose=0,
    mode="auto",
    restore_best_weights=True,
)


# In[ ]:


optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
loss = tf.keras.losses.categorical_crossentropy
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])


# In[ ]:


model.fit(dataset_train, validation_data=dataset_test,
          epochs=epochs, callbacks=[checkpoint_callback, lr_callback, early_stop_callback])


# In[ ]:


results = model.evaluate(dataset_test)


# In[ ]:


print("Accuracy: {}".format(results[1]))

