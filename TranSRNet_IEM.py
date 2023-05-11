import numpy as np
import catboost as cb
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from scipy.stats import gaussian_kde
import tensorflow
import cv2
import csv
import glob
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import SGD, Adadelta, RMSprop, Adagrad, Adam, Nadam
tf.random.set_seed(42)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from astropy.io import fits
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint,EarlyStopping
from sklearn import preprocessing
import pandas as pd
import gzip


patch_size_y1,num_heads ,transformer_layers,num_classes = 300,8,6,256


input_shape1 = (1,3700, 1)
learning_rate = 0.001
num_epochs = 200
image_dim = 1
image_size_x = 1
print('image_size_x=',image_size_x)
image_size_y1 = 3700
print('image_size_y=',image_size_y1)
patch_size_x = 1  # Patch Dimension
num_patches1 = (image_size_x // patch_size_x) * (image_size_y1 // patch_size_y1)
# num_patches=len(patch_size_y)
print(num_patches1)
projection_dim = 32
transformer_units = [projection_dim * 2,projection_dim]  # Size of the transformer layers
mlp_head_units = [2048, 1024]  # Size of the dense layers

class Patches(layers.Layer):
    def __init__(self, patch_size_x,patch_size_y):
        super(Patches, self).__init__()
        self.patch_size_x = patch_size_x
        self.patch_size_y = patch_size_y

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,#image: tesnsor of input data
            sizes=[1, self.patch_size_x, self.patch_size_y, 1],  #Sliding window size
            strides=[1, self.patch_size_x, self.patch_size_y, 1],#The distance between the center points of each patch area.
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patch_size_x': self.patch_size_x,
            'patch_size_y': self.patch_size_y,
        })
        return config

def multilayer_perceptron(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'layers.Dense(units=projection_dim)': self.projection,
            'layers.Embedding(input_dim=num_patches, output_dim=projection_dim)':self.position_embedding,
        })
        return config


def create_vit_classifier():
    inputs = layers.Input(shape=input_shape1)
    patches = Patches(patch_size_x,patch_size_y1)(inputs)
    encoded_patches = PatchEncoder(num_patches1, projection_dim)(patches)

    for _ in range(transformer_layers):

        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1,x1)

        x2 = layers.Add()([attention_output, encoded_patches])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        x3 = multilayer_perceptron(x3, hidden_units=transformer_units, dropout_rate=0.1)

        encoded_patches = layers.Add()([x3, 0.666*x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.25)(representation)

    features = multilayer_perceptron(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = layers.Dense(256)(features)
    logits2 = layers.Dense(1)(logits)
    model = tensorflow.keras.Model(inputs=inputs, outputs=logits2)
    # model.summary()
    return model


reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=4,
        mode='auto',
        verbose=1
    )
    # 保存最优模型
checkpoint = ModelCheckpoint(
        filepath='./bestmodel.hdf5',
        monitor='val_mae',
        save_weights_only=True,
        save_best_only=True,
        mode='auto',
        period=2
    )

earlystopping = EarlyStopping(monitor='val_mae',
                                  min_delta=0,
                                  patience=10,
                                  verbose=0,
                                  mode='auto',
                                  baseline=None,
                                  restore_best_weights=False)


def run_experiment(model):
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss = keras.losses.mean_absolute_error,metrics='mae')
    # model.load_weights=('./调参权重/300-8-6-256_test.hdf5') #
    history = model.fit(x=X_train,y=Y_train,batch_size=64,epochs=num_epochs,validation_data=(X_test,Y_test),callbacks=[reduce_lr,checkpoint,earlystopping]) #Pre-training models

    # history = model.fit_generator(generator=training_generator, steps_per_epoch=int(lenth // batch_size), epochs=200,
    #                               verbose=1, workers=2, validation_data=validation_generator,
    #                               validation_steps=int(lenth2 // batch_size),
    #                               callbacks=[reduce_lr, checkpoint, earlystopping])   #Train all data using a data generator
    #

    return history.history['mae'] ,history.history['val_mae']


vit_classifier = create_vit_classifier()
# history1,history2 = run_experiment(vit_classifier)

# print('Model weight saved')







