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
from TranSRNet_IEM import create_vit_classifier



#Information extraction module loading
model=create_vit_classifier()
# model.load_weights=('./bestweight.hdf5') #Load trained weights
dense1_layer_model = Model(inputs=model.input, outputs=model.layers[-2].output)
model = models.Sequential()
model.add(dense1_layer_model)
model.summary()

# Advanced features of spectra
# X_new = model.predict(X_all)
#Y = all label radii
# X_train, X_test,Y_train, Y_test= train_test_split(X_new,Y, shuffle=True, test_size=0.2, random_state=2)

PRM = cb.CatBoostRegressor(iterations=150000,
                                  learning_rate=0.001,
                                  use_best_model=True,
                                  snapshot_file='test.test',
                                  early_stopping_rounds=300,
                                  random_seed=42,
                                  task_type='GPU',
                                  eval_metric='MAE',
                                  loss_function='RMSE',
                                  )

# train_pool = cb.Pool(X_train, Y_train)
# test_pool = cb.Pool(X_test, Y_test)
# PRM train
# PRM.fit(train_pool, eval_set=test_pool, verbose=10,plot=True)





