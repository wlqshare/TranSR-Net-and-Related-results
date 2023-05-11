import numpy as np
import tensorflow
import cv2
import glob
import os
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow import keras
import gzip
from astropy.io import fits
import pandas as pd

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

class DataGenerator(tf.keras.utils.Sequence):
    """
    Custom Keras Data Generator Based on Sequence
    """
    def __init__(self, filepath,ypath, batch_size=8,imgshape1=(1,3700) ,
                 n_channels=1, n_classes=1, shuffle=True):
        """ Initialization method
        :param filepath :Address of spectral file
        :param batch_size: batch size
        :param imgshape: data size
        :param n_channels: data channel
        :param n_classes: data channel
        :param shuffle: Is the data scrambled after each epoch?

        """
        self.filepath=filepath
        self.ypath = ypath
        self.pathlist=glob.glob(os.path.join(self.filepath,'*.npy'))
        self.y_pathlist = glob.glob(os.path.join(self.ypath, '*.npy'))
        self.batch_size = batch_size
        self.imgshape1 = imgshape1
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        # Update index after each epoch
        self.on_epoch_end()
        # List of file addresses

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Index List
        batch_pathlist = [self.pathlist[k] for k in indexes]
        y_batch_pathlist=[self.y_pathlist[k] for k in indexes]
        # Generate data
        X = self._generate_X(batch_pathlist)
        y = self._generate_y(y_batch_pathlist)

        return X, y

    def __len__(self):
        """The number of batches under each epoch, which is the iteration of each epoch
        """
        return int(np.floor(len(self.pathlist) / self.batch_size))

    def _generate_X(self, batch_pathlist):
        X = np.empty((self.batch_size, *self.imgshape1, self.n_channels))
        for i, path in enumerate(batch_pathlist):
            X[i,] = self._load_image1(path)
        return X



    def _generate_y(self, batch_pathlist):
        y = np.empty((self.batch_size, ), dtype=int)
        # Generate data
        for i, path in enumerate(batch_pathlist):
            y[i,]=self._load_label(path)
        return y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.pathlist))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _load_image1(self, image_path):
        data1 = np.array(np.load(image_path))
        data1= data1.reshape(1,-1,1)
        return data1

    def _load_label(self,label_path):
        label = np.load(label_path)
        return label


if __name__=='__main__':
    params = {'batch_size': 64,
            'n_classes': 1,
            'n_channels': 1,
            'shuffle': True,
             'imgshape1' :(1,3700)
           }


    trainX_filepath = r"J:\Radius\npyflux"
    ypath = r"J:\Radius\npyRadius"
    files = os.listdir(trainX_filepath)
    # lenth = len(files)

    testX_filepath = r"\test\npyflux"
    testypath = r"J:\test\npyRadius"
    files2 = os.listdir(testX_filepath)
    # lenth2 = len(files2)


    training_generator = DataGenerator(trainX_filepath,ypath ,**params)
    validation_generator = DataGenerator(testX_filepath,testpath, **params)


##The following can be used to view the generated data situation

## validation_generator = DataGenerator(val_filepath,ypath, **params)
# for (x,x1),y in training_generator:
#     print((x.shape,x1.shape),y.shape)
