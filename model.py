import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2
import numpy as np

tf.enable_eager_execution()
print(tf.executing_eagerly())

def get_model():
    return keras.applications.vgg19.VGG19(include_top=True, weights=None, input_tensor=None, input_shape=None,
                                   pooling=None, classes=1)


def train_valid_test_split(x, y, test_ratio, valid_ratio, random_seed):
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=test_ratio, random_state=random_seed)
    x_train, x_valid, y_train, y_valid =\
        train_test_split(x_train, y_train, test_size=valid_ratio/(1-test_ratio), random_state=random_seed)

    return x_train, x_valid, x_test, y_train, y_valid, y_test

def train_model(model :keras.Model, x_train, y_train, x_valid, y_valid):
    epochs = 10
    batch_size = 32

    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss='mse')

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_valid, y_valid))


if __name__ == '__main__':

    data_folder = "data/small"


    data_df = pd.read_csv(data_folder + "/data.csv")
    data_np = data_df.values
    print(data_np.shape)
    x = np.array([cv2.imread(data_folder + "/" + l[2] + "_" + l[1]) / 255 for l in data_np])[:-1]
    y = np.array([l[7] for l in data_np])[1:]

    x_train, x_valid, x_test, y_train, y_valid, y_test = \
        train_valid_test_split(x, y, 0.10, 0.15, 42)

    model : keras.Model = get_model()
    # print(model.summary())
    train_model(model, x_train, y_train, x_valid, y_valid)
