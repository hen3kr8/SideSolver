#digit classifier

import tensorflow as tf
import matplotlib.pyplot as plt
# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# %matplotlib inline # Only use this if using iPython
import pickle 
import logging
from PIL import Image

def main():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    tf.get_logger().setLevel('INFO')
    # logging.getLogger('foo').debug('bah')
    # logging.getLogger().setLevel(logging.INFO)
    # logging.getLogger('foo').debug('bah')

    # train_classifier()
    filename = 'models/finalized_model.sav'
    model = load_model(filename)


def train_classifier():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


    # Reshaping the array to 4-dims so that it can work with the Keras API
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    input_shape = (28, 28, 1)
    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255
    # only train on subset of data, we dont need 60 000
    x_train = x_train[:10000]
    x_test = x_test[:2000]

    y_train = y_train[:10000]
    y_test = y_test[:2000]

    print('x_train shape:', x_train.shape)
    print('Number of images in x_train', x_train.shape[0])
    print('Number of images in x_test', x_test.shape[0])

    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))

    model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, epochs=10)

    model.evaluate(x_test, y_test)
    serialize_model(model)
    # return model

    # image_index = 500

    # plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
    # # pred = model.predict(x_test[image_index].reshape(1, img_rows, img_cols, 1))
    # plt.show()
    # pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))

    # print(pred.argmax())


def serialize_model(model):
    filename = 'models/finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    logging.info('model serialized')
 

def load_model(filename='models/finalized_model.sav'):
# load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    # image_index = 500

    # plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
    # # pred = model.predict(x_test[image_index].reshape(1, img_rows, img_cols, 1))
    # plt.show()
    # pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))

    # print(pred.argmax())
    logging.info('model loaded')
    return loaded_model


def predict_number(image, model=load_model()):

    plt.imshow(image.reshape(1, 28, 28, 1), cmap='Greys')
    plt.show()
    pred = model.predict(image.reshape(1, 28, 28, 1))
    print('prdeiction: ', pred.argmax())

    return None


if __name__ == "__main__":
    main()