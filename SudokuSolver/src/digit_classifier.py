from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
import logging
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    train_classifier()


def train_classifier():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Reshaping the array to 4-dims so that it can work with the Keras API
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    input_shape = (28, 28, 1)
    # Making sure that the values are float so that we can get decimal points
    #  after division
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255
    # only train on subset of data, we dont need 60 000
    x_train = x_train[:10000]
    x_test = x_test[:2000]

    y_train = y_train[:10000]
    y_test = y_test[:2000]

    image_index = 2

    plt.imshow(x_test[image_index].reshape(28, 28), cmap="Greys")
    plt.show()
    return None

    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation=tf.nn.softmax))

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(x=x_train, y=y_train, epochs=10)

    model.evaluate(x_test, y_test)
    serialize_model(model)
    return model


def serialize_model(model):
    filename = "models/finalized_model.sav"
    pickle.dump(model, open(filename, "wb"))
    logging.info("model serialized")


def load_model(filename="models/finalized_model.sav"):
    loaded_model = pickle.load(open(filename, "rb"))
    logging.debug("model loaded")
    return loaded_model


def predict_number(image, loaded_model=load_model()):

    # image = puzzle_extractor.apply_threshold(src_image=image, bin=True)
    # image = np.invert(image)
    image = image.astype("float64") / 255.0

    # plt.imshow(image, cmap="gray")
    # plt.title("final pred")
    # plt.show()
    pred = loaded_model.predict(image.reshape(1, 28, 28, 1))
    logging.debug("prediction %s", pred)
    logging.debug("prediction %d", pred.argmax())

    return pred.argmax()


if __name__ == "__main__":
    main()
