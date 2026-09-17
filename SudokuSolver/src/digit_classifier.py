from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
import logging
import glob
import os
import cv2
import numpy as np
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


def load_processed_digits(data_dir="data/processed_digits"):
    X, y, src = [], [], []
    for path in sorted(glob.glob(os.path.join(data_dir, "*", "*.png"))):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape != (32, 32):
            continue
        X.append(img)
        y.append(int(os.path.basename(os.path.dirname(path))) - 1)
        # "image10002_7.png" -> source photo "image1000"
        src.append(os.path.basename(path).split("_")[0][:-1])
    X = np.array(X, dtype="float32").reshape(-1, 32, 32, 1) / 255.0
    return X, np.array(y), np.array(src)


def train_on_processed_digits(model_path="models/digit_model.keras"):
    X, y, src = load_processed_digits()

    # hold out whole photos, so cells from one image never span the split
    photos = np.unique(src)
    rng = np.random.default_rng(0)
    test_photos = set(rng.choice(photos, len(photos) // 5, replace=False).tolist())
    is_test = np.array([s in test_photos for s in src])

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(32, 32, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(9, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(X[~is_test], y[~is_test], epochs=15,
              validation_data=(X[is_test], y[is_test]))
    model.save(model_path)
    return model


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

    plt.imshow(x_test[image_index].reshape(32, 32), cmap="Greys")
    plt.show()

    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape)) # should this be 32?
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
    filename = "models/finalized_model.keras"
    model.save(filename)
    logging.info("model serialized")


def load_model(filename="models/finalized_model.keras"):
    if filename.endswith(".keras"):
        loaded_model = tf.keras.models.load_model(filename)
        logging.debug("model loaded")
        return loaded_model

    try:
        with open(filename, "rb") as model_file:
            loaded_model = pickle.load(model_file)
    except (ModuleNotFoundError, AttributeError) as error:
        raise RuntimeError(
            "The legacy pickled model is incompatible with the current "
            "TensorFlow/Keras version. Retrain the classifier to create "
            "models/finalized_model.keras."
        ) from error

    logging.debug("legacy model loaded")
    return loaded_model


def predict_number(image, loaded_model=None):
    if loaded_model is None:
        loaded_model = load_model()

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
