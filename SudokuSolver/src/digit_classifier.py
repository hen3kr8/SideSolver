from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
import logging
import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import warnings



path_to_processed_training_digits_directory = "../data/processed_digits/training"
path_to_processed_test_digits_directory = "../data/processed_digits/test"
train_model = False
model_path="models/finalized_model.keras"

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)



def main():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    # train_classifier()
    X_train, y_train = load_processed_digits(path_to_processed_training_digits_directory)
    X_test, y_test = load_processed_digits(path_to_processed_test_digits_directory)
    if train_model:
        train_on_processed_digits(X_train, y_train)

        test_on_processed_digits(X_test, y_test)
    accuracy_per_class = get_accuracy_per_class(X_test, y_test, model_path=model_path, disp_confusion_matrix=True)

def load_processed_digits(data_dir:str) -> tuple[np.ndarray, np.ndarray]:
    '''
    Loads processed digits from the specified directory.
    The processed_digits/training folder contains digit folders 1 - 9 
    The name of the folder is the y label. Its contents the X.
    
    Args:
        data_dir (str): The path to the directory containing processed digits.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the features (X) and labels (y).
    '''
    y = []
    X = []

    for digit_folder in os.listdir(data_dir):
        for digit_image in os.listdir(data_dir + '/' + str(digit_folder)):
            digit_image_path = os.path.join(data_dir, digit_folder, digit_image)
            img = cv2.imread(digit_image_path, cv2.IMREAD_GRAYSCALE)
            if img is None or img.shape != (32, 32):
                continue
            y.append(int(digit_folder) -1)  # Convert folder name to valid class indices (0-8)
            X.append(img)
    X = np.array(X, dtype="float32").reshape(-1, 32, 32, 1) / 255.0
    y = np.array(y)

    # validate keys are 0 - 8 and each key has more than 10 values 
    if not set(range(0, 9)).issubset(y.tolist()):
        logging.error("Not all digits labels (0-8) were found in processed_digits folder.")
        raise KeyError

    return X, y

def train_on_processed_digits(X_train, y_train, model_path="models/finalized_model.keras"):
    '''
    train a CNN model on the processed digits dataset and save the model to disk.

    '''
    is_test = np.zeros(len(y_train), dtype=bool)
    # use 10% of the training data for validation
    is_test[np.random.choice(len(y_train), size=int(0.1 * len(y_train)), replace=False)] = True

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
    model.fit(X_train[~is_test], y_train[~is_test], epochs=15,
              validation_data=(X_train[is_test], y_train[is_test]),
              verbose=2
              )
    model.save(model_path)

    #show model architecture (layers, number of nodes per layer)
    model.summary()

    logging.info(f"Model trained and saved to {model_path}")
    return model

def test_on_processed_digits(X_test, y_test, model_path="models/finalized_model.keras"):
    '''
    test a CNN model on the processed digits dataset and print the accuracy.

    '''
    model = tf.keras.models.load_model(model_path)
    loss, accuracy = model.evaluate(X_test, y_test)
    logging.info(f"Test accuracy: {accuracy:.4f}, Test loss: {loss:.4f}")


    return accuracy


def get_accuracy_per_class(X_test, y_test, model_path="models/finalized_model.keras", disp_confusion_matrix=False):
    '''
    test a CNN model on the processed digits dataset and print the accuracy per class.

    '''
    model = tf.keras.models.load_model(model_path)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    accuracy_per_class = {}
    for i in range(9):
        class_indices = np.where(y_test == i)[0]
        class_accuracy = np.mean(y_pred_classes[class_indices] == y_test[class_indices])
        accuracy_per_class[i] = class_accuracy

    logging.info(f"Accuracy per class: {accuracy_per_class}")

    if disp_confusion_matrix:
        cm = confusion_matrix(y_test, y_pred_classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=accuracy_per_class.keys())
        disp.plot()
        plt.show()
        # save confusion matrix to file
        plt.savefig("confusion_matrix.png")
        print(classification_report(y_test, y_pred_classes, digits=3))

    return accuracy_per_class



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

    image = image.astype("float64") / 255.0

    # plt.imshow(image, cmap="gray")
    # plt.title("final pred")
    # plt.show()
    pred = loaded_model.predict(image.reshape(1, 32, 32, 1))
    logging.debug("prediction %s", pred)
    logging.debug("prediction %d", pred.argmax())

    return pred.argmax()


if __name__ == "__main__":
    main()
