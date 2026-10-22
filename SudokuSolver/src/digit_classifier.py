from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
import logging
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
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


def load_model(filename="models/finalized_model.keras"):
    loaded_model = tf.keras.models.load_model(filename)
    logging.debug("model loaded")
    return loaded_model


def predict_number(image, loaded_model=None):
    if loaded_model is None:
        loaded_model = load_model()

    image = image.astype("float64") / 255.0

    pred = loaded_model.predict(image.reshape(1, 32, 32, 1))
    logging.debug("prediction %s", pred)
    logging.debug("prediction %d", pred.argmax())

    # training labels are class index (0-8) = digit (1-9) - 1, so shift back.
    return pred.argmax() + 1


if __name__ == "__main__":
    main()
