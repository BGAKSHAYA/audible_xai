import os
import sys

sys.path.insert(0, os.path.join('/scratch/python_envs/annalyzer/python/lib/python3.8/site-packages/'))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from constants.dataset_constants import DATASET_TYPES, DATASETS
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import glob
import os
import h5py
import tensorflow as tf
import keras
import json

path_configuration = json.load(open("path_configuration.json"))
config = path_configuration[DATASETS.AUDIO_MNIST_GENDER.value][DATASET_TYPES.SPECTROGRAM.value]
preprocessed_data = config["dataset"]["path"]
dataset_files = glob.glob(os.path.join(preprocessed_data, '**/*.hdf5'))
dataset_info_list = glob.glob(os.path.join(preprocessed_data, "AlexNet_gender_0*.txt"))


def read_h5py_file(file):
    with h5py.File(file) as f:
        return f["data"][...][0], f["label"][...][1]


def read_dataset(files_type="train"):
    files = []
    info_list = list(filter(lambda filename: files_type in filename, dataset_info_list))
    for filename in info_list:
        with open(filename, "r") as file:
            files += list(file.read().splitlines())

    files = list(map(lambda file: read_h5py_file(file), files))
    print("Number of Instances", len(files))

    dataset = np.array([i[0] for i in files])
    class_labels = np.array([i[1] for i in files])

    dataset = dataset.astype(np.float32) / 255
    class_labels = class_labels.astype(np.int32)
    return dataset, class_labels


training_data, training_class_labels = read_dataset(files_type="train")
validation_data, validation_data_class_labels = read_dataset(files_type="validate")
testing_data, testing_class_labels = read_dataset(files_type="test")

# AlexNet Implementation
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=16, kernel_size=(2, 2), activation='relu', input_shape=(257, 251, 1)),
    keras.layers.BatchNormalization(momentum=0.9),
    keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

    keras.layers.Conv2D(filters=32, kernel_size=(2, 2), activation='relu'),
    keras.layers.BatchNormalization(momentum=0.9),
    keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

    keras.layers.Conv2D(filters=64, kernel_size=(2, 2), activation='relu'),
    keras.layers.BatchNormalization(momentum=0.9),
    keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation='softmax')
])

model.summary()

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.001),
              metrics=['accuracy'])

model.fit(training_data, training_class_labels, epochs=10, verbose=2, batch_size=32,
          validation_data=(validation_data, validation_data_class_labels), callbacks=[callback])

model.evaluate(testing_data, testing_class_labels)
model_weights_path = config["model"]["model_weights_path"]
model_architecture_path = config["model"]["model_architecture_path"]

model_json = model.to_json()
with open(model_architecture_path + ".json", "w") as f:
    f.write(model_json)
model.save_weights(model_weights_path + ".h5")
print("Model Saved")
