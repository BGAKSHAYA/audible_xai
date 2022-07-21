import keras
import tensorflow as tf
from utils.Singleton import Singleton
from utils.model_utils import model_wo_softmax
from constants.dataset_constants import DATASETS
import numpy as np


@Singleton
class ModelService:
    def __init__(self, model_weights_path, model_architecture_path):
        self.model_weights_path = model_weights_path
        self.model_architecture_path = model_architecture_path
        self.model = self.load_model()
        self.model_without_softmax = self.remove_softmax_layer()

    # Getters for Singleton class
    def get_model(self):
        return self.model

    def get_model_without_softmax(self):
        return self.model_without_softmax

    def load_model(self):
        json_file = open(self.model_architecture_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = keras.models.model_from_json(loaded_model_json, custom_objects={"tf_maximum": tf.math.maximum})
        model.load_weights(self.model_weights_path)
        return model

    def evaluate_model(self, data, class_label, dataset, dataset_params):
        if dataset == DATASETS.AUDIOSET.value: # For audioset, the task is topK classification instead of top-label
            model_input = dataset_params.reshape_model_input(data)
            predictions = self.model.predict(model_input)
            predictions = np.mean(predictions.reshape((-1, dataset_params.n_patches, dataset_params.num_classes)),
                                  axis=1)
            top_K = 5
            top_class_indices = np.flip(np.argsort(predictions), 1)[:, :top_K]
            predicted_labels = []
            for top_indices in top_class_indices:
                predicted_labels.append(class_label if class_label in top_indices else top_indices[0])
            predicted_labels = np.array(predicted_labels)
        else:
            predictions = self.model.predict(data)
            predicted_labels = np.argmax(predictions, axis=1)
            top_class_indices = predicted_labels
        return predicted_labels, top_class_indices

    # Strip softmax layer
    def remove_softmax_layer(self):
        model_without_softmax = model_wo_softmax(self.model)
        return model_without_softmax
