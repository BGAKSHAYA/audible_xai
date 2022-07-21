from utils.Singleton import Singleton
from dataset_preprocessing.dataset_preprocessor import preprocess_example
from constants.dataset_constants import DATASETS, DATASET_CLASSNAMES, DATASET_PARAMS, CLASS_LABEL_POSITION_IN_FILENAME
from services.file_service import FileService
from services.model_service import ModelService
import numpy as np
import pandas as pd
import glob
import os
import h5py


@Singleton
class DatasetService:
    def __init__(self, config, dataset, dataset_type, quick_analysis_mode):
        self.dataset_config = config["dataset"]
        self.dataset_path = self.dataset_config["path"]
        self.dataset = dataset
        self.dataset_type = dataset_type
        self.quick_analysis_mode = quick_analysis_mode
        self.file_service = FileService()

        dataset_file_paths = []
        dataset_params = self.get_dataset_params()

        if self.dataset == DATASETS.AUDIOSET.value:
            dataset_file_paths = glob.glob(os.path.join(self.dataset_path, '**/*.hdf5'))
        else:
            dataset_info_list = []
            if self.dataset == DATASETS.AUDIO_MNIST_DIGITS.value:
                dataset_info_list = glob.glob(os.path.join(self.dataset_path, "AlexNet_digit_0*.txt"))
            elif self.dataset == DATASETS.AUDIO_MNIST_GENDER.value:
                dataset_info_list = glob.glob(os.path.join(self.dataset_path, "AlexNet_gender_0*.txt"))
            elif self.dataset == DATASETS.GTZAN.value:
                dataset_info_list = glob.glob(os.path.join(self.dataset_path, "*.txt"))

            info_list = list(filter(lambda file_name: "test" in file_name, dataset_info_list))
            for filename in info_list:
                with open(filename, "r") as file:
                    dataset_file_paths += list(file.read().splitlines())

        dataset_file_paths = self.get_class_files(dataset_file_paths)
        if self.quick_analysis_mode["run"]:
            dataset_file_paths = self.get_subset_dataset(dataset_file_paths, self.quick_analysis_mode["max_per_class"])

        data, audio_file_paths = dict(), dict()
        for class_name, class_files in dataset_file_paths.items():
            audio_file_paths[class_name] = np.empty((0,), dtype="str")
            data[class_name] = np.empty((0, *tuple(dataset_params.preprocessed_input_shape[1:])), dtype=np.float64)
            for file in class_files:
                f = h5py.File(file, 'r')
                data[class_name] = np.append(data[class_name], f["data"][...], axis=0)
                audio_file_paths[class_name] = np.append(audio_file_paths[class_name], f["path"][...], axis=0)

        self.data = data
        self.audio_file_paths = audio_file_paths
        self.dataset_filenames = self.get_basenames(dataset_file_paths)
        self.prediction_details = self.get_prediction_details()

    def get_basenames(self, dataset_file_paths):
        dataset_basenames = {}
        for class_name, class_files in dataset_file_paths.items():
            dataset_basenames[class_name] = np.array([self.file_service.basename(path) for path in class_files])
        return dataset_basenames

    def get_class_files(self, dataset_file_paths):
        class_dataset_file_paths = dict()
        class_label_position = CLASS_LABEL_POSITION_IN_FILENAME[self.dataset]
        class_names = self.get_classnames()
        for class_name in class_names:
            class_files = filter(lambda file: str(class_name) in os.path.basename(file).split('_')[class_label_position],
                                 dataset_file_paths)
            class_files = np.array(list(class_files))
            if len(class_files) == 0:
                continue
            class_files = np.sort(class_files)
            class_dataset_file_paths[class_name] = class_files
        return class_dataset_file_paths

    def get_subset_dataset(self, dataset_file_paths, max_per_class=10):
        subset_dataset_file_paths = dict()
        STRATIFY, STRATIFY_ON, STRATIFY_GROUP_LEN = None, None, None

        if self.dataset == DATASETS.AUDIO_MNIST_GENDER.value:
            STRATIFY = True  # Get various digits spoken by both genders
            STRATIFY_ON = DATASETS.AUDIO_MNIST_DIGITS.value
            STRATIFY_GROUP_LEN = max_per_class // len(DATASET_CLASSNAMES[STRATIFY_ON])

        elif self.dataset == DATASETS.AUDIO_MNIST_DIGITS.value:
            STRATIFY = True  # Get various digits spoken by both genders
            STRATIFY_ON = DATASETS.AUDIO_MNIST_GENDER.value
            STRATIFY_GROUP_LEN = max_per_class // len(DATASET_CLASSNAMES[STRATIFY_ON])

        for class_name, class_files in dataset_file_paths.items():
            class_files = np.sort(class_files)
            if STRATIFY:
                class_label_position = CLASS_LABEL_POSITION_IN_FILENAME[STRATIFY_ON]
                class_files_stratify_info = map(lambda file: int(os.path.basename(file).split('_')[class_label_position]), class_files)
                class_files_stratify_info = np.array(list(class_files_stratify_info))
                class_subset = []
                for stratify_class in DATASET_CLASSNAMES[STRATIFY_ON]:
                    class_files_stratified = class_files[np.where(class_files_stratify_info == stratify_class)]
                    class_subset.extend(class_files_stratified[:STRATIFY_GROUP_LEN])
                subset_dataset_file_paths[class_name] = class_subset
            else:
                subset_dataset_file_paths[class_name] = class_files[:max_per_class]
        return subset_dataset_file_paths

    def read_classnames(self):
        if self.dataset not in DATASET_CLASSNAMES:
            """Read the class name definition file and return an array of strings as classnames."""
            classnames_csv = pd.read_csv(self.dataset_config["classnames_path"])
            classnames = np.array(classnames_csv["display_name"])
        else:
            classnames = DATASET_CLASSNAMES[self.dataset]
        return classnames

    def get_prediction_details(self):
        class_name_mapper = self.class_name_to_index_mapper()
        prediction_details = {
            "correct": {},
            "incorrect": {},
            "predictions": {},
            "topK_labels": {}
        }
        for class_name in self.data.keys():
            pred_index = class_name_mapper[class_name]
            predicted_labels, topK_labels = ModelService.instance.evaluate_model(
                                    self.data[class_name], pred_index, self.dataset, self.get_dataset_params())

            prediction_details["predictions"][class_name] = predicted_labels

            correct_positions = np.where(predicted_labels == pred_index)[0]
            prediction_details["correct"][class_name] = correct_positions
            prediction_details["incorrect"][class_name] = {}
            prediction_details["topK_labels"][class_name] = {}

            for label in np.unique(predicted_labels):
                if label == pred_index:
                    continue
                prediction_details["incorrect"][class_name][label] = np.where(predicted_labels == label)[0]
                prediction_details["topK_labels"][class_name][label] = topK_labels[np.where(predicted_labels == label)]

            classnames = self.get_classnames()
        return prediction_details

    def filter(self, x, filter_type):
        filtered_x = {}
        if filter_type == "correct":
            for class_name in x.keys():
                filtered_x[class_name] = {}
                positions = self.prediction_details["correct"][class_name]
                filtered_x[class_name] = x[class_name][positions]
        elif filter_type == "incorrect":
            for class_name in self.data.keys():
                filtered_x[class_name] = {}
                for label, positions in self.prediction_details["incorrect"][class_name].items():
                    filtered_x[class_name][label] = x[class_name][positions]
        else:
            filtered_x = x
        return filtered_x

    # Getters
    def get_dataset(self, filter_type="both"):
        return self.filter(self.data, filter_type)

    def get_audio_file_paths(self, filter_type="both"):
        return self.filter(self.audio_file_paths, filter_type)

    def get_dataset_filenames(self, filter_type="both"):
        return self.filter(self.dataset_filenames, filter_type)

    def get_dataset_params(self):
        return DATASET_PARAMS[self.dataset][self.dataset_type]

    def get_dataset_type(self):
        return self.dataset_type

    def get_dataset_name(self):
        return self.dataset

    def get_classnames(self):
        classnames = DATASET_CLASSNAMES.get(self.dataset, self.read_classnames())
        return classnames

    def class_name_to_index_mapper(self):
        classnames = self.get_classnames()
        class_name_mapper = {k: v for v, k in enumerate(classnames)}
        return class_name_mapper

    def get_descriptive_title(self, filename):
        title = ""
        if self.dataset == DATASETS.AUDIO_MNIST_GENDER.value or self.dataset == DATASETS.AUDIO_MNIST_DIGITS.value:
            digit, gender, speaker = filename.split("_")
            gender = "Male" if int(gender) == 0 else "Female"
            title = "Digit=" + digit + " Speaker=" + speaker + " Gender=" + gender
        elif self.dataset == DATASETS.GTZAN.value:
            label, example, _ = filename.split("_")
            title = "Genre " + label
        return title


