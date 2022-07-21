import numpy as np
import glob
import h5py
import librosa
import logging
import os
import json
from dataset_preprocessing.audio_mnist import create_splits
from dataset_preprocessing.audioset import preprocess_audioset_example
from constants.dataset_constants import DATASET_TYPES, DATASET_PARAMS, DATASETS
logger = logging.getLogger("audible_xai")


def preprocess_example(filepath, dataset, dataset_type, dataset_params):
    data, sr = librosa.core.load(filepath, sr=dataset_params.sr, mono=False)
    # We currently only support audio files with single channel, so we combine audio with multiple channels into one
    data = librosa.to_mono(data)
    preprocessed_data = None

    MIN_DATA_LEN = dataset_params.min_data_len
    if len(data) < MIN_DATA_LEN:
        embedded_data = np.zeros(MIN_DATA_LEN)  # Pad the waveform till MIN_DATA_LEN by adding silence at the end
        embedded_data[:len(data)] = data
        data = embedded_data

    """Preprocessing for new datasets has to be added here"""
    if dataset == DATASETS.AUDIOSET.value:  # Preprocessing specific to audioset
        preprocessed_data = preprocess_audioset_example(data, dataset_params)

    elif dataset_type == DATASET_TYPES.SPECTROGRAM.value:
        # Short Term Fourier Transform
        spectrogram = librosa.stft(data, n_fft=dataset_params.n_fft,
                                   win_length=dataset_params.win_length,
                                   hop_length=dataset_params.hop_length)
        shape = spectrogram.shape  # shape = (#Frequency bins, #Frames/Temporal bins)
        abs_spectrogram = np.abs(spectrogram[:, :dataset_params.shape[1]])
        amplitude_spectrogram_to_db = librosa.amplitude_to_db(abs_spectrogram, ref=np.max)
        preprocessed_data = amplitude_spectrogram_to_db
        preprocessed_data /= 80.0  # Normalization with 80, which is the Maximum possible db for AudioMNIST

    elif dataset_type == DATASET_TYPES.MELSPECTROGRAM.value:
        melspectrogram = librosa.feature.melspectrogram(y=data, sr=sr,
                                                        n_fft=dataset_params.n_fft,
                                                        win_length=dataset_params.win_length,
                                                        hop_length=dataset_params.hop_length, )
        shape = melspectrogram.shape  # shape = (#Mels, #Frames/Temporal bins)
        power_spectrogram_to_db = librosa.power_to_db(melspectrogram[:, :dataset_params.shape[1]], ref=np.max)
        preprocessed_data = power_spectrogram_to_db
        preprocessed_data /= 80.0  # Normalization with 80, which is the Maximum possible db for AudioMNIST

    # We currently only support audio files with single channel
    preprocessed_data = preprocessed_data[np.newaxis, ..., np.newaxis]

    if preprocessed_data.shape != dataset_params.preprocessed_input_shape:
        logger.error("Dataset Shape mismatch expected {} but got {}".format(dataset_params.preprocessed_input_shape,
                                                                            preprocessed_data.shape))
        raise Exception("Dataset Shape mismatch expected {} but got {}".format(dataset_params.preprocessed_input_shape,
                                                                               preprocessed_data.shape))
    return preprocessed_data


class DatasetPreprocessorService:
    def __init__(self, path_configuration, preprocessing_config):
        self.preprocessing_config = preprocessing_config
        self.path_configuration = path_configuration

    def run(self):
        for dataset, dataset_types in self.preprocessing_config.items():
            dataset_path_config = self.path_configuration[dataset]
            for dataset_type in dataset_types:
                dataset_params = DATASET_PARAMS[dataset][dataset_type]
                dataset_type_path_config = dataset_path_config[dataset_type]["preprocessing"]
                self.preprocess(dataset, dataset_type, dataset_params, dataset_type_path_config)

    @staticmethod
    def preprocess(dataset, dataset_type, dataset_params, path_config):
        source = path_config["source"]
        destination = path_config["destination"]
        # Some datasets have dataset info in meta file ex. audio_mnist
        metafile = json.load(open(path_config["meta"])) if "meta" in path_config else None

        folders = []
        for folder in os.listdir(source):
            # only process folders
            if not os.path.isdir(os.path.join(source, folder)):
                continue
            folders.append(folder)

        src_dst_list = [(os.path.join(source, folder), os.path.join(destination, folder))
                        for folder in sorted(folders)]

        logger.info("Started preprocessing for {} with {} type".format(dataset, dataset_type))
        for src_dst in src_dst_list:
            src, dst = src_dst
            logger.info("processing {}".format(src))

            # create folder for hdf5 files
            if not os.path.exists(dst):
                os.makedirs(dst)

            # loop over recordings
            for filepath in sorted(glob.glob(os.path.join(src, "*.wav"))):
                preprocessed_filepath, label = None, None
                preprocessed_data = preprocess_example(filepath, dataset, dataset_type, dataset_params)
                if dataset == DATASETS.AUDIO_MNIST_GENDER.value or dataset == DATASETS.AUDIO_MNIST_DIGITS.value:
                    digit, speaker, example = os.path.splitext(os.path.basename(filepath))[0].split("_")
                    digit = digit.split("\\")[-1]
                    label = np.array([int(digit), 0 if metafile[speaker]["gender"] == "male" else 1])
                    preprocessed_filepath = os.path.join(dst, "{}_{}_{}.hdf5".format(label[0], label[1], example))

                elif dataset == DATASETS.GTZAN.value or dataset == DATASETS.AUDIOSET.value:
                    label, example = os.path.splitext(os.path.basename(filepath))[0].split(".")
                    preprocessed_filepath = os.path.join(dst, "{}_{}.hdf5".format(label, example))
                    label = np.array([label.encode("utf-8")])

                filepath = np.array([filepath.encode("utf-8")])
                with h5py.File(preprocessed_filepath, "w") as f:
                    f["data"] = preprocessed_data
                    f["label"] = label
                    f["path"] = filepath

        # for audio_mnist also prepare train, test, validation splits suggested by dataset authors
        if dataset == DATASETS.AUDIO_MNIST_GENDER.value or dataset == DATASETS.AUDIO_MNIST_DIGITS.value:
            create_splits(source, destination)
