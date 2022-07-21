from enum import Enum
from constants import dataset_params

"""New datasets have to be added in this file"""


class DATASETS(Enum):
    AUDIO_MNIST_DIGITS = "audio_mnist_digits"
    AUDIO_MNIST_GENDER = "audio_mnist_gender"
    GTZAN = "gtzan"
    AUDIOSET = "audioset"


class DATASET_TYPES(Enum):
    WAVEFORM = "waveform"
    SPECTROGRAM = "spectrogram"
    MELSPECTROGRAM = "melspectrogram"


AUDIO_MNIST_PARAMS = {
    DATASET_TYPES.SPECTROGRAM.value: dataset_params.AudioMnistSpectrogramParams(),
    DATASET_TYPES.MELSPECTROGRAM.value: dataset_params.AudioMnistMelSpectrogramParams()
}

GTZAN_PARAMS = {
    DATASET_TYPES.SPECTROGRAM.value: dataset_params.GtzanSpectrogramParams(),
    DATASET_TYPES.MELSPECTROGRAM.value: dataset_params.GtzanMelSpectrogramParams()
}

AUDIOSET_PARAMS = {
    DATASET_TYPES.MELSPECTROGRAM.value: dataset_params.AudioSetMelSpectrogramParams()
}

DATASET_PARAMS = {
    DATASETS.AUDIO_MNIST_DIGITS.value: AUDIO_MNIST_PARAMS,
    DATASETS.AUDIO_MNIST_GENDER.value: AUDIO_MNIST_PARAMS,
    DATASETS.GTZAN.value: GTZAN_PARAMS,
    DATASETS.AUDIOSET.value: AUDIOSET_PARAMS
}


DATASET_CLASSNAMES = {
    DATASETS.AUDIO_MNIST_DIGITS.value: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # 0 - 9 digits
    DATASETS.AUDIO_MNIST_GENDER.value: [0, 1],  # Male, Female
    DATASETS.GTZAN.value: ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
}


CLASS_LABEL_POSITION_IN_FILENAME = {
    DATASETS.AUDIO_MNIST_DIGITS.value: 0,
    DATASETS.AUDIO_MNIST_GENDER.value: 1,
    DATASETS.GTZAN.value: 0,
    DATASETS.AUDIOSET.value: 0
}
