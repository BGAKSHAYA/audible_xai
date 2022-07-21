from dataclasses import dataclass
import math
import numpy as np

"""The dataset params have to be added in this file with which the dataset has been created"""


@dataclass(frozen=True)  # Instances of this class are immutable.
class AudioMnistSpectrogramParams:
    sr = 8000
    n_fft = 512  # frame_size
    win_length = 512  # win_length = frame_size
    hop_length = 32  # <=frame_size/2 for perfect reconstruction
    min_data_len = 8000 * 1  # sr*duration for audio_mnist each audio file is 1sec long
    # (#freq_bins #temporal_bins) & #freq_bins=(frame_size/2+1), #temporal_bins=(floor(#samples*duration/hop_length)+1)
    shape = ((512 // 2) + 1, math.floor(8000 / 32) + 1)  # (257, 251)
    time_unit_ms = 1 / 251
    min_x_units = round(0.1 / (1 / 251))
    preprocessed_input_shape: tuple = (1, 257, 251, 1)

    @property
    def saliency_shape(self):
        saliency_shape = self.preprocessed_input_shape[1:]
        return saliency_shape

    def reshape_model_input(self, input_example):
        input_example = input_example.reshape((-1, *tuple(self.preprocessed_input_shape[1:])))
        return input_example

    def reshape_saliencies(self, saliencies):
        reshaped_saliencies = saliencies.reshape((-1, *tuple(self.preprocessed_input_shape[1:])))
        return reshaped_saliencies


@dataclass(frozen=True)  # Instances of this class are immutable.
class AudioMnistMelSpectrogramParams:
    sr = 8000
    n_fft = 512  # frame_size
    win_length = 512  # win_length = frame_size
    hop_length = 32  # <=frame_size/2 for perfect reconstruction
    min_data_len = 8000 * 1  # sr*duration for audio_mnist each audio file is 1sec long
    mel_bands = 128,
    # (#mel_bands, #temporal_bins)
    shape = (128, math.floor(8000 / 32) + 1)  # (128, 251)
    time_unit_ms = 1 / 251
    min_x_units = round(0.1 / (1 / 251))
    preprocessed_input_shape: tuple = (1, 128, 251, 1)
    mel_min_hz: float = 0
    mel_max_hz: float = 4000.0

    @property
    def saliency_shape(self):
        saliency_shape = self.preprocessed_input_shape[1:]
        return saliency_shape

    def reshape_model_input(self, input_example):
        input_example = input_example.reshape((-1, *tuple(self.preprocessed_input_shape[1:])))
        return input_example

    def reshape_saliencies(self, saliencies):
        reshaped_saliencies = saliencies.reshape((-1, *tuple(self.preprocessed_input_shape[1:])))
        return reshaped_saliencies


@dataclass(frozen=True)  # Instances of this class are immutable.
class GtzanSpectrogramParams:
    sr = 8000
    n_fft = 512  # frame_size
    win_length = 512  # win_length = frame_size
    hop_length = 64  # <=frame_size/2 for perfect reconstruction
    min_data_len = 8000 * 3  # sr*duration, for gtzan each splitted audio file is 3secs long
    # (#freq_bins, #temporal_bins) & #freq_bins=(frame_size/2+1), #temporal_bins=(floor(#samples*duration/hop_length)+1)
    shape = ((512 // 2) + 1, math.floor(8000 * 3 / 64) + 1)  # (257, 376)
    time_unit_ms = 3 / 376
    min_x_units = round(0.1 / (3 / 376))
    preprocessed_input_shape: tuple = (1, 257, 376, 1)

    batchnorm_center: bool = True
    batchnorm_scale: bool = False
    batchnorm_epsilon: float = 1e-4

    @property
    def saliency_shape(self):
        saliency_shape = self.preprocessed_input_shape[1:]
        return saliency_shape

    def reshape_model_input(self, input_example):
        input_example = input_example.reshape((-1, *tuple(self.preprocessed_input_shape[1:])))
        return input_example

    def reshape_saliencies(self, saliencies):
        reshaped_saliencies = saliencies.reshape((-1, *tuple(self.preprocessed_input_shape[1:])))
        return reshaped_saliencies


@dataclass(frozen=True)  # Instances of this class are immutable.
class GtzanMelSpectrogramParams:
    sr = 8000
    n_fft = 512  # frame_size
    win_length = 512  # win_length = frame_size
    hop_length = 64  # <=frame_size/2 for perfect reconstruction
    min_data_len = 8000 * 3  # sr*duration, for gtzan each splitted audio file is 3secs long
    mel_bands = 128
    # (#mel_bands, #temporal_bins)
    shape = (128, math.floor(8000 * 3 / 64) + 1)  # (128, 376)
    time_unit_ms = 30 / 376
    min_x_units = round(0.1 / (3 / 376))
    preprocessed_input_shape: tuple = (1, 128, 376, 1)
    mel_min_hz: float = 0
    mel_max_hz: float = 4000.0

    @property
    def saliency_shape(self):
        saliency_shape = self.preprocessed_input_shape[1:]
        return saliency_shape

    def reshape_model_input(self, input_example):
        input_example = input_example.reshape((-1, *tuple(self.preprocessed_input_shape[1:])))
        return input_example

    def reshape_saliencies(self, saliencies):
        reshaped_saliencies = saliencies.reshape((-1, *tuple(self.preprocessed_input_shape[1:])))
        return reshaped_saliencies


# The following hyperparameters (except patch_hop_seconds) were used to train YAMNet,
# so expect some variability in performance if you change these. The patch hop can
# be changed arbitrarily: a smaller hop should give you more patches from the same
# clip and possibly better performance at a larger computational cost.
@dataclass(frozen=True)  # Instances of this class are immutable.
class AudioSetMelSpectrogramParams:
    sr: float = 16000
    stft_window_seconds: float = 0.025
    stft_hop_seconds: float = 0.010
    mel_bands: int = 64
    mel_min_hz: float = 125.0
    mel_max_hz: float = 7500.0
    log_offset: float = 0.001
    patch_window_seconds: float = 0.96
    patch_hop_seconds: float = 0.48
    min_data_len: int = 16000 * 10
    n_patches: int = 11
    patch_frame_len: int = 96
    preprocessed_input_shape: tuple = (1, 11, 96, 64, 1)
    min_x_units = round(0.1 / (10 / 1000))

    @property
    def patch_frames(self):
        return int(round(self.patch_window_seconds / self.stft_hop_seconds))

    @property
    def patch_bands(self):
        return self.mel_bands

    @property
    def win_length(self):
        return int(round(self.sr * self.stft_window_seconds))

    @property
    def hop_length(self):
        return int(round(self.sr * self.stft_hop_seconds))

    @property
    def n_fft(self):
        return 2 ** int(np.ceil(np.log(self.win_length) / np.log(2.0)))

    @property
    def saliency_shape(self):
        saliency_shape = (self.mel_bands, self.n_patches * self.patch_frame_len, 1)
        return saliency_shape

    def reshape_saliencies(self, saliencies):
        reshaped_saliencies = saliencies.reshape((-1, *tuple(self.preprocessed_input_shape[1:])))
        no_saliencies = reshaped_saliencies.shape[0]
        reshaped_saliencies = reshaped_saliencies.reshape((no_saliencies, self.n_patches*self.patch_frame_len,
                                                           self.mel_bands, 1))
        reshaped_saliencies = np.einsum("ijkl->ikjl", reshaped_saliencies)
        return reshaped_saliencies

    def reshape_model_input(self, input_example):
        input_example = input_example.reshape((-1, *tuple(self.preprocessed_input_shape[2:])))
        return input_example

    num_classes: int = 521
    conv_padding: str = 'same'
    batchnorm_center: bool = True
    batchnorm_scale: bool = False
    batchnorm_epsilon: float = 1e-4
    classifier_activation: str = 'sigmoid'
    tflite_compatible: bool = False
