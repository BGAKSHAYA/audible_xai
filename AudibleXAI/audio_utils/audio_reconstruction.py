import numpy as np
import librosa
import librosa.display
import logging
from constants.dataset_constants import DATASET_TYPES, DATASETS


def griffin_lim_reconstruction(spectrogram, region_mask, dataset_params):
    # Inverse Operations from magnitude spectrogram
    db_to_amplitude_spect = librosa.db_to_amplitude(spectrogram)
    db_to_amplitude_spect[region_mask == False] = 0
    reconstructed_audio = librosa.core.spectrum.griffinlim(db_to_amplitude_spect, hop_length=dataset_params.hop_length,
                                                           win_length=dataset_params.win_length)
    return reconstructed_audio


def istft_reconstruction(spectrogram, region_mask, dataset_params):
    # Inverse Operations of Short Term Fourier Transform
    spectrogram[region_mask == False] = 0
    _, reconstructed_audio = librosa.istft(spectrogram, hop_length=dataset_params.hop_length,
                                           win_length=dataset_params.win_length)
    return reconstructed_audio


def griffin_lim_melspectrogram_reconstruction(melspectrogram, region_mask, dataset_params):
    power = librosa.db_to_power(melspectrogram)
    power[region_mask == False] = 0
    spectrogram = librosa.feature.inverse.mel_to_stft(power, sr=dataset_params.sr, n_fft=dataset_params.n_fft)
    reconstructed_audio = librosa.griffinlim(spectrogram, hop_length=dataset_params.hop_length,
                                             win_length=dataset_params.win_length)
    return reconstructed_audio


def relevant_audio_reconstruction(input_example, audio_path, regions, dataset_name, dataset_type, dataset_params):
    region_mask = np.full(dataset_params.saliency_shape[:-1], False)
    for index, (row_slice, col_slice) in enumerate(regions):
        region_mask[row_slice, col_slice] = True
    example = dataset_params.reshape_saliencies(input_example)[-1, :, :, -1]
    reconstructed_audio = None
    if dataset_name != DATASETS.AUDIOSET.value:
        example = example*80.0
    if dataset_type == DATASET_TYPES.MELSPECTROGRAM.value: 
        """ We get poor audio quality when we use audio reconstruction strategy for melspectrogram. 
            We recommend using audio identification strategy """
        reconstructed_audio = griffin_lim_melspectrogram_reconstruction(example, region_mask, dataset_params)
    elif dataset_type == DATASET_TYPES.SPECTROGRAM.value:
        reconstructed_audio = griffin_lim_reconstruction(example, region_mask, dataset_params)
    return reconstructed_audio
