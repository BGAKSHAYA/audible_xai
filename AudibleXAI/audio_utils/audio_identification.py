import numpy as np
from audio_utils.utils import audio_characteristics, bandpass_window_sinc_function
import librosa


def relevant_audio_identification(audio_path, regions, dataset_name, dataset_type, dataset_params):
    y, sr = librosa.core.load(audio_path, sr=dataset_params.sr, mono=False)
    y = librosa.to_mono(y)
    shape = dataset_params.saliency_shape[:-1]

    frequencies, times, samples = audio_characteristics(dataset_type, dataset_params, shape)

    relevant_audio_segments = []
    truncated_signal = np.array([])

    for index, (row_slice, col_slice) in enumerate(regions):
        # Sample rate and desired cutoff frequencies (in Hz).
        start_freq, end_freq = frequencies[row_slice.start], frequencies[row_slice.stop - 1]

        # Desired cutoff times (in s)
        start_time, end_time = 1000 * times[col_slice.start], 1000 * times[col_slice.stop - 1]

        # Desired cutoff samples
        start_sample, end_sample = samples[col_slice.start]-samples[0], samples[col_slice.stop - 1]-samples[0]

        if start_sample > len(y):
            # Region selection in the last padded part of the audio. Ignore audio extraction as its just silence
            continue

        if end_sample > len(y):
            """ Region selection extends to the last padded part of the audio. 
                Change the end sample to the length of the audio, as there is silence after that """
            end_sample = len(y)

        filtered_signal, convolved_signal = bandpass_window_sinc_function(np.copy(y), sr, start_freq, end_freq,
                                                                          start_sample, end_sample)
        relevant_audio_segments.append(filtered_signal)
        # We might want to only listen to the concatenated audio of regions, which has lesser duration than the original
        truncated_signal = np.hstack((truncated_signal, convolved_signal))

    relevant_audio = sum(relevant_audio_segments) if len(relevant_audio_segments) != 0 else np.zeros(len(y))
    return relevant_audio, truncated_signal

