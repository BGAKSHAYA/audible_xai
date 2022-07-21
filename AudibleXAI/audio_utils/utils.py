import librosa
import numpy as np
from constants.dataset_constants import DATASET_TYPES
from constants.evaluation_constants import AUDIO_TYPES
from constants.relevant_region_extractor_constants import AUDIO_EXTRACTION_STRATEGY
import soundfile as sf
from utils.evaluation_utils import true_random_regions_generator
from audio_utils.audio_amplifier import amplify_sound
from scipy.signal import butter, lfilter, windows
import logging

logger = logging.getLogger("audible_xai")


def audio_characteristics(dataset_type, dataset_params, shape):
    frequencies, times, samples = None, None, None
    if dataset_type == DATASET_TYPES.MELSPECTROGRAM.value:
        frequencies, times, samples = melspectrogram_characteristics(shape, dataset_params)

    elif dataset_type == DATASET_TYPES.SPECTROGRAM.value:
        frequencies, times, samples = spectrogram_characteristics(shape, dataset_params)
    return frequencies, times, samples


def spectrogram_characteristics(spectrogram_shape, dataset_params):
    frequencies = librosa.fft_frequencies(sr=dataset_params.sr, n_fft=dataset_params.n_fft)
    times = librosa.frames_to_time(range(spectrogram_shape[1]), hop_length=dataset_params.hop_length,
                                   sr=dataset_params.sr)
    samples = librosa.frames_to_samples(range(spectrogram_shape[1]), hop_length=dataset_params.hop_length)
    return frequencies, times, samples


def melspectrogram_characteristics(melspectrogram_shape, dataset_params):
    frequencies = librosa.mel_frequencies(n_mels=melspectrogram_shape[0], fmin=dataset_params.mel_min_hz,
                                          fmax=dataset_params.mel_max_hz, htk=True)
    times = librosa.frames_to_time(range(melspectrogram_shape[1]), hop_length=dataset_params.hop_length,
                                   sr=dataset_params.sr)
    samples = librosa.frames_to_samples(range(melspectrogram_shape[1]), hop_length=dataset_params.hop_length)
    return frequencies, times, samples


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """IIR bandpass filter"""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered = lfilter(b, a, data)
    return filtered


def bandpass_window_sinc_function(signal, sr, low_freq, high_freq, start_sample, end_sample):
    fL = low_freq / sr  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    fH = high_freq / sr  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    b = 0.04  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
    N = int(np.ceil((4 / b)))
    if not N % 2:
        N += 1  # Make sure that N is odd.
    n = np.arange(N)

    # Compute a low-pass filter with cutoff frequency fH.
    hlpf = np.sinc(2 * fH * (n - (N - 1) / 2))
    hlpf *= np.blackman(N)
    hlpf = hlpf / np.sum(hlpf)

    # Compute a high-pass filter with cutoff frequency fL.
    hhpf = np.sinc(2 * fL * (n - (N - 1) / 2))
    hhpf *= np.blackman(N)
    hhpf = hhpf / np.sum(hhpf)

    # Create a high-pass filter from the low-pass filter through spectral inversion.
    hhpf = -hhpf
    hhpf[(N - 1) // 2] += 1

    # Convolve both filters.
    h = np.convolve(hlpf, hhpf)

    """ Get audio of a region that spans from start_sample to end_sample along time axis and 
                                   spans from fL to fH along frequency axis """
    window = windows.tukey(end_sample - start_sample, alpha=0.05)
    filtered_signal = np.copy(signal)
    filtered_signal[: start_sample], filtered_signal[end_sample:] = 0, 0
    filtered_signal[start_sample:end_sample] *= window
    filtered_signal = np.convolve(filtered_signal, h)
    return filtered_signal, filtered_signal[start_sample:end_sample]


def save_audio(input_example, saliency, regions, audio_extraction_strategy, random_selection_comparison_runs,
               dataset_name, dataset_type, dataset_params, audio_file_path, save_path, title, method=""):
    # Lazy loading to avoid circular imports
    from audio_utils.audio_identification import relevant_audio_identification
    from audio_utils.audio_reconstruction import relevant_audio_reconstruction

    if len(regions) == 0:
        logger.warning("no relevant regions selected")
        return None

    relevant_regions_audio, random_regions_audio = None, None
    if audio_extraction_strategy == AUDIO_EXTRACTION_STRATEGY.AUDIO_IDENTIFICATION.value:
        relevant_regions_audio, truncated_relevant_audio = relevant_audio_identification(
            audio_file_path, regions, dataset_name, dataset_type=dataset_type, dataset_params=dataset_params)
    elif audio_extraction_strategy == AUDIO_EXTRACTION_STRATEGY.AUDIO_RECONSTRUCTION.value:
        relevant_regions_audio = relevant_audio_reconstruction(input_example, audio_file_path, regions,
                                                               dataset_name, dataset_type=dataset_type,
                                                               dataset_params=dataset_params)

    random_audio_save_paths = [save_path + "random" + str(index + 1) + ".wav" for index in
                               range(random_selection_comparison_runs)]

    result_path = {
        AUDIO_TYPES.RELEVANT.value: save_path + "relevant.wav",
        AUDIO_TYPES.RANDOM.value: random_audio_save_paths
    }

    sf.write(result_path[AUDIO_TYPES.RELEVANT.value], relevant_regions_audio, dataset_params.sr, 'PCM_24')

    for random_audio_save_path in random_audio_save_paths:
        random_regions = true_random_regions_generator(regions, dataset_params.saliency_shape[:-1])
        if audio_extraction_strategy == AUDIO_EXTRACTION_STRATEGY.AUDIO_IDENTIFICATION.value:
            random_regions_audio, truncated_random_audio = relevant_audio_identification(
                 audio_file_path, random_regions, dataset_name, dataset_type=dataset_type, dataset_params=dataset_params)
        elif audio_extraction_strategy == AUDIO_EXTRACTION_STRATEGY.AUDIO_RECONSTRUCTION.value:
            random_regions_audio = relevant_audio_reconstruction(
                input_example, audio_file_path, random_regions, dataset_name, dataset_type=dataset_type,
                dataset_params=dataset_params)
        sf.write(random_audio_save_path, random_regions_audio, dataset_params.sr, 'PCM_24')
    amplify_sound(result_path[AUDIO_TYPES.RELEVANT.value], save_path + "amplified.wav")
    return result_path
