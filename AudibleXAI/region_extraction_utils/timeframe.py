import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from innvestigate_utils import utils_imagenet


def relevant_timeframe_extractor_naive_approach(arr):
    max = np.max(arr)
    max_index = np.argmax(arr)
    start = max_index
    end = max_index
    length = len(arr)
    while start >= 0 and arr[start - 1] > max - 50:
        start = start - 1
    while end < length and arr[end + 1] > max - 50:
        end = end + 1
    return start, end


def get_timeframe(original_example, input_example, saliency, save_path,
                  dataset_shape=None):
    relevance = np.copy(original_example)
    relevance = relevance.reshape(dataset_shape)
    rows, cols, _ = np.where(saliency[0] > 0)
    unique_col_counts = np.unique(cols, return_counts=True)
    relevant_col_counts = unique_col_counts[1]
    most_relevant_col = np.argmax(relevant_col_counts)
    start, end = relevant_timeframe_extractor_naive_approach(relevant_col_counts)
    mask = np.hstack((np.arange(0, start), np.arange(end, dataset_shape[-1])))

    relevance = relevance[:, unique_col_counts[0][start:end]]

    # Inverse Operations of Short Term Fourier Transform
    db_to_amplitude_spect = librosa.db_to_amplitude(relevance)
    audio_signal = librosa.core.spectrum.griffinlim(db_to_amplitude_spect, hop_length=50)
    # write output
    sf.write(save_path + '.wav', audio_signal, 8000, 'PCM_24')

    input_example[0][:, np.array((start, most_relevant_col, end)), :] = 0
    labels = ['start', 'max', 'end']

    plt.imshow(utils_imagenet.heatmap(saliency)[0])
    plt.imshow(input_example[0], cmap="gray", alpha=0.4)
    plt.xticks(np.array((start, most_relevant_col, end)), labels, rotation=45, fontsize=8)
    plt.yticks([])
    plt.gca().invert_yaxis()
    plt.axis("on")
    plt.title("Relevant Time frame", fontdict={'fontsize': 24})
    plt.savefig(save_path + 'relevant_regions.png')
    plt.show()
