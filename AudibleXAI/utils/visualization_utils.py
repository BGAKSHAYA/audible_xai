import numpy as np
import skimage.filters
import matplotlib.pyplot as plt
from innvestigate_utils import utils_imagenet
from audio_utils.utils import audio_characteristics
import scipy.fftpack
from matplotlib.patches import Rectangle
from constants.saliency_constants import saliency_name_mapper
from textwrap import wrap


def normalize(a):
    a = (a - a.min()) / (a.max() - a.min())
    return a


def blend_input(e, x):
    e = np.sum(np.abs(e), axis=-1, keepdims=True)
    # Add blur
    e = skimage.filters.gaussian(e[0], 3)[None]
    # Normalize
    e = normalize(e)
    # Get and apply colormap
    heatmap = plt.get_cmap('jet')(e[:, :, :, 0])[:, :, :, :1]
    # Overlap
    ret = (1.0 - e) * (x.copy() / 255) + e * heatmap
    return ret


def plot_spectrogram(example, save_path):
    plt.imshow(example)
    plt.gca().invert_yaxis()
    plt.savefig(save_path)


def plot_heatmap(input_example, saliency, title, save_path):
    fig, ax = plt.subplots(1, 2, figsize=(15, 10), sharey="all")
    # Plot
    ax[0].imshow(utils_imagenet.heatmap(saliency))

    ax[1].imshow(utils_imagenet.heatmap(saliency))
    ax[1].imshow(saliency, alpha=0.5)

    plt.axis("off")
    plt.gca().invert_yaxis()
    plt.title(title, fontdict={'fontsize': 24})
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_overview_saliency_regions(visualizations, save_path=None,
                                   dataset_type="melspectrogram", dataset_params=None):
    saliency_methods = list(visualizations.keys())
    class_names = list(visualizations[saliency_methods[0]].keys())

    n_cols = len(saliency_methods)
    n_rows = len(class_names)

    plt.clf()
    fig, ax = plt.subplots(n_rows, n_cols + 1, figsize=(7, 7), sharey="all",
                           gridspec_kw={'wspace': 0.0001, 'hspace': 0.1})

    shape = dataset_params.saliency_shape[:-1]
    frequencies, times, samples = audio_characteristics(dataset_type, dataset_params, shape)

    xticks = list(range(0, shape[1], 100)) + [shape[1] - 1]
    yticks = list(range(0, shape[0], 50)) + [shape[0] - 1]
    xlabels = [round(times[xtick] * 10) / 10 for xtick in xticks]
    ylabels = [int(frequencies[ytick]) for ytick in yticks]

    for col_index, saliency_method in enumerate(saliency_methods):
        for row_index, class_name in enumerate(class_names):
            class_example = visualizations[saliency_method][class_name][0]
            masked_regions = np.full(shape, False)
            for row_slice, col_slice in class_example["regions"]:
                # Create a Rectangle patch
                rect = Rectangle((col_slice.start, row_slice.start), col_slice.stop - col_slice.start,
                                 row_slice.stop - row_slice.start, color="#006400")
                # Add the patch to the Axes
                ax[row_index, col_index + 1].add_patch(rect)
                masked_regions[row_slice, col_slice] = True

            ax[row_index, 0].imshow(class_example["input"], cmap="gray")
            ax[row_index, 0].set_xticks([])
            ax[row_index, 0].set_yticks([])

            ax[row_index, col_index + 1].imshow(utils_imagenet.heatmap(class_example["saliency"]))
            ax[row_index, col_index + 1].imshow(class_example["input"], alpha=0.8, cmap="gray")

            ax[row_index, col_index + 1].set_xticks([])
            ax[row_index, col_index + 1].set_yticks([])

            ax[0, col_index + 1].set_xlabel(saliency_name_mapper[saliency_method], rotation=45,
                                            horizontalalignment="center")
            ax[0, col_index + 1].xaxis.set_label_position('top')
            ax[0, 0].set_xlabel("input", rotation=45)
            ax[0, 0].xaxis.set_label_position('top')

            ax[row_index, 0].set_ylabel(class_name, fontsize=12, rotation=0, labelpad=10, verticalalignment="center")
    plt.gca().invert_yaxis()
    plt.axis("off")
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    # fig.suptitle("Region extractions for \n various Saliency Methods", fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + '.pdf', bbox_inches='tight')
    plt.show()


def plot_saliency_method_regions(visualizations, saliency_method, save_path=None,
                                 dataset_type="melspectrogram", dataset_params=None):
    class_names = list(visualizations.keys())
    n_rows = len(class_names)
    n_cols = len(visualizations[class_names[0]])
    plt.clf()
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(7, 7), sharey="all",
                           gridspec_kw={'wspace': 0.0001, 'hspace': 0.1})

    shape = dataset_params.saliency_shape[:-1]
    frequencies, times, samples = audio_characteristics(dataset_type, dataset_params, shape)

    xticks = list(range(0, shape[1], 100)) + [shape[1] - 1]
    yticks = list(range(0, shape[0], 50)) + [shape[0] - 1]
    xlabels = [round(times[xtick] * 10) / 10 for xtick in xticks]
    ylabels = [int(frequencies[ytick]) for ytick in yticks]

    for row_index, class_name in enumerate(class_names):
        for col_index, class_example in enumerate(visualizations[class_name]):
            masked_regions = np.full(shape, False)
            for row_slice, col_slice in class_example["regions"]:
                # Create a Rectangle patch
                rect = Rectangle((col_slice.start, row_slice.start), col_slice.stop - col_slice.start,
                                 row_slice.stop - row_slice.start, fill=None, color="#006400")
                # Add the patch to the Axes
                ax[row_index, col_index].add_patch(rect)
                masked_regions[row_slice, col_slice] = True

            ax[row_index, col_index].imshow(utils_imagenet.heatmap(class_example["saliency"]))
            ax[row_index, col_index].set_xticks([])
            ax[row_index, col_index].set_yticks([])
        ax[row_index, 0].set_ylabel('\n'.join(wrap(str(class_name), 20)), rotation=0, fontsize=14, labelpad=10,
                                    verticalalignment="center")

    plt.gca().invert_yaxis()
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + '.pdf', bbox_inches='tight')
    plt.show()


def plot_heatmap_with_relevant_regions(example, saliency, results, save_path=None,
                                       dataset_type="melspectrogram", dataset_params=None, title=""):
    fig, ax = plt.subplots(1, len(results) + 1, sharey="all")
    fig.tight_layout()

    shape = dataset_params.saliency_shape[:-1]
    frequencies, times, samples = audio_characteristics(dataset_type, dataset_params, shape)

    ax[0].imshow(utils_imagenet.heatmap(saliency))
    ax[0].imshow(dataset_params.reshape_saliencies(example)[0], alpha=0.6, cmap="gray")

    for index, regions in enumerate(results):
        for row_slice, col_slice in regions:
            # Create a Rectangle patch
            rect = Rectangle((col_slice.start, row_slice.start), col_slice.stop - col_slice.start,
                             row_slice.stop - row_slice.start, color="#006400")
            # Add the patch to the Axes
            ax[index + 1].add_patch(rect)

        ax[index + 1].imshow(utils_imagenet.heatmap(saliency))
        ax[index + 1].imshow(dataset_params.reshape_saliencies(example)[0], alpha=0.6, cmap="gray")

    for index in range(0, len(results) + 1):
        xticks = list(range(0, shape[1], 100)) + [shape[1] - 1]
        yticks = list(range(0, shape[0], 50)) + [shape[0] - 1]

        xlabels = [round(times[xtick] * 10) / 10 for xtick in xticks]
        ylabels = [int(frequencies[ytick]) for ytick in yticks]

        ax[index].set_xticks(xticks)
        ax[index].set_xticklabels(xlabels)
        ax[index].set_xlabel("Time (s)")
        ax[0].set_yticks(yticks)
        ax[0].set_yticklabels(ylabels)
        ax[0].set_ylabel("Frequency (Hz)")

    plt.gca().invert_yaxis()
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + 'relevant_regions.png', bbox_inches='tight')
    plt.show()


def plot_fourier_transform(y, sr, title="Fourier Transform"):
    # Number of samplepoints
    N = len(y)
    # sample spacing
    T = 1 / sr
    x = np.linspace(0.0, N * T, N)

    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)

    fig, ax = plt.subplots()
    ax.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
    plt.title(title)
    plt.show()


def plot_saliency_methods(visualizations, save_path=None,
                          dataset_type="melspectrogram", dataset_params=None, title=None):
    plt.clf()
    example = visualizations["example"]
    plt.figure(figsize=(10, 6))
    plot_sequence = [
        (example, (3, 4, 1), "Input"),
    ]
    method_values = list(visualizations["methods"].items())
    start = 2
    for i in range(0, 9, 3):
        plot_sequence += [(value, (3, 4, start + index), saliency_name_mapper[method])
                          for index, (method, value) in enumerate(method_values[i:i + 3])]
        start += 4

    for i, (plot_x, plot_id, plt_title) in enumerate(plot_sequence):
        ax = plt.subplot(*plot_id)
        ax.set_title(plt_title)
        if i == 0:
            ax.imshow(plot_x, cmap="gray")
        else:
            ax.imshow(utils_imagenet.heatmap(plot_x))
        ax.invert_yaxis()
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + title + '.pdf', bbox_inches='tight')
    plt.show()


def plot_overview_mispredictions(visualizations, save_path=None,
                                 dataset_type="melspectrogram", dataset_params=None, title=None):
    plt.clf()
    plt.figure(figsize=(10, 6))
    start = 1
    total_rows = len(visualizations.keys())
    for predicted_classname, actual_details in visualizations.items():
        if len(actual_details) == 0: continue
        for index, actual_detail in enumerate(actual_details[:3]):
            ax = plt.subplot(total_rows, 3, start + index)
            if index == 0:
                ax.set_ylabel('\n'.join(wrap("$\itp=$" + str(predicted_classname), 20)), rotation=0, fontsize=14,
                              labelpad=60,
                              verticalalignment="center")
            for row_slice, col_slice in actual_detail["regions"]:
                # Create a Rectangle patch
                rect = Rectangle((col_slice.start, row_slice.start), col_slice.stop - col_slice.start,
                                 row_slice.stop - row_slice.start, color="#006400")
                # Add the patch to the Axes
                ax.add_patch(rect)
            ax.imshow(utils_imagenet.heatmap(actual_detail["saliency"]))
            ax.imshow(actual_detail["input"], alpha=0.6)
            ax.set_title("$\itt=$" + str(actual_detail["actual_class"]), fontsize=14)
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
        start += 3

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + '.pdf', bbox_inches='tight')
    plt.show()


def plot_overview_region_selections(visualizations, save_path=None):
    fig, axs = plt.subplots(2, 5, figsize=(2, 2.5), sharey="all", gridspec_kw={'wspace': 0.01, 'hspace': 0.01})
    for ax, visualization in zip(axs.reshape(-1), visualizations):
        for row_slice, col_slice in visualization["regions"]:
            # Create a Rectangle patch
            rect = Rectangle((col_slice.start, row_slice.start), col_slice.stop - col_slice.start,
                             row_slice.stop - row_slice.start, fill=None, color="#006400", lw=0.3)
            # Add the patch to the Axes
            ax.add_patch(rect)
        ax.imshow(utils_imagenet.heatmap(visualization["saliency"]))
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path + '.pdf', bbox_inches='tight')
    plt.show()
