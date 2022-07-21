from utils.visualization_utils import plot_heatmap_with_relevant_regions, plot_fourier_transform
import numpy as np
from sklearn import preprocessing
from utils.evaluation_utils import mse, mae, se
from audio_utils.utils import audio_characteristics
import math
import copy
import pandas as pd
import logging

logger = logging.getLogger("audible_xai")


def improve_regions_selections(input_example, saliency, selected_regions, class_region_relevance_score,
                               dataset_type=None, dataset_params=None, title=None, save_path=None):
    filtered_regions = []

    masked_filtered_regions = np.full(dataset_params.saliency_shape, False)
    masked_regions = np.full(dataset_params.saliency_shape, False)

    region_relevance_scores = []

    if len(selected_regions) == 0:
        logging.warning("Region optimization skipped as no regions were received")
        return selected_regions

    for row_slice, col_slice in selected_regions:
        region_relevance_score = np.sum(
            saliency[row_slice, col_slice] * class_region_relevance_score[row_slice, col_slice])
        region_relevance_scores.append(region_relevance_score)

    THRESHOLD = min(0, np.percentile(region_relevance_scores, 100))
    for index, (row_slice, col_slice) in enumerate(selected_regions):
        masked_regions[row_slice, col_slice] = True
        if region_relevance_scores[index] >= THRESHOLD:
            filtered_regions.append([row_slice, col_slice])
            masked_filtered_regions[row_slice, col_slice] = True

    if len(filtered_regions) == 0:
        plot_heatmap_with_relevant_regions(input_example[0], saliency, [masked_regions, masked_filtered_regions],
                                           dataset_type=dataset_type, dataset_params=dataset_params,
                                           save_path=save_path, title=title)
        logger.error("This should never happen, Received {} regions and {} regions after filtering".
                     format(len(selected_regions), len(filtered_regions)))
        raise Exception("This should never happen, Received {} regions and {} regions after filtering".
                        format(len(selected_regions), len(filtered_regions)))
    return filtered_regions


def get_filtered_regions(regions, saliency_results, dataset_filenames, data,
                         dataset_params, max_inter_class_penalties):
    saliency_shape = dataset_params.saliency_shape
    original_CH_index = get_CH_index_score(regions, saliency_shape)
    logger.debug("Before running optimized approach, CH Index = {}".format(original_CH_index))

    best_CH_index, best_hp, best_hp_regions = -math.inf, None, None
    runs_summary = []
    for max_penalty in max_inter_class_penalties:
        class_region_relevance_scores = get_class_region_relevance_scores(regions, saliency_shape, max_penalty)
        filtered_regions = {}
        removed_regions = 0
        no_regions = 0

        for class_name in data.keys():
            if len(data[class_name]) == 0:
                continue
            class_region_relevance_score = class_region_relevance_scores[class_name]
            filtered_regions[class_name] = {}

            for example, filename in zip(data[class_name], dataset_filenames[class_name]):
                input_example = dataset_params.reshape_saliencies(example)
                saliency = saliency_results[class_name][filename]

                filtered_regions[class_name][filename] = improve_regions_selections(
                    input_example, saliency, regions[class_name][filename],
                    class_region_relevance_score, dataset_params=dataset_params)

                no_regions += len(filtered_regions[class_name][filename])
                removed_regions += len(regions[class_name][filename]) - len(filtered_regions[class_name][filename])

        CH_index, CH_index_freq, inter_class_distance, intra_class_distance = get_CH_index_score(filtered_regions, saliency_shape)
        if CH_index > best_CH_index:
            best_CH_index, best_hp_regions = CH_index, copy.deepcopy(filtered_regions)
            best_hp = max_penalty
        runs_summary.append([max_penalty, no_regions, removed_regions, CH_index, CH_index_freq, inter_class_distance,
                             intra_class_distance])

    result_columns = ["CH_index", "CH_index_freq", "inter_class_distance", "intra_class_distance"]
    runs_summary = pd.DataFrame(runs_summary, columns=["max_penalty", "no_regions", "removed_regions"] + result_columns)
    logger.debug(runs_summary)
    return best_hp_regions


def combine_example_regions(regions, shape):
    example = np.zeros(shape)
    for row_slice, col_slice in regions:
        example[row_slice, col_slice] += 1
    return example


def grid_search_hps(search_space):
    keys, values = list(search_space.keys()), list(search_space.values())
    hps = grid_search(values, 0)
    for index, hp in enumerate(hps):
        hps[index] = dict(zip(keys, hp))
    if "min_cluster_size" and "min_samples" in hps[0]:
        removed_hps = []
        for index, hp in enumerate(hps):
            if hp["min_samples"] <= hp["min_cluster_size"]:
                removed_hps.append(hp)
        hps = removed_hps
    return hps


def grid_search(lists, index) -> list:
    if len(lists) - 1 == index:
        return [[el] for el in lists[index]]
    else:
        l1 = lists[index]
        l2 = grid_search(lists, index + 1)
        result = []
        for el in l1:
            result.extend([[el, *els2] for els2 in l2])
    return result


def get_class_region_relevance_scores(regions, dataset_shape, max_penalty):
    class_names = regions.keys()
    K = len(class_names)
    class_representatives = {}
    scaler = preprocessing.MinMaxScaler(feature_range=(-max_penalty, 1))

    for class_name, class_examples in regions.items():
        class_representative = np.zeros(dataset_shape)
        for example_index, regions in class_examples.items():
            class_representative += combine_example_regions(regions, dataset_shape)
        if len(class_examples) == 0: continue
        class_representative /= len(class_examples)
        class_representatives[class_name] = class_representative

    penalized_class_region_relevance_score = {}
    global_representative = np.array(list(class_representatives.values()))
    global_representative = np.mean(global_representative, axis=0)

    for class_name in class_representatives.keys():
        penalized_class_region_relevance_score[class_name] = class_representatives[class_name] - global_representative
        penalized_class_region_relevance_score[class_name] = scaler.fit_transform(
            np.squeeze(penalized_class_region_relevance_score[class_name]))
        penalized_class_region_relevance_score[class_name] = np.expand_dims(
            penalized_class_region_relevance_score[class_name], axis=-1)
    return penalized_class_region_relevance_score


def get_CH_index_score(class_regions, shape):
    class_names = class_regions.keys()
    class_representatives = dict()
    class_representatives_freq = dict()

    inter_class_distance, intra_class_distance = 0, 0
    inter_class_distance_freq, intra_class_distance_freq = 0, 0

    n, K = dict(), len(class_names)  # n = No. of instances in each class, K = No. of classes
    distance_func = mse
    for class_name, class_examples in class_regions.items():
        class_representative = np.zeros(shape)
        examples = np.empty((0, *tuple(shape)))
        for regions in class_examples.values():
            example = combine_example_regions(regions, shape)
            example = np.expand_dims(example, axis=0)
            examples = np.vstack((examples, example))
            class_representative += example[0]
        class_representative_freq = np.sum(class_representative, axis=1)
        n_k = len(class_examples)
        if n_k == 0:
            continue

        class_representative /= n_k
        class_representative_freq /= n_k

        n[class_name] = n_k
        for example in examples:
            intra_class_distance += distance_func(class_representative, example)
            intra_class_distance_freq += distance_func(class_representative_freq, np.sum(example, axis=1))

        class_representatives[class_name] = class_representative
        class_representatives_freq[class_name] = class_representative_freq

    N = sum(n.values())  # Total instances in the dataset
    intra_class_distance /= (N - K)
    intra_class_distance_freq /= (N - K)

    global_representative = np.array(list(class_representatives.values()))
    global_representative = np.mean(global_representative, axis=0)

    global_representative_freq = np.array(list(class_representatives_freq.values()))
    global_representative_freq = np.mean(global_representative_freq, axis=0)

    for class_name, class_representative in class_representatives.items():
        inter_class_distance += (n[class_name] * distance_func(class_representative, global_representative))

    for class_name, class_representative_freq in class_representatives_freq.items():
        inter_class_distance_freq += (n[class_name] * distance_func(class_representative_freq, global_representative_freq))

    inter_class_distance /= (K - 1)
    inter_class_distance_freq /= (K - 1)

    CH_index = inter_class_distance / intra_class_distance
    CH_index_freq = inter_class_distance_freq / intra_class_distance_freq

    return CH_index, CH_index_freq, inter_class_distance, intra_class_distance, class_representatives_freq


def interpret_class_frequencies(class_representatives, dataset_type, dataset_params):
    shape = dataset_params.saliency_shape[:-1]
    frequencies, _, _ = audio_characteristics(dataset_type, dataset_params, shape)

    for class_name, class_representative in class_representatives.items():
        normalized_class_representative = (class_representative - class_representative.min())/\
                                          (class_representative.max() - class_representative.min())
        max_positions = np.where(normalized_class_representative == 1.0)[0]
        relevant_frequencies = []
        if len(max_positions) == 0:
            continue

        start, end = max_positions[0], max_positions[0]
        logger.debug("\n******* Class {} *******".format(class_name))
        for position in max_positions[1:]:
            if position == end + 1:
                end += 1
            else:
                relevant_frequencies.append([start, end])
                start, end = position, position
        relevant_frequencies.append([start, end])
        logger.debug(relevant_frequencies)
        for start, end in relevant_frequencies:
            logger.debug(frequencies[start-1 if start != 0 else start], frequencies[end])
    return None
