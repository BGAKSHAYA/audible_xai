import warnings
import os
from typing import Union

import numpy as np
import logging
import pandas as pd
import copy
import math
from constants.relevant_region_extractor_constants import REGION_EXTRACTION_METHODS
from services.dataset_service import DatasetService
from region_extraction_utils.utils import improve_regions_selections, get_CH_index_score, get_filtered_regions, \
    grid_search_hps, interpret_class_frequencies
from audio_utils.utils import save_audio
from services.file_service import FileService

warnings.filterwarnings("ignore")
logger = logging.getLogger("audible_xai")


class RelevantRegionsExtractorService:
    def __init__(self, path_config, run_config):
        self.saliency_path = path_config["results"]["saliency_path"]
        self.regions_path = path_config["results"]["correct_regions_path"]

        self.data = DatasetService.instance.get_dataset(filter_type="correct")
        self.audio_file_paths = DatasetService.instance.get_audio_file_paths(filter_type="correct")
        self.dataset_filenames = DatasetService.instance.get_dataset_filenames(filter_type="correct")

        self.dataset_params = DatasetService.instance.get_dataset_params()
        self.dataset_type = DatasetService.instance.get_dataset_type()
        self.dataset_name = DatasetService.instance.get_dataset_name()
        self.file_service = FileService()
        self.class_names = list(self.data.keys())
        self.run_config = run_config

        region_extraction_method = self.run_config["method"]
        self.extraction_method = REGION_EXTRACTION_METHODS[region_extraction_method]

    def run(self, saliency_methods):
        reset_results = self.run_config["reset"]
        random_selection_comparison_runs = self.run_config["random_selection_comparison_runs"]
        audio_extraction_paths = {}

        for saliency_method in saliency_methods:
            saliency_results = np.load(os.path.join(self.saliency_path, saliency_method + ".npy"),
                                       allow_pickle=True).item()
            self.file_service.create_directory(self.regions_path, saliency_method, self.class_names)
            logger.debug("Started region extraction method for {}".format(saliency_method))
            regions_path = os.path.join(self.regions_path, saliency_method, "regions.npy")
            filtered_regions_path = os.path.join(self.regions_path, saliency_method, "filtered_regions.npy")
            runs_summary_path = os.path.join(self.regions_path, saliency_method, "runs_summary.csv")

            if self.run_config["reload_regions_last_run"] and self.file_service.isfile(regions_path):
                regions = np.load(regions_path, allow_pickle=True).item()
            else:
                regions = self.grid_search_regions(saliency_results, saliency_method, regions_path, runs_summary_path)

            if self.run_config["run_region_optimization"]:
                max_inter_class_penalties = self.run_config["search_space"]["max_inter_class_penalties"]
                filtered_regions = get_filtered_regions(regions, saliency_results, self.dataset_filenames, self.data,
                                                        self.dataset_params, max_inter_class_penalties)
                np.save(filtered_regions_path, filtered_regions)
                regions = filtered_regions

            if self.run_config["save_audio"]:
                for class_name in self.class_names:
                    if reset_results:
                        self.file_service.remove_directory_contents(os.path.join(self.saliency_path, saliency_method),
                                                                    class_name)
                    class_path = os.path.join(self.regions_path, saliency_method, str(class_name))
                    audio_extraction_path = self.save_region_audio(saliency_results, regions, class_name,
                                                                   class_path, random_selection_comparison_runs)
                    audio_extraction_paths[class_name] = audio_extraction_path
                np.save(os.path.join(self.regions_path, saliency_method + ".npy"), audio_extraction_paths)

    def grid_search_regions(self, saliency_results, saliency_method, regions_path, runs_summary_path):
        search_space = self.run_config["search_space"]
        saliency_shape = self.dataset_params.saliency_shape
        # Recursively explore search space and get all possible hyperparameter combinations
        hps = grid_search_hps(search_space)
        best_CH_index, best_hp, best_hp_regions = -math.inf, None, None
        regions = {}

        runs_summary = pd.DataFrame(hps)
        result_columns = ["CH_index", "CH_index_freq", "inter_class_distance",
                          "intra_class_distance", "average_cluster_quality"]
        runs_summary = runs_summary.reindex(columns=runs_summary.columns.tolist() + result_columns)
        remove_silence = True if saliency_method == "LIME" or saliency_method == "GRAD_CAM" else False
        logger.debug("Grid Search Started")
        for index, hp in enumerate(hps):
            skip_current_hp = False  # We skip hps that do not generate regions
            class_index = 0

            while class_index < len(self.class_names) and not skip_current_hp:
                class_name = self.class_names[class_index]
                regions[class_name] = {}
                saliency_computed_files = saliency_results[class_name].keys()
                for example, filename in zip(self.data[class_name], self.dataset_filenames[class_name]):
                    assert filename in saliency_computed_files, "Saliency is not computed for {}".format(filename)
                    saliency = np.copy(saliency_results[class_name][filename])
                    if len(saliency) == 0 or saliency.max() <= 0:
                        logger.warning("No positive relevant features found for {}".format(filename))
                        regions[class_name][filename] = []
                        continue
                    title = DatasetService.instance.get_descriptive_title(filename)
                    rectangle_regions = self.extraction_method(example, saliency, self.dataset_params, hp,
                                                               title, remove_silence)
                    """
                    Some hp combinations might lead to not selecting regions, then we skip such hps
                    For example a large value for min_cluster_size could be one reason
                    """
                    if len(rectangle_regions) == 0 and self.run_config["region_tuning_mode"]:
                        skip_current_hp = True
                        break
                    regions[class_name][filename] = rectangle_regions
                class_index += 1

            if skip_current_hp:
                logger.debug("Skipped processing for hp {}".format(hp))
                continue

            CH_index, CH_index_freq, inter_class_distance, intra_class_distance, class_representatives = \
                                                            get_CH_index_score(regions, saliency_shape)

            if not self.run_config["region_tuning_mode"]:
                interpret_class_frequencies(class_representatives, self.dataset_type, self.dataset_params)

            result = [CH_index, CH_index_freq, inter_class_distance, intra_class_distance]
            runs_summary.loc[index, result_columns] = result

            if CH_index > best_CH_index:
                best_CH_index, best_hp_regions = CH_index, copy.deepcopy(regions)
                best_hp = hp
            if index % 10 == 0:
                logger.debug("Best hp so far {} with CH Index {} after {} searches".format(
                    best_hp, best_CH_index, index + 1))
                runs_summary.to_csv(runs_summary_path, index=False)

        logger.debug(runs_summary)
        runs_summary.to_csv(runs_summary_path, index=False)
        regions = best_hp_regions
        np.save(regions_path, regions)
        return regions

    def save_region_audio(self, saliency_results, regions, class_name, class_path, random_selection_comparison_runs=1):
        result_audio_paths = {}
        for example, filename, audiopath in zip(self.data[class_name], self.dataset_filenames[class_name],
                                                self.audio_file_paths[class_name]):
            input_example = self.dataset_params.reshape_model_input(example)
            save_path = os.path.join(class_path, filename + "_")

            assert filename in saliency_results[class_name], "Saliency is not computed for {}".format(filename)
            saliency = saliency_results[class_name][filename]

            title = DatasetService.instance.get_descriptive_title(filename)
            result_audio_path = save_audio(input_example, saliency, regions[class_name][filename],
                                           self.run_config["audio_extraction_strategy"],
                                           random_selection_comparison_runs,
                                           self.dataset_name, self.dataset_type,
                                           self.dataset_params, audiopath, save_path, title)
            result_audio_paths[filename] = result_audio_path
        return result_audio_paths
