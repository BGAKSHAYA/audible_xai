import warnings
import os
import numpy as np
import logging
from constants.relevant_region_extractor_constants import REGION_EXTRACTION_METHODS
from services.dataset_service import DatasetService
from region_extraction_utils.utils import grid_search_hps
from audio_utils.utils import save_audio
from services.file_service import FileService

warnings.filterwarnings("ignore")
logger = logging.getLogger("audible_xai")


class MispredictionsRegionsExtractorService:
    def __init__(self, path_config, run_config):
        self.saliency_path = path_config["results"]["saliency_path"]
        self.regions_path = path_config["results"]["incorrect_regions_path"]

        self.data = DatasetService.instance.get_dataset(filter_type="incorrect")
        self.audio_file_paths = DatasetService.instance.get_audio_file_paths(filter_type="incorrect")
        self.dataset_filenames = DatasetService.instance.get_dataset_filenames(filter_type="incorrect")

        self.dataset_params = DatasetService.instance.get_dataset_params()
        self.dataset_type = DatasetService.instance.get_dataset_type()
        self.dataset_name = DatasetService.instance.get_dataset_name()
        self.file_service = FileService()
        self.class_names = list(self.data.keys())
        self.all_classnames = DatasetService.instance.get_classnames()
        self.run_config = run_config

        region_extraction_method = self.run_config["method"]
        self.extraction_method = REGION_EXTRACTION_METHODS[region_extraction_method]

    def run(self, saliency_methods):
        reset_results = self.run_config["reset"]
        audio_extraction_paths = {}

        for saliency_method in saliency_methods:
            saliency_results = np.load(os.path.join(self.saliency_path, saliency_method + ".npy"),
                                       allow_pickle=True).item()
            self.file_service.create_directory(self.regions_path, saliency_method, self.class_names)
            logger.debug("Started incorrect region extraction method for {}".format(saliency_method))
            regions_path = os.path.join(self.regions_path, saliency_method, "regions.npy")

            if self.run_config["reload_regions_last_run"] and self.file_service.isfile(regions_path):
                regions = np.load(regions_path, allow_pickle=True).item()
            else:
                regions = self.grid_search_regions(saliency_results, saliency_method, regions_path)
                np.save(regions_path, regions)

            for class_name in self.class_names:
                if reset_results:
                    self.file_service.remove_directory_contents(os.path.join(self.saliency_path, saliency_method),
                                                                class_name)
                class_path = os.path.join(self.regions_path, saliency_method, str(class_name))
                audio_extraction_path = self.save_region_audio(saliency_results, regions, class_name,
                                                               class_path, random_selection_comparison_runs=0)
                audio_extraction_paths[class_name] = audio_extraction_path

            np.save(os.path.join(self.regions_path, saliency_method + ".npy"), audio_extraction_paths)

    def grid_search_regions(self, saliency_results, saliency_method, regions_path):
        search_space = self.run_config["search_space"]
        # Recursively explore search space and get all possible hyperparameter combinations
        hp = grid_search_hps(search_space)[0]
        regions = {}
        remove_silence = True if saliency_method == "LIME" or saliency_method == "GRAD_CAM" else False

        for class_name in self.class_names:
            regions[class_name] = {}
            saliency_computed_files = saliency_results[class_name].keys()
            predicted_labels = self.data[class_name].keys()
            for predicted_label in predicted_labels:
                regions[class_name][predicted_label] = {}
                for example, filename in zip(self.data[class_name][predicted_label],
                                             self.dataset_filenames[class_name][predicted_label]):
                    assert filename in saliency_computed_files, "Saliency is not computed for {}".format(filename)
                    saliency = saliency_results[class_name][filename]
                    if saliency.max() == 0:
                        logger.warning("No positive relevant features found for {}".format(filename))
                        regions[class_name][predicted_label][filename] = []
                        continue

                    title = DatasetService.instance.get_descriptive_title(filename)

                    rectangle_regions = self.extraction_method(example, saliency, self.dataset_params, hp, title,
                                                               remove_silence=remove_silence)
                    regions[class_name][predicted_label][filename] = rectangle_regions
        np.save(regions_path, regions)
        return regions

    def save_region_audio(self, saliency_results, regions, class_name, class_path, random_selection_comparison_runs=0):
        result_audio_paths = {}
        predicted_labels = self.data[class_name].keys()
        for predicted_label in predicted_labels:
            for example, filename, audiopath in zip(
                    self.data[class_name][predicted_label], self.dataset_filenames[class_name][predicted_label],
                    self.audio_file_paths[class_name][predicted_label]):

                input_example = self.dataset_params.reshape_model_input(example)
                self.file_service.create_directory(class_path, str(self.all_classnames[predicted_label]))
                save_path = os.path.join(class_path, str(self.all_classnames[predicted_label]), filename + "_")

                assert filename in saliency_results[class_name], "Saliency is not computed for {}".format(filename)
                saliency = saliency_results[class_name][filename]

                title = DatasetService.instance.get_descriptive_title(filename)
                result_audio_path = save_audio(input_example, saliency, regions[class_name][predicted_label][filename],
                                               self.run_config["audio_extraction_strategy"],
                                               random_selection_comparison_runs,
                                               self.dataset_name, self.dataset_type,
                                               self.dataset_params, audiopath, save_path, title)
                result_audio_paths[filename] = result_audio_path
        return result_audio_paths
