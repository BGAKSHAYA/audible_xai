import os
import numpy as np
import random
from services.dataset_service import DatasetService
from services.file_service import FileService
from utils.visualization_utils import plot_overview_saliency_regions, plot_saliency_method_regions, \
    plot_saliency_methods, plot_overview_mispredictions, plot_overview_region_selections
from region_extraction_utils.utils import combine_example_regions
import logging

logger = logging.getLogger("audible_xai")


class VisualizationService:
    def __init__(self, path_config, run_config):
        self.saliency_path = path_config["results"]["saliency_path"]
        self.regions_path = path_config["results"]["correct_regions_path"]
        self.incorrect_regions_path = path_config["results"]["incorrect_regions_path"]

        self.visualizations_path = path_config["results"]["visualizations_path"]

        self.data = DatasetService.instance.get_dataset(filter_type="correct")
        self.dataset_filenames = DatasetService.instance.get_dataset_filenames(filter_type="correct")

        self.incorrect_data = DatasetService.instance.get_dataset(filter_type="incorrect")
        self.incorrect_dataset_filenames = DatasetService.instance.get_dataset_filenames(filter_type="incorrect")
        self.predicted_labels = DatasetService.instance.get_prediction_details()

        self.dataset_params = DatasetService.instance.get_dataset_params()
        self.dataset_type = DatasetService.instance.get_dataset_type()
        self.dataset_name = DatasetService.instance.get_dataset_name()
        self.all_classnames = DatasetService.instance.get_classnames()

        self.file_service = FileService()
        self.class_names = list(self.data.keys())
        self.run_config = run_config

    def run(self, saliency_methods):
        self.file_service.create_directory(self.visualizations_path)
        if "overview_saliencies" in self.run_config["methods"]:
            """ 
                Take a random example of a class, and make a grid plot to visualize saliencies of all the
                attribution methods 
            """
            visualizations = {}
            random_class_name = random.choice(self.class_names)
            random_example_index = random.choice(range(0, len(self.data[random_class_name])))
            logger.debug("Started grid plot for saliency methods")
            example = self.data[random_class_name][random_example_index]
            filename = self.dataset_filenames[random_class_name][random_example_index]
            title = DatasetService.instance.get_descriptive_title(filename)
            title = "Attribution Methods " + title
            visualizations["example"] = self.dataset_params.reshape_saliencies(example)[0]
            visualizations["methods"] = {}

            for saliency_method in saliency_methods:
                saliency_results = np.load(os.path.join(self.saliency_path, saliency_method + ".npy"),
                                           allow_pickle=True).item()

                visualizations["methods"][saliency_method] = saliency_results[random_class_name][filename]
            save_path = os.path.join(self.visualizations_path, "overview_saliency_methods")
            plot_saliency_methods(visualizations, save_path=save_path,
                                  dataset_type=self.dataset_type, dataset_params=self.dataset_params, title=title)

        if "overview_saliency_regions" in self.run_config["methods"]:
            """
                Create an overview grid plot for each saliency method visualizing class region selections. 
                Inspect the intra-class and inter-class similarity of region selections in the obtained plots
            """
            visualizations = {}
            for saliency_method in saliency_methods:
                visualizations[saliency_method] = {}
                saliency_results = np.load(os.path.join(self.saliency_path, saliency_method + ".npy"),
                                           allow_pickle=True).item()
                class_regions_path = os.path.join(self.regions_path, saliency_method, "regions.npy")
                class_regions = np.load(class_regions_path, allow_pickle=True).item()
                logger.debug("Started visualization for", saliency_method)
                shape = self.dataset_params.saliency_shape
                for class_name in class_regions.keys():
                    if len(self.data[class_name]) == 0: continue
                    visualizations[saliency_method][class_name] = []
                    no_exs = 10
                    class_example = np.zeros(shape)
                    for input_example, filename in zip(self.data[class_name][:no_exs],
                                                       self.dataset_filenames[class_name][:no_exs]):
                        saliency = saliency_results[class_name][filename]
                        regions = class_regions[class_name][filename]
                        saliency[np.where(saliency < 0)] = 0
                        THRESHOLD = np.percentile(saliency, 95)
                        saliency[np.where(saliency < THRESHOLD)] = 0
                        visualization = {
                            "saliency": saliency,
                            "input": self.dataset_params.reshape_saliencies(input_example)[0],
                            "regions": regions
                        }
                        visualizations[saliency_method][class_name].append(visualization)
                        class_example += combine_example_regions(regions, self.dataset_params.saliency_shape)

                    # Average class region selection for the 10 class examples
                    class_example /= 10
                    class_example[np.where(class_example < 0.6)] = 0
                    visualization = {
                        "saliency": np.zeros(shape),
                        "input": class_example,
                        "regions": []
                    }
                    visualizations[saliency_method][class_name].append(visualization)

                save_path = os.path.join(self.visualizations_path, saliency_method)
                plot_saliency_method_regions(visualizations[saliency_method], saliency_method, save_path=save_path,
                                             dataset_type=self.dataset_type, dataset_params=self.dataset_params)

            plot_overview_saliency_regions(visualizations, save_path=os.path.join(self.visualizations_path, "overview"),
                                           dataset_type=self.dataset_type, dataset_params=self.dataset_params)

        if "overview_mispredictions" in self.run_config["methods"]:
            """
                Create an overview grid plot for each saliency method visualizing region selections for mispredictions. 
            """
            visualizations = {}
            for saliency_method in saliency_methods:
                visualizations[saliency_method] = {}
                saliency_results = np.load(os.path.join(self.saliency_path, saliency_method + ".npy"),
                                           allow_pickle=True).item()
                regions_path = os.path.join(self.incorrect_regions_path, saliency_method, "regions.npy")
                regions = np.load(regions_path, allow_pickle=True).item()
                logger.debug("Started visualization for", saliency_method)
                for class_name in regions.keys():
                    predicted_labels = self.incorrect_data[class_name].keys()
                    for predicted_label in predicted_labels:
                        for example, filename in zip(
                                self.incorrect_data[class_name][predicted_label],
                                self.incorrect_dataset_filenames[class_name][predicted_label]):

                            saliency = saliency_results[class_name][filename]
                            predicted_class = self.all_classnames[predicted_label]
                            visualization = {
                                "saliency": saliency,
                                "actual_class": class_name,
                                "input": example,
                                "regions": regions[class_name][predicted_label][filename]
                            }
                            if predicted_class not in visualizations[saliency_method]:
                                visualizations[saliency_method][predicted_class] = []
                            visualizations[saliency_method][predicted_class].append(visualization)

                save_path = os.path.join(self.visualizations_path, "overview_mispredictions_" + saliency_method)
                plot_overview_mispredictions(
                    visualizations[saliency_method],
                    save_path=save_path,
                    dataset_type=self.dataset_type, dataset_params=self.dataset_params)
