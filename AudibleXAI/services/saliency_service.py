import numpy as np
import os
import logging

logger = logging.getLogger('audible_xai')

try:
    import innvestigate
    import innvestigate.utils
    from services.custom_attribution_methods_service import CustomAttributionMethodsService

except (ModuleNotFoundError, AttributeError):
    from services.custom_attribution_methods_service_tf2 import CustomAttributionMethodsService
    logger.warning("Skipped loading innvestigate modules, Only TFv2 supported saliency methods will be executed")
    pass

from services.dataset_service import DatasetService
from services.model_service import ModelService
from services.file_service import FileService
from utils.visualization_utils import plot_heatmap
from utils.model_utils import has_lambda_layer
from constants.saliency_constants import CUSTOM_SALIENCY_METHODS


class SaliencyService:
    def __init__(self, path_config, run_config):
        self.path_config = path_config
        self.saliency_path = path_config["results"]["saliency_path"]

        self.model = ModelService.instance.get_model()
        self.model_without_softmax = ModelService.instance.get_model_without_softmax()

        self.data = DatasetService.instance.get_dataset()
        self.dataset_params = DatasetService.instance.get_dataset_params()
        self.dataset_filenames = DatasetService.instance.get_dataset_filenames()
        self.class_name_to_index_mapper = DatasetService.instance.class_name_to_index_mapper()
        self.predicted_labels = DatasetService.instance.get_prediction_details()["predictions"]
        self.class_names = list(self.data.keys())

        self.file_service = FileService()
        self.run_config = run_config

    def run(self, saliency_methods):
        saliency_results = {}
        analyzer = None
        save_saliency_heatmaps = self.run_config["save_saliency_heatmaps"]
        allow_lambda_layers = has_lambda_layer(self.model_without_softmax)
        custom_attribution_methods_service = CustomAttributionMethodsService(self.path_config, saliency_methods)

        for saliency_method in saliency_methods:
            try:
                self.file_service.create_directory(self.saliency_path)
                # Create analyzer
                if saliency_method not in CUSTOM_SALIENCY_METHODS:
                    analyzer = innvestigate.create_analyzer(
                        saliency_method, self.model_without_softmax, neuron_selection_mode="index",
                        **(dict(allow_lambda_layers=True) if allow_lambda_layers else {}))
                    """ In the "index" as neuron selection mode, we can only provide one neuron index per batch. 
                    As we might want to calculate saliencies w.r.t diff neurons for each example in batch, 
                    we take batch size as 1"""
                    batch_size = 1
                else:
                    """For custom methods, we handle the above problem"""
                    batch_size = 32

                if save_saliency_heatmaps:
                    self.file_service.create_directory(self.saliency_path, saliency_method)
                    self.file_service.create_directory(self.saliency_path, saliency_method, self.class_names)

                for class_name in self.class_names:
                    saliency_results[class_name] = {}
                    class_path = os.path.join(self.saliency_path, saliency_method, str(class_name))
                    class_index = self.class_name_to_index_mapper[class_name]
                    class_pred_indices = self.predicted_labels[class_name]
                    logger.debug("Started saliency method {} for class name {} with index {}".
                                 format(saliency_method, class_name, class_index))

                    class_examples = self.data[class_name]
                    class_file_paths = self.dataset_filenames[class_name]

                    for example_index in range(0, len(class_file_paths), batch_size):
                        batch_examples = class_examples[example_index: example_index+batch_size]
                        batch_file_paths = class_file_paths[example_index: example_index+batch_size]
                        batch_pred_indices = class_pred_indices[example_index: example_index+batch_size]
                        input_examples = self.dataset_params.reshape_model_input(batch_examples)
                        if saliency_method in CUSTOM_SALIENCY_METHODS:
                            saliencies = custom_attribution_methods_service.run_attribution_method(
                                saliency_method, input_examples, batch_pred_indices)
                        else:
                            # Accepts only one neuron index per batch, so batch size should be 1
                            assert batch_size == 1, "Batch size should be 1 for saliency methods in innvestigate module"
                            saliencies = analyzer.analyze(input_examples, batch_pred_indices[0])

                        saliencies = self.dataset_params.reshape_saliencies(saliencies)
                        saliency_results[class_name].update(dict(zip(batch_file_paths, saliencies)))

                    if save_saliency_heatmaps:
                        for input_example, saliency, filepath in zip(self.data[class_name],
                                                                     saliencies, self.dataset_filenames[class_name]):
                            filename = self.file_service.basename(filepath)
                            save_path = os.path.join(class_path, filename + ".png")
                            plot_heatmap(input_example, saliency, saliency_method, save_path)
                np.save(os.path.join(self.saliency_path, saliency_method + ".npy"), saliency_results)
            except Exception as e:
                logger.warning("Skipping saliency computation as Exception Occurred", e)
