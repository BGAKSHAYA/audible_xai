import numpy as np
from services.dataset_service import DatasetService
from services.model_service import ModelService
from services.file_service import FileService
import matplotlib.cm as cm
from lime import lime_image
import time
import skimage.filters
import skimage.segmentation
from skimage.segmentation import felzenszwalb
import gc
import tensorflow as tf
import keras
import logging
import time

logger = logging.getLogger('audible_xai')


def felzenszwalb(image):
    """
    Best hyperparameters obtained after grid-search for
        min_size from 10 to 100 with step size 10 and 
        scale from 1 to 10 with step size 1 
    """
    min_size = 50 
    scale = 3
    return skimage.segmentation.felzenszwalb(image[:, :, -1], scale=scale, min_size=min_size)


class CustomAttributionMethodsService:
    def __init__(self, config, saliency_methods):
        self.saliency_path = config["results"]["saliency_path"]

        self.model = ModelService.instance.get_model()
        self.model_without_softmax = ModelService.instance.get_model_without_softmax()

        self.data = DatasetService.instance.get_dataset()
        self.dataset_params = DatasetService.instance.get_dataset_params()

        self.file_service = FileService()
        self.class_names = list(self.data.keys())
        self.lime_explainer = lime_image.LimeImageExplainer(verbose=False)

        self.CUSTOM_ATTRIBUTION_METHODS = dict(
            GRAD_CAM=self.grad_cam,
            OCCLUSION=self.occlusion,
            INTEGRATED_GRADIENTS=self.integrated_gradients,
            LIME=self.lime
        )
        self.dataset_shape = self.dataset_params.preprocessed_input_shape[-3:]
        if "OCCLUSION" in saliency_methods:
            self.occluded_masks, self.occluded_center_pixels_masks = self.initialize_occlusion_masks()

    def run_attribution_method(self, method, *args):
        result = self.CUSTOM_ATTRIBUTION_METHODS[method](*args)
        return result

    def initialize_occlusion_masks(self):
        # Choose a patch size
        patch_size = 5
        patch_indices = []
        # Occlude patch by patch and calculate activation for each patch
        for i in range(0, self.dataset_shape[0], patch_size):
            for j in range(0, self.dataset_shape[1], patch_size):
                patch_indices.append((i, j))

        occluded_masks = np.ones((len(patch_indices), *tuple(self.dataset_shape)))
        occluded_center_pixels_masks = np.zeros((len(patch_indices), *tuple(self.dataset_shape)))

        for index, (i, j) in enumerate(patch_indices):
            occluded_masks[index][i:i + patch_size, j:j + patch_size, :] = 0
            x_center_pixel = min(self.dataset_shape[0] - 1, i + (patch_size // 2))
            y_center_pixel = min(self.dataset_shape[1] - 1, j + (patch_size // 2))
            occluded_center_pixels_masks[index][x_center_pixel, y_center_pixel, :] = 1
            occluded_center_pixels_masks[index][i:i + patch_size, j:j + patch_size, :] = 1

        return occluded_masks, occluded_center_pixels_masks

    def grad_cam(self, input_examples, pred_indices=None):
        last_conv_layer_index = -1
        for index, layer in enumerate(self.model_without_softmax.layers):
            if isinstance(layer, keras.layers.Conv2D):
                last_conv_layer_index = index

        if last_conv_layer_index == -1:
            logger.warning("No Convolutional layers found, Skipping GradCAM computation")
            return

        grad_model = keras.models.Model(
            self.model_without_softmax.inputs,
            [self.model_without_softmax.layers[last_conv_layer_index].output, self.model_without_softmax.output])

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(input_examples)
            if pred_indices is None:
                pred_indices = tf.argmax(preds, axis=1)
            class_channel = [row[pred_index] for row, pred_index in zip(preds, pred_indices)]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        heatmaps = tf.einsum('ijkl,il->ijk', last_conv_layer_output, pooled_grads)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmaps = tf.maximum(heatmaps, 0) / tf.math.reduce_max(heatmaps)
        channels = input_examples.shape[-1]
        # Use jet colormap to colorize heatmap
        if channels == 1:
            jet = cm.get_cmap("gray")
        else:
            jet = cm.get_cmap("jet")

        # Use Gray/ RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :channels]
        heatmaps = np.uint8(255 * heatmaps)
        jet_heatmaps = jet_colors[heatmaps]

        saliencies = np.zeros(input_examples.shape)
        shape = input_examples.shape
        for index, jet_heatmap in enumerate(jet_heatmaps):
            jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
            jet_heatmap = jet_heatmap.resize((shape[2], shape[1]))
            jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
            jet_heatmap = jet_heatmap / 255.0
            saliencies[index] = jet_heatmap
        return saliencies

    def lime(self, input_examples, pred_indices=None):
        saliencies = np.zeros(input_examples.shape)
        start_time = time.time()
        for index, input_example in enumerate(input_examples):
            explanation = self.lime_explainer.explain_instance(image=input_example,
                                                               classifier_fn=self.model_without_softmax,
                                                               top_labels=10,
                                                               hide_color=0, num_samples=1000,
                                                               segmentation_fn=felzenszwalb)
            pred_index = explanation.top_labels[0] if pred_indices is None else pred_indices[index]
            # Map each explanation weight to the corresponding superpixel
            dict_heatmap = dict(explanation.local_exp[pred_index])
            heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
            saliency = heatmap[:, :, np.newaxis]
            saliencies[index] = saliency
        logger.debug("Batch Complete", time.time()-start_time)
        return saliencies

    def occlusion(self, input_examples, pred_indices=None):
        saliencies = np.zeros(input_examples.shape)
        model_predictions = self.model_without_softmax.predict(input_examples)

        if pred_indices is None:
            pred_indices = tf.argmax(model_predictions, axis=1)
        start_time = time.time()
        initial_activation_values = [row[index] for row, index in zip(model_predictions, pred_indices)]
        for example_index, (input_example, pred_index) in enumerate(zip(input_examples, pred_indices)):
            occluded_images = self.occluded_masks * input_example
            occlusion_predictions = self.model_without_softmax.predict(occluded_images, batch_size=32)
            occlusion_predictions = [row[pred_index] for row in occlusion_predictions]

            initial_example_activation = np.repeat(initial_activation_values[example_index], len(occlusion_predictions))
            saliency = np.einsum("ijkl, i-> ijkl", self.occluded_center_pixels_masks,
                                 initial_example_activation - occlusion_predictions)
            saliency = np.sum(saliency, axis=0)
            saliencies[example_index] = saliency
        logger.debug("Batch Complete", time.time()-start_time)
        del occlusion_predictions
        del occluded_images
        del input_examples
        gc.collect()  # Force garbage collection to avoid running to OOM errors
        return saliencies

    def integrated_gradients(self, input_examples, pred_indices=None):
        # Nr. of steps along paths
        steps = 100
        # Take as reference a black image,
        # i.e., lowest number networks input value range.
        x_ref = np.zeros_like(input_examples)
        # Sum gradients along the path from x to x_ref
        gradient_sum = np.zeros_like(input_examples)
        input_examples = tf.Variable(input_examples)
        for step in range(steps):
            x_step = x_ref + (input_examples - x_ref) * step / steps
            with tf.GradientTape() as tape:
                tape.watch(x_step)
                preds = self.model_without_softmax(x_step)
                if pred_indices is None:
                    class_channel = tf.reduce_max(preds, axis=1)
                else:
                    class_channel = [row[pred_index] for row, pred_index in zip(preds, pred_indices)]
            gradient_sum += tape.gradient(class_channel, x_step).numpy()
        # Integrated Gradients formula
        analysis = gradient_sum * (input_examples - x_ref)
        analysis = analysis.numpy()
        return analysis
