""" Implementations adapted from https://github.com/albermax/interpretable_ai_book__sw_chapter/tree/master/notebooks"""

import numpy as np
import matplotlib.pyplot as plt
from innvestigate_utils import utils_imagenet
from scipy.ndimage.filters import gaussian_filter
from services.dataset_service import DatasetService
from services.model_service import ModelService
from services.file_service import FileService
import matplotlib.cm as cm
import skimage.filters
import skimage.segmentation
import sklearn.linear_model
import sklearn.metrics
from skimage.color import gray2rgb

import tensorflow as tf
import keras
import logging

session = keras.backend.get_session()
logger = logging.getLogger('audible_xai')


class CustomAttributionMethodsService:
    def __init__(self, config, saliency_methods):
        self.saliency_path = config["results"]["saliency_path"]

        self.model = ModelService.instance.get_model()
        self.model_without_softmax = ModelService.instance.get_model_without_softmax()

        self.data = DatasetService.instance.get_dataset()
        self.file_service = FileService()
        self.class_names = list(self.data.keys())

        self.CUSTOM_ATTRIBUTION_METHODS = dict(
            INPUT_TIMES_GRADIENT=self.input_times_gradient,
            INTEGRATED_GRADIENTS=self.integrated_gradients,
            GRAD_CAM=self.grad_cam,
            OCCLUSION=self.occlusion,
            LIME=self.lime
        )

    def run_attribution_method(self, method, *args):
        result = self.CUSTOM_ATTRIBUTION_METHODS[method](*args)
        return result

    # Input * Gradient
    def input_times_gradient(self, x, pred_index):
        input, output = self.model_without_softmax.inputs[0], self.model_without_softmax.outputs[0]
        # Reduce output to response of neuron with largest activation
        max_output = tf.reduce_max(output, axis=1)
        # Take gradient of output neuron w.r.t. to input
        gradient = tf.gradients(max_output, input)[0]
        # and multiply it with the input
        input_t_gradient = input * gradient
        # Run the code with TF
        a = session.run(input_t_gradient, {input: x})
        # a = gaussian_filter(a, sigma=1)
        a = (a - a.min()) / (a.max() - a.min())
        return a

    def integrated_gradients(self, x, pred_index):
        input, output = self.model_without_softmax.inputs[0], self.model_without_softmax.outputs[0]
        # Reduce output to response of neuron with largest activation
        max_output = tf.reduce_max(output, axis=1)

        # Nr. of steps along path
        steps = 100
        # Take as reference a black image,
        # i.e., lowest number networks input value range.
        x_ref = (np.ones_like(x) * 0)[0]
        # Take gradient of output neuron w.r.t. to input
        gradient = tf.gradients(max_output, input)[0]

        # Sum gradients along the path from x to x_ref
        gradient_sum = np.zeros_like(x)
        for step in range(steps):
            # Create intermediate input
            x_step = x_ref + (x - x_ref) * step / steps
            # Compute and add the gradient for intermediate input
            gradient_sum += session.run(gradient, {input: x_step})

        # Integrated Gradients formula
        A2 = gradient_sum * (x - x_ref)

        # Postprocess
        a = A2
        # a = gaussian_filter(a, sigma=1)
        a = (a - a.min()) / (a.max() - a.min())
        return a

    def grad_cam(self, x, pred_index):
        last_conv_layer_index = -1
        for index, layer in enumerate(self.model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                last_conv_layer_index = index

        if last_conv_layer_index == -1:
            logger.warning("No Convolutional layers found, Skipping GradCAM computation")
            return

        input, conv_output, output = self.model.inputs[0], self.model.layers[last_conv_layer_index], \
                                     self.model.outputs[0]

        gradient = tf.gradients(output[pred_index], conv_output.output)
        grads = session.run(gradient, {input: x})[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = session.run((conv_output.output, input), {input: x})[0]
        heatmap = tf.einsum('mij,jk->mik', tf.constant(last_conv_layer_output[0]), pooled_grads[..., tf.newaxis])
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.eval(session=session)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("gray")
        # Use Gray values of the colormap
        jet_colors = jet(np.arange(256))[:, :1]
        heatmap = np.uint8(255 * heatmap)
        jet_heatmap = jet_colors[heatmap]

        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((x.shape[2], x.shape[1]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
        jet_heatmap = jet_heatmap.reshape(1, *tuple(jet_heatmap.shape))
        jet_heatmap = jet_heatmap / 255.0
        return jet_heatmap

    def occlusion(self, x, pred_index):
        diff = np.zeros_like(x)
        # Choose a patch size
        patch_size = 50
        input, output = self.model_without_softmax.inputs[0], self.model_without_softmax.outputs[0]
        # Reduce output to response of neuron with largest activation
        max_output = tf.reduce_max(output, axis=1)
        # Occlude patch by patch and calculate activation for each patch
        for i in range(0, self.dataset_shape[0], patch_size):
            for j in range(0, self.dataset_shape[1], patch_size):
                # Create image with the patch occluded
                occluded_x = x.copy()
                occluded_x[:, i:i + patch_size, j:j + patch_size, :] = 0

                # Store activation of occluded image
                diff[:, i:i + patch_size, j:j + patch_size, :] = session.run(max_output, {input: occluded_x})[0]

        # Normalize with initial activation value
        A3 = session.run(max_output, {input: x})[0] - diff

        # Postprocess
        a = A3
        b = (a - a.min()) / (a.max() - a.min())
        # Displaying the explanation
        plt.imshow(utils_imagenet.heatmap(b)[0])
        plt.show()
        return b

    def lime(self, x, pred_index):
        # Segment (not pre-processed) image
        rgb_x = x
        if x.shape[-1] == 1:  # Convert gray scale image to RGB
            rgb_x = gray2rgb(x[:, :, :, -1]).astype("float64")
        segments = skimage.segmentation.felzenszwalb(rgb_x[0], scale=1, min_size=20)
        nr_segments = np.max(segments) + 1

        input, output = self.model_without_softmax.inputs[0], self.model_without_softmax.outputs[0]
        # Reduce output to response of neuron with largest activation
        max_output = tf.reduce_max(output, axis=1)

        # Create dataset
        nr_samples = 100
        # Randomly switch segments on and off
        features = np.random.randint(0, 2, size=(nr_samples, nr_segments))
        features[0, :] = 1

        # Get labels for features
        labels = []
        for sample in features:
            tmp = x.copy()
            # Switch segments on and off
            for segment_id, segment_on in enumerate(sample):
                if segment_on == 0:
                    tmp[0][segments == segment_id] = 0
            # Get predicted value for this sample
            labels.append(session.run(max_output, {input: tmp})[0])

        # Compute sample weights
        distances = sklearn.metrics.pairwise_distances(
            features,
            features[0].reshape(1, -1),
            metric='cosine',
        ).ravel()
        kernel_width = 0.25
        sample_weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))

        # Fit L1-regressor
        regressor = sklearn.linear_model.Ridge(alpha=1, fit_intercept=True)
        regressor.fit(features, labels, sample_weight=sample_weights)
        weights = regressor.coef_

        # Map weights onto segments
        A4 = np.zeros_like(x)
        for segment_id, w in enumerate(weights):
            A4[0][segments == segment_id] = w

        # Postprocess
        a = A4
        a = (a - a.min()) / (a.max() - a.min())
        plt.imshow(utils_imagenet.heatmap(a)[0])
        plt.show()
        return a



