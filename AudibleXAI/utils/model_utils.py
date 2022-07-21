import keras
import tensorflow as tf
import logging
logger = logging.getLogger('audible_xai')


def has_lambda_layer(model):
    for layer in model.layers:
        if isinstance(layer, keras.layers.Lambda):
            return True


def pre_softmax_tensors(output_layers, should_find_softmax: bool = True):
    """Finds the tensors that were preceeding a potential softmax."""
    softmax_sigmoid_found = False

    ret = []
    for output_layer in output_layers:
        layer, node_index, _tensor_index = output_layer._keras_history
        if layer.activation.__name__ == "softmax" or layer.activation.__name__ == "sigmoid":
            softmax_sigmoid_found = True
            if isinstance(layer, keras.layers.Activation):
                ret.append(layer.get_input_at(node_index))
            else:
                layer.activation = keras.activations.linear
                ret.append(layer(layer.get_input_at(node_index)))

    if should_find_softmax and not softmax_sigmoid_found:
        logger.error("No softmax or sigmoid layer found.")
        raise Exception("No softmax or sigmoid layer found.")

    return ret


def model_wo_softmax(model):
    """Creates a new model w/o the final softmax activation."""
    cloned_model = model.__class__.from_config(model.get_config(), custom_objects={"tf_maximum": tf.math.maximum})

    # As Keras only copies architecture of the model, not the weights. Need to set_weights as follows.
    cloned_model.set_weights(model.get_weights())

    return keras.models.Model(
        inputs=cloned_model.inputs, outputs=pre_softmax_tensors(cloned_model.outputs), name=model.name
    )
