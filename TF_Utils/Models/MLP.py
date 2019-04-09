"""
Created on 1/25/19
@author: MRChou

Scenario: simple multi-layer perceptron
"""

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


class MLP:
    """Used for creating MLP model_fn

    Args:
        hidden_units: a list of integers specifies the number of hidden units
        activations: a callable or list of callable specifies the activation functions
        n_classes: number of classes for the output
        batchnorm: turn on the batchnorm layer or not, note that we place bn after activation functions
        scope: name scope for model
    """

    def __init__(self, hidden_units, activations, n_classes, batchnorm=False, scope="MLP"):
        if not isinstance(hidden_units, list):
            raise TypeError("hidden_units must be list of integers")
        if isinstance(activations, list):
            activations = list(activations)
            assert len(hidden_units) == len(activations), "hidden_units and activations must be equal length"
        elif isinstance(activations, str):
            activations = [activations for _ in range(len(hidden_units))]
        else:
            raise TypeError("activations must be string or list of strings")

        # create dense layers
        with tf.variable_scope(scope, tf.float32):
            layercount = 0
            self.layers = []

            for unit, activation in zip(hidden_units, activations):
                self.layers.append(tf.layers.Dense(units=unit,
                                                   activation=activation,
                                                   name="layer-%s" % layercount))
                if batchnorm:
                    self.layers.append(tf.layers.BatchNormalization(momentum=_BATCH_NORM_DECAY,
                                                                    epsilon=_BATCH_NORM_EPSILON,
                                                                    name="layerBN-%s" % layercount))
                layercount += 1

            self.layers.append(tf.layers.Dense(units=n_classes,
                                               name="layer-output"))

    def __call__(self, inputs, training):
        for layer in self.layers:
            if isinstance(layer, tf.layers.BatchNormalization):
                inputs = layer(inputs, training=training)
            else:
                inputs = layer(inputs)
        return inputs


if __name__ == "__main__":
    pass
