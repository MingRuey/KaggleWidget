from MLBOX.Models.TF.Keras.UNet import _up_sample, _conv3_relu, UNetPadType
from MLBOX.Models.TF.Keras.UNet import unet_loss, dice_coef

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Cropping2D, Concatenate, Activation


class ResUnet:

    def __init__(self, inputs: tf.Tensor, n_class: int, load_pretrained: bool=False):
        model = keras.applications.ResNet50(
            include_top=False,
            weights="imagenet" if load_pretrained else None,
            input_tensor=inputs,
            pooling=None
        )
        self._res_model = model
        self._nClass = n_class

    @property
    def _res_blocks(self):
        self._conv1 = self._res_model.layers[4]
        self._conv2 = self._res_model.layers[38]
        self._conv3 = self._res_model.layers[80]
        self._conv4 = self._res_model.layers[142]
        self._conv5 = self._res_model.layers[174]
        blocks = [self._conv1, self._conv2, self._conv3, self._conv4, self._conv5]
        assert all(isinstance(layer, Activation) for layer in blocks)
        return blocks

    def forward(self) -> tf.Tensor:
        res_outputs = [layer.output for layer in self._res_blocks]

        up1 = _up_sample(
            inputs=res_outputs[-1], feat_to_concat=res_outputs[-2],
            n_filter=res_outputs[-2].shape[-1], pad_type=UNetPadType.zero
        )

        up2 = _up_sample(
            inputs=up1, feat_to_concat=res_outputs[-3],
            n_filter=res_outputs[-3].shape[-1], pad_type=UNetPadType.zero
        )

        up3 = _up_sample(
            inputs=up2, feat_to_concat=res_outputs[-4],
            n_filter=res_outputs[-4].shape[-1], pad_type=UNetPadType.zero
        )

        up4 = _up_sample(
            inputs=up3, feat_to_concat=res_outputs[-5],
            n_filter=res_outputs[-5].shape[-1], pad_type=UNetPadType.zero
        )

        up5 = UpSampling2D(size=(2, 2))(up4)
        conv5 = _conv3_relu(up5, n_filter=res_outputs[-5].shape[-1], padding=UNetPadType.zero.value)
        conv5 = _conv3_relu(conv5, n_filter=res_outputs[-5].shape[-1], padding=UNetPadType.zero.value)
        output = Conv2D(filters=self._nClass, kernel_size=1, kernel_initializer='he_uniform')(conv5)
        return output


def get_resunet(softmax_activate=False):
    inputs = keras.Input(shape=(256, 1600, 3), name="data")
    resunet = ResUnet(inputs=inputs, n_class=5, load_pretrained=True)
    outputs = resunet.forward()

    if softmax_activate:
        outputs = K.softmax(outputs, axis=-1)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_resunet_sigmoid(sigmoid_activate=False, n_class=5):
    inputs = keras.Input(shape=(256, 1600, 3), name="data")
    resunet = ResUnet(inputs=inputs, n_class=n_class, load_pretrained=True)
    outputs = resunet.forward()

    if sigmoid_activate:
        outputs = K.sigmoid(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == "__main__":
    get_resunet()
