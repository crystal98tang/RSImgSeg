from utils.config import cfg
import tensorflow as tf
# If can not work, add "tensorflow.python."
# from tensorflow.python.keras.models import *
# from tensorflow.python.keras.layers import *
# from tensorflow.python.keras.optimizers import *
#
from keras.models import *
from keras.layers import *
from keras.optimizers import *
"""
    Image Size
"""
imageSize = cfg.global_image_size
"""
    Image Channels
"""
Channels = cfg.global_image_channels
"""
    label Classes
"""
Classes = cfg.global_label_classes
"""
    VGG Block Layer
"""
def vgg_like(filter):
    def _create_vgg_like_block(inputs):
        conv = Conv2D(filter, 3, padding='same')(inputs)
        conv = Conv2D(filter, 3, padding='same')(conv)
        conv = BatchNormalization()(conv, training=False)
        conv = Activation('relu')(conv)
        return conv
    return _create_vgg_like_block
"""
    ShortCutBlock Layer
"""
def shortcutblock(filter):
    def _create_shortcut_block(inputs):
        conv_main = Conv2D(filter, 1, padding='same')(inputs)
        conv_main = Conv2D(filter, 3, padding='same')(conv_main)
        conv_main = Conv2D(filter, 1, padding='same')(conv_main)
        conv_main = BatchNormalization()(conv_main, training=False)

        conv_fine = Conv2D(filter, 1, padding='same')(inputs)
        conv_fine = BatchNormalization()(conv_fine, training=False)

        merge = Add()([conv_main, conv_fine])

        conv = Activation('relu')(merge)

        return conv
    return _create_shortcut_block

"""
    ShortCutBlock fix Layer
"""
def shortcutblock_fix(filter):
    def _create_shortcut_block(inputs):
        conv_main = Conv2D(filter, 1, padding='same')(inputs)
        conv_main = Conv2D(filter, 3, padding='same')(conv_main)
        conv_main = Conv2D(filter * 4, 1, padding='same')(conv_main)
        conv_main = BatchNormalization()(conv_main, training=False)

        conv_fine = Conv2D(filter * 4, 1, padding='same')(inputs)
        conv_fine = BatchNormalization()(conv_fine, training=False)

        merge = Add()([conv_main, conv_fine])

        conv = Activation('relu')(merge)

        return conv
    return _create_shortcut_block
