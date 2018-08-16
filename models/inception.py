from keras.layers import Input, GlobalAveragePooling2D, AveragePooling2D, Flatten, ZeroPadding2D, Dense, Concatenate, Merge, Lambda, Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.models import Model, Sequential
from keras.backend import concatenate

def Inception1x1(input, conv_1x1=64, strides_1x1=(1,1)):
    x = Conv2D(conv_1x1, 1, strides=strides_1x1, padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def Inception3x3(input, conv_1x1=96, conv_3x3=128, strides_1x1 =(1,1), strides_3x3 =(1,1)):
    x = Inception1x1(input, conv_1x1, strides_1x1=strides_1x1)
    x = Conv2D(conv_3x3, 3, strides=strides_3x3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def Inception5x5(input, conv_1x1=16, conv_5x5=32, strides_1x1 =(1,1), strides_5x5 =(1,1)):
    x = Inception1x1(input, conv_1x1, strides_1x1=strides_1x1)
    x = Conv2D(conv_5x5, 5, strides=strides_5x5, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def InceptionPooling(input, conv_1x1=32, strides=(1,1), pool_type='max'):
    if pool_type == 'max':
        x = MaxPooling2D(pool_size=3, strides=strides, padding='same')(input)
    elif pool_type == 'l2':
        x = AveragePooling2D(pool_size=3, strides=strides, padding='same')(input)
    else:
        raise NotImplementedError('pool_type = {0}. '
                                  'This type of pooling is not available.'.format(pool_type))
    if conv_1x1:
        x = Inception1x1(x, conv_1x1=conv_1x1, strides_1x1=strides)
    return x

def InceptionLayer(input, conv_1x1, conv3x3_reduce, conv_3x3, conv_5x5_reduce, conv_5x5, pool_proj):
    to_concatenate = []
    if conv_1x1:
        inception_1x1 = Inception1x1(input, conv_1x1=conv_1x1[0], strides_1x1= conv_1x1[1])
        to_concatenate.append(inception_1x1)
    if conv_3x3:
        inception_3x3 = Inception3x3(input, conv_1x1=conv3x3_reduce[0], conv_3x3=conv_3x3[0],
                                     strides_1x1 =conv3x3_reduce[1], strides_3x3 =conv_3x3[1])
        to_concatenate.append(inception_3x3)
    if conv_5x5:
        inception_5x5 = Inception5x5(input, conv_1x1=conv_5x5_reduce[0], conv_5x5=conv_5x5[0],
                                     strides_1x1 =conv_5x5_reduce[1], strides_5x5 =conv_5x5[1])
        to_concatenate.append(inception_5x5)
    if pool_proj:
        inception_pool = InceptionPooling(input, conv_1x1=pool_proj[1], strides=pool_proj[2], pool_type=pool_proj[0])
        to_concatenate.append(inception_pool)
    inception = Concatenate()(to_concatenate)
    return inception