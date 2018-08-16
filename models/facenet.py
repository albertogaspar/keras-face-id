import keras.backend as K
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Flatten, Dense, Lambda, Conv2D, MaxPooling2D, \
    BatchNormalization, Activation
from models.inception import Inception3x3, InceptionLayer


def facenet(input_shape):
    input = input_shape

    x = Conv2D(64, 7, strides=(2,2), padding='same')(input)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=3, strides=(2,2), padding='same')(x)

    x = Inception3x3(x, conv_1x1=64, conv_3x3=192)
    x = MaxPooling2D(pool_size = 3, strides = 2, padding='same')(x)


    inception_3a = InceptionLayer(x, conv_1x1=(64,(1,1)), conv3x3_reduce=(96,(1,1)), conv_3x3=(128,(1,1)),
                                  conv_5x5_reduce=(16,(1,1)), conv_5x5=(32,(1,1)), pool_proj=('max',32,1))
    inception_3b = InceptionLayer(inception_3a, conv_1x1=(64,(1,1)), conv3x3_reduce=(96,(1,1)), conv_3x3=(128,(1,1)),
                                  conv_5x5_reduce=(32,(1,1)), conv_5x5=(64,(1,1)), pool_proj=('l2',64,1))
    inception_3c = InceptionLayer(inception_3b, conv_1x1=None, conv3x3_reduce=(128,(1,1)), conv_3x3=(256,(2,2)),
                                  conv_5x5_reduce=(32,(1,1)), conv_5x5=(64,(2,2)), pool_proj=('max',None,2))

    inception_4a = InceptionLayer(inception_3c, conv_1x1=(256,(1,1)), conv3x3_reduce=(96,(1,1)), conv_3x3=(192,(1,1)),
                                  conv_5x5_reduce=(32,(1,1)), conv_5x5=(64,(1,1)), pool_proj=('l2',128,1))
    inception_4b = InceptionLayer(inception_4a, conv_1x1=(224,(1,1)), conv3x3_reduce=(112,(1,1)), conv_3x3=(224,(1,1)),
                                  conv_5x5_reduce=(32,(1,1)), conv_5x5=(64,(1,1)), pool_proj=('l2',128,1))
    inception_4c = InceptionLayer(inception_4b, conv_1x1=(192,(1,1)), conv3x3_reduce=(128,(1,1)), conv_3x3=(256,(1,1)),
                                  conv_5x5_reduce=(32,(1,1)), conv_5x5=(64,(1,1)), pool_proj=('l2',128,1))
    inception_4d = InceptionLayer(inception_4c, conv_1x1=(160,(1,1)), conv3x3_reduce=(144,(1,1)), conv_3x3=(288,(1,1)),
                                  conv_5x5_reduce=(32,(1,1)), conv_5x5=(64,(1,1)), pool_proj=('l2',128,1))
    inception_4e = InceptionLayer(inception_4d, conv_1x1=None, conv3x3_reduce=(160,(1,1)), conv_3x3=(256,(2,2)),
                                  conv_5x5_reduce=(64,(1,1)), conv_5x5=(128,(2,2)), pool_proj=('max',None,2))

    inception_5a = InceptionLayer(inception_4e, conv_1x1=(384,(1,1)), conv3x3_reduce=(192,(1,1)), conv_3x3=(384,(1,1)),
                                  conv_5x5_reduce=(48,(1,1)), conv_5x5=(128,(1,1)), pool_proj=('l2',128,1))
    inception_5b = InceptionLayer(inception_5a, conv_1x1=(384,(1,1)), conv3x3_reduce=(192,(1,1)), conv_3x3=(384,(1,1)),
                                  conv_5x5_reduce=(48,(1,1)), conv_5x5=(128,(1,1)), pool_proj=('max',128,1))

    x = GlobalAveragePooling2D()(inception_5b)
    # x = Flatten()(x)
    x = Dense(128)(x)
    x = Lambda(lambda emb: K.l2_normalize(emb, axis=1))(x)

    return x