from keras import backend as K
from keras.layers import Input, Lambda, merge, Layer
from keras.models import Model, load_model
from keras.optimizers import Adam
from models.facenet import facenet
from data_loader import *
from config import IMG_SIZE, N_CHANNELS

def euclidean_distance(inputs):
    assert len(inputs) == 2, \
        'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    u, v = inputs
    return K.sqrt(K.sum((K.square(u - v)), axis=1, keepdims=True))

def emb_distance(inputs):
    anchor, sample = inputs
    diff = K.square(K.sqrt(anchor[..., :] - sample[..., :]))
    return diff

def t_loss(inputs):
    diff_pos, diff_neg = inputs
    loss = K.mean(diff_pos[..., :] - diff_neg[..., :] + 0.2)
    return loss

def indenty_loss(y_true, y_pred):
    return K.mean(y_pred)

def contrastive_loss(y_true, y_pred):
    margin = 1.
    return K.mean((1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0.)))

def main_triplet_loss():
    """
    Train network with triplet loss
    """
    input_shape = Input(shape=(IMG_SIZE, IMG_SIZE, N_CHANNELS))
    embeddings = facenet(input_shape)
    facenet_model = Model(inputs=input_shape, outputs=embeddings)
    facenet_model.summary()

    anchor = Input(shape=(IMG_SIZE, IMG_SIZE, N_CHANNELS))
    positive = Input(shape=(IMG_SIZE, IMG_SIZE, N_CHANNELS))
    negative = Input(shape=(IMG_SIZE,IMG_SIZE,N_CHANNELS))

    emb_anchor = facenet_model(anchor)
    emb_positive = facenet_model(positive)
    emb_negative = facenet_model(negative)

    anchor_positive_distance = Lambda(emb_distance)([emb_anchor, emb_positive])
    anchor_negative_distance = Lambda(emb_distance)([emb_anchor, emb_negative])
    merge_layer = Lambda(t_loss)([anchor_positive_distance, anchor_negative_distance])

    model = Model(inputs=[anchor, positive, negative], outputs=merge_layer)
    model.summary()

    adam = Adam(lr=0.00005)
    model.compile(optimizer=adam, loss=indenty_loss)

    generator = get_triple(8)
    outputs = model.fit_generator(generator, steps_per_epoch=30, epochs=50)

    facenet_model.save("faceid_triplet_loss.h5")

def main_contrastive_loss():
    """
    Train network with contrastive loss.
    """
    input_shape = Input(shape=(IMG_SIZE, IMG_SIZE, N_CHANNELS))
    embeddings = facenet(input_shape)
    facenet_model = Model(inputs=input_shape, outputs=embeddings)
    facenet_model.summary()

    im1 = Input(shape=(IMG_SIZE, IMG_SIZE, N_CHANNELS))
    im2 = Input(shape=(IMG_SIZE, IMG_SIZE, N_CHANNELS))

    emb_im1 = facenet_model(im1)
    emb_im2 = facenet_model(im2)

    lambda_merge = Lambda(euclidean_distance)([emb_im1, emb_im2])

    model = Model(inputs=[im1, im2], outputs=lambda_merge)
    model.summary()

    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss=contrastive_loss)

    generator = get_batch(8)
    outputs = model.fit_generator(generator, steps_per_epoch=20, epochs=30)

    facenet_model.save("faceid_contrastive_loss.h5")

def get_model(mode = 'contrastive'):
    """
    Load pretrained model
    :param mode: str
        contrastive or triplet
    """
    if mode == 'contrastive':
        model = load_model("faceid_contrastive_loss.h5")
    elif mode == 'triplet':
        model = load_model("faceid_triplet_loss.h5")
    else:
        raise NotImplementedError("{0} is not a valid mode for inference.".format(mode))
    return model

def get_embeddings(model, img):
    """
    Get embeddings for an image
    :param model: keras model
    :param img: np.array
    :return:
    """
    img = img.reshape([1, IMG_SIZE, IMG_SIZE, N_CHANNELS])
    return model.predict(img)[0]

def get_distance(anchor, test, model=None, precomputed_emb_an=None, precomputed_emb_test=None):
    """
    Get euclidean distance between two embedded faces.
    :param anchor: np.array
        [224,224,3] image
    :param test: np.array
        [224,224,3] image
    :param model:
        keras model
    :param precomputed_emb_an: np.array
        [128,] precomputed embeddings for anchor image
    :param precomputed_emb_test: np.array
        [128,] precomputed embeddings for test image
    :return:
    """
    if precomputed_emb_an is not None:
        emb_an = precomputed_emb_an
    else:
        emb_an = get_embeddings(model, anchor)
    if precomputed_emb_test is not None:
        emb_test = precomputed_emb_test
    else:
        emb_test = get_embeddings(model, test)
    distance = np.sum(np.sqrt((emb_an - emb_test) ** 2))
    return distance

def simple_test():
    """
    Simple test using an hold out set of faces
    """
    model = get_model()
    for i in range(10):
        an, pos = get_validation_pos()
        distance = get_distance(an, pos, model)
        print('Distance between same class: {0}'.format(distance))

        an, neg = get_validation_neg()
        distance = get_distance(an, neg, model)
        print ('Distance between different class: {0}'.format(distance))

if __name__ == '__main__':
    main_contrastive_loss()
    simple_test(mode='contrastive')
    # main_triplet_loss()
    # simple_test(mode='triplet')


