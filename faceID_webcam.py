import numpy as np
import os
from data_loader import read_bmp_img
from faceId import get_model, get_embeddings, get_distance
from argparse import ArgumentParser
from config import IMG_SIZE, N_CHANNELS
import cv2
import time
from matplotlib import pyplot as plt
from copy import deepcopy
import pickle

PADDING = 40

parser = ArgumentParser()
parser.add_argument('--true_dir', type=str, default='./data_test/true')
parser.add_argument('--test_dir', type=str, default='./data_test/test')
args = parser.parse_args()

def load_obj(name):
    """
    Load dict from disk
    :param name: str
        name of the file
    :return: dict
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def capture_and_align(filename, resize=True, plot=True):
    """
    Take one image from a webcom and extract the face.
    It saves it in filename.
    :param filename: str
        filepath e.g. ./data-test/test/img.png
    :param resize: bool
        if true, resize img to 224x224
    :param plot: bool
        if true, plot the captured img.
    :return:
    """
    face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    start = time.time()
    cam = cv2.VideoCapture(0)
    while True:
        img = cam.read()[1]
        img_copy = deepcopy(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Save face
        if (time.time() - start) > 5. and len(faces) > 0:
            x, y, w, h = faces[0]
            tmp = max(h, w)
            h = tmp
            w = tmp
            img_cropped = img_copy[max(0, y - PADDING):y + h + PADDING,
                                   max(0, x - PADDING):x + w + PADDING,
                                   :]
            if resize:
                img_cropped = cv2.resize(img_cropped, dsize=(IMG_SIZE, IMG_SIZE))
            if plot:
                plt.imshow(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
                plt.show()
                plt.close()
            cv2.imwrite(filename, img_cropped)
            break
    # When everything is done, release the capture
    cam.release()
    cv2.destroyAllWindows()

def capture_identify(model, precomputed_emb_truth):
    """
    Live webcam Face recognition
    :param model: keras model
    :param precomputed_emb_truth: np.array()
        (n, dim_embeddings) where n is the number of images in the data_test/true dir.
    """
    face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)
    while True:
        img = cam.read()[1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # identify face
        if len(faces) > 0:
            # for each face in image
            for (x, y, w, h) in faces:
                tmp = max(h, w)
                h = tmp
                w = tmp
                img_cropped = img[max(0, y - PADDING):y + h + PADDING,
                                       max(0, x - PADDING):x + w + PADDING,
                                       :]
                img_cropped = cv2.resize(img_cropped, dsize=(IMG_SIZE, IMG_SIZE))
                img_cropped = (img_cropped - np.mean(img_cropped)) / np.std(img_cropped)
                precomputed_emb_test = get_embeddings(model, img_cropped)
                scores=[]
                for file in os.listdir(args.true_dir):
                    true_path = os.path.join(args.true_dir, file)
                    true_img = read_bmp_img(true_path)
                    if true_img.shape == (IMG_SIZE, IMG_SIZE, N_CHANNELS):
                        score = get_distance(None, None, model=None,
                                             precomputed_emb_an=precomputed_emb_truth[true_path],
                                             precomputed_emb_test=precomputed_emb_test)
                        scores.append(score)
                mean_score = np.mean(scores)
                print(mean_score)
                # Draw a rectangle around the faces
                if mean_score >= 5.0:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                else:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Display the resulting frame
        cv2.imshow('Video', img)
    # When everything is done, release the capture
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if not os.path.exists(args.test_dir):
        os.mkdir(args.test_dir)

    model = get_model()
    precomputed_emb_truth = load_obj('embeddings')
    capture_identify(model, precomputed_emb_truth)

