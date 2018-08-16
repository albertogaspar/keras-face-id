import cv2
import os
from config import IMG_SIZE
import time
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from data_loader import read_bmp_img
import numpy as np
from faceId import get_embeddings, get_model
import pickle

parser = ArgumentParser()
parser.add_argument('--register', action='store_true', help='Save images from webcam')
parser.add_argument('--dest_dir', type=str, default='./data_test/true/')
parser.add_argument('--n_images', type=int, default=50, help='Number of images to capture')
parser.add_argument('--no_embeddings', action='store_true', help='Th embeddings are not precomputed')
args = parser.parse_args()

PADDING = 40

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def capture_img(dest_dir, n_images):
    idx = 0
    start = time.time()
    cam = cv2.VideoCapture(0)
    while True:
        img = cam.read()[1]
        cv2.imshow("Window",img)
        # first image has too much light
        if idx == 0:
            WAIT = 3.
        else:
            WAIT = 1.

        if (time.time() - start) > WAIT:
            cv2.imwrite(os.path.join(dest_dir, 'img_{0}.png'.format(idx)), img)
            start = time.time()
            idx += 1
        if idx == n_images:
            break
        if cv2.waitKey(5) == 32:
            break

def align_face(filename, plot=True, resize=True):
    face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    img = cv2.imread(filename)
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        x, y, w, h = faces[0]
        tmp = max(h, w)
        h = tmp
        w = tmp

        img_cropped = img[y - PADDING:y + h + PADDING, x - PADDING:x + w + PADDING, :]
        if resize:
            img_cropped = cv2.resize(img_cropped, dsize=(IMG_SIZE, IMG_SIZE))
        if plot:
            plt.imshow(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
            plt.show()
            plt.close()
        cv2.imwrite(filename, img_cropped)
        return True
    except Exception as e:
        print('Unable to process {0} '.format(filename))
        os.remove(filename)
        return False


if __name__ == '__main__':
    if not os.path.exists(args.dest_dir):
        os.mkdir(args.dest_dir)

    if True:
        capture_img(args.dest_dir, args.n_images)

    # face alignment
    src_files = list(filter(lambda x: 'crop' not in x, os.listdir(args.dest_dir)))
    model = get_model()
    embeddings = []
    for filename in src_files:
        filename = os.path.join(args.dest_dir, filename)

        ret = align_face(filename)
        if not args.no_embeddings and ret:
            embeddings.append((filename, get_embeddings(model, read_bmp_img(filename))))

    if len(embeddings) > 0:
        save_obj(dict(embeddings), 'embeddings')

