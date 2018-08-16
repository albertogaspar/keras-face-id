import numpy as np
import cv2
import os
from config import DATA_PATH

IMG_HEIGHT = 480
IMG_WIDTH = 640

val_path = './data/validation'

def read_bmp_img(filepath, normalize=True):
    rgb_img = cv2.imread(filepath)

    if normalize:
        rgb_img = (rgb_img - np.mean(rgb_img)) / np.std(rgb_img)

    return rgb_img


def get_batch(batch_size):
    paths =  list(filter(lambda x: x != val_path, [os.path.join(DATA_PATH, d) for d in os.listdir(DATA_PATH)]))
    while True:
        X = []
        y = []
        switch = True
        for _ in range(batch_size):
            idx = np.random.randint(0,len(paths))
            files = os.listdir(paths[idx])
            anchor_path = os.path.join(paths[idx], files[np.random.randint(0,len(files))][:-5])
            if switch:
                positive_path = os.path.join(paths[idx], files[np.random.randint(0, len(files))][:-5])
                while anchor_path == positive_path:
                    positive_path = os.path.join(paths[idx], files[np.random.randint(0, len(files))][:-5])
                anchor = read_bmp_img(anchor_path + 'c.bmp')
                positive = read_bmp_img(positive_path + 'c.bmp')
                X.append(np.array([anchor,positive]))
                y.append(np.array([0.]))
            else:
                idx_neg = np.random.randint(0, len(paths))
                while idx_neg == idx:
                    idx_neg = np.random.randint(0, len(paths))
                files_neg = os.listdir(paths[idx_neg])
                negative_path = os.path.join(paths[idx_neg], files_neg[np.random.randint(0, len(files))][:-5])
                anchor = read_bmp_img(anchor_path + 'c.bmp')
                negative = read_bmp_img(negative_path + 'c.bmp')
                X.append(np.array([anchor,negative]))
                y.append(np.array([1.]))
            switch = not switch
        X = np.asarray(X)
        y = np.asarray(y)
        yield [X[:, 0], X[:, 1]], y

def get_triple(batch_size):
    paths =  list(filter(lambda x: x != val_path, [os.path.join(DATA_PATH, d) for d in os.listdir(DATA_PATH)]))
    while True:
        X = []
        for _ in range(batch_size):
            idx = np.random.randint(0,len(paths))
            files = os.listdir(paths[idx])
            anchor_path = os.path.join(paths[idx], files[np.random.randint(0,len(files))][:-5])
            positive_path = os.path.join(paths[idx], files[np.random.randint(0, len(files))][:-5])
            while anchor_path == positive_path:
                positive_path = os.path.join(paths[idx], files[np.random.randint(0, len(files))][:-5])
            idx_neg = np.random.randint(0, len(paths))
            while idx_neg == idx:
                idx_neg = np.random.randint(0, len(paths))
            files_neg = os.listdir(paths[idx_neg])
            negative_path = os.path.join(paths[idx_neg], files_neg[np.random.randint(0, len(files))][:-5])

        anchor = read_bmp_img(anchor_path + 'c.bmp')
        positive = read_bmp_img(positive_path + 'c.bmp')
        negative = read_bmp_img(negative_path + 'c.bmp')
            X.append(np.array([anchor, positive, negative]))
        X = np.asarray(X)
        y = np.arange(batch_size).reshape((batch_size,1))
        yield [X[:,0], X[:,1], X[:,2]], y

def get_validation_pos():
    files = os.listdir(val_path)
    anchor_path = os.path.join(val_path, files[np.random.randint(0, len(files))][:-5])
    positive_path = os.path.join(val_path, files[np.random.randint(0, len(files))][:-5])
    while anchor_path == positive_path:
        positive_path = os.path.join(val_path, files[np.random.randint(0, len(files))][:-5])
    anchor = read_bmp_img(anchor_path + 'c.bmp')
    positive = read_bmp_img(positive_path + 'c.bmp')
    print(anchor_path, positive_path)
    return np.asarray(anchor), np.asarray(positive)


def get_validation_neg():
    val_files = os.listdir(val_path)
    anchor_path = os.path.join(val_path, val_files[np.random.randint(0, len(val_files))][:-5])

    paths =  list(filter(lambda x: x != val_path, [os.path.join(DATA_PATH, d) for d in os.listdir(DATA_PATH)]))
    idx_neg = np.random.randint(0, len(paths))
    files_neg = os.listdir(paths[idx_neg])
    negative_path = os.path.join(paths[idx_neg], files_neg[np.random.randint(0, len(files_neg))][:-5])

    anchor = read_bmp_img(anchor_path + 'c.bmp')
    negative = read_bmp_img(negative_path + 'c.bmp')
    print(anchor_path, negative_path)
    return np.asarray(anchor), np.asarray(negative)



