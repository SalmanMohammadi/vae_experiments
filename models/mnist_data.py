#!/usr/bin/env python

from os.path import dirname, join as pjoin

import numpy as np
import torch
from torch.utils.data import Dataset as D
from numpy.random import random, choice, permutation

from skimage.io import imread
from skimage.transform import rotate, resize, rescale

import pandas as pd


def _euclidean_diff(x1, x2):
    return np.linalg.norm(x1 - x2)

class Dataset(object):
    def __init__(self, Y, T, standardize=True, meta_data={}):
        self.Y = Y
        self.T = T
        self.meta_data = meta_data
        for k, v in meta_data.items():
            setattr(self, k, v)

        # Normalize 
        # self.mean = self.Y.mean()
        # self.std = self.Y.std() + 1e-10
        self.standardize = standardize
        if standardize:
            Y /= 255.
        # for Y in [self.Y, self.Y]:
            # Y -= self.mean
            # Y /= self.std

    def __getitem__(self, args):
        # FIXME ensure that subclasses are instantiated
        Y_sub = self.Y[args]
        T_sub = self.T.iloc[args]

        sub_data = Dataset(Y_sub.copy(), T_sub.copy(), standardize=False)
        sub_data.mean = self.mean
        sub_data.std = self.std
        sub_data.standardize = self.standardize

        return sub_data

    @staticmethod
    def load(npz_path, **kwargs):
        npz = np.load(npz_path)
        try:
            meta_data = npz["meta_data"]
        except KeyError:
            meta_data = {}
        T = pd.DataFrame(npz["T"], columns=npz["T_columns"])
        return Dataset(npz["Y"], T, meta_data=meta_data, **kwargs)

    def save(self, npz_path):
        T_columns = self.T.columns

        # TODO perform inverse standardization
        np.savez(npz_path, Y=self.Y, T=self.T, T_columns=T_columns)

    @property
    def columns(self):
        return self.T.columns

    @property
    def shape(self):
        return self.Y.shape

    def __len__(self):
        return self.shape[0]

    def subsample(self, n_observations):
        idxs = choice(self.shape[0], n_observations, replace=False)
        return self[idxs]


class MNIST(Dataset):
    def __init__(self, digits=None, n_obs=1000, standardize=True,
                 target_shape=(28, 28), csv_path="../data/mnist_train.csv",
                 balance_classes=True):

        if digits is None:
            digits = range(10)

        # Load and select digits
        folder = dirname(__file__)
        # mnist = np.loadtxt(pjoin(folder, csv_path), delimiter=",")
        mnist = pd.read_csv(pjoin(folder, csv_path), header=None).values
        T_arr, Y = mnist[:, 0].astype(int), np.array(mnist[:, 1:], dtype=np.float32)
        
        # Selects from dataset only the digits we are interested in.
        idxs = [np.nonzero(T_arr == i)[0] for i in digits]
        num_samples = min(n_obs // len(digits), min([len(x) for x in idxs]))

        print("DATASET: num_samples per digit: %d" %(num_samples))
        # assert np.all([num_samples <= len(x) for x in idxs]),\
        #         "num_samples shouldn't exceed the number of available samples per digit"
        idxs = np.array([permutation(x)[:min(num_samples, len(x))] for x in idxs]).flatten()
        Y = Y[idxs]
        T_arr = T_arr[idxs]
        
        T = pd.DataFrame(T_arr, columns=["Digit"])
        v = int(np.sqrt(Y.shape[-1]))
        orig_shape = (v, v)

        if target_shape is not None:
            target_shape = target_shape
        else:
            target_shape = orig_shape

        # Scale
        if orig_shape != target_shape:
            Y_ = np.zeros((len(Y), np.prod(target_shape)))
            for i, y in enumerate(Y):
                Y_[i] = resize(
                    y.reshape(*orig_shape), target_shape).ravel()
            Y = Y_

        # Normalize dataset?
        # Y /= Y.max()

        super().__init__(Y, T, standardize=standardize)

        self.orig_shape = orig_shape
        self.target_shape = target_shape

    def __getitem__(self, args):
        Y_sub = np.array(self.Y[args], dtype=np.float32)
        T_sub = self.T.iloc[args].values

        return (Y_sub, T_sub)

    def create_transformations(self, trans_per_image=1, max_angle=45,
                               max_brightness=0.0, max_noise=0.0,
                               max_scale=3.0):
        # FIXME prior standardization messes up noise
        # FIXME noise not looking correct

        assert self.T.shape[-1] == 1

        Y_, T_ = [], []
        columns = np.array(["Digit", "Rotation", "Brightness", "Noise",
                            "Inv. scale"])
        for y, t in zip(self.Y, self.T.values.ravel()):
            im = y.copy().reshape(*self.target_shape)
            for _ in range(trans_per_image):
                r = random() * max_angle * 2 - max_angle
                b = random() * max_brightness
                w = random() * max_noise
                s = random() * (max_scale - 1) + 1

                im_new = rotate(im, r)
                scaled = rescale(im_new.copy(), 1./s)
                height, width = scaled.shape
                dh = (self.target_shape[0] - height) // 2
                dw = (self.target_shape[1] - width) // 2
                im_new = np.zeros_like(im_new)
                im_new[dh:dh+height, dw:dw+width] = scaled

                im_new = np.clip(im_new + random(self.target_shape) * w, 0, 1)

                Y_.append(im_new.ravel())
                T_.append([t, r, b, w, s])

        self.Y = np.array(Y_, dtype=np.float32)
        # if self.standardize:
        #     self.Y /= 255.
            # self.Y -= self.Y.mean(axis=1)[:, None]
            # self.Y /= self.Y.std(axis=1)[:, None] + 1e-5
        T_ = np.array(T_, dtype=np.float32)
        T_[:,1] = T_[:,1]/45.
        # js = np.where(T_.std(axis=0) > 0)[0]
        # wot
        self.T = pd.DataFrame(T_, columns=columns)
        # # Scramble
        # idxs = permutation(range(len(self)))
        # self.Y = self.Y[idxs]
        # self.T = self.T.iloc[idxs]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Loads raw mnist with specified digits with n_obs datapoints
    data = MNIST(digits=[0, 1], n_obs=2)
    data.create_transformations(trans_per_image=2)
    # print(data.T.groupby('Digit').count())
    attrs = data.columns

    k = 9
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.55)
    for i, ax in enumerate(np.ravel(axes)):
        y = data.Y[i]
        # print(y.shape)
        im = y.reshape(int(np.sqrt(len(y))), -1)
        # print(im.shape)
        # print(im)
        metadata = data.T.iloc[i]

        l = ["{}: {:.2f}".format(c, v) for c, v in zip(data.columns, metadata)]
        ax.set_title("\n".join(l))
        ax.imshow(im, cmap="Greys")

    plt.show()
