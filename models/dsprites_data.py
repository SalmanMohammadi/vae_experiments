import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from torch.utils.data import Dataset, DataLoader

class DSprites(Dataset):
    def __init__(self, latents_counts=None, npz_path="../data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
        ):
        dataset_zip = np.load(npz_path, allow_pickle=True, encoding='latin1')
        
        self.imgs = np.reshape(dataset_zip['imgs'], (-1, 4096))
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']
        self.metadata = dataset_zip['metadata'][()]
        self.latents_sizes = self.metadata['latents_sizes']

        # An array to convert latent indices to indices in imgs
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:],
                                np.array([1,])))

        if latents_counts == None:
            latents_counts = [-1 for _ in self.latents_sizes]

        x_indices, self.Y = self.get_latents(latents_counts)
        self.X = self.latent_to_index(x_indices)

        # Normalize data
        self.Y[:, 3] /= 2 * math.pi



    def latent_to_index(self, latents):
        """
        latents - an array of shape (-1, 6) which indexes values of latents

        returns the correnspoding indices in img[]
        """
        return np.dot(latents, self.latents_bases).astype(int)

    def train_test_latents(self, test_index=-1, test_split=0.1):    
        """
        Parameters
        ----------
        test_index - index of the latent to split for train and test
        test_split - poriton of test_index latents to split
        
        Returns
        ----------
        train_latent_indexes, train_labels
        """
        latents_sizes = self.metadata['latents_sizes']
        latents_counts = [y if x==-1 else x for x,y in zip(latents_counts, latents_sizes)]

        assert test_index in range(0, len(latents_sizes-1))
        assert len(latents_counts) == len(latents_sizes)
        assert all([x <= y for x, y in zip(latents_counts, latents_sizes)])
        
        keys_ = ['color', 'shape', 'scale', 'orientation', 'posX', 'posY']
        latents_possible_values = [self.metadata['latents_possible_values'][x] for x in keys_]

        num_samples = np.cumprod(latents_counts)[-1]
        samples = np.zeros((num_samples, len(latents_sizes)))
        labels = np.zeros((num_samples, len(latents_sizes)), dtype=np.float32)

        for i, (size, latent_size) in enumerate(zip(latents_counts, latents_sizes)):
            selection = np.random.choice(np.random.randint(latent_size, size=size), size=num_samples)
            samples[:, i] = selection
            labels[:, i] = np.array([latents_possible_values[i][x] for x in selection])
        return samples, labels

    def get_latents(self, latents_counts):
        """
        Parameters
        ----------
        latent_counts - a list with 6 elements where each element corresponds to
        [ 1  3  6 40 32 32]
        ['color', 'shape', 'scale', 'orientation', 'posX', 'posY']
        the number of ways that the latent variable can _randomly_ vary
        (between 1 and n, or -1 for all)

        Returns
        ----------
        train_latent_indexes - an array of shape len(cumprod(latent_counts))
        train_latent_labels - an dictionary of {latent_labels:latent_indexes} for retrieving metadata
        """
        latents_sizes = self.metadata['latents_sizes']
        latents_counts = [y if x==-1 else x for x,y in zip(latents_counts, latents_sizes)]

        assert len(latents_counts) == len(latents_sizes)
        assert all([x <= y for x, y in zip(latents_counts, latents_sizes)])

        keys_ = ['color', 'shape', 'scale', 'orientation', 'posX', 'posY']
        latents_possible_values = [self.metadata['latents_possible_values'][x] for x in keys_]

        num_samples = np.cumprod(latents_counts)[-1]
        samples = np.zeros((num_samples, len(latents_sizes)))
        labels = np.zeros((num_samples, len(latents_sizes)), dtype=np.float32)

        for i, (size, latent_size) in enumerate(zip(latents_counts, latents_sizes)):
            selection = np.random.choice(np.random.randint(latent_size, size=size), size=num_samples)
            samples[:, i] = selection
            labels[:, i] = np.array([latents_possible_values[i][x] for x in selection])
        return samples, labels


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        
        X_new = np.array(self.imgs[self.X[idx]], dtype=np.float32)
        Y_new = np.array(self.Y[idx])

        return (X_new, Y_new)

if __name__ == "__main__":
    dsprites = DSprites([-1, 1, 1, 9, 1, 1])
    data = DataLoader(dsprites, 1, shuffle=True)

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.55)
    for idx, (x, y) in enumerate(data):
        x = x.view(-1, 64, 64).squeeze()
        np.ravel(axes)[idx].imshow(x, cmap="Greys")
    plt.show()


    # DATA_PATH = "../data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    # dataset_zip = np.load(DATA_PATH, allow_pickle=True, encoding='latin1')
    # print(list(dataset_zip.keys()))
    # print('Keys in the dataset:', [x for x in dataset_zip.keys()])
    # imgs = dataset_zip['imgs']
    # latents_values = dataset_zip['latents_values']
    # latents_classes = dataset_zip['latents_classes']
    # metadata = dataset_zip['metadata']

    # print('Metadata: \n', metadata)