import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from torch.utils.data import Dataset, DataLoader

class DSpritesRaw():
    def __init__(self, npz_path="../data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
        test_index=-1, test_split=0.1):

        dataset_zip = np.load(npz_path, allow_pickle=True, encoding='latin1')
        
        self.imgs = np.reshape(dataset_zip['imgs'], (-1, 4096))
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']
        self.metadata = dataset_zip['metadata'][()]
        self.latents_sizes = self.metadata['latents_sizes']

        # An array to convert latent indices to indices in imgs
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:],
                                np.array([1,])))

        train_indices, self.train_labels, test_indices, self.test_labels = self.train_test_latents(test_index, 
                                                                                                  test_split)
        
        self.train_indices = self.latent_to_index(train_indices)
        self.test_indices = self.latent_to_index(test_indices)

    def get_train_test_datasets(self):
        return (DSprites(self.train_indices, self.train_labels, self), 
               DSprites(self.test_indices, self.test_labels, self))

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
        train_latent_indexes, train_labels, test_latent_indexes, test_labels
        """
        latents_sizes = self.latents_sizes
        keys_ = ['color', 'shape', 'scale', 'orientation', 'posX', 'posY']
        latents_possible_values = [self.metadata['latents_possible_values'][x] for x in keys_]

        latents = np.random.permutation(latents_sizes[test_index])
        train_latents = latents[:int(len(latents) * (1 - test_split))]
        test_latents = latents[-int(len(latents) * (test_split)):]

        train_latents_sizes, test_latents_sizes  = list(latents_sizes), list(latents_sizes)
        train_latents_sizes[test_index], test_latents_sizes[test_index] = len(train_latents), len(test_latents)

        n_train_samples, n_test_samples = np.cumprod(train_latents_sizes)[-1], np.cumprod(test_latents_sizes)[-1]

        def sample_latents(sizes, latent_choices, n_samples):
            samples = np.zeros((n_samples, len(sizes)))
            labels = np.zeros((n_samples, len(sizes)), dtype=np.float32)
            for i, latent_size in enumerate(sizes):
                choices = latent_choices if i == test_index else latent_size
                cur_latents = np.random.choice(choices, size=n_samples)
                samples[:, i] = cur_latents
                labels[:, i] = np.array([latents_possible_values[i][x] for x in cur_latents])
            return samples, labels

        train_samples, train_labels = sample_latents(train_latents_sizes, train_latents, n_train_samples)
        test_samples, test_labels = sample_latents(test_latents_sizes, test_latents, n_test_samples)
        
        return train_samples, train_labels, test_samples, test_labels

class DSPritesIID(Dataset):
    def __init__(self, npz_path="../data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
                    size=10000):
        with np.load(npz_path, allow_pickle=True, encoding='latin1') as dataset_zip:
        
            self.X = np.reshape(dataset_zip['imgs'], (-1, 4096))
            self.y = dataset_zip['latents_values']
            # self.latents_classes = dataset_zip['latents_classes']
            # self.metadata = dataset_zip['metadata'][()]
            # self.latents_sizes = self.metadata['latents_sizes']

            # # An array to convert latent indices to indices in imgs
            # self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:],
            #                         np.array([1,])))    

class DSpritesIIDDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass


class DSprites(Dataset):
    def __init__(self, x_indices, y, dataset):
        self.X = x_indices
        self.Y = y
        self.dataset = dataset
        # Normalize data
        self.Y[:, 3] /= 2 * math.pi
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        
        X_new = np.array(self.dataset.imgs[self.X[idx]], dtype=np.float32)
        Y_new = np.array(self.Y[idx])

        return (X_new, Y_new)




if __name__ == "__main__":
    raw_dataset = DSpritesRaw(test_index=-1)
    train_data, test_data = raw_dataset.get_train_test_datasets()

    print(train_data.Y)
    print(test_data.Y)
    # dsprites = DSprites([-1, 1, 1, 9, 1, 1])
    data = DataLoader(train_data, 512, shuffle=True)

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    plt.tight_layout()
    batch, y = next(iter(data))
    batch = batch[:9]
    print(batch)
    plt.subplots_adjust(top=0.9, hspace=0.55)
    for idx, x in enumerate(batch):
        x = x.view((-1, 64, 64)).squeeze()
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