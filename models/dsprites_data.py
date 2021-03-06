import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from torch.utils.data import Dataset, DataLoader, RandomSampler, Sampler

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


class DSpritesLoader():
    def __init__(self, npz_path="../data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"):
        with np.load(npz_path, allow_pickle=True, encoding='latin1') as dataset_zip:
            self.metadata = dataset_zip['metadata'][()]
            self.X = np.reshape(dataset_zip['imgs'], (-1, 4096))
            self.Y = dataset_zip['latents_values']
            self.Y[:, 3] /= 2 * math.pi
            self.Y[:, 1] -= 1
        
class DSPritesIID(Dataset):
    # onehot - index(es) of labels to be converted to onehot.
    def __init__(self, dsprites_loader, size=10000):

        self.size = size

        self.dsprites_loader = dsprites_loader
        # self.X = np.reshape(dataset_zip['imgs'], (-1, 4096))
        # self.Y = dataset_zip['latents_values']
        # self.Y[:, 3] /= 2 * math.pi
        self.metadata = self.dsprites_loader.metadata
        self.latents_sizes = self.metadata['latents_sizes']
        # An array to convert latent indices to indices in imgs
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:],
                                                np.array([1,])))
            # self.latents_classes = dataset_zip['latents_classes']
            # # An array to convert latent indices to indices in imgs
            # self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:],
            #                         np.array([1,])))    
        self.samples = self.sample_latent()
        self.indices = self.latent_to_index(self.samples)

    def __len__(self):
        return self.size

    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)
    
    def sample_latent(self):
        samples = np.zeros((self.size, self.latents_sizes.size))
        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=self.size)

        return samples

    def __getitem__(self, idx):
        idx = self.indices[idx]
        X_new = np.array(self.dsprites_loader.X[idx], dtype=np.float32)
        Y_new = np.array(self.dsprites_loader.Y[idx], dtype=np.long)
        return (X_new, Y_new.squeeze())

class IIDSampler(Sampler):
    def __init__(self, data_source, num_samples):
        self.data_source = data_source
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        n = len(self.data_source)
        return iter(self.latent_to_index(self.sample_latent()))
    
    def latent_to_index(self, latents):
        return np.dot(latents, self.data_source.latents_bases).astype(int)
    
    def sample_latent(self):
        samples = np.zeros((self.num_samples, self.data_source.latents_sizes.size))
        for lat_i, lat_size in enumerate(self.data_source.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=self.num_samples)

        return samples


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


#testing IID
if __name__ == "__main__":
    dsprites_loader = DSpritesLoader()
    dataset = DSPritesIID(size=5000, dsprites_loader=dsprites_loader)
    batch_size = 512
    data = DataLoader(dataset, batch_size)#, sampler=IIDSampler(dataset, num_samples=batch_size))
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    plt.tight_layout()
    batch, y = next(iter(data))
    print("hello")
    print(y.shape)
    batch = batch[:9]
    exit()
    # plot individual images
    plt.subplots_adjust(top=0.9, hspace=0.55)
    for idx, x in enumerate(batch):
        x = x.view((-1, 64, 64)).squeeze()
        np.ravel(axes)[idx].imshow(x, interpolation='nearest', cmap="Greys_r")

    # plot the mean of batches
    batches, ys = zip(*[next(iter(data)) for _ in range(9)])
    print(batches[0].shape)
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    plt.subplots_adjust(top=0.9, hspace=0.55)
    for idx, batch in enumerate(batches):
        batch = batch.view((-1, 64, 64)).squeeze()
        np.ravel(axes)[idx].imshow(batch.mean(axis=0), interpolation='nearest', cmap="Greys_r")
    plt.show()



# testing raw
# if __name__ == "__main__":
#     raw_dataset = DSpritesRaw(test_index=5)
#     train_data, test_data = raw_dataset.get_train_test_datasets()

#     print(train_data.Y)
#     print(test_data.Y)
#     # dsprites = DSprites([-1, 1, 1, 9, 1, 1])
#     data = DataLoader(train_data, 1024, shuffle=True)

#     fig, axes = plt.subplots(3, 3, figsize=(8, 8))
#     plt.tight_layout()
#     batch, y = next(iter(data))
#     batch = batch[:9]
#     print(batch)
#     plt.subplots_adjust(top=0.9, hspace=0.55)
#     for idx, x in enumerate(batch):
#         x = x.view((-1, 64, 64)).squeeze()
#         np.ravel(axes)[idx].imshow(x, cmap="Greys")
    
#     batches, ys = zip(*[next(iter(data)) for _ in range(9)])
#     print(batches[0].shape)
#     fig, axes = plt.subplots(3, 3, figsize=(8, 8))
#     plt.subplots_adjust(top=0.9, hspace=0.55)
#     for idx, batch in enumerate(batches):
#         batch = batch.view((-1, 64, 64)).squeeze()
#         np.ravel(axes)[idx].imshow(batch.mean(axis=0), interpolation='nearest', cmap="Greys_r")
#     plt.title("Train data density")
    
#     test_data = DataLoader(test_data, 1024, shuffle=True)
#     batches, ys = zip(*[next(iter(test_data)) for _ in range(9)])
#     fig, axes = plt.subplots(3, 3, figsize=(8, 8))
#     plt.subplots_adjust(top=0.9, hspace=0.55)
#     for idx, batch in enumerate(batches):
#         batch = batch.view((-1, 64, 64)).squeeze()
#         np.ravel(axes)[idx].imshow(batch.mean(axis=0), interpolation='nearest', cmap="Greys_r")
#     plt.title("Test data density")
#     plt.show()
    # DATA_PATH = "../data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    # dataset_zip = np.load(DATA_PATH, allow_pickle=True, encoding='latin1')
    # print(list(dataset_zip.keys()))
    # print('Keys in the dataset:', [x for x in dataset_zip.keys()])
    # imgs = dataset_zip['imgs']
    # latents_values = dataset_zip['latents_values']
    # latents_classes = dataset_zip['latents_classes']
    # metadata = dataset_zip['metadata']

    # print('Metadata: \n', metadata)