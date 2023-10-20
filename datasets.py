"""Definitions and helpers for datasets."""

import jax
import gzip
import pickle
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import os
import torch as to
import torchvision as tv
import torchvision.transforms as tvt
from scipy.io import loadmat


def pad_dataset_for_equal_batches(dataset, batch_size=None):
    if batch_size is None:
        return dataset

    padding_len = batch_size - len(dataset[0]) % batch_size
    return (np.concatenate((dataset[0],
                            np.full((padding_len, *dataset[0].shape[1:]),
                                    np.nan,
                                    dataset[0].dtype))),
            np.concatenate((dataset[1],
                            np.full((padding_len, *dataset[1].shape[1:]),
                                    np.nan,
                                    dataset[1].dtype))))


def make_split_datasets(name,
                        validation_proportion=0,
                        pad_to_equal_training_batches=None,
                        **dataset_kwargs):
    normalise_inputs = dataset_kwargs.pop('normalise_inputs', False)
    normalise_outputs = dataset_kwargs.pop('normalise_outputs', False)
    dataset = globals()[name]

    train_val_dataset = dataset(train=True, **dataset_kwargs)
    train_val_sizes = len(train_val_dataset) * np.array(
        [1 - validation_proportion, validation_proportion])
    train_val_sizes = np.rint(train_val_sizes).astype(np.int)
    # If proportional split isn't exact, may need to adjust indices to avoid
    # overflowing the dataset
    train_val_sizes[-1] -= sum(train_val_sizes) - len(train_val_dataset)
    test_dataset = dataset(train=False, **dataset_kwargs)

    if normalise_inputs:
        input_data = to.stack([point[0] for point in train_val_dataset])
        normalise_dimension = (0, 1) if input_data.ndim == 3 else 0
        means = input_data.mean(dim=normalise_dimension)
        standard_deviations = input_data.std(dim=normalise_dimension)
        standard_deviations[standard_deviations == 0] = 1
        normaliser = Normaliser(means, standard_deviations)
        for dataset in train_val_dataset, test_dataset:
            if isinstance(dataset.transform, tvt.Compose):
                dataset.transform.transforms.append(normaliser)
            elif dataset.transform is not None:
                dataset.transform = tvt.Compose([dataset.transform,
                                                 normaliser])
            else:
                dataset.transform = normaliser

    if normalise_outputs:
        output_data = to.stack([point[1] for point in train_val_dataset])
        normalise_dimension = (0, 1)
        means = output_data.mean(dim=normalise_dimension)
        standard_deviations = output_data.std(dim=normalise_dimension)
        standard_deviations[standard_deviations == 0] = 1
        normaliser = Normaliser(means, standard_deviations)
        for dataset in train_val_dataset, test_dataset:
            if isinstance(dataset.target_transform, tvt.Compose):
                dataset.target_transform.transforms.append(normaliser)
            elif dataset.target_transform is not None:
                dataset.target_transform = tvt.Compose([dataset.target_transform,
                                                        normaliser])
            else:
                dataset.target_transform = normaliser
            setattr(dataset, 'target_unnormaliser',
                    lambda x: (x * standard_deviations) + means)

    # TODO: Ensure reproducibility of random sampler
    if validation_proportion == 0:
        datasets = (to.utils.data.Subset(train_val_dataset, range(len(train_val_dataset))),
                    None,
                    test_dataset)
    else:
        datasets = (to.utils.data.Subset(train_val_dataset, range(train_val_sizes[0])),
                    to.utils.data.Subset(train_val_dataset, range(train_val_sizes[0],
                                                                  sum(train_val_sizes))),
                    test_dataset)
    datasets = [[component.numpy() for component in dataset[:]]
                for dataset in datasets]
    datasets[0] = pad_dataset_for_equal_batches(datasets[0],
                                                pad_to_equal_training_batches)
    return datasets


def make_batches(dataset, batch_size, rng):
    steps_per_epoch = max(1, int(np.ceil(len(dataset[0]) / batch_size)))
    permutations = jax.random.permutation(rng, len(dataset[0]))
    permutations = np.split(
        permutations,
        [batch_size * batch_num for batch_num in range(1, steps_per_epoch)])

    for permutation in permutations:
        yield dataset[0][permutation], dataset[1][permutation]


class Normaliser():
    """Reimplementation of the torchvision `Normalize` transform, supporting a
    broader range of data sizes.
    """

    def __init__(self, means, standard_deviations):
        self.means = means
        self.standard_deviations = standard_deviations

    def __call__(self, unnormalised_data):
        return (unnormalised_data - self.means) / self.standard_deviations


class ExternalDataset(to.utils.data.Dataset):
    has_target_data = True

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        super().__init__()
        self.root = root

        if download:
            self.download()
            self.process()

        self.data = to.load(os.path.join(root, 'data.pt'))
        if self.has_target_data:
            self.targets = to.load(os.path.join(root, 'targets.pt'))
        else:
            self.targets = []

        if train:
            self.data = self.data[type(self).train_val_slice]
            self.targets = self.targets[type(self).train_val_slice]
        else:  # Test
            self.data = self.data[type(self).test_slice]
            self.targets = self.targets[type(self).test_slice]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        data = self.data[index]
        targets = self.targets[index]

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            targets = self.target_transform(targets)

        return data, targets

    def download(self):
        """Download the necessary dataset files from the Internet."""
        for name, url in self.download_urls.items():
            download_path = os.path.join(self.root, 'raw', name)
            Path(download_path).parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url) as response, open(download_path, 'wb') as download_file:
                shutil.copyfileobj(response, download_file)

    def process(self):
        """Process the downloaded files into `data.pt` and `targets.pt`."""
        pass


class SVHN(ExternalDataset):
    download_urls = {'train_32x32.mat': 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat',
                     'test_32x32.mat': 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'}
    track_accuracies = True
    train_val_slice = slice(73257)
    test_slice = slice(-26032, None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, root='./data/SVHN', **kwargs)

    def process(self):
        train_mat = loadmat(os.path.join(self.root, 'raw', 'train_32x32.mat'))
        test_mat = loadmat(os.path.join(self.root, 'raw', 'test_32x32.mat'))

        data = np.concatenate((train_mat['X'],
                               test_mat['X']), axis=3)
        # Cast labels to np.int64 so they become PyTorch longs
        targets = np.concatenate((train_mat['y'].astype(np.int64).squeeze(),
                                  test_mat['y'].astype(np.int64).squeeze()), axis=0)

        # PyTorch requires labels from 0 to 9, not 1 to 10:
        np.place(targets, targets == 10, 0)
        targets = to.from_numpy(targets)
        data = to.from_numpy(
            np.transpose(
                data,
                (3, 2, 0, 1))
            ).contiguous().float()

        to.save(data, os.path.join(self.root, 'data.pt'))
        to.save(targets, os.path.join(self.root, 'targets.pt'))


class CIFAR10(ExternalDataset):
    download_urls = {'base_archive.tar.gz': 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'}
    track_accuracies = True
    train_val_slice = slice(50000)
    test_slice = slice(-10000, None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         root='./data/CIFAR10',
                         **kwargs)

    def process(self):
        # Logic informed by https://cs.toronto.edu/~kriz/cifar.html
        with tarfile.open(os.path.join(self.root, 'raw', 'base_archive.tar.gz'),
                          'r') as archive_file:
            archive_file.extractall(os.path.join(self.root, 'raw'))

        data, targets = [], []
        batch_files = ('data_batch_1',
                       'data_batch_2',
                       'data_batch_3',
                       'data_batch_4',
                       'data_batch_5',
                       'test_batch')
        for batch_file in batch_files:
            batch_path = os.path.join(self.root,
                                      'raw',
                                      'cifar-10-batches-py',
                                      batch_file)
            with open(batch_path, 'rb') as batch_data:
                batch_dict = pickle.load(batch_data, encoding='bytes')
            data.append(
                to.from_numpy(
                    batch_dict[b'data']
                    .reshape(-1, 3, 32, 32)
                    .transpose((0, 2, 3, 1))))
            targets.append(
                to.tensor(
                    batch_dict[b'labels']))

        data = to.cat(data, dim=0) / 255
        data_mean = to.tensor([[[[0.4914, 0.4822, 0.4465]]]])
        data_std = to.tensor([[[[0.247, 0.243, 0.262]]]])
        data = (data - data_mean) / data_std
        targets = to.cat(targets, dim=0)
        to.save(data, os.path.join(self.root, 'data.pt'))
        to.save(targets, os.path.join(self.root, 'targets.pt'))


class CIFAR100(ExternalDataset):
    download_urls = {'base_archive.tar.gz': 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'}
    track_accuracies = True
    train_val_slice = slice(50000)
    test_slice = slice(-10000, None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         root='./data/CIFAR100',
                         transform=tvt.Compose([
                             tvt.Normalize((0.5074, 0.4867, 0.4411),
                                           (0.2675, 0.2566, 0.2763))]),
                         **kwargs)

    def process(self):
        # Logic informed by https://cs.toronto.edu/~kriz/cifar.html
        with tarfile.open(os.path.join(self.root, 'raw', 'base_archive.tar.gz'),
                          'r') as archive_file:
            archive_file.extractall(os.path.join(self.root, 'raw'))

        data, targets = [], []
        batch_files = ('train',
                       'test')
        for batch_file in batch_files:
            batch_path = os.path.join(self.root,
                                      'raw',
                                      'cifar-100-python',
                                      batch_file)
            with open(batch_path, 'rb') as batch_data:
                batch_dict = pickle.load(batch_data, encoding='bytes')
            data.append(
                to.from_numpy(
                    batch_dict[b'data']
                    .reshape(-1, 3, 32, 32)))
            targets.append(
                to.tensor(
                    batch_dict[b'fine_labels']))

        data = to.cat(data, dim=0) / 255
        targets = to.cat(targets, dim=0)
        to.save(data, os.path.join(self.root, 'data.pt'))
        to.save(targets, os.path.join(self.root, 'targets.pt'))


class FashionMNIST(ExternalDataset):

    download_urls = {'train_data.gz': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
                     'train_targets.gz': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
                     'test_data.gz': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
                     'test_targets.gz': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz'}
    track_accuracies = True
    train_val_slice = slice(60000)
    test_slice = slice(-10000, None)

    def __init__(self, *args, **kwargs):
        root = kwargs.pop('root', './data/FashionMNIST')
        super().__init__(*args, root=root, **kwargs)

    def process(self):
        # Logic borrowed from https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
        with gzip.open(os.path.join(self.root, 'raw', 'train_data.gz'),
                       'rb') as raw_path:
            train_data = np.frombuffer(raw_path.read(),
                                       dtype=np.uint8,
                                       offset=16).reshape(-1, 784)
        with gzip.open(os.path.join(self.root, 'raw', 'train_targets.gz'),
                       'rb') as raw_path:
            train_targets = np.frombuffer(raw_path.read(),
                                          dtype=np.uint8,
                                          offset=8)
        with gzip.open(os.path.join(self.root, 'raw', 'test_data.gz'),
                       'rb') as raw_path:
            test_data = np.frombuffer(raw_path.read(),
                                      dtype=np.uint8,
                                      offset=16).reshape(-1, 784)
        with gzip.open(os.path.join(self.root, 'raw', 'test_targets.gz'),
                       'rb') as raw_path:
            test_targets = np.frombuffer(raw_path.read(),
                                         dtype=np.uint8,
                                         offset=8)

        data = to.cat((to.from_numpy(train_data),
                       to.from_numpy(test_data)),
                      dim=0)
        targets = to.cat((to.from_numpy(train_targets),
                          to.from_numpy(test_targets)),
                         dim=0)
        data = data.float()
        to.save(data, os.path.join(self.root, 'data.pt'))
        to.save(targets, os.path.join(self.root, 'targets.pt'))


class UCIDataset(ExternalDataset):
    """Extend `ExternalDataset` with UCI data processing function."""

    def process(self):
        raw = np.loadtxt(
            os.path.join(self.root, 'raw', 'data_targets.txt'),
            dtype='float32')
        permutation_indices = np.loadtxt(
            os.path.join(self.root, 'permutation_indices.txt'),
            dtype='int')

        raw_permuted = to.from_numpy(raw[permutation_indices])
        to.save(raw_permuted[:, :-1], os.path.join(self.root, 'data.pt'))
        to.save(raw_permuted[:, -1:], os.path.join(self.root, 'targets.pt'))


class UCI_Energy(UCIDataset):

    train_val_slice = slice(614 + 78)
    test_slice = slice(-76, None)
    download_urls = {'data_targets.txt': 'https://raw.githubusercontent.com/yaringal/DropoutUncertaintyExps/master/UCI_Datasets/energy/data/data.txt'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, root='./data/UCI_Energy', **kwargs)


class Kin8nm(UCIDataset):

    train_val_slice = slice(5898 + 1475)
    test_slice = slice(-819, None)
    download_urls = {'data_targets.txt': 'https://raw.githubusercontent.com/yaringal/DropoutUncertaintyExps/master/UCI_Datasets/kin8nm/data/data.txt'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, root='./data/Kin8nm', **kwargs)


class UCI_Protein(UCIDataset):

    train_val_slice = slice(41157)
    test_slice = slice(-4573, None)
    download_urls = {'data_targets.txt': 'https://raw.githubusercontent.com/yaringal/DropoutUncertaintyExps/master/UCI_Datasets/protein-tertiary-structure/data/data.txt'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, root='./data/UCI_Protein', **kwargs)
