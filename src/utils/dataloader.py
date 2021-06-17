"""
Unsupervised Multi-Source Domain Adaptation with No Observable Source Data
    - DEMS (Data-free Exploitation of Multiple Sources)

Authors:
    - Hyunsik Jeon (jeon185@snu.ac.kr)
    - Seongmin Lee (ligi214@snu.ac.kr)
    - U Kang (ukang@snu.ac.kr)

File: DEMS/src/utils/dataloader.py
"""
import pickle as pkl

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets.mnist import MNIST
from torchvision.datasets.svhn import SVHN
from torchvision.datasets.usps import USPS

from utils.utils import *


class DataSet(torch.utils.data.Dataset):
    """
    DataSet definition for dataset.
    """
    def __init__(self, x, y=None, transformer=transforms.ToTensor()):
        """
        Class initializer.
        """
        self.x = x
        self.y = y
        self.tf = transformer

    def __getitem__(self, idx):
        """
        Transformation for getting each item.
        """
        if self.y is not None:
            return self.tf(np.uint8(self.x[idx])), torch.tensor(self.y[idx]).long()
        else:
            return self.tf(self.x[idx]), torch.zeros(1)

    def __len__(self):
        """
        Length of dataset.
        """
        return self.x.shape[0]


def in_torchvision(data, batch_size, num_val=5000):
    """
    Dataset definition for which are in torchvision library.
    """
    train, valid, test = None, None, None

    # for MNIST dataset
    if data == DATA_MNIST:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])   # (-1, 1)
        train = MNIST('../data/digit',
                      train=True,
                      download=True,
                      transform=transform)
        valid = MNIST('../data/digit',
                      train=True,
                      download=True,
                      transform=transform)
        test = MNIST('../data/digit',
                     train=False,
                     download=True,
                     transform=transform)

    # for SVHN dataset
    elif data == DATA_SVHN:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])   # (-1, 1)
        train = SVHN('../data/digit/SVHN',
                     split='train',
                     download=True,
                     transform=transform)
        valid = SVHN('../data/digit/SVHN',
                     split='train',
                     download=True,
                     transform=transform)
        test = SVHN('../data/digit/SVHN',
                    split='test',
                    download=True,
                    transform=transform)

    # for USPS dataset
    elif data == DATA_USPS:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])   # (-1, 1)
        train = USPS('../data/digit/USPS',
                     train=True,
                     download=True,
                     transform=transform)
        valid = USPS('../data/digit/USPS',
                     train=True,
                     download=True,
                     transform=transform)
        test = USPS('../data/digit/USPS',
                    train=False,
                    download=True,
                    transform=transform)
        num_val = 1000

    # define random sampler for training dataset
    num_train = len(train)
    idx = list(range(num_train))
    np.random.shuffle(idx)
    trn_idx, val_idx = idx[num_val:], idx[:num_val]
    trn_sampler, val_sampler = SubsetRandomSampler(
        trn_idx), SubsetRandomSampler(val_idx)

    # define dataloaders
    trn_loader = DataLoader(train, batch_size=batch_size, sampler=trn_sampler,
                            num_workers=8)
    val_loader = DataLoader(valid, batch_size=batch_size, sampler=val_sampler,
                            num_workers=8)
    test_loader = DataLoader(test, batch_size=batch_size, num_workers=8)
    return trn_loader, val_loader, test_loader


def main(data, batch_size):
    """
    Dataloader for dataset.
    """

    # for datasets where in torchvision library
    if data in [DATA_MNIST, DATA_SVHN, DATA_USPS]:
        return in_torchvision(data, batch_size)

    # for MNISTM datatset
    elif data in [DATA_MNISTM]:
        mnistm = pkl.load(open(f'../data/digit/MNISTM/mnistm_data.pkl', 'rb'))
        train_x = mnistm['train_x']
        train_y = mnistm['train_y']
        valid_x = mnistm['valid_x']
        valid_y = mnistm['valid_y']
        test_x = mnistm['test_x']
        test_y = mnistm['test_y']
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])   # (-1, 1)
        train = DataSet(train_x, train_y, transformer=transform)
        valid = DataSet(valid_x, valid_y, transformer=transform)
        test = DataSet(test_x, test_y, transformer=transform)
        trn_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                                num_workers=8)
        val_loader = DataLoader(valid, batch_size=batch_size, num_workers=8)
        test_loader = DataLoader(test, batch_size=batch_size, num_workers=8)
        return trn_loader, val_loader, test_loader

    # for SynDigits dataset
    elif data in [DATA_SYNDIGITS]:
        root_dir = '../data/digit/SYNDIGITS/'
        train_x = np.load(root_dir + 'train_x.npz')['arr_0']
        train_y = np.load(root_dir + 'train_y.npz')['arr_0']
        valid_x = np.load(root_dir + 'val_x.npz')['arr_0']
        valid_y = np.load(root_dir + 'val_y.npz')['arr_0']
        test_x = np.load(root_dir + 'test_x.npz')['arr_0']
        test_y = np.load(root_dir + 'test_y.npz') ['arr_0']

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])   # (-1, 1)

        train = DataSet(train_x, train_y, transformer=transform)
        valid = DataSet(valid_x, valid_y, transformer=transform)
        test = DataSet(test_x, test_y, transformer=transform)
        trn_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                                num_workers=8)
        val_loader = DataLoader(valid, batch_size=batch_size, num_workers=8)
        test_loader = DataLoader(test, batch_size=batch_size, num_workers=8)
        return trn_loader, val_loader, test_loader
