"""
Unsupervised Multi-Source Domain Adaptation with No Observable Source Data
    - DEMS (Data-free Exploitation of Multiple Sources)

Authors:
    - Hyunsik Jeon (jeon185@snu.ac.kr)
    - Seongmin Lee (ligi214@snu.ac.kr)
    - U Kang (ukang@snu.ac.kr)

File: DEMS/src/main.py
"""
import os

import click

from translator.test import main as test_adaptation
from translator.train import main as train_adaptation
from utils.utils import *


@click.command()
@click.option('--target', type=str, default=DATA_MNISTM)
@click.option('--batch_size', type=int, default=256)
@click.option('--epsilon', type=float, default=0.9)
@click.option('--temperature', type=float, default=1)
@click.option('--train', type=bool, default=True)
def main(target, batch_size, epsilon, temperature, train):
    """
    Main function for training and testing DEMS with given hyperparameters.
    """
    source = [DATA_MNIST, DATA_MNISTM, DATA_SVHN, DATA_SYNDIGITS, DATA_USPS]
    source.remove(target)
    out_path = f'../out/{target}'
    os.makedirs(out_path, exist_ok=True)
    path_model = os.path.join(out_path, 'adaptation_models.pt')
    config = {
        'source': source,
        'target': target,
        'out_path': out_path,
        'batch_size': batch_size,
        'epsilon': epsilon,
        'temperature': temperature,
        'path_model': path_model,
    }
    if train:
        # training
        train_adaptation(**config)

    # inference
    test_adaptation(**config)


if __name__ == '__main__':
    main()
