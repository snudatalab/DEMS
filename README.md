# DEMS (Data-free Exploitation of Multiple Sources)

This repository is a PyTorch implementation of "Unsupervised Multi-Source Domain Adaptation with No Observable Source Data" which is submitted to PLOS ONE.

<p align="center">
    <img src="docs/Architecture.tif" width="750"\>
</p>



## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
> CUDA settings are highly encouraged for the efficient training of DEMS which is constructed based on CNN structure.
> Detailed instruction for CUDA-based PyTorch settings is described in <https://pytorch.org/get-started/previous-versions/>.



## Pre-processed Datasets

For easy usage, we provide pre-processed datasets of {MNIST, MNIST-M, SVHN, SynDigits, USPS} which is uploaded in <https://doi.org/10.6084/m9.figshare.14790036.v1>.

The datasets should be placed in this package as follows:
```
DEMS
└───README.md
│
└───demo.sh
│
└───requirements.txt
│
└───docs
│   └─── ...
│
└───src
│   └─── ...
│
└───out
│   └─── ...
│
└───pretrained
│   └─── ...   
│
└───data
    └───digit
        └───MNIST   
        └───MNISTM
        └───SVHN
        └───SYNDIGITS
        └───USPS
```



## Pre-trained Classifiers

Each pre-trained classifier for <source_domain>, which is necessary for training and testing DEMS, is provided in ```./pretrained/<source_domain>.pt```.



## Demo

You can run the demo script by ```bash demo.sh```, which simply runs ```src/main.py``` with USPS target in training mode.



## Training

Run this command to train DEMS:

```train
python main.py --target <target_domain> --train True
```
> Train DEMS and save the best model to ```./out/<target_domain>/adaptation_models.pt```
>
> <target_domain> should be one of whole datasets: {mnist, mnistm, svhn, syndigits, usps}.
>
> The remains excluding <target_domain> among the whole datasets are selected as sources automatically.
>
> The optimal hyperparameters are set as default.
>
> - ```epsilon```: 0.9
> - ```temperature``` (```lambda``` in the paper): 1.0



## Evaluation

Run this command to test DEMS:

```eval
python main.py --target <target_domain> --train False
```
> Load a saved model from ```./out/<target_domain>/adaptation_models.pt``` and test the model on the target test dataset.
>
> A pre-trained adaptation network is needed to run this code. We thus provide the pre-trained adaptation network for users that want to evaluate DEMS instantly without any training process.
>
> <target_domain> should be one of whole datasets: {mnist, mnistm, svhn, syndigits, usps}.
>
> The remains excluding <target_domain> among the whole datasets are selected as sources automatically.



## Results

DEMS achieves the following classification performance:

| Target dataset | MNIST  | MNIST-M | SVHN   | SynDigits | USPS   |
| -------------- | ------ | ------- | ------ | --------- | ------ |
| Accuracy       | 98.64% | 82.20%  | 77.22% | 94.57%    | 95.57% |


## Reference
You can copy the following information to cite the paper:
```
@article{jeon2021unsupervised,
  title={Unsupervised multi-source domain adaptation with no observable source data},
  author={Jeon, Hyunsik and Lee, Seongmin and Kang, U},
  journal={Plos one},
  volume={16},
  number={7},
  pages={e0253415},
  year={2021},
  publisher={Public Library of Science San Francisco, CA USA}
}
```


## Contact us
- Hyunsik Jeon (jeon185@snu.ac.kr)
- Seongmin Lee (ligi214@snu.ac.kr)
- U Kang (ukang@snu.ac.kr)
- Data Mining Lab at Seoul National University
