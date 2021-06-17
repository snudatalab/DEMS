"""
Unsupervised Multi-Source Domain Adaptation with No Observable Source Data
    - DEMS (Data-free Exploitation of Multiple Sources)

Authors:
    - Hyunsik Jeon (jeon185@snu.ac.kr)
    - Seongmin Lee (ligi214@snu.ac.kr)
    - U Kang (ukang@snu.ac.kr)

File: DEMS/src/utils/utils.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from PIL import Image
from torchvision.utils import save_image

from classifier.resnet import ResNet
from translator.models import Encoder, Decoder, DomainWeight

irange = range
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_MNIST = 'mnist'
DATA_MNISTM = 'mnistm'
DATA_SVHN = 'svhn'
DATA_SYNDIGITS = 'syndigits'
DATA_USPS = 'usps'


def load_model(model, path, device):
    """
    Load model from the given path.
    """
    checkpoint = torch.load(path, map_location=device)
    model_state = checkpoint.get('model_state', None)
    model.load_state_dict(model_state)


def set_seed(seed):
    """
    Set a random seed for numpy and PyTorch.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def load_classifier(dataset):
    """
    Load classifier with a ResNet structure.
    """
    network = ResNet(num_channels=3)
    path = f'../pretrained/{dataset}.pt'
    load_model(network, path, DEVICE)
    return network.eval()


def init_encoder(in_channels, out_channels, num_down, num_blocks):
    """
    Initialize an encoder.
    """
    model = Encoder(in_channels=in_channels, out_channels=out_channels,
                    num_down=num_down, num_blocks=num_blocks)
    return model


def init_decoder(in_channels, out_channels, num_up, num_blocks):
    """
    Initialize a decoder.
    """
    model = Decoder(in_channels=in_channels, out_channels=out_channels,
                    num_up=num_up, num_blocks=num_blocks)
    return model


def init_domain_weight(num_domain, dim, temperature):
    """
    Initialize domain embedding factors.
    """
    emb = DomainWeight(num_domain=num_domain, dim=dim, temperature=temperature)
    return emb


def visualize_samples(encoder, decoders, samples, path_sample):
    """
    Visualize the given samples and save them to the given path.
    """
    decoder_t, decoder_s = decoders
    encoder.eval()
    decoder_t.eval()
    for decoder in decoder_s:
        decoder.eval()

    samples = samples.to(DEVICE)
    num_samples = samples.size(0)
    encoded = encoder(samples)
    repeat_encoded = encoded.repeat(len(decoder_s)+1, 1, 1, 1)    # including target
    split_encoded = torch.split(repeat_encoded, num_samples, dim=0)
    recon_tgt = decoder_t(split_encoded[0])
    batches = split_encoded[1:]
    output = torch.cat([decoder_s[i](batches[i]) for i in range(len(decoder_s))], dim=3)
    images = torch.cat((samples, recon_tgt, output), dim=3)
    images = denorm(images.data.cpu())
    save_image(images, path_sample+'.jpg', nrow=1)


def to_source(src_net, decoder_s, batches):
    """
    Adapt the bathes to source domains and estimate logits via source networks.
    """
    output = torch.cat([src_net[i](decoder_s[i](batches[i])).unsqueeze(1) for i in
                        range(len(src_net))], 1)
    return output


def to_source_direct(src_net, images):
    """
    Estimate logits via source networks directly, without any adaptations.
    """
    output = torch.cat([src_net[i](images).unsqueeze(1) for i in range(len(src_net))], 1)
    return output


def denorm(x):
    """
    Convert tensor range from [-1, 1] to [0, 1].
    """
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def sample_images(target, loader, num_class, num_sample):
    """
    Sample one image for each class from the loader.
    """
    x, y, transform, logic = None, None, None, None

    if target in [DATA_MNIST, DATA_USPS]:
        x = loader.dataset.data
        y = loader.dataset.targets
        transform = loader.dataset.transform
        logic = lambda img: transform(Image.fromarray(np.array(img), mode='L'))

    elif target in [DATA_SVHN]:
        x = loader.dataset.data
        y = loader.dataset.labels
        transform = loader.dataset.transform
        logic = lambda img: transform(Image.fromarray(np.transpose(img, (1, 2, 0))))

    elif target in [DATA_MNISTM, DATA_SYNDIGITS]:
        x = loader.dataset.x
        y = loader.dataset.y
        transform = loader.dataset.tf
        logic = lambda img: transform(np.uint8(img))

    images = []
    for i in range(num_class):
        images.append(get_specific(x, y, logic, i, num_sample))
    images = torch.cat(images, dim=0)

    return images


def get_specific(x, y, logic, label, num_sample):
    """
    Get one image that corresponds to the label.
    """
    samples = []
    idx = (np.array(y) == label).squeeze().nonzero()[0]
    where = torch.randperm(idx.shape[0])[:num_sample]
    images = x[idx[where]]
    if num_sample == 1:
        images = np.expand_dims(images, axis=0)
    for i in range(images.shape[0]):
        samples.append(logic(images[i]))
    samples = torch.stack(samples, dim=0)
    return samples
