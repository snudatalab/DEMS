"""
Unsupervised Multi-Source Domain Adaptation with No Observable Source Data
    - DEMS (Data-free Exploitation of Multiple Sources)

Authors:
    - Hyunsik Jeon (jeon185@snu.ac.kr)
    - Seongmin Lee (ligi214@snu.ac.kr)
    - U Kang (ukang@snu.ac.kr)

File: DEMS/src/translator/test.py
"""
import os
import time

from translator import loss as tran_loss
from utils.dataloader import main as loader
from utils.utils import *


def main(source, target, out_path, batch_size, epsilon, temperature, path_model):
    """
    Main function for testing adaptation functions.
    """
    # for model settings
    img_channels = 3
    first_conv_kernel = 32
    num_scales = 2
    last_conv_kernel = 32 * (2 ** num_scales)
    num_blocks = 3

    # initialize networks
    src_net = list(map(lambda x: load_classifier(x).to(DEVICE), source))
    encoder = init_encoder(in_channels=img_channels, out_channels=first_conv_kernel,
                           num_down=num_scales, num_blocks=num_blocks).to(DEVICE)
    decoder_t = init_decoder(in_channels=last_conv_kernel, out_channels=img_channels,
                             num_up=num_scales, num_blocks=num_blocks).to(DEVICE)
    decoder_s = []
    for i in range(len(src_net)):
        decoder_s.append(init_decoder(in_channels=last_conv_kernel, out_channels=img_channels,
                                      num_up=num_scales, num_blocks=num_blocks).to(DEVICE))
    decoders = (decoder_t, decoder_s)
    emb = init_domain_weight(num_domain=len(src_net)+1, dim=10, temperature=temperature).to(DEVICE)

    # load models
    checkpoint = torch.load(path_model)
    emb.load_state_dict(checkpoint['emb'])
    encoder.load_state_dict(checkpoint['encoder'])
    decoder_t.load_state_dict(checkpoint['decoder_t'])
    for i in range(len(decoder_s)):
        decoder_s[i].load_state_dict(checkpoint[f'decoder_s{i}'])

    # define data loaders
    _, _, test_loader = loader(target, batch_size)

    # get sample images for visualization
    test_samples = sample_images(target, test_loader, num_class=10, num_sample=1)

    # define losses
    loss1 = tran_loss.ReconstructionLoss(method='l1').to(DEVICE)
    loss2 = tran_loss.ConsistencyLoss().to(DEVICE)
    loss3 = tran_loss.InstanceEntropyLoss().to(DEVICE)
    loss4 = tran_loss.BatchEntropyLoss().to(DEVICE)
    loss5 = tran_loss.SupervisedLoss().to(DEVICE)
    losses = (loss1, loss2, loss3, loss4, loss5)

    # test the given models
    path_sample = os.path.join(out_path, 'test_samples')
    t1 = time.time()
    visualize_samples(encoder, decoders, test_samples, path_sample)
    test_loss1, test_loss2, test_loss3, test_loss4, test_loss5, test_floss, test_acc = inference(
        src_net, encoder, decoders, emb, test_loader, losses, epsilon)
    duration = time.time() - t1
    print(f'TestResults: [{duration:5.2f}s], TestLoss:{test_loss1:.4f}/{test_loss2:.4f}/{test_loss3:.4f}/{test_loss4:.4f}/{test_loss5:.4f}/{test_floss:.4f}, TestAcc:{test_acc:.4f}')


def inference(src_net, encoder, decoders, emb, loader, losses, epsilon):
    """
    Infer adaptation functions.
    """
    decoder_t, decoder_s = decoders
    reconstruction, consistency, instance_entropy, batch_entropy, supervised = losses
    epoch_reconstruction_loss, epoch_consistency_loss,\
    epoch_instance_entropy_loss, epoch_batch_instance_loss = 0., 0., 0., 0.
    epoch_supervised_loss, epoch_full_loss = 0., 0.
    num_total, total_hit = 0, 0
    num_src = len(src_net)

    # evaluation modes of networks
    for src in src_net:
        src.eval()
    emb.eval()
    encoder.eval()
    decoder_t.eval()
    for decoder in decoder_s:
        decoder.eval()

    # forward and backward process
    for i, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        num_batch = images.size(0)

        # forward process
        encoded = encoder(images)
        repeat_encoded = encoded.repeat(num_src+1, 1, 1, 1) # including target
        split_encoded = torch.split(repeat_encoded, num_batch, dim=0)
        recon_tgt = decoder_t(split_encoded[0])
        output_src = to_source(src_net, decoder_s, split_encoded[1:])

        # reconstruction
        reconstruction_loss = reconstruction(recon_tgt, images)

        # consistency loss
        consistency_loss = consistency(output_src, emb)

        # instance entropy loss
        instance_entropy_loss, instance_H = instance_entropy(output_src)

        # batch entropy loss
        batch_entropy_loss, batch_H = batch_entropy(output_src)

        # supervised loss
        tgt_idx = torch.tensor([0]).long().to(DEVICE)
        src_idx = torch.arange(num_src+1)[1:].long().to(DEVICE)
        domain_weight = emb(tgt_idx, src_idx).unsqueeze(0).unsqueeze(2)
        pseudo_logits = to_source_direct(src_net, images)
        preds = torch.softmax(pseudo_logits, dim=2)
        preds = weighted_averge(preds, domain_weight)
        pseudo_idx = (preds.max(1)[0] > epsilon)
        pseudo_label = preds.max(1)[1][pseudo_idx].detach()
        weighted_pseudo_output = (domain_weight * output_src[pseudo_idx]).sum(1)
        supervised_loss = supervised(weighted_pseudo_output, pseudo_label) if pseudo_label.size(0) > 0\
            else torch.tensor([0.]).to(DEVICE)   # for pseudo label

        # total loss
        loss = reconstruction_loss + instance_entropy_loss + batch_entropy_loss + 0.1*consistency_loss + supervised_loss

        # for epoch-wise loss evaluation
        epoch_reconstruction_loss += reconstruction_loss.item()
        epoch_consistency_loss += consistency_loss.item()
        epoch_instance_entropy_loss += instance_entropy_loss.item()
        epoch_batch_instance_loss += batch_entropy_loss.item()
        epoch_supervised_loss += supervised_loss.item()
        epoch_full_loss += loss.item()

        # for epoch-wise accuracy
        pred_prob, pred_label = weigted_average_preds(output_src, domain_weight)
        total_hit += pred_label.eq(labels.view_as(pred_label)).sum()
        num_total += labels.shape[0]

    recon_loss = epoch_reconstruction_loss/(i+1)
    cons_loss = epoch_consistency_loss/(i+1)
    inst_ent_loss = epoch_instance_entropy_loss/(i+1)
    batch_ent_loss = epoch_batch_instance_loss/(i+1)
    sup_loss = epoch_supervised_loss/(i+1)    # for real label
    f_loss = epoch_full_loss/(i+1)
    acc = float(total_hit)/num_total

    return recon_loss, cons_loss, inst_ent_loss, batch_ent_loss, sup_loss, f_loss, acc


def weigted_average_preds(output_src, weight):
    """
    Evaluate a weighted average of predictions and estimate the probabilities and labels.
    """
    preds = torch.softmax(output_src, dim=2)
    preds = weighted_averge(preds, weight)
    prob, label = torch.max(preds, dim=1)
    return prob, label


def weighted_averge(input, weight):
    """
    Evaluate a weighted average of predictions.
    """
    return (input*weight).sum(dim=1)
