"""
Unsupervised Multi-Source Domain Adaptation with No Observable Source Data
    - DEMS (Data-free Exploitation of Multiple Sources)

Authors:
    - Hyunsik Jeon (jeon185@snu.ac.kr)
    - Seongmin Lee (ligi214@snu.ac.kr)
    - U Kang (ukang@snu.ac.kr)

File: DEMS/src/translator/train.py
"""
import os
import time

import torch.nn.functional as F
from torch import optim

from translator import loss as tran_loss
from utils.dataloader import main as loader
from utils.utils import *


def main(source, target, out_path, batch_size, epsilon, temperature, path_model, index=0):
    """
    Main function for training adaptation functions.
    """
    # set random seed
    set_seed(seed=3000+index)

    # for training settings
    num_epochs = 100
    full_step = 5
    lr = 1e-3
    decay = 1e-4

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
    emb_tmp = init_domain_weight(num_domain=len(src_net)+1, dim=10, temperature=temperature).to(DEVICE)
    emb = init_domain_weight(num_domain=len(src_net)+1, dim=10, temperature=temperature).to(DEVICE)

    # define data loaders
    trn_loader, val_loader, test_loader = loader(target, batch_size)

    # get sample images for visualization
    trn_samples = sample_images(target, trn_loader, num_class=10, num_sample=1)

    # define losses
    loss1 = tran_loss.ReconstructionLoss(method='l1').to(DEVICE)
    loss2 = tran_loss.ConsistencyLoss().to(DEVICE)
    loss3 = tran_loss.InstanceEntropyLoss().to(DEVICE)
    loss4 = tran_loss.BatchEntropyLoss().to(DEVICE)
    loss5 = tran_loss.SupervisedLoss().to(DEVICE)
    losses = (loss1, loss2, loss3, loss4, loss5)

    # initialize domain weight embeddings
    params = list(emb_tmp.parameters())
    dw_optimizer = optim.Adam(params, lr=1e-3, weight_decay=1e-2)
    pretrain_domain_weight(src_net, emb_tmp, trn_loader, dw_optimizer, (loss3, loss4), temperature)
    emb.emb.weight.data.copy_(emb_tmp.emb.weight.data)

    # define an optimizer
    params = list(emb.parameters()) + list(encoder.parameters()) + list(decoder_t.parameters())
    for decoder in decoder_s:
        params += list(decoder.parameters())
    optimizer = optim.Adam(params, lr=lr, weight_decay=decay)
    nn.utils.clip_grad_norm_(params, 5)

    # path for outputs
    os.makedirs(out_path, exist_ok=True)
    path_loss = os.path.join(out_path, 'loss.txt')
    with open(path_loss, 'w') as f:
        f.write('Epoch\tDur\t'
                'TrL1\tTrL2\tTrL3\tTrL4\tTrL5\tTrFL\tTrA\t'
                'VaL1\tVaL2\tVaL3\tVaL4\tVaL5\tVaFL\tVaA\t'
                'TeL1\tTeL2\tTeL3\tTeL4\tTeL5\tTeFL\tTeA\n')

    # train the networks
    min_val_loss = np.inf
    for epoch in range(1, num_epochs + 1):
        path_sample = os.path.join(out_path, f'{epoch:03}')
        t1 = time.time()
        trn_loss1, trn_loss2, trn_loss3, trn_loss4, trn_loss5, trn_floss, trn_acc = update(
            epoch, src_net, encoder, decoders, emb, trn_loader, optimizer, losses,
            train=True, full_step=full_step, epsilon=epsilon)
        visualize_samples(encoder, decoders, trn_samples, path_sample)
        val_loss1, val_loss2, val_loss3, val_loss4, val_loss5, val_floss, val_acc = update(
            epoch, src_net, encoder, decoders, emb, val_loader, None, losses,
            train=False, full_step=full_step, epsilon=epsilon)
        test_loss1, test_loss2, test_loss3, test_loss4, test_loss5, test_floss, test_acc = update(
            epoch, src_net, encoder, decoders, emb, test_loader, None, losses,
            train=False, full_step=full_step, epsilon=epsilon)
        duration = time.time() - t1
        print(f'Epoch:{epoch:3d} [{duration:5.2f}s], TrnLoss:{trn_loss1:.4f}/{trn_loss2:.4f}/{trn_loss3:.4f}/{trn_loss4:.4f}/{trn_loss5:.4f}/{trn_floss:.4f}, TrnAcc:{trn_acc:.4f}, '
              f'ValLoss:{val_loss1:.4f}/{val_loss2:.4f}/{val_loss3:.4f}/{val_loss4:.4f}/{val_loss5:.4f}/{val_floss:.4f}, ValAcc:{val_acc:.4f}, '
              f'TestLoss:{test_loss1:.4f}/{test_loss2:.4f}/{test_loss3:.4f}/{test_loss4:.4f}/{test_loss5:.4f}/{test_floss:.4f}, TestAcc:{test_acc:.4f}')
        with open(path_loss, 'a') as f:
            f.write(f'{epoch:3d}\t{duration:5.2f}\t'
                    f'{trn_loss1:.4f}\t{trn_loss2:.4f}\t{trn_loss3:.4f}\t{trn_loss4:.4f}\t{trn_loss5:.4f}\t{trn_floss:.4f}\t{trn_acc:.4f}\t'
                    f'{val_loss1:.4f}\t{val_loss2:.4f}\t{val_loss3:.4f}\t{val_loss4:.4f}\t{val_loss5:.4f}\t{val_floss:.4f}\t{val_acc:.4f}\t'
                    f'{test_loss1:.4f}\t{test_loss2:.4f}\t{test_loss3:.4f}\t{test_loss4:.4f}\t{test_loss5:.4f}\t{test_floss:.4f}\t{test_acc:.4f}\n')

        # save the adaptation models
        if val_floss < min_val_loss:
            min_val_loss = val_floss
            model_dict = {}
            model_dict['emb'] = emb.state_dict()
            model_dict['encoder'] = encoder.state_dict()
            model_dict['decoder_t'] = decoder_t.state_dict()
            for i in range(len(decoder_s)):
                model_dict[f'decoder_s{i}'] = decoder_s[i].state_dict()
            torch.save(model_dict, path_model)


def pretrain_domain_weight(src_net, emb, loader, optimizer, losses, temperature):
    """
    Pretrain the domain embedding factors.
    """
    num_epochs = 1000
    instance_entropy, batch_entropy = losses

    # evaluation modes of source networks
    for src in src_net:
        src.eval()

    # evaluate the logits
    logits = []
    for i, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)
        pseudo_logits = to_source_direct(src_net, images)
        bl, bh = batch_entropy(pseudo_logits)
        il, ih = instance_entropy(pseudo_logits)
        logits.append((1-bh-ih).detach().unsqueeze(0))
    logits = torch.cat(logits, dim=0).mean(dim=0)
    target_weight = torch.pow(logits, 1/temperature)
    target = (target_weight/target_weight.sum()).detach()

    # train the embedding factors
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        tgt_idx = torch.tensor([0]).long().to(DEVICE)
        src_idx = torch.arange(len(src_net)+1)[1:].long().to(DEVICE)
        pred = emb(tgt_idx, src_idx)
        loss = F.kl_div(pred, target, reduction='batchmean')
        loss.backward()
        optimizer.step()


def update(epoch, src_net, encoder, decoders, emb, loader, optimizer, losses,
           train, full_step, epsilon):
    """
    Update (or infer) adaptation functions for a single epoch.
    """
    decoder_t, decoder_s = decoders
    reconstruction, consistency, instance_entropy, batch_entropy, supervised = losses
    epoch_reconstruction_loss, epoch_consistency_loss,\
    epoch_instance_entropy_loss, epoch_batch_instance_loss = 0., 0., 0., 0.
    epoch_supervised_loss, epoch_full_loss = 0., 0.
    num_total, total_hit = 0, 0
    num_src = len(src_net)

    # initialize decoders
    if epoch == full_step + 1:
        for decoder in decoder_s:
            decoder.load_state_dict(decoder_t.state_dict())

    # evaluation (or training) modes of networks
    for src in src_net:
        src.eval()
    if train:
        emb.train()
        encoder.train()
        decoder_t.train()
        for decoder in decoder_s:
            decoder.train()
    else:
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

        if train:
            optimizer.zero_grad()

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

        # define the total loss
        init_loss = reconstruction_loss
        full_loss = reconstruction_loss + instance_entropy_loss + batch_entropy_loss + 0.1*consistency_loss + supervised_loss
        loss = init_loss if epoch <= full_step else full_loss

        # optimize the parameters
        if train:
            loss.backward()
            optimizer.step()

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
