"""
Unsupervised Multi-Source Domain Adaptation with No Observable Source Data
    - DEMS (Data-free Exploitation of Multiple Sources)

Authors:
    - Hyunsik Jeon (jeon185@snu.ac.kr)
    - Seongmin Lee (ligi214@snu.ac.kr)
    - U Kang (ukang@snu.ac.kr)

File: DEMS/src/translator/loss.py
"""
import torch.nn.functional as F

from utils.utils import *


class ConsistencyLoss(nn.Module):
    """
    Label consistency loss.
    """
    def __init__(self, num_select=2):
        """
        Class initializer.
        """
        super(ConsistencyLoss, self).__init__()
        self.num_select = num_select

    def forward(self, prob, emb):
        """
        Forward propagation.
        """
        dl = 0.
        count = 0
        num_src = prob.shape[1]
        src_idx = torch.arange(num_src+1)[1:].long().to(DEVICE)
        for i in range(prob.shape[1]-1):
            i_selected = torch.index_select(src_idx, 0, torch.tensor([i]).to(DEVICE))
            before_i = src_idx[:i]
            after_i = src_idx[i+1:]
            remains = torch.cat((before_i, after_i))
            weights = emb(i_selected, remains)
            for j in range(i+1, prob.shape[1]):
                r_j = src_idx[j]
                where = ((remains == r_j).nonzero()).squeeze()
                weight = weights[where]
                dl += weight * self.jensen_shanon(prob[:, i, :], prob[:, j, :], dim=1)
                count += 1
        return dl/count

    @staticmethod
    def jensen_shanon(pred1, pred2, dim):
        """
        Jensen-Shannon Divergence.
        """
        m = (torch.softmax(pred1, dim=dim) + torch.softmax(pred2, dim=dim)) / 2
        pred1 = F.log_softmax(pred1, dim=dim)
        pred2 = F.log_softmax(pred2, dim=dim)
        return (F.kl_div(pred1, m.detach(), reduction='batchmean') + F.kl_div(pred2, m.detach(), reduction='batchmean')) / 2


class BatchEntropyLoss(nn.Module):
    """
    Batch-entropy loss.
    """
    def __init__(self):
        """
        Class initializer.
        """
        super(BatchEntropyLoss, self).__init__()

    def forward(self, prob):
        """
        Forward propagation.
        """
        batch_entropy = F.softmax(prob, dim=2).mean(dim=0)
        batch_entropy = batch_entropy * (-batch_entropy.log())
        batch_entropy = -batch_entropy.sum(dim=1)
        loss = batch_entropy.mean()
        return loss, batch_entropy


class InstanceEntropyLoss(nn.Module):
    """
    Instance-entropy loss.
    """
    def __init__(self):
        """
        Class initializer.
        """
        super(InstanceEntropyLoss, self).__init__()

    def forward(self, prob):
        """
        Forward propagation.
        """
        instance_entropy = F.softmax(prob, dim=2) * F.log_softmax(prob, dim=2)
        instance_entropy = -1.0 * instance_entropy.sum(dim=2)
        instance_entropy = instance_entropy.mean(dim=0)
        loss = instance_entropy.mean()
        return loss, instance_entropy


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss.
    """
    def __init__(self, method):
        """
        Class initializer.
        """
        super(ReconstructionLoss, self).__init__()
        if method == 'l1':
            self.loss = nn.L1Loss()
        elif method == 'l2':
            self.loss = nn.MSELoss()

    def forward(self, output, target):
        """
        Forward propagation.
        """
        return self.loss(output, target)


class SupervisedLoss(nn.Module):
    """
    Supervised loss with pseudo labels.
    """
    def __init__(self):
        """
        Class initializer.
        """
        super(SupervisedLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, prob, label):
        """
        Forward propagation.
        """
        return self.loss(prob, label)