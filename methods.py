from torch import nn
from torch.nn.functional import one_hot
import torch
from torch.nn import Softmax


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')

class NonDecisiveCrossEntropyLoss(nn.Module):
    def __init__(self, c=.0, w=1., n_classes=2):
        super(NonDecisiveCrossEntropyLoss, self).__init__()
        self.c = c
        self.w = w
        self.n_classes = n_classes
        
    def __str__(self):
        return f'NonDecisiveCrossEntropyLoss(c={self.c}, w={self.w})'

    def forward(self, output, target):
        # Softmax
        softmax_test = Softmax(dim=1)
        soft_output = softmax_test(output)
        oh_target = one_hot(target, self.n_classes)    

        # Cross entropy
        a = oh_target
        b = torch.log(soft_output)    
        
        # Table correction
        apred = target
        bpred = soft_output.argmax(1)
        mask = apred == bpred      
        
        c_loss = -((a[mask]*b[mask]).sum())/mask.sum()
        w_loss = -(a[~mask]*b[~mask]).sum()/(~mask).sum()

        loss = c_loss*self.c + w_loss * self.w

        return loss