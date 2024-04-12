import math
import torch
from torch import nn


def get_loss(name):
    if name == "cosface":
        return CosFace()
    elif name == "arcface":
        return ArcFace()
    elif name == 'adaface':
        return AdaFace()
    else:
        raise ValueError()

class AdaFace(torch.nn.Module):
    def __init__(self,
                 embedding_size=512,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1.0,
                 ):
        super(AdaFace, self).__init__()

        # initial kernel
        # self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m 
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_std', torch.ones(1)*100)

        print('\n\AdaFace with the following property')
        print('self.m', self.m)
        print('self.h', self.h)
        print('self.s', self.s)
        print('self.t_alpha', self.t_alpha)

    def forward(self, cosine, norms, label):
        # print(label)
        index_positive = torch.where(label != -1)[0]
        # target_logits  = cosine[index_positive]
        # target_norms   = norms[index_positive]
        # target_labels  = label[index_positive]

        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)
        # g_angular
        m_arc = torch.zeros(index_positive.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label[index_positive].view(-1, 1), 1.0)
        g_angular = self.m * margin_scaler[index_positive] * -1
        m_arc = m_arc * g_angular
        cosine.acos_()
        cosine[index_positive] = torch.clip(cosine[index_positive] + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine.cos_()

        # g_additive
        
        m_cos = torch.zeros(index_positive.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label[index_positive].view(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler[index_positive])
        m_cos = m_cos * g_add
        cosine[index_positive] -= m_cos

        # scale
        ret = cosine * self.s
        return ret

class CosFace(nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine[index] -= m_hot
        ret = cosine * self.s
        return ret


class ArcFace(nn.Module):
    def __init__(self, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine: torch.Tensor, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine.acos_()
        cosine[index] += m_hot
        cosine.cos_().mul_(self.s)
        return cosine
