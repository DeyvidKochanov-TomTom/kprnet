import torch
import torch.nn as nn
from torch.nn import functional as F


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_index=-1, thresh=0.6, min_kept=350000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thresh
        self.min_kept = max(1, min_kept)
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_index, reduction="none"
        )

    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode="bilinear")
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_index

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_index] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()
