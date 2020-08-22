import numpy as np
from sklearn.metrics import confusion_matrix


class Eval:
    def __init__(self, n_classes: int, ignore_class: int):
        self.confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int32)
        self.ignore_class = ignore_class
        self.n_classes = n_classes

    def update(self, predictions, labels):
        labels_flat = labels.reshape(-1)
        predictions_flat = predictions.reshape(-1)
        predictions_flat = predictions_flat[labels_flat < self.ignore_class]
        labels_flat = labels_flat[labels_flat < self.ignore_class]

        self.confusion_matrix += confusion_matrix(
            labels_flat, predictions_flat, labels=np.arange(self.n_classes)
        )

    def getIoU(self):
        conf = np.copy(self.confusion_matrix).astype(np.float64)
        tp = np.diag(conf)
        fp = conf.sum(axis=1) - tp
        fn = conf.sum(axis=0) - tp

        ious = tp / (tp + fp + fn + 1e-5)

        return ious.mean(), ious
