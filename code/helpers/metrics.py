import numpy as np
from scipy.interpolate import interp1d

class Evaluator(object):
    def reset(self):
        """ Reset internal variables between epochs (or validation runs) """
        raise NotImplementedError
    def add_batch(self, gt_image, pre_image, **kwargs):
        """ Add data from batch for matric accumulation """
        raise NotImplementedError
    def compute_stats(self, **kwargs):
        """ Compute/print metric and return main matric valie """
        raise NotImplementedError

class SegmEvaluator(Evaluator):
    def __init__(self, cfg, **kwargs):
        self.num_class = cfg.MODEL.NUM_CLASSES
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image_t, output):
        gt_image = gt_image_t.cpu().numpy()
        pred = output.logits.cpu().numpy()
        pre_image = np.argmax(pred, axis=-1)
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def compute_stats(self, **kwargs):
        Acc = self.Pixel_Accuracy()
        Acc_class = self.Pixel_Accuracy_Class()
        mIoU = self.Mean_Intersection_over_Union()
        FWIoU = self.Frequency_Weighted_Intersection_over_Union()

        printflag = kwargs.get("printflag", None)
        if printflag is not None: 
            epoch = kwargs.get("epoch", 0)
            print(f"EVAL_METRIC: Acc={Acc:0.4f}, Acc_class={Acc_class:0.4f}, mIoU={mIoU:0.4f}, fwIoU={FWIoU:0.4f}")
        return mIoU
