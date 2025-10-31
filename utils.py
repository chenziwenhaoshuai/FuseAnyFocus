import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        # target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list


class GALoss(nn.Module):
    """
    The Class of GALoss
    """

    def __init__(self):
        super(GALoss, self).__init__()
        self._smooth = 1

    def _dice_loss(self, predict, target):
        """
        Compute the dice loss of the prediction decision map and ground-truth label
        :param predict: tensor, the prediction decision map
        :param target: tensor, ground-truth label
        :return:
        """
        target = target.float()
        intersect = torch.sum(predict * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(predict * predict)
        loss = (2 * intersect + self._smooth) / (z_sum + y_sum + self._smooth)
        loss = 1 - loss
        return loss

    def _qg_soft(self, img1, img2, fuse, k):
        """
        Compute the Qg for the given two image and the fused image.
        The calculation of Qg is modified to the python version based on the
        matlab version from https://github.com/zhengliu6699/imageFusionMetrics
        :param img1: tensor, input image A
        :param img2: tensor, input image B
        :param fuse: tensor, fused image
        :param k: softening factor
        :return:
        """
        # 1) get the map
        img1_gray = img1
        img2_gray = img2
        buf = 0.000001
        flt1 = torch.FloatTensor(np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1], ])).reshape((1, 1, 3, 3)).cuda(img1.device)
        flt2 = torch.FloatTensor(np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1], ])).reshape((1, 1, 3, 3)).cuda(img1.device)
        fuseX = F.conv2d(fuse, flt1, padding=1) + buf
        fuseY = F.conv2d(fuse, flt2, padding=1)
        fuseG = torch.sqrt(torch.mul(fuseX, fuseX) + torch.mul(fuseY, fuseY))
        buffer = (fuseX == 0)
        buffer = buffer.float()
        buffer = buffer * buf
        fuseX = fuseX + buffer
        fuseA = torch.atan(torch.div(fuseY, fuseX))

        img1X = F.conv2d(img1_gray, flt1, padding=1)
        img1Y = F.conv2d(img1_gray, flt2, padding=1)
        img1G = torch.sqrt(torch.mul(img1X, img1X) + torch.mul(img1Y, img1Y))
        buffer = (img1X == 0)
        buffer = buffer.float()
        buffer = buffer * buf
        img1X = img1X + buffer
        img1A = torch.atan(torch.div(img1Y, img1X))

        img2X = F.conv2d(img2_gray, flt1, padding=1)
        img2Y = F.conv2d(img2_gray, flt2, padding=1)
        img2G = torch.sqrt(torch.mul(img2X, img2X) + torch.mul(img2Y, img2Y))
        buffer = (img2X == 0)
        buffer = buffer.float()
        buffer = buffer * buf
        img2X = img2X + buffer
        img2A = torch.atan(torch.div(img2Y, img2X))

        # 2) edge preservation estimation

        buffer = (img1G == 0)
        buffer = buffer.float()
        buffer = buffer * buf
        img1G = img1G + buffer
        buffer1 = torch.div(fuseG, img1G)

        buffer = (fuseG == 0)
        buffer = buffer.float()
        buffer = buffer * buf
        fuseG = fuseG + buffer
        buffer2 = torch.torch.div(img1G, fuseG)

        bimap = torch.sigmoid(-k * (img1G - fuseG))
        bimap_1 = torch.sigmoid(k * (img1A - fuseA))
        Gaf = torch.mul(bimap, buffer2) + torch.mul((1 - bimap), buffer1)
        Aaf = torch.abs(torch.abs(img1A - fuseA) - np.pi / 2) * 2 / np.pi

        # -------------------
        buffer = (img2G == 0)
        buffer = buffer.float()
        buffer = buffer * buf
        img2G = img2G + buffer
        buffer1 = torch.div(fuseG, img2G)

        buffer = (fuseG == 0)
        buffer = buffer.float()
        buffer = buffer * buf
        fuseG = fuseG + buffer
        buffer2 = torch.div(img2G, fuseG)

        # bimap = torch.sigmoid(-k * (img2G-fuseG))
        bimap = torch.sigmoid(-k * (img2G - fuseG))
        bimap_2 = torch.sigmoid(k * (img2A - fuseA))
        Gbf = torch.mul(bimap, buffer2) + torch.mul((1 - bimap), buffer1)
        Abf = torch.abs(torch.abs(img2A - fuseA) - np.pi / 2) * 2 / np.pi

        # some parameter
        gama1 = 1
        gama2 = 1
        k1 = -10
        k2 = -20
        delta1 = 0.5
        delta2 = 0.75

        Qg_AF = torch.div(gama1, (1 + torch.exp(k1 * (Gaf - delta1))))
        Qalpha_AF = torch.div(gama2, (1 + torch.exp(k2 * (Aaf - delta2))))
        Qaf = torch.mul(Qg_AF, Qalpha_AF)

        Qg_BF = torch.div(gama1, (1 + torch.exp(k1 * (Gbf - delta1))))
        Qalpha_BF = torch.div(gama2, (1 + torch.exp(k2 * (Abf - delta2))))
        Qbf = torch.mul(Qg_BF, Qalpha_BF)

        # 3) compute the weighting matrix
        L = 1
        Wa = torch.pow(img1G, L)
        Wb = torch.pow(img2G, L)
        res = torch.mean(torch.div(torch.mul(Qaf, Wa) + torch.mul(Qbf, Wb), (Wa + Wb)))

        return res

    def forward(self, img1, img2, img_clear, mask, mask_BGF, gt_mask, k=10e4):
        """
        Compute the GALoss
        :param img1: tensor, input image A
        :param img2: tensor, input image B
        :param mask: tensor, the prediction decision map without bounary guider filter
        :param mask_BGF: tensor, the prediction decision map with bounary guider filter
        :param gt_mask: tensor, the ground-truth decision map
        :param k: the softening factor of loss_qg
        :return:
        """
        fused = torch.mul(mask_BGF, img1) + torch.mul((1 - mask_BGF), img2)
        loss_qg = 1 - self._qg_soft(img1, img2, fused, k)
        loss_dice = self._dice_loss(mask, gt_mask)

        return loss_dice + 0.1*loss_qg, loss_dice, loss_qg