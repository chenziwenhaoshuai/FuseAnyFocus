import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .dinov2 import DINOv2
from .util.blocks import FeatureFusionBlock, _make_scratch
from torch.autograd import Variable

def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )



class FAFHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False
    ):
        super(FAFHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.FPEM = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            # nn.ReLU(True),
            nn.Identity(),
        )
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.FPEM(out)
        
        return out

class GaussBlur(nn.Module):
    """
     Gaussian blurring
    """
    def __init__(self, sigma, filter_size):
        super(GaussBlur, self).__init__()
        self.radius = filter_size
        sigma2 = sigma ** 2
        sum_val = 0
        x = torch.tensor(np.arange(-self.radius, self.radius + 1), dtype=torch.float).expand(1, 2 * self.radius + 1)
        y = x.t().expand(2 * self.radius + 1, 2 * self.radius + 1)
        x = x.expand(2 * self.radius + 1, 2 * self.radius + 1)
        self.kernel = torch.exp(-(torch.mul(x, x) + torch.mul(y, y)) / (2 * sigma2))
        self.kernel = self.kernel / torch.sum(self.kernel)
        self.weight = self.kernel.expand(1, 1, 2 * self.radius + 1, 2 * self.radius + 1)

    def forward(self, data):
        _, c, _, _ = data.shape
        self.weight = self.weight.expand(c, 1, 9, 9)
        if str(self.weight.device) != str(data.device):
            self.weight = self.weight.to(data.device)
        blurred = F.conv2d(data, self.weight, padding=[self.radius], groups=c)
        return blurred
class BoxFilter(nn.Module):
    """
    The BoxFilter is copied from https://github.com/wuhuikai/DeepGuidedFilter.git
    """
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def forward(self, x):
        assert x.dim() == 4

        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)


def diff_x(input, r):
    assert input.dim() == 4

    left = input[:, :, r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:] - input[:, :, :-2 * r - 1]
    right = input[:, :, -1:] - input[:, :, -2 * r - 1: -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output


def diff_y(input, r):
    assert input.dim() == 4

    left = input[:, :, :, r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:] - input[:, :, :, :-2 * r - 1]
    right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1: -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output
class GuidedFilter(nn.Module):
    """
    The GuidedFilter is copied from https://github.com/wuhuikai/DeepGuidedFilter.git
    """
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.box_filter = BoxFilter(r)

    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()
        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        # N
        N = self.box_filter(Variable(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)))

        # mean_x
        mean_x = self.box_filter(x) / N
        # mean_y
        mean_y = self.box_filter(y) / N
        # cov_xy
        cov_xy = self.box_filter(x * y) / N - mean_x * mean_y
        # var_x
        var_x = self.box_filter(x * x) / N - mean_x * mean_x

        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x

        # mean_A; mean_b
        mean_A = self.box_filter(A) / N
        mean_b = self.box_filter(b) / N

        return mean_A * x + mean_b
class FAF(nn.Module):
    def __init__(
        self, 
        encoder='vits',
        features=64,
        out_channels=[48, 96, 192, 384],
        use_bn=False, 
        use_clstoken=False,
        in_chans=6
    ):
        super(FAF, self).__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder, in_chans=in_chans)
        self.FAF_head = FAFHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
        self.gaussian = GaussBlur(8, 4)
        self.guided_filter = GuidedFilter(3, 0.1)
    
    def forward(self, x):

        img1 = x[:, :3, :, :]
        img2 = x[:, 3:, :, :]
        x = F.interpolate(x, (518, 518), mode="bilinear", align_corners=True)  # 224 for pretrain 518 for finetune
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        logits = self.FAF_head(features, patch_h, patch_w)
        logits = F.interpolate(logits, (512, 512), mode="bilinear", align_corners=True)# 224 for pretrain 512 for finetune
        output_origin, output_bgf = self.BRM(logits, img1, img2)

        return logits, output_origin, output_bgf

    def BRM(self, DoF_logits, img1, img2):
        # Boundary Refinement Module
        output_origin = torch.sigmoid(1000 * DoF_logits)
        output_blur = self.gaussian(output_origin)
        zeros = torch.zeros_like(output_blur)
        ones = torch.ones_like(output_blur)
        half = ones / 2
        mask_1 = torch.where(output_blur > 0.8, ones, zeros)
        mask_2 = torch.where(output_blur < 0.1, ones, zeros)
        mask_3 = mask_1 * output_blur + mask_2 * (1 - output_blur)
        boundary_map = 1 - torch.abs(2 * (output_blur * mask_3 + (1 - mask_3) * half) - 1)
        output_origin = output_origin.repeat(1, 3, 1, 1)
        temp_fused = img1 * output_origin + (1 - output_origin) * img2
        output_gf = self.guided_filter(temp_fused, output_origin)
        output_bgf = output_gf * boundary_map + output_origin * (1 - boundary_map)
        output_origin = output_origin[:, 0, :, :].unsqueeze(1)
        output_bgf = output_bgf[:, 0, :, :].unsqueeze(1)
        output_origin = F.interpolate(output_origin, (512, 512), mode="bilinear", align_corners=True)
        output_bgf = F.interpolate(output_bgf, (512, 512), mode="bilinear", align_corners=True)
        return output_origin, output_bgf