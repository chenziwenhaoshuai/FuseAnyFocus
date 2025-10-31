import argparse
import os
import numpy as np
import torch
import cv2
from torchvision import transforms
from FAF_Model.FAF import FAF

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default=r'G:\project\Fuse_any_focus\get_MFIF_dataset\MFIF-imagenet-depth2_overfit', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='MFIF', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=3, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=0, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--input_channel', type=int,
                    default=6, help='input image channel, for single image input should be 3, for two image input should be 6')
args = parser.parse_args()



if __name__ == '__main__':


    net = FAF().cuda()
    ckp = torch.load('./model/FAF-Finetune.pth')
    # del module.
    for key in list(ckp.keys()):
        if 'module' in key:
            ckp[key.replace('module.', '')] = ckp[key]
            del ckp[key]
    net.load_state_dict(ckp)
    print('load model')

    img_dir = r'./FAF-1M/FAF-1M-val'
    new_round_dir = r'./FAF-1M/FAF-1M-val/results'
    if not os.path.exists(new_round_dir):
        os.makedirs(new_round_dir)
    img_list = os.listdir(img_dir+'/image_A')
    for img_name in img_list:
        img_num = img_name.split('.')[0]
        img_A_name = 'image_A/'+img_name
        img_B_name = 'image_B/'+img_name

        img_A_ori = cv2.imread(os.path.join(img_dir, img_A_name))
        img_B_ori = cv2.imread(os.path.join(img_dir, img_B_name))
        img_shape = img_A_ori.shape
        resize = (512,512)
        try:
            if resize is not None:
                img_A = cv2.resize(img_A_ori, resize)
                img_B = cv2.resize(img_B_ori, resize)
            else:
                img_A = img_A_ori
                img_B = img_B_ori
        except:
            print(img_A_name)
            print(img_B_name)
        img = torch.cat([transforms.ToTensor()(img_A), transforms.ToTensor()(img_B)], dim=0).unsqueeze(0).cuda()
        net.eval()
        with torch.no_grad():
            logits, mask, mask_BGF = net(img)
            fused_mask = mask_BGF.cpu().numpy()
            mask = mask.cpu().numpy()
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            fused_mask = cv2.resize(fused_mask[0, 0, :, :], (img_shape[1], img_shape[0]))
            mask = cv2.resize(mask[0, 0, :, :], (img_shape[1], img_shape[0]))
            cv2.imwrite(os.path.join(new_round_dir, img_num+'_DM.jpg'), mask*255)
            fake_label = torch.tensor(mask).unsqueeze(0)
            fused_mask = np.repeat(fused_mask[:, :, np.newaxis], 3, axis=2)
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            fused_image = img_A_ori * fused_mask + img_B_ori * (1 - fused_mask)
            cv2.imwrite(os.path.join(new_round_dir, img_num+'.jpg'), fused_image)
