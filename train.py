import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from trainer import trainer_FAF
from FAF_Model.FAF import FAF


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default=r'./FAF-1M', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='MFIF', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_FAF', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=10, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=2, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--DDP', action='store_true', default=False, help='DistributedDataParallel')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.000001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=2024, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=0, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--input_channel', type=int,
                    default=6, help='input image channel, for single image input should be 3, for two image input should be 6')
parser.add_argument('--resume', type=str,
                    default=r'./model/FAF-Finetune.pth', help='resume checkpoint')
parser.add_argument('--is_pretrain', type=bool,
                    default=False, help='whether use pretrain model')
parser.add_argument('--freeze', type=bool,
                    default=False, help='whether use pretrain model')
args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'MFIF': {
            'root_path': args.root_path,
            'list_dir': None,
            'num_classes': 1,
    }}
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "./model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # DDP
    if args.DDP:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda",args.local_rank)
        torch.distributed.init_process_group(backend='gloo', init_method="env://")
        net = FAF().to(device)
    else:
        net = FAF().cuda()

    if args.resume:
        ckp = torch.load(args.resume,weights_only=True)
        for k in list(ckp.keys()):
            if k.startswith('module.'):
                ckp[k[7:]] = ckp[k]
                del ckp[k]
        msg = net.load_state_dict(ckp,strict=True)
        print(msg)

    if args.freeze:
        pass
    for name, param in net.named_parameters():
        if param.requires_grad:
            print("trainable:", name)

    if args.is_pretrain:
        trainer = {'MFIF': trainer_FAF, }
        trainer[dataset_name](args, net, snapshot_path)
    else:
        trainer = {'MFIF': trainer_FAF, }
        trainer[dataset_name](args, net, snapshot_path)