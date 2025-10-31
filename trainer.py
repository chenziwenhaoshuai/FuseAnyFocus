import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, GALoss
import time
from torch.utils.data.distributed import DistributedSampler


def trainer_FAF(args, model, snapshot_path):
    from datasets.dataset_FAF import MFIF_dataset_finetune_aug
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    db_train = MFIF_dataset_finetune_aug(base_dir=args.root_path, split="train")
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    if args.n_gpu > 1 and args.DDP:
        train_sampler = DistributedSampler(db_train)
        trainloader = DataLoader(db_train, batch_size=batch_size, sampler=train_sampler, num_workers=0, pin_memory=True)
    else:
        trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    if args.n_gpu > 1 and args.DDP:
        model = nn.parallel.DistributedDataParallel(model)#, device_ids=[0, 1, 2, 3], output_device=args.local_rank)
    elif args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss(num_classes)
    galoss = GALoss()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        if args.n_gpu >1 and args.DDP:
            train_sampler.set_epoch(epoch_num)
        for i_batch, sampled_batch in enumerate(trainloader):
            start = time.time()
            image_A, image_B, image_batch, label_batch, image_C = sampled_batch['image_A'], sampled_batch['image_B'], sampled_batch['image'], sampled_batch['label'], sampled_batch['image_C']
            image_A, image_B, image_batch, label_batch, image_C = image_A.cuda(), image_B.cuda(), image_batch.cuda(), label_batch.cuda(), image_C.cuda()
            logits, output_origin, output_bgf = model(image_batch)
            # rgb2gary
            image_A_gary = 0.299 * image_A[:, 0, :, :] + 0.587 * image_A[:, 1, :, :] + 0.114 * image_A[:, 2, :, :]
            image_B_gary = 0.299 * image_B[:, 0, :, :] + 0.587 * image_B[:, 1, :, :] + 0.114 * image_B[:, 2, :, :]
            image_C_gary = 0.299 * image_C[:, 0, :, :] + 0.587 * image_C[:, 1, :, :] + 0.114 * image_C[:, 2, :, :]
            image_A_gary = image_A_gary.unsqueeze(1)
            image_B_gary = image_B_gary.unsqueeze(1)
            image_C_gary = image_C_gary.unsqueeze(1)
            loss, dice, qg = galoss(image_A_gary, image_B_gary, image_C_gary, output_origin, output_bgf, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_qg', qg, iter_num)
            writer.add_scalar('info/loss_dice', dice, iter_num)
            end = time.time()
            logging.info('Epcoch:%d, iteration %d : loss : %f, loss_qg: %f, loss_dice: %f, Time:%f' % (epoch_num, iter_num, loss.item(), qg.item(), dice.item(), end - start))
            if iter_num % 50 == 0:
                image = image_batch[0, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs_bgf = output_bgf[0, 0:1, :, :]# * 255
                writer.add_image('train/Prediction_bgf', outputs_bgf, iter_num)
                outputs_log = logits[0, 0:1, :, :]
                outputs_log = (outputs_log - outputs_log.min()) / (outputs_log.max() - outputs_log.min())
                writer.add_image('train/Prediction_logits', outputs_log, iter_num)
                labs = label_batch[0, ...] * 255
                writer.add_image('train/GroundTruth', labs, iter_num)
                output_ori = output_origin[0, 0:1, :, :]
                writer.add_image('train/Prediction_ori', output_ori, iter_num)
            if iter_num % 50 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"