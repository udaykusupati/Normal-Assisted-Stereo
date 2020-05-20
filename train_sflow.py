from models import MVDNet as MVDNet

import argparse
import time
import csv

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import custom_transforms
from utils import tensor2array, save_checkpoint, save_path_formatter, adjust_learning_rate
from loss_functions import compute_errors_train, compute_errors_test, compute_angles
from inverse_warp_ import check_depth

from logger import TermLogger, AverageMeter
from itertools import chain
from tensorboardX import SummaryWriter
from data_loader import SequenceFolder

import matplotlib.pyplot as plt
from scipy.misc import imsave
from path import Path
import os
from inverse_warp import inverse_warp
#from sync_batchnorm import convert_model

parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', dest='dataset', default='dataset', metavar='PATH',
                    help='dataset')
parser.add_argument('--dataset-format', default='sequential', metavar='STR',
                    help='dataset format, stacked: stacked frames (from original TensorFlow code) \
                    sequential: sequential folders (easier to convert to with a non KITTI/Cityscape dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--p-dps', dest='pretrained_dps', action='store_true',
                    help='finetune on pretrained_dps')
parser.add_argument('--pretrained-mvdn', dest='pretrained_mvdn', default=None, metavar='PATH',
                    help='path to pre-trained mvdnet model')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH',
                    help='csv where to save per-gradient descent train stats')
parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs and warped imgs at validation step')
parser.add_argument('--ttype', default='train.txt', type=str, help='Text file indicates input data')
parser.add_argument('--ttype2', default='test.txt', type=str, help='Text file indicates input data')
parser.add_argument('-f', '--training-output-freq', type=int, help='frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output',
                    metavar='N', default=100)
parser.add_argument('--nlabel', type=int ,default=64, help='number of label')
parser.add_argument('--mindepth', type=float ,default=0.5, help='minimum depth')

parser.add_argument('--exp', default='default', type=str, help='Experiment name')
parser.add_argument('-sv', dest='skip_v', action='store_true',
                    help='Skip validation')

parser.add_argument('-nw','--n-weight', type=float ,default=1, help='weight of nmap loss')
parser.add_argument('-dw','--d-weight', type=float ,default=1, help='weight of depth loss')
parser.add_argument('-np', dest='no_pool', action='store_true',
                    help='Use less pool layers in nnet')
parser.add_argument('--index', type=int, default=0, help='')
parser.add_argument('--output-dir', default='test_result', type=str, help='Output directory for saving predictions in a big 3D numpy file')
parser.add_argument('--output-print', action='store_true', help='print output depth')

parser.add_argument('--scale', type=float ,default=1, help='scale sceneflow')
parser.add_argument('--crop-h', type=int ,default=240, help='crop h sceneflow')
parser.add_argument('--crop-w', type=int ,default=320, help='crop w sceneflow')

n_iter = 0


def main():
    global n_iter
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    save_path = save_path_formatter(args, parser)
    args.save_path = 'checkpoints'/(args.exp+'_'+save_path)
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)

    training_writer = SummaryWriter(args.save_path)
    output_writers = []
    for i in range(3):
        output_writers.append(SummaryWriter(args.save_path/'valid'/str(i)))

    # Data loading code
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
    if args.dataset == 'sceneflow':
        train_transform = custom_transforms.Compose([
            custom_transforms.RandomCrop(scale = args.scale, h = args.crop_h, w = args.crop_w),
            custom_transforms.ArrayToTensor(),
            normalize
        ])
    else:
        train_transform = custom_transforms.Compose([
            custom_transforms.RandomScaleCrop(),
            custom_transforms.ArrayToTensor(),
            normalize
        ])

    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

    
    print("=> fetching scenes in '{}'".format(args.data))
    val_set = SequenceFolder(
        args.data,
        transform=valid_transform,
        seed=args.seed,
        ttype=args.ttype2,
        dataset = args.dataset,
	index = args.index
    )
    
    train_set = SequenceFolder(
        args.data,
        transform=train_transform,
        seed=args.seed,
        ttype=args.ttype,
        dataset = args.dataset
    )


    train_set.samples = train_set.samples[:len(train_set) - len(train_set)%args.batch_size]



    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")

    if args.dataset == 'sceneflow':
        args.mindepth = 5.45
    mvdnet = MVDNet(args.nlabel, args.mindepth)
    #mvdnet = convert_model(mvdnet)
    
    if args.pretrained_mvdn:
        print("=> using pre-trained weights for MVDNet")
        weights = torch.load(args.pretrained_mvdn)   
        mvdnet.init_weights()
        mvdnet.load_state_dict(weights['state_dict'])
    elif args.pretrained_dps:
        print("=> using pre-trained DPS weights for MVDNet")
        weights = torch.load('pretrained/dpsnet_updated.pth.tar')   
        mvdnet.init_weights()
        mvdnet.load_state_dict(weights['state_dict'], strict = False)
    else:
        mvdnet.init_weights()

    print('=> setting adam solver')

    optimizer = torch.optim.Adam(mvdnet.parameters(), args.lr,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)
    
    cudnn.benchmark = True
    mvdnet = torch.nn.DataParallel(mvdnet)
    mvdnet = mvdnet.cuda()

    print(' ==> setting log files')
    with open(args.save_path/args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_abs_rel', 'validation_abs_diff','validation_sq_rel', 'validation_a1', 'validation_a2', 'validation_a3', 'mean_angle_error'])

    with open(args.save_path/args.log_full, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss'])


    
    print(' ==> main Loop')
    for epoch in range(args.epochs):
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        if args.evaluate:
            train_loss = 0
        else:
            train_loss = train(args, train_loader, mvdnet, optimizer, args.epoch_size, training_writer, epoch)
        if not args.evaluate and (args.skip_v or (epoch+1)%3 != 0):
            error_names = ['abs_rel', 'abs_diff', 'sq_rel', 'a1', 'a2', 'a3', 'angle']
            errors = [0]*7
        else:
            errors, error_names = validate_with_gt(args, val_loader, mvdnet, epoch, output_writers)

        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))

        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, epoch)

        # Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)
        decisive_error = errors[0]
        with open(args.save_path/args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error, errors[1], errors[2], errors[3], errors[4], errors[5], errors[6]])
        if args.evaluate:
            break
        save_checkpoint(
            args.save_path, {
                'epoch': epoch + 1,
                'state_dict': mvdnet.module.state_dict()
            },
            epoch, file_prefixes = ['mvdnet'])
        


def train(args, train_loader, mvdnet, optimizer, epoch_size, train_writer, epoch):
    global n_iter
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    d_losses = AverageMeter(precision=4)
    nmap_losses = AverageMeter(precision=4)

    # switch to training mode
    mvdnet.train()

    print("Training")
    end = time.time()
    for i, (tgt_img, ref_imgs, gt_nmap, ref_poses, intrinsics, intrinsics_inv, tgt_depth) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img_var = Variable(tgt_img.cuda())
        ref_imgs_var = [Variable(img.cuda()) for img in ref_imgs]
        gt_nmap_var = Variable(gt_nmap.cuda())
        ref_poses_var = [Variable(pose.cuda()) for pose in ref_poses]
        intrinsics_var = Variable(intrinsics.cuda())
        intrinsics_inv_var = Variable(intrinsics_inv.cuda())
        tgt_depth_var = Variable(tgt_depth.cuda()).cuda()

        # compute output
        pose = torch.cat(ref_poses_var,1)

        if args.dataset == 'sceneflow':
            factor = (1.0/args.scale)*intrinsics_var[:,0,0]/1050.0
            factor = factor.view(-1,1,1)
        else:
            factor = torch.ones((tgt_depth_var.size(0),1,1)).type_as(tgt_depth_var)

        # get mask
        mask = (tgt_depth_var <= args.nlabel*args.mindepth*factor*3) & (tgt_depth_var >= args.mindepth*factor) & (tgt_depth_var == tgt_depth_var)
        mask.detach_()
        if mask.any() == 0:
            continue

        targetimg = inverse_warp(ref_imgs_var[0], tgt_depth_var.unsqueeze(1), pose[:,0], intrinsics_var, intrinsics_inv_var)#[B,CH,D,H,W,1]

        outputs = mvdnet(tgt_img_var, ref_imgs_var, pose, intrinsics_var, intrinsics_inv_var, no_pool = args.no_pool, factor = factor.unsqueeze(1))
        
        nmap = outputs[2].permute(0,3,1,2)
        depths = outputs[0:2]


        disps = [args.mindepth*args.nlabel/(depth) for depth in depths] # correct disps
        if args.dataset == 'sceneflow':
            disps = [(args.mindepth*args.nlabel)*3/(depth) for depth in depths] # correct disps
            depths = [(args.mindepth*factor)*(args.nlabel*3)/disp for disp in disps]
        loss = 0.
        d_loss = 0.
        nmap_loss = 0.
        if args.dataset == 'sceneflow':
            tgt_disp_var = ((1.0/args.scale)*intrinsics_var[:,0,0].view(-1,1,1)/tgt_depth_var)
            for l, disp in enumerate(disps):
                output = torch.squeeze(disp,1)
                d_loss = d_loss + F.smooth_l1_loss(output[mask], tgt_disp_var[mask]) * pow(0.7, len(disps)-l-1)
        else:
            for l, depth in enumerate(depths):
                output = torch.squeeze(depth,1)
                d_loss = d_loss + F.smooth_l1_loss(output[mask], tgt_depth_var[mask]) * pow(0.7, len(depths)-l-1)
        
        n_mask = mask.unsqueeze(1).expand(-1,3,-1,-1)
        nmap_loss = nmap_loss + F.smooth_l1_loss(nmap[n_mask], gt_nmap_var[n_mask])

        loss = loss + args.d_weight*d_loss + args.n_weight*nmap_loss

        if i > 0 and n_iter % args.print_freq == 0:
            train_writer.add_scalar('total_loss', loss.item(), n_iter)
        # record loss and EPE
        losses.update(loss.item(), args.batch_size)
        d_losses.update(d_loss.item(), args.batch_size)
        nmap_losses.update(nmap_loss.item(), args.batch_size)
        
        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.item()])
        if i % args.print_freq == 0:
            print('Train: Time {} Data {} Loss {} NmapLoss {} DLoss {} Iter {}/{} Epoch {}/{}'.format(batch_time, data_time, losses, nmap_losses, d_losses, i, len(train_loader), epoch, args.epochs))
        if i >= epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0]


def validate_with_gt(args, val_loader, mvdnet, epoch, output_writers=[]):
    batch_time = AverageMeter()
    error_names = ['abs_rel', 'abs_diff', 'sq_rel', 'a1', 'a2', 'a3', 'mean_angle']
    test_error_names = ['abs_rel','abs_diff','sq_rel','rms','log_rms','a1','a2','a3', 'mean_angle']
    errors = AverageMeter(i=len(error_names))
    test_errors = AverageMeter(i=len(test_error_names))
    log_outputs = len(output_writers) > 0

    output_dir= Path(args.output_dir)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # switch to evaluate mode
    mvdnet.eval()

    end = time.time()
    with torch.no_grad():
        for i, (tgt_img, ref_imgs, gt_nmap, ref_poses, intrinsics, intrinsics_inv, tgt_depth) in enumerate(val_loader):
            tgt_img_var = Variable(tgt_img.cuda())
            ref_imgs_var = [Variable(img.cuda()) for img in ref_imgs]
            gt_nmap_var = Variable(gt_nmap.cuda())
            ref_poses_var = [Variable(pose.cuda()) for pose in ref_poses]
            intrinsics_var = Variable(intrinsics.cuda())
            intrinsics_inv_var = Variable(intrinsics_inv.cuda())
            tgt_depth_var = Variable(tgt_depth.cuda())

            pose = torch.cat(ref_poses_var,1)

            if (pose != pose).any():
                continue
            
            if args.dataset == 'sceneflow':
                factor = (1.0/args.scale)*intrinsics_var[:,0,0]/1050.0
                factor = factor.view(-1,1,1)
            else:
                factor = torch.ones((tgt_depth_var.size(0),1,1)).type_as(tgt_depth_var)

            # get mask
            mask = (tgt_depth_var <= args.nlabel*args.mindepth*factor*3) & (tgt_depth_var >= args.mindepth*factor) & (tgt_depth_var == tgt_depth_var)
            
            if not mask.any():
                continue

            output_depth, nmap = mvdnet(tgt_img_var, ref_imgs_var, pose, intrinsics_var, intrinsics_inv_var, no_pool = True, factor = factor.unsqueeze(1))
            output_disp = args.nlabel*args.mindepth/(output_depth)  
            if args.dataset == 'sceneflow':
                output_disp = (args.nlabel*args.mindepth)*3/(output_depth)  
                output_depth = (args.nlabel*3)*(args.mindepth*factor)/output_disp

            tgt_disp_var = ((1.0/args.scale)*intrinsics_var[:,0,0].view(-1,1,1)/tgt_depth_var)

            

            if args.dataset == 'sceneflow':
                output = torch.squeeze(output_disp.data.cpu(),1)
                errors_ = compute_errors_train(tgt_disp_var.cpu(), output, mask)
                test_errors_ = list(compute_errors_test(tgt_disp_var.cpu()[mask], output[mask]))
            else:
                output = torch.squeeze(output_depth.data.cpu(),1)
                errors_ = compute_errors_train(tgt_depth, output, mask)
                test_errors_ = list(compute_errors_test(tgt_depth[mask], output[mask]))

            n_mask = (gt_nmap_var.permute(0,2,3,1)[0,:,:] != 0)
            n_mask = n_mask[:,:,0] | n_mask[:,:,1] | n_mask[:,:,2]
            total_angles_m = compute_angles(gt_nmap_var.permute(0,2,3,1)[0], nmap[0])
            
            mask_angles = total_angles_m[n_mask]
            total_angles_m[~ n_mask] = 0
            errors_.append(torch.mean(mask_angles).item())#/mask_angles.size(0)#[torch.sum(mask_angles).item(), (mask_angles.size(0)),  torch.sum(mask_angles < 7.5).item(), torch.sum(mask_angles < 15).item(), torch.sum(mask_angles < 30).item(), torch.sum(mask_angles < 45).item()]
            test_errors_.append(torch.mean(mask_angles).item())
            errors.update(errors_)
            test_errors.update(test_errors_)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if args.output_print:
                np.save(output_dir/'{:04d}{}'.format(i,'_depth.npy'), output.numpy()[0])
                plt.imsave(output_dir/'{:04d}_gt{}'.format(i,'.png'), tgt_depth.numpy()[0], cmap='rainbow')
                imsave(output_dir/'{:04d}_aimage{}'.format(i,'.png'), np.transpose(tgt_img.numpy()[0],(1,2,0)))
                np.save(output_dir/'{:04d}_cam{}'.format(i,'.npy'),intrinsics_var.cpu().numpy()[0])
                np.save(output_dir/'{:04d}{}'.format(i,'_normal.npy'), nmap.cpu().numpy()[0])

            if i % args.print_freq == 0:
                print('valid: Time {} Abs Error {:.4f} ({:.4f}) Abs angle Error {:.4f} ({:.4f}) Iter {}/{}'.format(batch_time, test_errors.val[0], test_errors.avg[0], test_errors.val[-1], test_errors.avg[-1], i, len(val_loader)))
    if args.output_print:
        np.savetxt(output_dir/args.ttype+'errors.csv', test_errors.avg, fmt='%1.4f', delimiter=',')
        np.savetxt(output_dir/args.ttype+'angle_errors.csv', test_errors.avg, fmt='%1.4f', delimiter=',')
    return errors.avg, error_names


if __name__ == '__main__':
    main()
