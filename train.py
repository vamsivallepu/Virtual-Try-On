# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="GMM")
    # parser.add_argument("--name", default="TOM")

    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)

    parser.add_argument("--dataroot", default="data")

    parser.add_argument("--datamode", default="train")

    parser.add_argument("--stage", default="GMM")
    # parser.add_argument("--stage", default="TOM")

    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate for adam')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='model checkpoint for initialization')
    # parser.add_argument("--display_count", type=int, default=20)
    parser.add_argument("--save_count", type=int, default=5000)
    parser.add_argument("--keep_step", type=int, default=100000)
    parser.add_argument("--shuffle", action='store_true',
                        help='shuffle input data')

    opt = parser.parse_args()
    return opt


def train_gmm(opt, train_loader, model):
    model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    
    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0 -
                                                  max(0, step - opt.keep_step) / float(opt.decay_step + 1))

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c = inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()

        grid, theta = model(agnostic, cm)    # can be added c too for new training
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')


        # loss for warped cloth
        Lwarp = criterionL1(warped_cloth, im_c)    # changing to previous code as it corresponds to the working code
        # Actual loss function as in the paper given below (comment out previous line and uncomment below to train as per the paper)
        # Lwarp = criterionL1(warped_mask, cm)    # loss for warped mask thanks @xuxiaochun025 for fixing the git code.
        
        # grid regularization loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(
                opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))


def train_tom(opt, train_loader, model):
    model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0 -
                                                  max(0, step - opt.keep_step) / float(opt.decay_step + 1))

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        pcm = inputs['parse_cloth_mask'].cuda()

        outputs = model(torch.cat([agnostic, c], 1))  
        p_rendered, m_composite = torch.split(outputs, 3, 1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        """visuals = [[im_h, shape, im_pose],
                   [c, cm*2-1, m_composite*2-1],
                   [p_rendered, p_tryon, im]]""" 

        loss_l1 = criterionL1(p_tryon, im)
        loss_vgg = criterionVGG(p_tryon, im)
        loss_mask = criterionMask(m_composite, cm)
        loss = loss_l1 + loss_vgg + loss_mask
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(
                opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))


def main():
    opt = get_opt()
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))

    # create dataset
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # create model & train & save the final checkpoint
    if opt.stage == 'GMM':
        model = GMM(opt)
        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_gmm(opt, train_loader, model)
        save_checkpoint(model, os.path.join(
            opt.checkpoint_dir, opt.name, 'gmm_final.pth'))
    elif opt.stage == 'TOM':
        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d) 
        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_tom(opt, train_loader, model)
        save_checkpoint(model, os.path.join(
            opt.checkpoint_dir, opt.name, 'tom_final.pth'))
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)

    print('Finished training %s, named: %s!' % (opt.stage, opt.name))


if __name__ == "__main__":
    main()

# python train.py --name GMM --stage GMM --workers 4 --save_count 5000 --shuffle
