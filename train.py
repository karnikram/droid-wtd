import sys
sys.path.append('droid_slam')

import cv2
import numpy as np
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_readers.factory import dataset_factory

from lietorch import SO3, SE3, Sim3
from geom import losses
from geom.losses import geodesic_loss, residual_loss, flow_loss
from geom.graph_utils import build_frame_graph

# network
from droid_net import DroidNet
from logger import Logger

# DDP training
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_ddp(gpu, args):
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',     
    	world_size=args.world_size,                              
    	rank=gpu)

    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()

def train(gpu, args):
    """ Test to make sure project transform correctly maps points """

    # coordinate multiple GPUs
    setup_ddp(gpu, args)
    rng = np.random.default_rng(12345)

    N = args.n_frames
    model = DroidNet()
    model.cuda()
    model.train()

    model = DDP(model, device_ids=[gpu], find_unused_parameters=False)

    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt))

    # fetch dataloader
    db = dataset_factory(['tartan'], datapath=args.datapath, n_frames=args.n_frames, fmin=args.fmin, fmax=args.fmax)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        db, shuffle=True, num_replicas=args.world_size, rank=gpu)

    train_loader = DataLoader(db, batch_size=args.batch, sampler=train_sampler, num_workers=args.num_workers)

    # fetch optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
        args.lr, args.steps, pct_start=0.01, cycle_momentum=False)

    logger = Logger(args.name, scheduler)
    should_keep_training = True
    total_steps = 0
    flow_coeff = 1.0
    res_coeff = 1.0
    ro_coeff = 4.5
    flow_coeff_ratios = []
    res_coeff_ratios = []
    ro_coeff_ratios = []

    while should_keep_training:
        for i_batch, item in enumerate(train_loader):
            optimizer.zero_grad()

            images, poses, disps, intrinsics = [x.to('cuda') for x in item]

            # convert poses w2c -> c2w
            Ps = SE3(poses).inv()
            Gs = SE3.IdentityLike(Ps)

            # randomize frame graph
            if np.random.rand() < 0.5:
                graph = build_frame_graph(poses, disps, intrinsics, num=args.edges)
            
            else:
                graph = OrderedDict()
                for i in range(N):
                    graph[i] = [j for j in range(N) if i!=j and abs(i-j) <= 2]
            
            # fix first to camera poses
            Gs.data[:,0] = Ps.data[:,0].clone()
            Gs.data[:,1:] = Ps.data[:,[1]].clone()
            disp0 = torch.ones_like(disps[:,:,3::8,3::8])

            # perform random restarts
            r = 0
            while r < args.restart_prob:
                r = rng.random()
                
                intrinsics0 = intrinsics / 8.0
                poses_est, disps_est, residuals, traj, stats = model(Gs, images, disp0, intrinsics0, 
                    disps[:, :, 3::8, 3::8], graph, num_steps=args.iters, fixedp=2)

                geo_loss, geo_metrics, tr_loss, ro_loss = losses.geodesic_loss(Ps, poses_est, graph, ro_coeff, do_scale=False)
                res_loss, res_metrics = losses.residual_loss(residuals, traj)
                flo_loss, flo_metrics = losses.flow_loss(Ps, disps, poses_est, disps_est, intrinsics, graph, traj)

                #if total_steps > 1000 and total_steps % 20 == 0:
                #    flow_grad = torch.autograd.grad(flo_loss, model.module.update.gru.convq_glo.weight, retain_graph=True)[0].norm().item()
                #    pose_grad = torch.autograd.grad(geo_loss, model.module.update.gru.convq_glo.weight, retain_graph=True)[0].norm().item()
                #    res_grad = torch.autograd.grad(res_loss, model.module.update.gru.convq_glo.weight, retain_graph=True)[0].norm().item()
                #    try:
                #        flow_coeff_ratios = flow_coeff_ratios[-50:] + [max(min(pose_grad/flow_grad, 10*flow_coeff), 0.1*flow_coeff)]
                #    except:
                #        flow_coeff_ratios = flow_coeff_ratios[-50:] + [max(10*flow_coeff, 0.1*flow_coeff)]
                #    try:
                #        res_coeff_ratios = res_coeff_ratios[-50:] + [max(min(pose_grad/res_grad, 10*res_coeff), 0.1*res_coeff)]
                #    except:
                #        res_coeff_ratios = res_coeff_ratios[-50:] + [max(10*res_coeff, 0.1*res_coeff)]
                #    res_coeff = np.mean(res_coeff_ratios)
                #    flow_coeff = np.mean(flow_coeff_ratios)
                    #tr_grad = torch.autograd.grad(tr_loss, model.module.update.gru.convq_glo.weight, retain_graph=True)[0].norm().item()
                    #ro_grad = torch.autograd.grad(ro_loss, model.module.update.gru.convq_glo.weight, retain_graph=True)[0].norm().item()
                    #try:
                    #    ro_coeff_ratios = ro_coeff_ratios[-50:] + [max(min(tr_grad/ro_grad, 10*ro_coeff), 0.1*ro_coeff)]
                    #except:
                    #    ro_coeff_ratios = ro_coeff_ratios[-50:] + [max(10*ro_coeff, 0.1*ro_coeff)]
                    #ro_coeff = np.mean(ro_coeff_ratios)

                loss = args.w1 * geo_loss + args.w2 * res_loss + args.w3 * flo_loss
                loss.backward()

                Gs = poses_est[-1].detach()
                disp0 = disps_est[-1][:,:,3::8,3::8].detach()

            metrics = {
                "loss": loss.item(),
                "res_coeff": res_coeff,
                "flow_coeff": flow_coeff,
                "ro_coeff": ro_coeff
            }
            metrics.update(geo_metrics)
            metrics.update(res_metrics)
            metrics.update(flo_metrics)
            metrics.update(stats)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            
            total_steps += 1

            if gpu == 0:
                logger.push(metrics)

            if total_steps % 10000 == 0 and gpu == 0:
                PATH = 'checkpoints/%s_%06d.pth' % (args.name, total_steps)
                torch.save(model.state_dict(), PATH)

            if total_steps >= args.steps:
                should_keep_training = False
                break

    dist.destroy_process_group()
                

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bla', help='name your experiment')
    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--datasets', nargs='+', help='lists of datasets for training')
    parser.add_argument('--datapath', default='datasets/TartanAir', help="path to dataset directory")
    parser.add_argument('--gpus', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--iters', type=int, default=15)
    parser.add_argument('--steps', type=int, default=250000)
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--clip', type=float, default=2.5)
    parser.add_argument('--n_frames', type=int, default=7)

    parser.add_argument('--w1', type=float, default=10.0)
    parser.add_argument('--w2', type=float, default=0.01)
    parser.add_argument('--w3', type=float, default=0.05)

    parser.add_argument('--fmin', type=float, default=8.0)
    parser.add_argument('--fmax', type=float, default=96.0)
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--edges', type=int, default=24)
    parser.add_argument('--restart_prob', type=float, default=0.2)
    parser.add_argument('--upsample', action='store_true')

    args = parser.parse_args()

    args.world_size = args.gpus
    print(args)

    import os
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    args = parser.parse_args()
    args.world_size = args.gpus

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    mp.spawn(train, nprocs=args.gpus, args=(args,))

