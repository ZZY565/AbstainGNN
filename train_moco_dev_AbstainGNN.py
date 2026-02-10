from __future__ import print_function

import moco.CSC
from moco.CSC import MoCo

import pandas as pd
import argparse
import os
import time
import random
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, closefig
from loss import SelfAdativeTraining, deep_gambler_loss, log_margin_loss, OursProtoLoss,cwr_loss


from graph_models.util import load_data, separate_data
from graph_models.graph_dataset_adapter import (
    GraphListDataset, collate_graph_batch
)
from graph_models.confidence_head import GCNConfidenceHead,GATConfidenceHead,ReadoutConfidenceHead

from graph_models.graphppd import GraphPPD

graph_backbones = ["gin", "gcn", "mpnn"]


parser = argparse.ArgumentParser(description='Selective Classification for Self-Adaptive Training')
parser.add_argument(
    '-d', '--dataset', metavar='DATASET', default='cifar100',
    choices=[
        'MUTAG', 'PROTEINS', 'ENZYMES', 'DD', 'NCI1',
        'COLLAB', 'IMDBBINARY', 'IMDBMULTI',
        'PTC', 'REDDITBINARY', 'REDDITMULTI5K'
    ],
    help='dataset (default: cifar100)'
)


parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--mode', default='train', type=str, choices=['train', 'tuning'],
                    help='mode: tuning refers to 80/20 split of the training data for hyperparameter tuning')

parser.add_argument('-t', '--train', dest='evaluate', action='store_true',
                    help='train the model. When evaluate is true, training is ignored and trained models are loaded.')

parser.add_argument('-r', '--resume', dest='resume', action='store_true',
                    help='resume the model. When resume is true, training is ignored and trained models are loaded.')
parser.add_argument('--start_epochs', default=0, type=int, metavar='N',
                    help='resume epochs to run')

parser.add_argument('-w', '--warmup', dest='warmup', action='store_true',
                    help='warmup the reward.')
parser.add_argument('--ir', '--initialreward', default=1e-6, type=float,
                    metavar='IR', help='initial reward')
parser.add_argument('--warmup_epochs', default=0, type=int, metavar='WN',
                    help='warm-up iterations')

parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--save_model_step', default=25, type=int, metavar='N',
                    help='number of epochs to run before a model is saved')
parser.add_argument('--train-batch', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--num_classes', default=2, type=int, metavar='N',
                    help='Number of Classes for ImageNetSubset ONLY')
parser.add_argument('--num_valid', default=1000, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500],
                        help='Multiply learning rate by gamma at the scheduled epochs (default: 25,50,75,100,125,150,175,200,225,250,275)')
parser.add_argument('--gamma', type=float, default=0.5, help='LR is multiplied by gamma on schedule (default: 0.5)') 
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--sat-momentum', default=0.9, type=float, help='momentum for sat')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-o', '--rewards', dest='rewards', type=float, nargs='+', default=[4.6],#fixed for gamlber 【4.6】fixed for lcsc【0.5】
                    metavar='o', help='The reward o for a correct prediction; Abstention has a reward of 1. Provided parameters would be stored as a list for multiple runs.')
parser.add_argument('--pretrain', type=int, default=100,
                    help='Number of pretraining epochs using the cross entropy loss, so that the learning can always start. Note that it defaults to 100 if dataset==cifar10 and reward<6.1, and the results in the paper are reproduced.')
parser.add_argument('--coverage', type=float, nargs='+',default=[100.,99.,98.,97.,95.,90.,85.,80.,75.,70.,60.,50.,40.,30.,20.,10.],
                    help='the expected coverages used to evaluated the accuracies after abstention')

parser.add_argument('-s', '--save', default='save', type=str, metavar='PATH',
                    help='path to save checkpoint (default: save)')

parser.add_argument('--loss', default='sat', type=str,
    choices=['sat', 'ce', 'gambler', 'sat_entropy', 'cwr','csc', 'log_ml', 'csc_entropy', 'csc_sat_entropy', 'ours_proto','graphppd'],

    help='loss function (sat, ce, gambler, sat_entropy, csc,cwr, log_ml, csc_entropy, csc_sat_entropy, ours)')
parser.add_argument('--entropy', type=float, default=0.0, help='Entropy Coefficient for the SAT Loss (default: 0.0)') 
parser.add_argument(
    '--arch', '-a', metavar='ARCH', default='gcn',
    choices=graph_backbones,
    help='model architecture: ' +
         ' | '.join(graph_backbones) +
         ' (default: gcn)'
)

parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate trained models on validation set, following the paths defined by "save", "arch" and "rewards"')

parser.add_argument(
    "--moco-k",
    default=300,
    type=int,
    help="queue size; number of negative keys (default: 65536)",
)
parser.add_argument(
    "--moco-m",
    default=0.99,
    type=float,
    help="moco momentum of updating key encoder (default: 0.999)",
)
parser.add_argument(
    "--moco-t", default=0.07, type=float, help="softmax temperature (default: 0.07)"
)

parser.add_argument('--alpha', type=float, default=0.1,
                    help='weight for Lvar in Lours (Lce + alpha * Lvar)')


parser.add_argument('--ours_warmup_epochs', type=int, default=30,
                    help='Warm-up epochs Es for Ours method (CE only before prototype constraints).')

parser.add_argument('--ours_rho', type=float, default=0.05,
                    help='EMA momentum rho for updating class prototypes.')

parser.add_argument('--ours_gamma', type=float, default=0.2,
                    help='Shrinking coefficient gamma in: h_tilde = (1 - gamma) * h + gamma * mu_y.')

parser.add_argument('--ours_alpha_intra', type=float, default=1.0,
                    help='Weight alpha for intra-class variance loss L_intra.')



parser.add_argument(
    '--conf_weight', type=float, default=1.0,
    help='Weight of the confidence loss in the total loss function'
)


parser.add_argument('--ppd_context_size', type=int, default=64,
                    help='GraphPPD: number of context graphs K sampled from training set per step.')

parser.add_argument('--ppd_P', type=int, default=10,
                    help='GraphPPD: number of Monte Carlo context samples at inference.')

parser.add_argument('--ppd_dh', type=int, default=64,
                    help='GraphPPD: attention head hidden dim dh.')

parser.add_argument('--ppd_heads', type=int, default=1,
                    help='GraphPPD: number of attention heads.')

parser.add_argument('--ppd_mlp_hidden', type=int, default=128,
                    help='GraphPPD: hidden size of the MLP after attention.')

parser.add_argument('--ppd_mlp_layers', type=int, default=2,
                    help='GraphPPD: number of MLP layers after attention.')






args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}



expected_coverage = args.coverage
if args.dataset == 'cifar10':
    expected_coverage = [100.,95.,90.,85.,80.,75.,70.]
if args.dataset == 'cifar100':
    expected_coverage = [100.,95.,90.,85.,80.,75.,70.,60, 50., 40., 30., 20., 10.]
if args.dataset == 'svhn':
    expected_coverage = [100.,95.,90.,85.,80.,75.,70.,60, 50., 40., 30., 20., 10.]
if args.dataset == 'celeba':
    expected_coverage = [100.,95.,90.,85.,80.,75.,70.,60, 50., 40., 30., 20., 10.]
reward_list = args.rewards

graph_datasets = {
    "MUTAG", "PROTEINS", "ENZYMES", "DD", "NCI1",
    "COLLAB", "IMDBBINARY", "IMDBMULTI",
    "PTC", "REDDITBINARY", "REDDITMULTI5K"
}


use_cuda = torch.cuda.is_available()


if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
hidden_features = None
hidden_features_k = None

def hook_fn(module, input, output):
    global hidden_features
    hidden_features = output

def hook_fn_k(module, input, output):
    global hidden_features_k
    hidden_features_k = output


def main():
    print(args)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not resume_path and not os.path.isdir(save_path):
        mkdir_p(save_path)

    
    if args.dataset in graph_datasets:
        print(f"=> Using graph dataset: {args.dataset}")

        # ---- 1. load data ----
        use_degree = args.dataset in [
            "IMDBBINARY", "IMDBMULTI",
            "REDDITBINARY", "REDDITMULTI5K",
            "COLLAB"
        ]

        all_graphs, num_classes = load_data(args.dataset, degree_as_tag=use_degree)

        dataset_class_map = {
            "MUTAG": 2,
            "PROTEINS": 2,
            "ENZYMES": 6,
            "DD": 2,
            "NCI1": 2,
            "COLLAB": 3,
            "IMDBBINARY": 2,
            "IMDBMULTI": 3,
            "PTC": 2,
            "REDDITBINARY": 2,
            "REDDITMULTI5K": 5
        }
        num_classes = dataset_class_map.get(args.dataset, num_classes)
        args.num_classes = num_classes

        train_graphs, test_graphs = separate_data(
            all_graphs,
            seed=args.manualSeed or 0,
            fold_idx=0
        )

        trainset = GraphListDataset(train_graphs)
        testset  = GraphListDataset(test_graphs)

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.train_batch, shuffle=True,
            num_workers=args.workers, collate_fn=collate_graph_batch
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch, shuffle=False,
            num_workers=args.workers, collate_fn=collate_graph_batch
        )
        devloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.test_batch, shuffle=False,
            num_workers=args.workers, collate_fn=collate_graph_batch
        )

        
    print("==> creating model '{}'".format(args.arch))
    
    if args.arch.lower() == 'gcn':
        print("=> Building PYG-GCN model...")
        from graph_models.pyg_backbones import PYG_GCN

        input_dim = trainset[0][0].node_features.shape[1]
        hidden_dim = 64

        model = PYG_GCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=args.num_classes if args.loss in ['ce', 'log_ml', 'csc', 'csc_entropy', 'ours_proto','graphppd'] else args.num_classes + 1,
            num_layers=3,
            dropout=0.5,
            device="cuda"
        ).cuda()


        model_k = copy.deepcopy(model)

        print("==> Registering GCN hooks (train mode)")

        def hook_gcn(module, input, output):
            global hidden_features
            cached = getattr(module, "cached_hidden", None)
            if isinstance(cached, torch.Tensor):
                hidden_features = cached.detach()

        def hook_gcn_k(module, input, output):
            global hidden_features_k
            cached = getattr(module, "cached_hidden", None)
            if isinstance(cached, torch.Tensor):
                hidden_features_k = cached.detach()

        args.moco_dim = hidden_dim
        print(f"[GCN] MoCo feature dim = {hidden_dim}")
        
    model = model.cuda() 

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    if args.pretrain: criterion = nn.CrossEntropyLoss()
    if args.loss == 'ce' or args.loss == 'csc':
        criterion = nn.CrossEntropyLoss() 
        
    elif args.loss == 'gambler':
        criterion = deep_gambler_loss
    elif args.loss == 'cwr':
        args.cwr_d = 0.5
        criterion = cwr_loss
        
    elif args.loss == 'sat' or args.loss == 'sat_entropy':
        criterion = SelfAdativeTraining(num_examples=len(trainset), num_classes=num_classes, mom=args.sat_momentum)
    elif args.loss == 'log_ml':
        criterion = log_margin_loss
    elif args.loss == 'ours_proto':
  
        criterion = OursProtoLoss()

    optimizer = optim.SGD(model.parameters(), lr=state['lr'], momentum=args.momentum, weight_decay=args.weight_decay)


    title = args.dataset + '-' + args.arch + ' o={:.2f}'.format(reward)
    logger = Logger(os.path.join(save_path, 'eval.txt' if args.evaluate else 'log.txt'), title=title)
    logger.set_names(['Epoch', 'Learning Rate', 'Train Loss', 'Train Loss2','Test Loss', 'Train Err.', 'Test Err.', 'MOCO Err.'])
    


    if args.evaluate:
        print('\n[Evaluation Mode] Loading trained model for evaluation...')
        assert os.path.isfile(resume_path), f'❌ no model exists at "{resume_path}"'

        checkpoint = torch.load(resume_path, map_location='cuda' if use_cuda else 'cpu')
        
        ppd_module = None
        confidence_head = None

        if isinstance(checkpoint, dict):
            if 'net_state_dict' in checkpoint:
                print("✅ Found key 'net_state_dict' in checkpoint.")
                state_dict = checkpoint['net_state_dict']
            elif 'state_dict' in checkpoint:
                print("✅ Found key 'state_dict' in checkpoint.")
                state_dict = checkpoint['state_dict']
            else:
                print("⚠️ No key found, treating as raw state_dict.")
                state_dict = checkpoint

            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)

            if args.loss == "graphppd":
                ppd_module = GraphPPD(
                    dx=hidden_dim,
                    num_classes=args.num_classes,
                    dh=args.ppd_dh,
                    heads=args.ppd_heads,
                    mlp_hidden=args.ppd_mlp_hidden,
                    mlp_layers=args.ppd_mlp_layers,
                    dropout=0.0,
                ).cuda()


                assert "ppd_module" in checkpoint, "❌ checkpoint missing key 'ppd_module' for graphppd"
                ppd_module.load_state_dict(checkpoint["ppd_module"], strict=True)
                ppd_module.eval()

                print(f"[GraphPPD] Loaded ppd_module weights. dx={hidden_dim}, C={args.num_classes}")
                n_ppd = sum(p.numel() for p in ppd_module.parameters())
                print(f"[GraphPPD] ppd_module params: {n_ppd/1e6:.3f}M")
                
                
            if args.loss == "graphppd":
                print("[GraphPPD] ppd_module created:", type(ppd_module))
                n_ppd = sum(p.numel() for p in ppd_module.parameters())
                print(f"[GraphPPD] ppd_module params: {n_ppd/1e6:.3f}M")

            if args.loss == "ours_proto":
              
                args.prototypes_h = checkpoint['prototypes_h']
                args.prototypes_u = checkpoint['prototypes_u']
                args.ours_gamma = checkpoint.get('ours_gamma', args.ours_gamma)
                args.prototype_momentum = checkpoint.get('prototype_momentum', 0.99)

                print(f"✅ Loaded prototypes_h: {args.prototypes_h.shape}")
                print(f"✅ Loaded prototypes_u: {args.prototypes_u.shape}")
                print(f"✅ Loaded ours_gamma: {args.ours_gamma}")
                print(f"✅ Loaded prototype_momentum: {args.prototype_momentum}")

       
                confidence_head = GCNConfidenceHead(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=1,     
                    dropout=0.0,   
                    mlp_hidden=hidden_dim,
                    mlp_layers=2,
                    device="cuda",
                ).cuda()


                # confidence_head = GATConfidenceHead(
                #     input_dim=input_dim,
                #     hidden_dim=hidden_dim,
                #     num_layers=1,      
                #     dropout=0.0,   
                #     mlp_hidden=hidden_dim,
                #     mlp_layers=2,     
                #     heads=1,          
                #     attn_dropout=0.0,   
                #     concat=False,    
                #     device="cuda",
                # ).cuda() 

                # confidence_head = ReadoutConfidenceHead(
                #     input_dim=input_dim,   
                #     mlp_hidden=hidden_dim,  
                #     mlp_layers=2,        
                #     dropout=0.0,
                #     readout="sum",   
                #     device="cuda",
                # ).cuda()

                confidence_head.load_state_dict(checkpoint['confidence_head'])
                if use_cuda:
                    confidence_head = confidence_head.cuda()
                confidence_head.eval()
            else:
                confidence_head = None

        else:
            print("⚠️ Checkpoint is a full model object (old format). Loading directly.")
            model = checkpoint


        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        if use_cuda:
            model = model.cuda()

        print("✅ Model loaded successfully. Starting evaluation...")
        test(devloader, testloader, model, criterion, args.epochs, use_cuda, evaluation=True,confidence_head=confidence_head,ppd_module=ppd_module)
        return

    
    archive = moco.CSC.MoCo(
        args.moco_dim,
        args.moco_k,
        args.moco_m,
        args.moco_t,
        num_class = num_classes
    )
    for epoch in range(args.start_epochs, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        print('\n'+save_path)
        print('Epoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        train_loss, train_loss2, train_acc, moco_top1 = train(trainloader, archive, model, model_k, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(devloader, testloader, model, criterion, epoch, use_cuda)
        print(train_acc, train_loss2, test_acc, moco_top1 * 100)


        if (epoch+1) % args.save_model_step == 0:
            filepath = os.path.join(save_path, "{:d}".format(epoch+1) + ".pth")
            torch.save(model, filepath)
        
        logger.append([epoch+1, state['lr'], train_loss, train_loss2, test_loss, 100-train_acc, 100-test_acc, 100-moco_top1 * 100])

    filepath = os.path.join(save_path, "{:d}".format(args.epochs) + ".pth")
    torch.save(model, filepath)
    last_path = os.path.join(save_path, "{:d}".format(args.epochs-1) + ".pth")
    if os.path.isfile(last_path): os.remove(last_path)
    logger.plot(['Train Loss', 'Test Loss'])
    savefig(os.path.join(save_path, 'logLoss.eps'))
    closefig()
    logger.plot(['Train Err.', 'Test Err.'])
    savefig(os.path.join(save_path, 'logErr.eps'))
    closefig()
    logger.plot(['Train Loss', 'Train Loss2'])
    savefig(os.path.join(save_path, 'loss2.eps'))
    closefig()
    logger.close()

def sample_context_from_trainset(trainset, K):

    assert trainset is not None
    K = min(int(K), len(trainset))
    idxs = random.sample(range(len(trainset)), K)
    ctx_graphs, yC = [], []
    for idx in idxs:
        g, y, _ = trainset[idx]  
        ctx_graphs.append(g)
        yC.append(int(y))
    yC = torch.tensor(yC, dtype=torch.long)  
    return ctx_graphs, yC

def _move_graphs_to_device(graphs, device):
    
    if graphs is None:
        return
    if isinstance(graphs, list):
        for g in graphs:
            if hasattr(g, "to"):
                g.to(device)
            else:
                if hasattr(g, "node_features") and isinstance(g.node_features, torch.Tensor):
                    g.node_features = g.node_features.to(device, non_blocking=True)
def forward_get_emb(model, graphs):
    out = model(graphs)
    h = getattr(model, "cached_hidden", None)
    if h is not None and h.dim() > 2:
        h = torch.flatten(h, 1)
    return out, h

def test(devloader, testloader, model, criterion, epoch, use_cuda, evaluation = False,confidence_head=None, ppd_module=None):
    global best_acc

    if evaluation:
        evaluate(devloader, testloader, model, use_cuda, confidence_head=confidence_head, ppd_module=ppd_module)
        return

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    abstention_results = []
    sr_results = []
    abstention_results_nosoftmax = []

    confhead_results = [] 

    

    for batch_idx, batch_data in enumerate(testloader):
        if len(batch_data) == 3:
            inputs, targets, indices = batch_data
        else:
            inputs, targets = batch_data
            indices = None
        data_time.update(time.time() - end)


        if use_cuda:
            for g in inputs:
                if hasattr(g, "node_features") and isinstance(g.node_features, torch.Tensor):
                    g.node_features = g.node_features.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        targets = torch.autograd.Variable(targets)

        with torch.no_grad():
            output_logits = model(inputs).cpu()
            outputs = output_logits
            values, predictions = outputs.data.max(1)
            if epoch >= args.pretrain:

                if args.loss == 'gambler':
                    loss = criterion(outputs, targets, reward)
                elif args.loss == 'sat' or args.loss == 'sat_entropy':
                    loss = F.cross_entropy(outputs[:, :-1], targets)
                elif args.loss == 'log_ml':
                    loss = F.cross_entropy(outputs, targets)

                else:
                    loss = criterion(outputs, targets)
                outputs = F.softmax(outputs, dim=1)

                if args.loss == 'ce' or args.loss == 'csc'  :  
                    outputs, reservation = outputs, (outputs * torch.log(outputs)).sum(-1) 
                elif args.loss == 'log_ml':
                    top_logits, top_indices = torch.topk(outputs, k=2, dim=-1)

                    first_max_indices = top_indices[:, 0]
                    second_max_indices = top_indices[:, 1]

                    first_max_logits = torch.gather(outputs, 1, first_max_indices.unsqueeze(1))
                    second_max_logits = torch.gather(outputs, 1, second_max_indices.unsqueeze(1))

                    first_max_logits_nosoftmax = torch.gather(output_logits, 1, first_max_indices.unsqueeze(1))
                    second_max_logits_nosoftmax = torch.gather(output_logits, 1, second_max_indices.unsqueeze(1))


                    difference = first_max_logits - second_max_logits
                    difference_nosoftmax = first_max_logits_nosoftmax - second_max_logits_nosoftmax

                    reservation = (-difference).view(-1)
                    reservation_nosoftmax = (-difference_nosoftmax).view(-1)
                else:
                    outputs, reservation = outputs[:,:-1], outputs[:,-1]
                abstention_results.extend(zip(list( reservation.numpy() ),list( predictions.eq(targets.to(predictions.device)).numpy() )))
                if args.loss == 'log_ml':
                    abstention_results_nosoftmax.extend(zip(list( reservation_nosoftmax.numpy() ),list( predictions.eq(targets.to(predictions.device)).numpy() )))
                if args.loss == 'ce' or args.loss == 'log_ml' or args.loss == 'csc' or args.loss == 'ours': 
                    pred_logits = nn.functional.softmax(output_logits, -1)
                else:
                    pred_logits = nn.functional.softmax(output_logits[:,:-1], -1)
                sr_results.extend(zip(list(pred_logits.max(-1)[0].numpy()), list( predictions.eq(targets.to(predictions.device)).numpy() )))
            else:
                if args.loss == 'ce' or args.loss == 'log_ml' or args.loss == 'csc':
                    loss = F.cross_entropy(outputs.cpu(), targets)
                else:
                    loss = F.cross_entropy(outputs[:,:-1].cpu(), targets)

            if args.dataset != 'catsdogs':
                prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))
            else:
                prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    if epoch >= args.pretrain:
        abstention_results.sort(key = lambda x: x[0])
        sorted_correct = list(map(lambda x: int(x[1]), abstention_results))
        size = len(sorted_correct)
        print('Abstention Logit: accuracy of coverage ',end='')
        for coverage in expected_coverage:
            covered_correct = sorted_correct[:round(size/100*coverage)]
            print('{:.0f}: {:.3f}, '.format(coverage, sum(covered_correct)/len(covered_correct)*100.), end='')
        print('')


        sr_results.sort(key = lambda x: -x[0])

        sorted_correct = list(map(lambda x: int(x[1]), sr_results))
        size = len(sorted_correct)
        print('Softmax Response: accuracy of coverage ',end='')
        for coverage in expected_coverage:
            covered_correct = sorted_correct[:round(size/100*coverage)]
            print('{:.0f}: {:.3f}, '.format(coverage, sum(covered_correct)/len(covered_correct)*100.), end='')
        print('')
        if args.loss == "log_ml":
            abstention_results_nosoftmax.sort(key = lambda x: x[0])
            sorted_correct = list(map(lambda x: int(x[1]), abstention_results_nosoftmax))
            size = len(sorted_correct)
            print('Abstention Logit_nonsoftmax: accuracy of coverage ',end='')
            for coverage in expected_coverage:
                covered_correct = sorted_correct[:round(size/100*coverage)]
                print('{:.0f}: {:.3f}, '.format(coverage, sum(covered_correct)/len(covered_correct)*100.), end='')
            print('')
    return (losses.avg, top1.avg)

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']



def evaluate(devloader, testloader, model, use_cuda,confidence_head=None, ppd_module=None):

    global hidden_features
    hidden_features = None
    model.eval()


    abortion_results = [[], []]
    abortion_results_nosoftmax = [[], []]
    sr_results = [[], []]
    if args.loss == "ce":
        mc_ce_results = [[], []]  


    if args.loss == "ours_proto":
        sr_tilde_results = [[], []] 

        confhead_results = [[], []]

    feature_train = []
    prediction_list_train = []
    correct_list_train = []
    target_list = []

    feature = []
    prediction_list = []
    correct_list = []



    start_t = time.time()

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(testloader):
            inputs, targets = batch_data[:2]

            if args.arch.lower() in ['gin', 'gcn', 'mpnn']:
                if isinstance(inputs, list):
                    for g in inputs:
                        if hasattr(g, 'node_features') and isinstance(g.node_features, torch.Tensor):
                            g.node_features = g.node_features.cuda(non_blocking=True)
                        if hasattr(g, 'edge_mat') and isinstance(g.edge_mat, torch.Tensor):
                            g.edge_mat = g.edge_mat.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            hidden_features = None
            output_logits = model(inputs)

            if args.arch.lower() in ['gin', 'gcn', 'mpnn']:
                batch_feats = model.cached_hidden.detach()
            else:
                batch_feats = hidden_features
                if args.arch != 'vgg16_bn':
                    batch_feats = torch.flatten(batch_feats, 1)

            batch_feats = torch.nan_to_num(
                batch_feats, nan=0.0, posinf=0.0, neginf=0.0
            )

            for i in range(batch_feats.size(0)):
                feature.append(batch_feats[i].cpu())


            probs_ppd = None 
            output = F.softmax(output_logits, dim=1)

            if args.loss == "graphppd":
                assert ppd_module is not None, "ppd_module is None but loss==graphppd"
                trainset_local = getattr(devloader, "dataset", None)
                assert trainset_local is not None, "GraphPPD evaluate needs devloader.dataset for sampling contexts"

                x_T = batch_feats 
                x_T = torch.nan_to_num(x_T, nan=0.0, posinf=0.0, neginf=0.0)

                P = int(args.ppd_P)
                probs_accum = None
                device = x_T.device

                for _p in range(P):
                    ctx_graphs, y_C = sample_context_from_trainset(trainset_local, args.ppd_context_size)
                    _move_graphs_to_device(ctx_graphs, device)

                    _, x_C = forward_get_emb(model, ctx_graphs)
                    if x_C is None:
                        continue
                    x_C = torch.nan_to_num(x_C, nan=0.0, posinf=0.0, neginf=0.0)

                    y_C = y_C.to(x_C.device, non_blocking=True).long()
                    y_C_onehot = F.one_hot(y_C, num_classes=args.num_classes).to(device=x_C.device, dtype=x_C.dtype)

                    logits_p = ppd_module(x_T, x_C, y_C_onehot)   
                    probs_p = torch.softmax(logits_p, dim=-1)   

                    probs_accum = probs_p if probs_accum is None else (probs_accum + probs_p)

                if probs_accum is not None:
                    probs_ppd = probs_accum / float(P)         
                    output = probs_ppd                            
                    output_logits = torch.log(probs_ppd.clamp_min(1e-12))  



            if args.loss == "graphppd":
                reservation = (-(output * torch.log(output.clamp_min(1e-12))).sum(-1)).cpu()

            elif args.loss == 'ce':
                reservation = 1 - output.data.max(1)[0].cpu()

            elif args.loss in ['log_ml', 'csc', 'csc_entropy']:
                top_logits, top_indices = torch.topk(output, k=2, dim=-1)
                first = top_indices[:, 0]
                second = top_indices[:, 1]

                f1 = torch.gather(output, 1, first.unsqueeze(1))
                f2 = torch.gather(output, 1, second.unsqueeze(1))

                diff = f1 - f2
                reservation = (-diff).view(-1)

                f1_ns = torch.gather(output_logits, 1, first.unsqueeze(1))
                f2_ns = torch.gather(output_logits, 1, second.unsqueeze(1))
                reservation_nosoftmax = (-(f1_ns - f2_ns)).view(-1)

            else:
    
                output, reservation = output[:, :-1], output[:, -1].cpu()
               


            if args.loss != "ours_proto":
                values, preds = output.data.max(1)
                preds = preds.cpu()

                h_tilde_batch = None 

            else:
                gamma = args.ours_gamma
                num_classes = args.num_classes
                raw_probs = F.softmax(output_logits[:, :num_classes], dim=1)  
                y0 = raw_probs.argmax(dim=1)                    

                h = batch_feats                                        
                prototypes_h = args.prototypes_h.to(h.device)    
                mu_batch_h = prototypes_h[y0]   

                h_tilde = (1 - gamma) * h + gamma * mu_batch_h    

                logits_tilde = model.linear(h_tilde)                 
                logits_tilde_main = logits_tilde[:, :num_classes]        

                preds = torch.argmax(logits_tilde_main, dim=1).cpu()
    

            correct_bool = list(preds.eq(targets.cpu()))
            prediction_list.extend(list(preds))
            correct_list.extend(correct_bool)


            if args.loss == "ce":
                T = getattr(args, "mc_dropout_T", 10)   
                B = targets.size(0)

                probs_all = [[] for _ in range(B)]
                was_training = model.training
                model.train()
                for m in model.modules():
                    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                        m.eval()
                for t in range(T):
                    logits_t = model(inputs)          
                    p_t = F.softmax(logits_t, dim=1)  

                    for i in range(B):
                        probs_all[i].append(p_t[i].detach().cpu())

                if not was_training:
                    model.eval()

                mc_scores_batch = []
                for i in range(B):
                    probs = torch.stack(probs_all[i], dim=0) 
                    p_bar = probs.mean(dim=0)          
                    y_hat = torch.argmax(p_bar).item()    

                    var = ((probs[:, y_hat] - p_bar[y_hat]) ** 2).mean()
                    mc_scores_batch.append(var.item())

                mc_ce_results[0].extend(mc_scores_batch)
                mc_ce_results[1].extend([bool(x) for x in correct_bool])



            if (
                args.loss == "ours_proto"
                and confidence_head is not None
                and args.arch.lower() in ['gin', 'gcn', 'mpnn']
            ):
              
                h_graph = batch_feats  
                giGT_batch = confidence_head(inputs)

                conf_scores_batch = (-giGT_batch).detach().cpu().view(-1).tolist()

                confhead_results[0].extend(conf_scores_batch)
                confhead_results[1].extend([bool(x) for x in correct_bool])

            abortion_results[0].extend(reservation.tolist())
            abortion_results[1].extend([bool(x) for x in correct_bool])

            if args.loss == 'log_ml':
                abortion_results_nosoftmax[0].extend(reservation_nosoftmax.tolist())
                abortion_results_nosoftmax[1].extend([bool(x) for x in correct_bool])

            pred_logits = F.softmax(output_logits, dim=-1)

            if args.loss == "graphppd" and probs_ppd is not None:
                pred_logits = probs_ppd

            sr_batch = (-pred_logits.max(dim=-1)[0]).view(-1).tolist()
            sr_results[0].extend(sr_batch)
            sr_results[1].extend([bool(x) for x in correct_bool])


    print("Eval time:", time.time() - start_t)


    abortion_scores = torch.tensor(abortion_results[0]).cpu().float()
    abortion_correct = torch.tensor(abortion_results[1]).cpu().float()

    sr_scores = torch.tensor(sr_results[0]).cpu().float()
    sr_correct = torch.tensor(sr_results[1]).cpu().float()

    abortion_curve = []
    bisection_method(abortion_scores, abortion_correct, abortion_curve)

    sr_curve = []
    bisection_method(sr_scores, sr_correct, sr_curve)

   
    if args.loss == "ours_proto" and confhead_results is not None:
        conf_scores = torch.tensor(confhead_results[0]).cpu().float()
        conf_correct = torch.tensor(confhead_results[1]).cpu().float()

        conf_curve = []
        bisection_method(conf_scores, conf_correct, conf_curve)


    if args.loss == "ce":
        mc_ce_scores = torch.tensor(mc_ce_results[0]).cpu().float()
        mc_ce_correct = torch.tensor(mc_ce_results[1]).cpu().float()

        mc_ce_curve = []
        bisection_method(mc_ce_scores, mc_ce_correct, mc_ce_curve)


    end = str(args.epochs) + 'selective risk.txt'
    with open(os.path.join(save_path, end), 'w') as file:

        file.write("\nAbstention\tLogit\tTest\tCoverage\tError")
        print("\nAbstention\tLogit\tTest\tCoverage\tError")
        for cov, acc in abortion_curve:
            err = (1 - acc) * 100
            print(f"{err:.3f}")
            file.write(f"\n{err:.3f}")

        file.write("\n\nSoftmax\tResponse\tTest\tCoverage\tError")
        print("Softmax\tResponse\tTest\tCoverage\tError")
        for cov, acc in sr_curve:
            err = (1 - acc) * 100
            print(f"{err:.3f}")
            file.write(f"\n{err:.3f}")


        if args.loss == "log_ml":
            abortion_scores_nosoftmax = torch.tensor(abortion_results_nosoftmax[0]).cpu().float()
            abortion_correct_nosoftmax = torch.tensor(abortion_results_nosoftmax[1]).cpu().float()

            curve_ns = []
            bisection_method(abortion_scores_nosoftmax, abortion_correct_nosoftmax, curve_ns)

            file.write("\nAbstention\tLogit\tTest\tCoverage\tError")
            print("\nAbstention_nosoftmax\tLogit\tTest\tCoverage\tError")

            for idx, (cov, acc) in enumerate(curve_ns):
                print('{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], cov * 100., (1 - acc) * 100))
                file.write('\n{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], cov * 100., (1 - acc) * 100))


        if args.loss == "ours_proto" and confhead_results is not None:
            file.write("\n\nGCN_ConfHead\tScore\tTest\tCoverage\tError")
            print("\nGCN_ConfHead\tScore\tTest\tCoverage\tError")

            for cov, acc in conf_curve:
                err = (1 - acc) * 100
                print(f"{err:.3f}")
                file.write(f"\n{err:.3f}")

        if args.loss == "ce":
            file.write("\n\nMC_Dropout\tScore\tTest\tCoverage\tError")
            print("\nMC_Dropout\tScore\tTest\tCoverage\tError")
            for cov, acc in mc_ce_curve:
                err = (1 - acc) * 100
                print(f"{err:.3f}")
                file.write(f"\n{err:.3f}")

    return


def cal_sim(average_features, features, prediction_list, correct_list):
    proto = torch.stack([torch.from_numpy(x) for x in average_features]) 
    proto = proto.to(features[0].device if torch.is_tensor(features[0]) else 'cpu')

    sims = []
    sims_per_class = [[[], []] for _ in range(args.num_classes)]
    nce_list = []
    dot_list = []

    for feat, pred, corr in zip(features, prediction_list, correct_list):
        if not torch.is_tensor(feat):
            feat = torch.from_numpy(feat)
        feat = feat.to(proto.device).float()
        pred = int(pred.item()) if hasattr(pred, 'item') else int(pred)

        s = F.cosine_similarity(
            feat.unsqueeze(0),
            proto[pred].unsqueeze(0),
            dim=1
        ).item()

        sims.append(-s) 
        sims_per_class[pred][0].append(-s)
        sims_per_class[pred][1].append(torch.tensor(float(corr), dtype=torch.float32))

        dot_list.append((feat * proto[pred]).sum().item())

        logits = (feat.unsqueeze(0) @ proto.t()) 
        labels = torch.tensor([pred], dtype=torch.long, device=proto.device)
        loss = F.cross_entropy(logits, labels)
        nce_list.append(loss.item())

    print("class-wise mean sim:", np.array([
        np.mean(c) if len(c) > 0 else 0.0
        for c in [[v for v in vlist[0]] for vlist in sims_per_class]
    ]))

    return torch.tensor(sims), sims_per_class, torch.tensor(nce_list), torch.tensor(dot_list)


def bisection_method(score, correct, results): 

    if score is None or (hasattr(score, "numel") and score.numel() == 0):
        print("⚠️ [bisection_method] score is empty — skip threshold computation.")
        return

    def calc_threshold(val_tensor, cov):
        threshold = np.percentile(np.array(val_tensor), 100 - cov * 100)
        return threshold

    neg_score = -score
    for coverage in expected_coverage: 
        threshold = calc_threshold(neg_score, coverage / 100)

        mask = (neg_score >= threshold)
        nData = len(correct)
        nSelected = mask.long().sum().item()
        if nSelected == 0:
            continue  
        isCorrect = correct[mask]
        nCorrectSelected = isCorrect.long().sum().item()
        passed_acc = nCorrectSelected / nSelected
        results.append((nSelected / nData, passed_acc))



if __name__ == '__main__':
    if args.loss in ['sat_entropy', 'csc_entropy', 'csc_sat_entropy']:
        if args.mode == 'tuning':
            base_path = os.path.join(args.save, args.dataset, args.loss, args.mode,
                                     f'entropy_coeff-{args.entropy}', args.arch)
        else:
            base_path = os.path.join(args.save, args.dataset, args.loss,
                                     f'entropy_coeff-{args.entropy}', args.arch)
    else:
        base_path = os.path.join(args.save, args.dataset, args.loss, args.arch)

    baseLR = state['lr']
    base_pretrain = args.pretrain
    for reward in reward_list:
        state['lr'] = baseLR

        save_path = os.path.join(
            base_path,
            f"o{reward:.2f}",
            f"k{args.moco_k}",
            f"m{args.moco_m:.4f}",
            f"t{args.moco_t:.3f}",
            f"pretrain{args.pretrain}",
            f"seed-{args.manualSeed}"
        )

        if args.evaluate:
        
            resume_path = os.path.join(save_path, "150.pth")
        elif args.resume:
            resume_path = os.path.join(base_path, 'resume', f"{args.start_epochs}.pth")

        print(f"🔍 Looking for checkpoint at: {resume_path}")
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f'❌ No model exists at "{resume_path}"')

        args.pretrain = base_pretrain

        if args.loss == 'gambler' and args.pretrain == 0:
            if args.dataset == 'cifar10' and reward < 6.3:
                args.pretrain = 100
            elif args.dataset == 'svhn' and reward < 6.0:
                args.pretrain = 50
            elif args.dataset == 'catsdogs':
                args.pretrain = 50

        main()
