from __future__ import print_function
import moco.CSC
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
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, closefig
from loss import SelfAdativeTraining, deep_gambler_loss, log_margin_loss, OursProtoLoss,cwr_loss
import time
import os, csv

from graph_models.util import load_data, separate_data
from graph_models.graph_dataset_adapter import GraphListDataset, collate_graph_batch
from graph_models.confidence_head import GCNConfidenceHead,GATConfidenceHead,ReadoutConfidenceHead

from graph_models.graphppd import GraphPPD

from typing import Dict, Optional, Tuple
import matplotlib
import matplotlib.pyplot as plt



graph_backbones = ["gin", "gcn", "mpnn"]

parser = argparse.ArgumentParser(description='Selective Classification for Self-Adaptive Training')
parser.add_argument('-d', '--dataset', metavar='DATASET', default='MUTAG',
                    choices=[
                        'MUTAG', 'PROTEINS', 'ENZYMES', 'DD', 'NCI1',
                        'COLLAB', 'IMDBBINARY', 'IMDBMULTI',
                        'PTC', 'REDDITBINARY', 'REDDITMULTI5K'
                    ],
                    help='dataset (default: MUTAG)')

parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (defaulst: 0)')
parser.add_argument('--mode', default='train', type=str, choices=['train', 'tuning'],
                    help='mode: tuning refers to 80/20 split of the training data for hyperparameter tuning')

parser.add_argument('-t', '--train', dest='evaluate', action='store_true',
                    help='train the model. When evaluate is true, training is ignored and trained models are loaded.')

parser.add_argument('-r', '--resume', dest='resume', action='store_true',
                    help='resume the model. When resume is true, training is ignored and trained models are loaded.')

parser.add_argument('-w', '--warmup', dest='warmup', action='store_true',
                    help='warmup the reward.')
parser.add_argument('--ir', '--initialreward', default=1e-6, type=float,
                    metavar='IR', help='initial reward')
parser.add_argument('--warmup_epochs', default=0, type=int, metavar='WN',
                    help='warm-up iterations')

parser.add_argument('--start_epochs', default=0, type=int, metavar='N',
                    help='resume epochs to run')
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
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500], #[25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500]
                        help='Multiply learning rate by gamma at the scheduled epochs (defaßult: 25,50,75,100,125,150,175,200,225,250,275)')
parser.add_argument('--gamma', type=float, default=0.5, help='LR is multiplied by gamma on schedule (default: 0.5)')  #0.5
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--sat-momentum', default=0.9, type=float, help='momentum for sat')#fixed for sat
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-o', '--rewards', dest='rewards', type=float, nargs='+', default=[4.6],#fixed for gambler【4.6】 fixed for CCL-SC 【0.5】
                    metavar='o', help='The reward o for a correct prediction; Abstention has a reward of 1. Provided parameters would be stored as a list for multiple runs.')
parser.add_argument('--pretrain', type=int, default=200,#fixed
                    help='Number of pretraining epochs using the cross entropy loss, so that the learning can always start. Note that it defaults to 100 if dataset==cifar10 and reward<6.1, and the results in the paper are reproduced.')
parser.add_argument('--coverage', type=float, nargs='+',default=[100.,99.,98.,97.,95.,90.,85.,80.,75.,70.,60.,50.,40.,30.,20.,10.],
                    help='the expected coverages used to evaluated the accuracies after abstention')

parser.add_argument('-s', '--save', default='save', type=str, metavar='PATH',
                    help='path to save checkpoint (default: save)')


parser.add_argument('--loss', default='sat', type=str,
    choices=['sat', 'ce', 'gambler', 'sat_entropy', 'cwr','csc', 'log_ml', 'csc_entropy', 'csc_sat_entropy', 'ours_proto', 'graphppd'],
    help='loss function (sat, ce, gambler, sat_entropy, cwr,csc, log_ml, csc_entropy, csc_sat_entropy, ours)')
parser.add_argument('--alpha', type=float, default=0.1,
                    help='weight for Lvar in Lours (Lce + alpha * Lvar)')

parser.add_argument('--entropy', type=float, default=0.0, help='Entropy Coefficient for the SAT Loss (default: 0.0)') 
parser.add_argument(
    '--arch', '-a', metavar='ARCH', default='gcn',
    choices=graph_backbones,
    help='model architecture: ' +
         ' | '.join( graph_backbones) +
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
print(f"[INFO] Using loss = {args.loss}, alpha = {args.alpha}")

state = {k: v for k, v in args._get_kwargs()}


expected_coverage = args.coverage
reward_list = args.rewards


use_cuda = torch.cuda.is_available()


if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
hidden_features = None
hidden_features_k = None
full = False
num_classes=2 

def hook_fn(module, input, output):
    global hidden_features
    hidden_features = output

def hook_fn_k(module, input, output):
    global hidden_features_k
    hidden_features_k = output


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


@torch.no_grad()
def extract_embeddings_batch(
    model,
    inputs,
    epoch: int,
    device,
    args,
):

    model.eval()

    if isinstance(inputs, torch.Tensor):
        inputs = inputs.to(device, non_blocking=True)
    elif isinstance(inputs, list):
        _move_graphs_to_device(inputs, device)
    _ = model(inputs)

    h = getattr(model, "cached_hidden", None)
    if h is None:
        return {} 

    if h.dim() > 2:
        h = torch.flatten(h, 1)

    h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

    out: Dict[str, torch.Tensor] = {}
    out["h_raw"] = h

    eps = getattr(args, "ours_eps", 1e-8)
    h_norm = h / (h.norm(p=2, dim=1, keepdim=True) + eps)
    h_norm = torch.nan_to_num(h_norm, nan=0.0, posinf=0.0, neginf=0.0)
    out["h_norm"] = h_norm

    if (
        getattr(args, "tsne_with_shrink", False)
        and args.loss == "ours_proto"
        and epoch >= args.pretrain
        and hasattr(args, "prototypes_h")
    ):
        
        logits = model.linear(h) if hasattr(model, "linear") else None
        if logits is not None:
            y0 = torch.softmax(logits, dim=1).argmax(dim=1) 
            mu = args.prototypes_h[y0].to(h.device)         
            h_shrink = (1 - args.ours_gamma) * h + args.ours_gamma * mu
            h_shrink = torch.nan_to_num(h_shrink, nan=0.0, posinf=0.0, neginf=0.0)
            out["h_shrink"] = h_shrink

           
            h_shrink_norm = h_shrink / (h_shrink.norm(p=2, dim=1, keepdim=True) + eps)
            h_shrink_norm = torch.nan_to_num(h_shrink_norm, nan=0.0, posinf=0.0, neginf=0.0)
            out["h_shrink_norm"] = h_shrink_norm

    return out


@torch.no_grad()
def collect_embeddings_from_loader(
    model,
    loader,
    epoch: int,
    device,
    args,
    which: str = "h_raw",
    max_points: int = 7000,
):

    model.eval()

    X_list = []
    y_list = []
    n_collected = 0

    for batch_idx, batch_data in enumerate(loader):
        inputs, targets, indices = batch_data

     
        if isinstance(targets, torch.Tensor):
            targets_t = targets
        else:
            targets_t = torch.tensor(targets)

     
        emb_dict = extract_embeddings_batch(
            model=model,
            inputs=inputs,
            epoch=epoch,
            device=device,
            args=args,
        )

        if (emb_dict is None) or (which not in emb_dict):

            continue

        h = emb_dict[which]  
        if h is None:
            continue

        B = h.size(0)
        remain = max_points - n_collected
        if remain <= 0:
            break

        take = min(B, remain)

        X_list.append(h[:take].detach().cpu())
        y_list.append(targets_t[:take].detach().cpu())

        n_collected += take

        if n_collected >= max_points:
            break

    if len(X_list) == 0:
        return None, None

    X = torch.cat(X_list, dim=0).numpy()
    y = torch.cat(y_list, dim=0).numpy().astype(int)

    return X, y


@torch.no_grad()
def compute_epoch_lintra_hnorm(
    model,
    loader,
    epoch: int,
    device,
    args,
    max_points: int = 7000,
):

    model.eval()

    X, y = collect_embeddings_from_loader(
        model=model,
        loader=loader,
        epoch=epoch,
        device=device,
        args=args,
        which="h_norm",
        max_points=max_points,
    )
    if X is None or y is None or len(y) < 5:
        return None

    X = np.asarray(X, dtype=np.float32)  
    y = np.asarray(y, dtype=np.int64) 

    lintra_per_class = {}
    n_per_class = {}

    lintra_sum = 0.0
    c_eff = 0

    for c in range(args.num_classes):
        idx = (y == c)
        n_c = int(idx.sum())
        n_per_class[c] = n_c

        if n_c <= 1:
            continue

        Xc = X[idx]
        mu = Xc.mean(axis=0, keepdims=True)

        lintra_c = float(((Xc - mu) ** 2).sum(axis=1).mean())

        lintra_per_class[c] = lintra_c
        lintra_sum += lintra_c
        c_eff += 1

    lintra_mean = lintra_sum / max(1, c_eff)

    return {
        "lintra_sum": lintra_sum,
        "lintra_mean": lintra_mean,
        "lintra_per_class": lintra_per_class,
        "n_per_class": n_per_class,
        "c_eff": c_eff,
        "N": int(len(y)),
    }

def save_lintra_curve(lintra_epochs, lintra_records, save_path: str, num_classes: int):
    os.makedirs(save_path, exist_ok=True)
    csv_path = os.path.join(save_path, "lintra_curve.csv")
    header = ["epoch", "lintra_mean", "lintra_sum", "c_eff", "N"]
    header += [f"lintra_c{c}" for c in range(num_classes)]
    header += [f"n_c{c}" for c in range(num_classes)]

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        for e, rec in zip(lintra_epochs, lintra_records):
            row = [
                int(e),
                float(rec["lintra_mean"]),
                float(rec["lintra_sum"]),
                int(rec["c_eff"]),
                int(rec["N"]),
            ]

            per = rec["lintra_per_class"]
            row += [float(per[c]) if c in per else "" for c in range(num_classes)]

            nper = rec["n_per_class"]
            row += [int(nper.get(c, 0)) for c in range(num_classes)]

            w.writerow(row)

    print(f"[Lintra] saved csv: {csv_path}")

    lintra_mean_values = [rec["lintra_mean"] for rec in lintra_records]

    plt.figure(figsize=(7, 4))
    plt.plot(lintra_epochs, lintra_mean_values)
    plt.xlabel("Epoch")
    plt.ylabel("Mean intra-class variance on h_norm")
    plt.title("Mean intra-class variance curve (trainloader, h_norm)")
    plt.tight_layout()

    fig_path = os.path.join(save_path, "lintra_mean_curve.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[Lintra] saved figure: {fig_path}")


    plt.figure(figsize=(7, 4))
    for c in range(num_classes):
        vals_c = []
        for rec in lintra_records:
            vals_c.append(rec["lintra_per_class"].get(c, None))
        vals_c = [v if v is not None else float("nan") for v in vals_c]
        plt.plot(lintra_epochs, vals_c, label=f"class {c}")

    plt.xlabel("Epoch")
    plt.ylabel("Intra-class variance on h_norm")
    plt.title("Per-class intra-class variance (trainloader, h_norm)")
    plt.legend(fontsize=9)
    plt.tight_layout()

    fig_path2 = os.path.join(save_path, "lintra_per_class_curve.png")
    plt.savefig(fig_path2, dpi=200)
    plt.close()
    print(f"[Lintra] saved figure: {fig_path2}")


def main():
    print(args)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not resume_path and not os.path.isdir(save_path):
        mkdir_p(save_path)

    confidence_head = None  
    ppd_module = None

    graph_datasets = {
    "MUTAG", "PROTEINS", "ENZYMES", "DD", "NCI1",
    "COLLAB", "IMDBBINARY", "IMDBMULTI",
    "PTC", "REDDITBINARY", "REDDITMULTI5K"
    }

    if args.dataset in graph_datasets:
        print(f"=> Using graph dataset: {args.dataset}")
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
        train_graphs, test_graphs = separate_data(all_graphs, seed=args.manualSeed or 0, fold_idx=0)

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
        devloader = trainloader


    print("==> creating model '{}'".format(args.arch))

    hidden_features = None
    hidden_features_k = None


    if args.arch.lower() == "gcn":
        # print("=> Building PYG-GCN model...")
        print("=> Building PYG-GraphTransformer model...")

        from graph_models.pyg_backbones import PYG_GCN,PYG_GraphTransformer

        input_dim = trainset[0][0].node_features.shape[1]
        hidden_dim = 64

        model = PYG_GCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=args.num_classes if args.loss in ['ce', 'log_ml', 'csc', 'csc_entropy', 'ours_proto', 'graphppd']
                        else args.num_classes + 1,
            num_layers=3,
            dropout=0.5,
            device="cuda"
        ).cuda()

        # model = PYG_GraphTransformer(
        #     input_dim=input_dim,
        #     hidden_dim=hidden_dim,
        #     output_dim=args.num_classes if args.loss in ['ce', 'log_ml', 'csc', 'csc_entropy', 'ours_proto', 'graphppd']
        #                 else args.num_classes + 1,
        #     num_layers=3,
        #     dropout=0.5,
        #     device="cuda"
        # ).cuda()

        args.moco_dim = hidden_dim

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
            print(f"[GraphPPD] Build ppd_module with dx={hidden_dim}, C={args.num_classes}")
        if args.loss == "graphppd":
            print("[GraphPPD] ppd_module created:", type(ppd_module))
            n_ppd = sum(p.numel() for p in ppd_module.parameters())
            print(f"[GraphPPD] ppd_module params: {n_ppd/1e6:.3f}M")




        if args.loss == "ours_proto":

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
    else:
        raise ValueError(f"Unknown architecture {args.arch}. Only gin / gcn / mpnn are supported.")


    model = model.cuda()
    model_k = copy.deepcopy(model)
    cudnn.benchmark = True

    print('    Total params: %.2fM' % (
        sum(p.numel() for p in model.parameters()) / 1e6
    ))


    print("==> Registering graph backbone hooks for MoCo...")

    def hook_q(module, inp, out):
        global hidden_features
        cached = getattr(module, "cached_hidden", None)
        if isinstance(cached, torch.Tensor):
            if args.loss in ["ours_proto", "graphppd"]:
                hidden_features = cached    
            else:
                hidden_features = cached.detach() 


    def hook_k(module, inp, out):
        global hidden_features_k
        cached = getattr(module, "cached_hidden", None)
        if isinstance(cached, torch.Tensor):
            hidden_features_k = cached.detach()

    model.register_forward_hook(hook_q)
    model_k.register_forward_hook(hook_k)

    print(f"[Graph-{args.arch}] MoCo feature dim = {args.moco_dim}")



    if args.loss == 'ours_proto':
        args.feature_dim = args.moco_dim
        args.prototypes_h = torch.zeros(args.num_classes, args.feature_dim, device="cuda")
        args.prototypes_u = torch.zeros(args.num_classes, args.feature_dim, device="cuda")
        args.prototype_sigma = torch.zeros(args.num_classes, device="cuda")
        args.prototype_momentum = args.ours_rho

        print(
            f"[OURS] Init prototypes_h {tuple(args.prototypes_h.shape)}, "
            f"prototypes_u {tuple(args.prototypes_u.shape)}, rho={args.prototype_momentum}"
        )

    criterion2 = None
    if args.pretrain:
        criterion = nn.CrossEntropyLoss()

    if args.loss in ['ce', 'csc', 'csc_entropy']:
        criterion = nn.CrossEntropyLoss()

    elif args.loss == 'gambler':
        criterion = deep_gambler_loss

        

    elif args.loss == 'cwr':
        args.cwr_d = 0.5
        criterion = cwr_loss

    elif args.loss in ['sat', 'sat_entropy']:
        criterion = SelfAdativeTraining(num_examples=len(trainset),
                                        num_classes=args.num_classes,
                                        mom=args.sat_momentum)

    elif args.loss == 'log_ml':
        criterion = log_margin_loss

    elif args.loss == 'csc_sat_entropy':
        criterion = nn.CrossEntropyLoss()
        criterion2 = SelfAdativeTraining(num_examples=len(trainset),
                                        num_classes=args.num_classes,
                                        mom=args.sat_momentum)

    elif args.loss == 'ours_proto':
        criterion = OursProtoLoss()



   
    if args.loss == "ours_proto" and confidence_head is not None:
        optim_params = list(model.parameters()) + list(confidence_head.parameters())
        print("[OPT] training model + confidence_head")
    elif args.loss == "graphppd" and ppd_module is not None:
        optim_params = list(model.parameters()) + list(ppd_module.parameters())
        print("[OPT] training model + ppd_module")
    else:
        optim_params = model.parameters()
        print("[OPT] training model only")




    optimizer = optim.SGD(
        optim_params,
        lr=state['lr'],
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    title = args.dataset + '-' + args.arch + ' o={:.2f}'.format(reward)
    logger = Logger(os.path.join(save_path, 'eval.txt' if args.evaluate else 'log.txt'), title=title)
    logger.set_names(['Epoch', 'Learning Rate', 'Train Loss', 'Train Loss2','Test Loss', 'Train Err.', 'Test Err.', 'MOCO Err.'])
    

    archive = moco.CSC.MoCo(
        args.moco_dim, 
        args.moco_k,  
        args.moco_m,  
        args.moco_t,  
        num_class = args.num_classes
    )

    best_acc = 0

    stage2_best_reset_done = False

    lintra_epochs = []
    lintra_records = []

    for epoch in range(args.start_epochs, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        print('\n'+save_path)
        print('Epoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        
       
        train_loss, train_loss2, train_acc, moco_top1 = train(trainloader, archive, model, model_k, criterion, criterion2, optimizer, epoch, use_cuda,confidence_head,ppd_module=ppd_module,trainset=trainset) 

        test_loss, test_acc = test(trainloader, testloader, model, criterion, epoch, use_cuda,confidence_head=confidence_head,ppd_module=ppd_module)


        epoch_human = epoch + 1
        lintra_start = 1 

        if epoch_human >= lintra_start:
            device = torch.device("cuda" if use_cuda else "cpu")
            rec = compute_epoch_lintra_hnorm(
                model=model,
                loader=trainloader,
                epoch=epoch_human,
                device=device,
                args=args,
                max_points=7000,
            )
            if rec is not None:
                lintra_epochs.append(epoch_human)
                lintra_records.append(rec)
                print(
                    f"[Lintra] epoch={epoch_human} "
                    f"mean={rec['lintra_mean']:.6f} sum={rec['lintra_sum']:.6f} "
                    f"c_eff={rec['c_eff']} N={rec['N']} "
                    f"per_class={rec['lintra_per_class']}"
                )
            else:
                print(f"[Lintra] epoch={epoch_human} skip (too few points / no embeddings)")


        
        print(train_acc, train_loss2, test_acc, moco_top1 * 100)

        if (not stage2_best_reset_done) and (epoch >= args.pretrain):
            best_acc = 0
            stage2_best_reset_done = True
            print(f"[RESET] Enter Stage2 at epoch {epoch}. best_acc reset to 0.")

        if best_acc <= test_acc:
            filepath = os.path.join(save_path, "{:d}".format(619) + ".pth")
            save_dict = {"state_dict": model.state_dict()}

            if args.loss == "ours_proto":
                save_dict["prototypes_h"] = args.prototypes_h.detach().cpu()
                save_dict["prototypes_u"] = args.prototypes_u.detach().cpu()

                save_dict["prototype_momentum"] = args.prototype_momentum
                save_dict["ours_gamma"] = args.ours_gamma
                save_dict["ours_alpha_intra"] = args.ours_alpha_intra

                save_dict["confidence_head"] = confidence_head.state_dict()
            elif args.loss == "graphppd":
                assert ppd_module is not None, "ppd_module is None but loss==graphppd"
                save_dict["ppd_module"] = ppd_module.state_dict()
                save_dict["graphppd_hparams"] = {
                    "dx": args.moco_dim,
                    "dh": args.ppd_dh,
                    "heads": args.ppd_heads,
                    "mlp_hidden": args.ppd_mlp_hidden,
                    "mlp_layers": args.ppd_mlp_layers,
                    "dropout": 0.0,
                    "K": args.ppd_context_size,
                    "P": args.ppd_P,
                }

            torch.save(save_dict, filepath)

            best_acc = test_acc
            print("best_acc: ", best_acc)


        if (epoch+1) % args.save_model_step == 0:
            filepath = os.path.join(save_path, "{:d}".format(epoch+1) + ".pth")
            save_dict = {"state_dict": model.state_dict()}

            if args.loss == "ours_proto":
                save_dict["prototypes_h"] = args.prototypes_h.detach().cpu()
                save_dict["prototypes_u"] = args.prototypes_u.detach().cpu()

                save_dict["prototype_momentum"] = args.prototype_momentum
                save_dict["ours_gamma"] = args.ours_gamma
                save_dict["ours_alpha_intra"] = args.ours_alpha_intra

                save_dict["confidence_head"] = confidence_head.state_dict()
            elif args.loss == "graphppd":
                assert ppd_module is not None, "ppd_module is None but loss==graphppd"
                save_dict["ppd_module"] = ppd_module.state_dict()
                save_dict["graphppd_hparams"] = {
                    "dx": args.moco_dim,
                    "dh": args.ppd_dh,
                    "heads": args.ppd_heads,
                    "mlp_hidden": args.ppd_mlp_hidden,
                    "mlp_layers": args.ppd_mlp_layers,
                    "dropout": 0.0,
                    "K": args.ppd_context_size,
                    "P": args.ppd_P,
                }

            torch.save(save_dict, filepath)
        
        logger.append([epoch+1, state['lr'], train_loss, train_loss2, test_loss, 100-train_acc, 100-test_acc, 100-moco_top1 * 100])

    filepath = os.path.join(save_path, "{:d}".format(args.epochs) + ".pth")
    save_dict = {"state_dict": model.state_dict()}

    if args.loss == "ours_proto":
        save_dict["prototypes_h"] = args.prototypes_h.detach().cpu()
        save_dict["prototypes_u"] = args.prototypes_u.detach().cpu()

        save_dict["prototype_momentum"] = args.prototype_momentum
        save_dict["ours_gamma"] = args.ours_gamma
        save_dict["ours_alpha_intra"] = args.ours_alpha_intra

        save_dict["confidence_head"] = confidence_head.state_dict()
    elif args.loss == "graphppd":
        assert ppd_module is not None, "ppd_module is None but loss==graphppd"
        save_dict["ppd_module"] = ppd_module.state_dict()
        save_dict["graphppd_hparams"] = {
            "dx": args.moco_dim,
            "dh": args.ppd_dh,
            "heads": args.ppd_heads,
            "mlp_hidden": args.ppd_mlp_hidden,
            "mlp_layers": args.ppd_mlp_layers,
            "dropout": 0.0,
            "K": args.ppd_context_size,
            "P": args.ppd_P,
        }

    torch.save(save_dict, filepath)

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

    if len(lintra_epochs) > 0:
        save_lintra_curve(lintra_epochs, lintra_records, save_path, num_classes=args.num_classes)
    else:
        print("[Lintra] No lintra points collected, skip saving.")


def linear_warmup(current_epoch, warmup_epochs, initial_reward, final_reward):
    if current_epoch < warmup_epochs:
        warmup_increment = (final_reward - initial_reward) / warmup_epochs
        return initial_reward + warmup_increment * current_epoch
    else:
        return final_reward
       

def compute_L_intra(feats, targets, prototypes, logits):
    with torch.no_grad():
        probs = F.softmax(logits, dim=1)
        w_r = probs.max(dim=1)[0]

    num_classes = prototypes.size(0)
    device = feats.device
    L_intra = feats.new_tensor(0.0)

    for c in range(num_classes):
        mask = (targets == c)
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue

        h_c = feats[idx]
        mu_c = prototypes[c].to(device)
        w_c = w_r[idx]

        diff = h_c - mu_c
        dist_sq = diff.pow(2).sum(dim=1)

        L_c = (w_c * dist_sq).sum() / (idx.numel() + 1e-8)
        L_intra = L_intra + L_c

    return L_intra





def train(trainloader, archive, model, model_k, criterion, criterion2, optimizer, epoch, use_cuda,confidence_head=None, ppd_module=None, trainset=None):
    loss = 0.0

    cwr_d = 0.5
    global hidden_features, hidden_features_k


    if args.loss == "ours_proto" and confidence_head is not None:
        pass
  
    model.train()
    model_k.train()
    if args.arch =='vgg16_bn':
        feature_extractor = model.features

    if epoch == args.pretrain:
        for param_q, param_k in zip(
                model.parameters(), model_k.parameters()
            ):
                param_k.data = param_q.data 
    if epoch > args.pretrain:
        for param_q, param_k in zip(
                model.parameters(), model_k.parameters()
            ):
                param_k.data = param_k.data * args.moco_m + param_q.data * (1.0 - args.moco_m)
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses2 = AverageMeter()

    top1 = AverageMeter()
    top5 = AverageMeter()
    moco_top1 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    print("TrainLoader Length:", len(trainloader))

    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    train_t0 = time.perf_counter()

    current_reward = reward 
    if args.warmup and epoch >= args.pretrain:
        current_reward = linear_warmup(epoch - args.pretrain, args.warmup_epochs, args.ir, reward)
        print('current_reward: ', current_reward)
        

    for batch_idx,  batch_data in tqdm(enumerate(trainloader)):
        inputs, targets, indices = batch_data 
        data_time.update(time.time() - end)

        if use_cuda:
            device = torch.device("cuda")
            for g in inputs:     
                if hasattr(g, "to"):
                    g.to(device)
            targets = targets.cuda()


        inputs = inputs
        targets = torch.autograd.Variable(targets)

        with torch.no_grad():
            outputs_k = model_k(inputs)
       
        outputs = model(inputs) 
        
        global full_k1
        global full_k2
        if epoch >= args.pretrain and full_k1 and full_k2:
            if args.loss in ['csc', 'csc_entropy', 'csc_sat_entropy']:
                if args.arch != 'vgg16_bn':
                    temp_full_k1, temp_full_k2, moco_error, loss2 = archive(torch.flatten(hidden_features, 1), torch.flatten(hidden_features_k, 1), targets, outputs, outputs_k, epoch + 1, args.pretrain, full_k1 and full_k2)

                else:
                    temp_full_k1, temp_full_k2, moco_error, loss2 = archive(outputs_projection, hidden_features_k, targets, outputs, outputs_k, epoch + 1, args.pretrain, full_k1 and full_k2)

                full_k1 = full_k1 or temp_full_k1
                full_k2 = full_k2 or temp_full_k2

            if args.loss == 'gambler':
                loss = criterion(outputs, targets, reward)
              

            elif args.loss == 'cwr':
                loss = cwr_loss(outputs, targets, d=cwr_d)     

            elif args.loss == 'sat':
                loss = criterion(outputs, targets, indices)
              
            elif args.loss == 'sat_entropy':
                softmax = nn.Softmax(-1)
                loss = criterion(outputs, targets, indices) + (args.entropy * (-softmax(outputs[:, :-1]) * outputs[:, :-1]).sum(-1)).mean()
            elif args.loss == 'log_ml':
                loss = F.cross_entropy(outputs, targets) + log_margin_loss(outputs, targets, reward)

            elif args.loss == 'csc':
                if full_k1 and full_k2:
                    batch_size = inputs.size(0) if isinstance(inputs, torch.Tensor) else len(inputs)
                    losses2.update(loss2.item(), batch_size)
                    moco_top1.update(moco_error, batch_size)
                  
                    loss = criterion(outputs, targets) + loss2 * current_reward
                else:
                    loss = F.cross_entropy(outputs, targets)
            elif args.loss == 'csc_entropy':
                if full_k1 and full_k2:
                    softmax = nn.Softmax(-1)
                    batch_size = inputs.size(0) if isinstance(inputs, torch.Tensor) else len(inputs)
                    losses2.update(loss2.item(), batch_size)
                    moco_top1.update(moco_error, batch_size)
                    loss = criterion(outputs, targets) + loss2 * current_reward + (args.entropy * (-softmax(outputs) * outputs).sum(-1)).mean()
                else:
                    loss = F.cross_entropy(outputs, targets)
            elif args.loss == 'csc_sat_entropy':
                if full_k1 and full_k2:
                    softmax = nn.Softmax(-1)
                    batch_size = inputs.size(0) if isinstance(inputs, torch.Tensor) else len(inputs)
                    losses2.update(loss2.item(), batch_size)
                    moco_top1.update(moco_error, batch_size)
                    loss = criterion(outputs, targets) + loss2 * current_reward + criterion2(outputs, targets, indices) + (args.entropy * (-softmax(outputs[:, :-1]) * outputs[:, :-1]).sum(-1)).mean()
                else:
                    loss = F.cross_entropy(outputs[:, :-1], targets)
                    outputs = outputs[:, :-1]
            


            elif args.loss == "ours_proto":
                feats = hidden_features  
                if feats is None:
                    print("hook is not working, using CE loss")
                    loss = F.cross_entropy(outputs, targets)
                else:
                    if feats.dim() > 2:
                        feats = torch.flatten(feats, 1)

                    feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

                    eps = getattr(args, "ours_eps", 1e-8)  
                    u = feats / (feats.norm(p=2, dim=1, keepdim=True) + eps)
                    u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)


                    rho = args.ours_rho

                    with torch.no_grad():
                        if args.prototypes_h.device != feats.device:
                            args.prototypes_h = args.prototypes_h.to(feats.device)
                        if args.prototypes_u.device != feats.device:
                            args.prototypes_u = args.prototypes_u.to(feats.device)

                        batch_centers_u = torch.zeros(args.num_classes, u.size(1), device=u.device, dtype=u.dtype)

                        for c in range(args.num_classes):
                            mask = (targets == c)
                            if mask.sum() > 0:
                                batch_mu_h = feats[mask].mean(dim=0) 
                                batch_mu_u = u[mask].mean(dim=0)  

                                batch_centers_u[c] = u[mask].mean(dim=0) 

                                args.prototypes_h[c] = (1 - rho) * args.prototypes_h[c] + rho * batch_mu_h
                                args.prototypes_u[c] = (1 - rho) * args.prototypes_u[c] + rho * batch_mu_u

                        args.prototypes_u = args.prototypes_u / (args.prototypes_u.norm(p=2, dim=1, keepdim=True) + eps)
                        args.prototypes_u = torch.nan_to_num(args.prototypes_u, nan=0.0, posinf=0.0, neginf=0.0)


                    if epoch < args.pretrain:
                        logits = outputs
                        ce_loss = F.cross_entropy(logits, targets)

                        L_intra = logits.new_tensor(0.0)
                        confidence_loss = logits.new_tensor(0.0)

                    else:
                     
                        mu_batch_h = args.prototypes_h[targets]                  
                        h_shrink = (1 - args.ours_gamma) * feats + args.ours_gamma * mu_batch_h

                        logits = model.linear(h_shrink)                      
                        ce_loss = F.cross_entropy(logits, targets)

                    
                        L_intra = compute_L_intra(
                            feats=u,
                            targets=targets,
                            prototypes=args.prototypes_u,#or prototypes=batch_centers_u
                            logits=logits,           
                        )


                        confidence_loss = logits.new_tensor(0.0)
                        if confidence_head is not None and isinstance(inputs, list):
                            feats_detached = feats.detach()         
                            giGT = confidence_head(inputs)     

                            ti = (logits.argmax(dim=1) == targets).float()      
                            confidence_loss = F.binary_cross_entropy(giGT, ti)

                            if batch_idx == 0:
                                pos_mask = (ti == 1)
                                neg_mask = (ti == 0)
                                if pos_mask.any() and neg_mask.any():
                                    print(
                                        f"[DEBUG][{args.dataset}][epoch {epoch}] "
                                        f"ConfBCE={confidence_loss.item():.4f} "
                                        f"g_mean={giGT.mean().item():.4f} "
                                        f"g_pos={giGT[pos_mask].mean().item():.4f} "
                                        f"g_neg={giGT[neg_mask].mean().item():.4f} "
                                        f"pos_ratio={pos_mask.float().mean().item():.2f}"
                                    )

                    loss = (
                        ce_loss
                        + args.ours_alpha_intra * L_intra
                        + args.conf_weight * confidence_loss
                    )

                    if batch_idx == 0 or batch_idx % 1 == 0:
                        phase = "PRETRAIN" if epoch < args.pretrain else "STAGE2"
                        print(
                            f"[OursProto][{phase}] epoch {epoch} batch {batch_idx} | "
                            f"CE={ce_loss.item():.4f} "
                            f"L_intra={L_intra.item():.4f} "
                            f"Conf={confidence_loss.item():.4f} "
                            f"Total={loss.item():.4f}"
                        )
            elif args.loss == "graphppd":
                if epoch < args.pretrain:
                    logits = outputs
                    loss = F.cross_entropy(logits, targets)
                else:
                    if ppd_module is None:
                        raise ValueError("ppd_module is None but loss==graphppd")

                    x_T = hidden_features         
                    y_T = targets        

                    if x_T is None:
                        break
                    else:
                        if x_T.dim() > 2:
                            x_T = torch.flatten(x_T, 1)
                        x_T = torch.nan_to_num(x_T, nan=0.0, posinf=0.0, neginf=0.0)

                        ctx_graphs, y_C = sample_context_from_trainset(trainset, args.ppd_context_size)

                        if use_cuda:
                            device = torch.device("cuda")
                            for g in ctx_graphs:
                                if hasattr(g, "to"):
                                    g.to(device)

                        y_C = y_C.to(x_T.device, non_blocking=True).long() 

                    
                        hf_saved = hidden_features 

                        with torch.no_grad():
                            _ = model(ctx_graphs)       
                            x_C = hidden_features    

                        hidden_features = hf_saved     


                        if x_C.dim() > 2:
                            x_C = torch.flatten(x_C, 1)
                        x_C = torch.nan_to_num(x_C, nan=0.0, posinf=0.0, neginf=0.0)


                        y_C_onehot = F.one_hot(y_C, num_classes=args.num_classes).to(
                            device=x_C.device, dtype=x_C.dtype
                        ) 
      
                        logits = ppd_module(x_T, x_C, y_C_onehot)  
                        loss = F.cross_entropy(logits, y_T)

                if batch_idx == 0:
                    print(f"[GraphPPD] epoch={epoch} stage={'PRETRAIN' if epoch < args.pretrain else 'STAGE2'} "
                        f"logits={tuple(outputs.shape)} target[min,max]=({int(targets.min())},{int(targets.max())}) "
                        f"loss={loss.item():.4f}")



            else:
                loss = criterion(outputs, targets)


        else:
            moco_error = 1/ num_classes


            if args.arch != 'vgg16_bn' and epoch >= args.pretrain:
                temp_full_k1, temp_full_k2, moco_error = archive(torch.flatten(hidden_features, 1), torch.flatten(hidden_features_k, 1), targets, outputs, outputs_k, epoch + 1, args.pretrain, full_k1 and full_k2)

            else:
                if epoch >= args.pretrain:
                    temp_full_k1, temp_full_k2, moco_error = archive(outputs_projection, hidden_features_k, targets, outputs, outputs_k, epoch + 1, args.pretrain, full_k1 and full_k2)


            try:
                full_k1 = full_k1 or temp_full_k1
            except NameError:
                full_k1 = None

            try:
                full_k2 = full_k2 or temp_full_k2
            except NameError:
                full_k2 = None


            if args.loss == 'ce' or args.loss == 'log_ml' or args.loss == 'csc' or args.loss == 'csc_entropy':
                loss = F.cross_entropy(outputs, targets)

            elif args.loss == "ours_proto":
                feats = hidden_features 
                if feats is None:
                    print("hook is not working, using CE loss")
                    loss = F.cross_entropy(outputs, targets)
                else:
                    if feats.dim() > 2:
                        feats = torch.flatten(feats, 1)

                    feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

                  
                    eps = getattr(args, "ours_eps", 1e-8) 
                    u = feats / (feats.norm(p=2, dim=1, keepdim=True) + eps)
                    u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)

                    rho = args.ours_rho

                    with torch.no_grad():
                        if args.prototypes_h.device != feats.device:
                            args.prototypes_h = args.prototypes_h.to(feats.device)
                        if args.prototypes_u.device != feats.device:
                            args.prototypes_u = args.prototypes_u.to(feats.device)

                        batch_centers_u = torch.zeros(args.num_classes, u.size(1), device=u.device, dtype=u.dtype)#加入消融EMA

                        for c in range(args.num_classes):
                            mask = (targets == c)
                            if mask.sum() > 0:
                                batch_mu_h = feats[mask].mean(dim=0) 
                                batch_mu_u = u[mask].mean(dim=0)    

                                batch_centers_u[c] = u[mask].mean(dim=0)  

                                args.prototypes_h[c] = (1 - rho) * args.prototypes_h[c] + rho * batch_mu_h
                                args.prototypes_u[c] = (1 - rho) * args.prototypes_u[c] + rho * batch_mu_u

                        args.prototypes_u = args.prototypes_u / (args.prototypes_u.norm(p=2, dim=1, keepdim=True) + eps)
                        args.prototypes_u = torch.nan_to_num(args.prototypes_u, nan=0.0, posinf=0.0, neginf=0.0)

                    if epoch < args.pretrain:
                        logits = outputs
                        ce_loss = F.cross_entropy(logits, targets)

                        L_intra = logits.new_tensor(0.0)
                        confidence_loss = logits.new_tensor(0.0)

                    else:
                        mu_batch_h = args.prototypes_h[targets]               
                        h_shrink = (1 - args.ours_gamma) * feats + args.ours_gamma * mu_batch_h

                        logits = model.linear(h_shrink)                       
                        ce_loss = F.cross_entropy(logits, targets)

                    
                        L_intra = compute_L_intra(
                            feats=u,
                            targets=targets,
                            prototypes=args.prototypes_u,
                            logits=logits,        
                        )
                        

                
                        confidence_loss = logits.new_tensor(0.0)
                        if confidence_head is not None and isinstance(inputs, list):
                            feats_detached = feats.detach()     
                            giGT = confidence_head(inputs) 

                            ti = (logits.argmax(dim=1) == targets).float()     
                            confidence_loss = F.binary_cross_entropy(giGT, ti)

                            if batch_idx == 0:
                                pos_mask = (ti == 1)
                                neg_mask = (ti == 0)
                                if pos_mask.any() and neg_mask.any():
                                    print(
                                        f"[DEBUG][{args.dataset}][epoch {epoch}] "
                                        f"ConfBCE={confidence_loss.item():.4f} "
                                        f"g_mean={giGT.mean().item():.4f} "
                                        f"g_pos={giGT[pos_mask].mean().item():.4f} "
                                        f"g_neg={giGT[neg_mask].mean().item():.4f} "
                                        f"pos_ratio={pos_mask.float().mean().item():.2f}"
                                    )
                    loss = (
                        ce_loss
                        + args.ours_alpha_intra * L_intra
                        + args.conf_weight * confidence_loss
                    )

                    if batch_idx == 0 or batch_idx % 1 == 0:
                        phase = "PRETRAIN" if epoch < args.pretrain else "STAGE2"
                        print(
                            f"[OursProto][{phase}] epoch {epoch} batch {batch_idx} | "
                            f"CE={ce_loss.item():.4f} "
                            f"L_intra={L_intra.item():.4f} "
                            f"Conf={confidence_loss.item():.4f} "
                            f"Total={loss.item():.4f}"
                        )


            elif args.loss == "graphppd":
                if epoch < args.pretrain:
                    logits = outputs
                    loss = F.cross_entropy(logits, targets)
                else:
                    if ppd_module is None:
                        raise ValueError("ppd_module is None but loss==graphppd")

                    x_T = hidden_features      
                    y_T = targets          

                    if x_T is None:
                        break
                    else:
                        if x_T.dim() > 2:
                            x_T = torch.flatten(x_T, 1)
                        x_T = torch.nan_to_num(x_T, nan=0.0, posinf=0.0, neginf=0.0)

                        ctx_graphs, y_C = sample_context_from_trainset(trainset, args.ppd_context_size)

                        if use_cuda:
                            device = torch.device("cuda")
                            for g in ctx_graphs:
                                if hasattr(g, "to"):
                                    g.to(device)

                        y_C = y_C.to(x_T.device, non_blocking=True).long() 

                        hf_saved = hidden_features

                        with torch.no_grad():
                            _ = model(ctx_graphs)     
                            x_C = hidden_features       

                        hidden_features = hf_saved    


                        if x_C.dim() > 2:
                            x_C = torch.flatten(x_C, 1)
                        x_C = torch.nan_to_num(x_C, nan=0.0, posinf=0.0, neginf=0.0)

                        y_C_onehot = F.one_hot(y_C, num_classes=args.num_classes).to(
                            device=x_C.device, dtype=x_C.dtype
                        ) 

                        logits = ppd_module(x_T, x_C, y_C_onehot)  
                        loss = F.cross_entropy(logits, y_T)

                if batch_idx == 0:
                    print(f"[GraphPPD] epoch={epoch} stage={'PRETRAIN' if epoch < args.pretrain else 'STAGE2'} "
                        f"logits={tuple(outputs.shape)} target[min,max]=({int(targets.min())},{int(targets.max())}) "
                        f"loss={loss.item():.4f}")



            else:
                loss = F.cross_entropy(outputs[:, :-1], targets)
                outputs = outputs[:, :-1]

            batch_size = inputs.size(0) if isinstance(inputs, torch.Tensor) else len(inputs)
            moco_top1.update(moco_error, batch_size)


        if args.dataset != 'celeba':
           
            if args.loss == 'ours_proto':
                if epoch < args.pretrain:
                    acc_logits = outputs  
                else:
                   
                    acc_logits = logits      
            else:
                acc_logits = outputs

            prec1, prec5 = accuracy(acc_logits.data, targets.data, topk=(1, 5))

         
            batch_size = inputs.size(0) if isinstance(inputs, torch.Tensor) else len(inputs)
            
            losses.update(loss.item(), batch_size)
            top1.update(prec1.item(), batch_size)
            top5.update(prec5.item(), batch_size)
        else:
            prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]
            batch_size = inputs.size(0) if isinstance(inputs, torch.Tensor) else len(inputs)
            losses.update(loss.item(), batch_size)
            top1.update(prec1.item(), batch_size)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss2=losses2.avg,
                    moco_top1=moco_top1.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()

    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    train_time = time.perf_counter() - train_t0
    nb = len(trainloader)
    print(f"[TRAIN TIME] epoch={epoch+1} train_only={train_time:.3f}s "
          f"({nb} batches, {train_time/max(nb,1):.3f}s/batch)")
    return (losses.avg, losses2.avg, top1.avg, moco_top1.avg)


def forward_get_emb(model, graphs):
    out = model(graphs)
    h = getattr(model, "cached_hidden", None)
    if h is not None and h.dim() > 2:
        h = torch.flatten(h, 1)
    return out, h


def test(trainloader, testloader, model, criterion, epoch, use_cuda, evaluation = False, confidence_head=None,ppd_module=None):
    if evaluation:
        evaluate(trainloader, testloader, model, use_cuda)
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
        inputs, targets, indices = batch_data 
        data_time.update(time.time() - end)

        if use_cuda:

            for g in inputs:
                if hasattr(g, 'node_features') and isinstance(g.node_features, torch.Tensor):
                    g.node_features = g.node_features.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        targets = torch.autograd.Variable(targets)

    

        with torch.no_grad():
            probs_ppd = None
            output_logits = model(inputs) 
            outputs = output_logits

            h_graph = getattr(model, "cached_hidden", None)

            if h_graph is not None and h_graph.dim() > 2:
                h_graph = torch.flatten(h_graph, 1)


            if targets.device != outputs.device:
                targets = targets.to(outputs.device)


            if args.loss == "graphppd" and epoch >= args.pretrain:
                assert ppd_module is not None, "ppd_module is None but loss==graphppd"
                assert trainloader is not None, "GraphPPD test needs trainloader (for sampling contexts)"
               
                trainset_local = getattr(trainloader, "dataset", None)
                assert trainset_local is not None

             
                x_T = h_graph
                if x_T is None:
                    break
                else:
                    x_T = torch.nan_to_num(x_T, nan=0.0, posinf=0.0, neginf=0.0)

                    probs_accum = None
                    P = int(args.ppd_P)

                    for _p in range(P):
                        ctx_graphs, y_C = sample_context_from_trainset(trainset_local, args.ppd_context_size)

                        if use_cuda:
                            device = x_T.device
                            for g in ctx_graphs:
                                if hasattr(g, "to"):
                                    g.to(device)

                        _ctx_logits, x_C = forward_get_emb(model, ctx_graphs)

                        x_C = torch.nan_to_num(x_C, nan=0.0, posinf=0.0, neginf=0.0)
                        y_C = y_C.to(x_C.device, non_blocking=True).long()
                        y_C_onehot = F.one_hot(y_C, num_classes=args.num_classes).to(dtype=x_C.dtype)

                        logits_p = ppd_module(x_T, x_C, y_C_onehot) 
                        probs_p = torch.softmax(logits_p, dim=-1)

                        probs_accum = probs_p if probs_accum is None else (probs_accum + probs_p)

                    if probs_accum is not None:
                        probs_ppd = probs_accum / float(P)      
                        outputs = torch.log(probs_ppd.clamp_min(1e-12))
                        output_logits = outputs
                        print("Using probs_accum as outputs")
                        print("P",P)


            if args.loss in ["gambler", "sat", "sat_entropy", "cwr"]:
                values, predictions = outputs[:, :-1].data.max(1)
            else:
                values, predictions = outputs.data.max(1)

            if args.loss == "ours_proto":

                h = h_graph 
                if h.dim() > 2:
                    h = torch.flatten(h, 1)
                h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

                raw_probs = F.softmax(output_logits, dim=1)
                y0 = raw_probs.argmax(dim=1)    

                mu_batch_h = args.prototypes_h[y0]   
                h_tilde = (1 - args.ours_gamma) * h + args.ours_gamma * mu_batch_h


                final_logits = model.linear(h_tilde)   
                outputs = final_logits            

                values, predictions = final_logits.data.max(1)

                loss = F.cross_entropy(final_logits, targets)

            else:

              
                if args.loss == "graphppd":
                    loss = F.cross_entropy(outputs, targets) 

                elif args.loss == 'gambler':
                    loss = criterion(outputs, targets, reward)

                elif args.loss == 'cwr':
                    loss = cwr_loss(outputs, targets, d=0.5)

                elif args.loss == 'sat' or args.loss == 'sat_entropy':
                    loss = F.cross_entropy(outputs[:, :-1], targets)
                elif args.loss == 'log_ml':
                    loss = F.cross_entropy(outputs, targets)


                else:
                    loss = criterion(outputs, targets)


            if args.loss == "graphppd" and epoch >= args.pretrain and probs_ppd is not None:
                outputs = probs_ppd 
            else:
                outputs = F.softmax(outputs, dim=1)

            if args.loss == "ours_proto":
                reservation = (outputs * torch.log(outputs)).sum(-1)


            elif args.loss == 'ce' or args.loss == 'csc' or args.loss == 'csc_entropy' or args.loss== "graphppd":
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
                outputs, reservation, reservation_nosoftmax = outputs, -difference.squeeze(), -difference_nosoftmax.squeeze()

            else:
                outputs, reservation = outputs[:,:-1], outputs[:,-1]

            

            reservation_cpu = reservation.detach().cpu().numpy().tolist()
            correct_cpu     = predictions.eq(targets).detach().cpu().numpy().tolist()

            abstention_results.extend(zip(reservation_cpu, correct_cpu))
            
            correct_bool = predictions.eq(targets).detach().cpu().tolist()  


     
            if args.loss == "ours_proto" and confidence_head is not None:  
                giGT_batch = confidence_head(inputs)   
                conf_scores = giGT_batch.detach().cpu().view(-1).tolist()

                confhead_results.extend(zip(conf_scores, correct_bool))



     
            if args.loss == 'log_ml':
                abstention_results_nosoftmax.extend(zip(list( reservation_nosoftmax.numpy() ),list( predictions.eq(targets.data).numpy() )))

            if args.loss == 'ours_proto':
                pred_logits = outputs                  
            elif args.loss == "graphppd" and epoch >= args.pretrain and probs_ppd is not None:
                pred_logits = probs_ppd
            elif args.loss in ['ce', 'log_ml', 'csc', 'csc_entropy']:
                pred_logits = nn.functional.softmax(output_logits, -1)
            else:
                pred_logits = nn.functional.softmax(output_logits[:, :-1], -1)

            sr_results.extend(
                zip(
                    list(pred_logits.max(-1)[0].detach().cpu().numpy()),
                    list(predictions.eq(targets.data).detach().cpu().numpy())
                )
            )

            if args.dataset != 'celeba':
                prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))

                batch_size = inputs.size(0) if isinstance(inputs, torch.Tensor) else len(inputs)

                losses.update(loss.item(), batch_size)
                top1.update(prec1.item(), batch_size)
                top5.update(prec5.item(), batch_size)
            else:
                prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]
                
                batch_size = inputs.size(0) if isinstance(inputs, torch.Tensor) else len(inputs)
                losses.update(loss.item(), batch_size)
                top1.update(prec1.item(), batch_size)

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


    if True:
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


        if args.loss == "ours_proto" and confidence_head is not None and len(confhead_results) > 0:
            confhead_results.sort(key=lambda x: -x[0])  # 对应方案A
            sorted_correct = [int(c) for _, c in confhead_results]
            size = len(sorted_correct)
            print('ConfidenceHead giGT: accuracy of coverage ', end='')
            for coverage in expected_coverage:
                k = max(1, round(size / 100 * coverage))
                covered_correct = sorted_correct[:k]
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
    if epoch in args.schedule and args.dataset != 'celeba':
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']






if __name__ == '__main__':
    if args.loss == 'sat_entropy' or args.loss == 'csc_entropy' or args.loss == 'csc_sat_entropy':
        if args.mode == 'tuning':
            base_path = os.path.join(args.save, args.dataset, args.loss, args.mode, f'entropy_coeff-{str(args.entropy)}', args.arch)
        else:
            base_path = os.path.join(args.save, args.dataset, args.loss, f'entropy_coeff-{str(args.entropy)}', args.arch)
            base_path2 = os.path.join(args.save, args.dataset, args.loss)
    else:
        base_path = os.path.join(args.save, args.dataset, args.loss, args.arch)

        
    baseLR = state['lr']
    base_pretrain = args.pretrain
    resume_path = ""
    for i in range(len(reward_list)): 
        state['lr'] = baseLR
        reward = reward_list[i]
        if args.loss == 'csc':
            full_k1 = False
            full_k2 = False
        else:
            full_k1 = True
            full_k2 = True             
        if "imagenet_subset" == args.dataset:
            base_path = os.path.join(base_path, f"nClasses-{args.num_classes}")

        if args.warmup:
            save_path = os.path.join(base_path, 'o{:.2f}'.format(reward), 'k{:.0f}'.format(args.moco_k), 'm{:.4f}'.format(args.moco_m), 't{:.3f}'.format(args.moco_t), 'pretrain{:.0f}'.format(args.pretrain), 'warmup_epochs{:.0f}'.format(args.warmup_epochs), 'initialreward{:.0e}'.format(args.ir))
        else:
            save_path = os.path.join(base_path, 'o{:.2f}'.format(reward), 'k{:.0f}'.format(args.moco_k), 'm{:.4f}'.format(args.moco_m), 't{:.3f}'.format(args.moco_t), 'pretrain{:.0f}'.format(args.pretrain), f"seed-{args.manualSeed}")

        if args.evaluate:
            resume_path= os.path.join(save_path,'{:d}.pth'.format(args.epochs))
        if args.resume:
            if args.loss == 'csc_entropy' or args.loss == 'csc_sat_entropy':
                resume_path = os.path.join(base_path2, 'resume','{:d}.pth'.format(args.start_epochs))
            else:
                resume_path = os.path.join(base_path, 'resume','{:d}.pth'.format(args.start_epochs))
        args.pretrain = base_pretrain
        
        if args.loss == 'gambler' and args.pretrain == 0:
            if  args.dataset == 'cifar10' and reward < 6.3:
                args.pretrain = 100
            elif args.dataset == 'svhn' and reward < 6.0:
                args.pretrain = 50
            elif args.dataset == 'catsdogs':
                args.pretrain = 50
        
        main()
