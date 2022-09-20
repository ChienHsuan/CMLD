import argparse
import os.path as osp
import random
import csv
import sys
import collections

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import normalized_mutual_info_score
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as T

from uda import datasets
from uda import models
from uda.trainers import ClusterBaseTrainer
from uda.evaluators import Evaluator, extract_features
from uda.utils.data import IterLoader
from uda.utils.data import RandomErasing
from uda.utils.data.sampler import RandomMultipleGallerySampler
from uda.utils.data.preprocessor import Preprocessor
from uda.utils.logging import Logger
from uda.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from uda.utils.rerank import compute_jaccard_dist


start_epoch = best_mAP = 0

def get_data(name, data_dir):
    dataset = datasets.create(name, data_dir)
    return dataset

def get_train_loader(dataset, batch_size, workers, num_instances, iters,
                     transform=None, trainset=None):

    train_set = sorted(dataset.train) if trainset is None else trainset
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
                DataLoader(Preprocessor(train_set, root=dataset.images_dir,
                                        transform=transform, mutual=False, triplet=False),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def get_test_loader(dataset, batch_size, workers, transform=None, testset=None):

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=transform),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def create_model(args, classes, device=torch.device('cuda')):
    model_1 = models.create(args.arch_teacher, num_features=args.features, dropout=args.dropout, num_classes=classes)

    model_1.to(device)
    model_1 = nn.DataParallel(model_1)

    initial_weights = load_checkpoint(args.init_t1)
    copy_state_dict(initial_weights['state_dict'], model_1)

    return model_1

def calScores(clusters, labels):
    """
    compute pair-wise precision pair-wise recall
    """
    from scipy.special import comb
    if len(clusters) == 0:
        return 0, 0
    else:
        curCluster = []
        for curClus in clusters.values():
            curCluster.append(labels[curClus])
        TPandFP = sum([comb(len(val), 2) for val in curCluster])
        TP = 0
        for clusterVal in curCluster:
            for setMember in set(clusterVal):
                if sum(clusterVal == setMember) < 2: continue
                TP += comb(sum(clusterVal == setMember), 2)
        FP = TPandFP - TP
        # FN and TN
        TPandFN = sum([comb(labels.tolist().count(val), 2) for val in set(labels)])
        FN = TPandFN - TP
        # cal precision and recall
        precision, recall = TP / (TP + FP), TP / (TP + FN)
        fScore = 2 * precision * recall / (precision + recall)
        return precision, recall, fScore

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    main_worker(args)

def main_worker(args):
    global start_epoch, best_mAP

    torch.set_default_tensor_type(torch.FloatTensor)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_file_name = f'clust_{args.dataset_source}-2-{args.dataset_target}_{args.arch_teacher}_log.txt'
    sys.stdout = Logger(osp.join(args.logs_dir, log_file_name))
    print("==========\nArgs:{}\n==========".format(args))

    if args.labels_quality_mesurement:
        pseudo_labels_quality_csv_path = osp.join(args.logs_dir, 'pseudo_labels_quality.csv')
        with open(pseudo_labels_quality_csv_path, 'w') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['Epoch', 'Precision', 'Recall', 'F-score', 'NMI'])

    # Create data loaders
    iters = args.iters if (args.iters>0) else None
    dataset_source = get_data(args.dataset_source, args.data_dir)
    dataset_target = get_data(args.dataset_target, args.data_dir)
    
    height = args.height
    width = args.width
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
             RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
         ])
    test_transform = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    target_clust_loader = get_test_loader(dataset_target, args.batch_size, args.workers, transform=test_transform, testset=dataset_target.train)
    if not args.use_lab314_dataset:
        target_test_loader = get_test_loader(dataset_target, args.batch_size, args.workers, transform=test_transform)
    
    # Create model
    model_1 = create_model(args, len(dataset_target.train), device=device)

    # Evaluator
    evaluator = Evaluator(model_1)

    for epoch in range(args.epochs):
        # extract features
        t_features, _ = extract_features(model_1, target_clust_loader, print_freq=100)
        t_f = torch.stack(list(t_features.values()))

        # re-ranking
        rerank_dist = compute_jaccard_dist(t_f, args.k1, args.k2, use_gpu=args.rr_gpu).numpy()

        # DBSCAN
        if (epoch==0):
            tri_mat = np.triu(rerank_dist, 1) # tri_mat.dim=2
            tri_mat = tri_mat[np.nonzero(tri_mat)] # tri_mat.dim=1
            tri_mat = np.sort(tri_mat,axis=None)
            rho = args.rho
            top_num = np.round(rho*tri_mat.size).astype(int)
            eps = tri_mat[:top_num].mean()
            print('eps for cluster: {:.3f}'.format(eps))
            cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)

        print('Clustering and labeling...')
        pseudo_labels = cluster.fit_predict(rerank_dist)
        num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
        args.num_clusters = num_ids

        # generate new dataset and adjust pseudo labels
        new_dataset = []
        cluster_centers = collections.defaultdict(list)
        if not args.labels_quality_mesurement:
            for i, ((fname, _, cid), label) in enumerate(zip(dataset_target.train, pseudo_labels)):
                if label == -1: continue
                new_dataset.append((fname, label, cid))
                cluster_centers[label].append(t_f[i])
        else:
            realIDs_dict, realIDs, fakeIDs = collections.defaultdict(list), [], []
            index = 0
            for i, ((fname, pid, cid), label) in enumerate(zip(dataset_target.train, pseudo_labels)):
                if label == -1: continue
                new_dataset.append((fname, label, cid))
                cluster_centers[label].append(t_f[i])
                realIDs_dict[pid].append(index)
                realIDs.append(pid)
                fakeIDs.append(label)
                index += 1
            # F-score
            precision, recall, fscore = calScores(realIDs_dict, np.asarray(fakeIDs))
            print(f'\nPrecision: {100*precision}, Recall: {100*recall}, F-score: {fscore} .')

            # NMI
            NMI_value = normalized_mutual_info_score(realIDs, fakeIDs)
            print(f'NMI: {NMI_value} .\n')

            with open(pseudo_labels_quality_csv_path, 'a') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow([epoch, precision, recall, fscore, NMI_value])

        print(f'\n Clustered into {num_ids} classes, {len(new_dataset)} total target training samples. \n')

        cluster_centers = [torch.stack(cluster_centers[idx]).mean(0) for idx in sorted(cluster_centers.keys())]
        cluster_centers = torch.stack(cluster_centers)

        # initial classifier
        model_1.module.classifier.weight.data[:args.num_clusters].copy_(F.normalize(cluster_centers, dim=1).float().cuda())

        # train loader
        target_train_loader = get_train_loader(dataset_target, args.batch_size, args.workers,
                                               args.num_instances, iters, transform=train_transformer,
                                               trainset=new_dataset)

        # optimizer
        params = []
        for key, value in model_1.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]

        optimizer = torch.optim.Adam(params)

        # Trainer
        trainer = ClusterBaseTrainer(model_1, target_train_loader, optimizer,
                                     num_cluster=args.num_clusters)

        iters = len(target_train_loader)
        result = trainer.train(epoch, print_freq=args.print_freq, train_iters=iters)

        def save_model(model_ema, is_best, best_mAP):
            save_checkpoint({
                'state_dict': model_ema.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'model_checkpoint.pth.tar'))

        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
            model_1.eval()
            with torch.no_grad():
                if not args.use_lab314_dataset:
                    mAP = evaluator.evaluate(target_test_loader, dataset_target.query, dataset_target.gallery, cmc_flag=False)
                else:
                    mAP = result
                is_best = (mAP>best_mAP) 
                best_mAP = max(mAP, best_mAP)
                save_model(model_1, is_best, best_mAP)

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%} best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

    if not args.use_lab314_dataset:
        print ('Test on the best model.')
        model_1.eval()
        with torch.no_grad():
            checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
            copy_state_dict(checkpoint['state_dict'], model_1)
            evaluator.evaluate(target_test_loader, dataset_target.query, dataset_target.gallery, cmc_flag=True)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MMT Training")
    # data
    parser.add_argument('-ds', '--dataset-source', type=str, default='dukemtmc',
                        choices=datasets.names())
    parser.add_argument('-dt', '--dataset-target', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256,
                        help="input height")
    parser.add_argument('--width', type=int, default=128,
                        help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    parser.add_argument('--use-lab314-dataset', action='store_true',
                        help='use lab314 dataset for training')
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--k1', type=int, default=20,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    # model
    parser.add_argument('--arch-teacher', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=0.999)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--weight', type=float, default=0.5)
    parser.add_argument('--weight_ms', type=float, default=0.8)
    parser.add_argument('--weight_tf', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--iters', type=int, default=400)
    # training configs
    parser.add_argument('--init-t1', type=str, default='', metavar='PATH')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--lambda-value', type=float, default=0)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--rho', type=float, default=0.0016, help='dbscan hyper-parameter')
    parser.add_argument('--milestones', nargs='+', type=int, default=[],
                        help='milestones for the learning rate decay')
    parser.add_argument('--rr-gpu', action='store_true', 
                        help="use GPU for accelerating clustering")
    # evaluation
    parser.add_argument('--labels-quality-mesurement', action='store_true',
                        help='pseudo labels quality mesurement (F-score and NMI metric)')
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/data/per-id/data')  # root
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
                        
    main()
