import argparse
import os.path as osp
import random
import math
import csv

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as T

from uda import datasets
from uda import models
from uda.evaluators import extract_features
from uda.utils.data.preprocessor import Preprocessor
from uda.utils.logging import Logger
from uda.utils.serialization import load_checkpoint, copy_state_dict


def get_data(name, data_dir, height, width, batch_size, workers):
    dataset = datasets.create(name, data_dir)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, test_loader

def calculate_class_variance(all_features, all_labels, features, labels, select_ids, file_name=None):
    total_class_variance_csv_path = file_name[:-4] + '_total_class_variance.csv'
    with open(total_class_variance_csv_path, 'w') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Within_class_variance', 'Between_class_variance'])

    select_ids_variance = np.zeros(select_ids.shape[0], dtype=np.float64)
    for i, label in enumerate(select_ids):
        select_ids_variance[i] = np.var(features[labels == label], axis=0).sum()
    
    Within_class_variance = 0.
    unique_all_labels = np.unique(all_labels)
    classes_mean_center = np.zeros((unique_all_labels.shape[0], all_features.shape[1]), dtype=np.float64)
    for i, label in enumerate(unique_all_labels):
        label_features = all_features[all_labels == label]
        classes_mean_center[i, :] = label_features.mean(axis=0)
        Within_class_variance += np.var(label_features, axis=0).sum()
    Within_class_variance /= unique_all_labels.shape[0]
    between_class_variance = np.var(classes_mean_center, axis=0).sum()

    with open(total_class_variance_csv_path, 'a') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([Within_class_variance, between_class_variance])
    return select_ids_variance

def plot_figure(norm_tsne_results, labels, select_ids, select_ids_variance,
                point_label_type='interval', out_file_path=None):
    fig, ax = plt.subplots()
    camp = plt.get_cmap('rainbow')
    for i, label in enumerate(select_ids):
        label_coord = norm_tsne_results[labels == label]

        color = i / select_ids.shape[0]
        ax.scatter(label_coord[:, 0], label_coord[:, 1], color=camp(color),
                   label=f'{label}({select_ids_variance[i]:.2f})')

        if point_label_type == 'interval':
            size = label_coord.shape[0]
            for j in range(0, size, math.ceil(size/3)):
                ax.annotate(label, (label_coord[j, 0], label_coord[j, 1]))
        elif point_label_type == 'mean':
            mean_center = np.mean(label_coord, axis=0)
            ax.annotate(label, (mean_center[0], mean_center[1]))
        else:
            raise NameError(f'Unsupported point_label_type method: {point_label_type}.')

    box = ax.get_position()
    ax.legend(title='class(variance)', bbox_to_anchor=(0, box.ymin-box.height+0.02, 1, 0.6), loc='upper left',
              ncol=4, mode='expand', borderaxespad=0)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(out_file_path, bbox_inches='tight')

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
    torch.set_default_tensor_type(torch.FloatTensor)

    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    dataset_target, test_loader_target = \
        get_data(args.dataset_target, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers)

    # Create model
    model = models.create(args.arch, pretrained=False, num_features=args.features,
                          dropout=args.dropout, num_classes=0)
    model.cuda()
    model = nn.DataParallel(model)

    # Load from checkpoint
    checkpoint = load_checkpoint(args.resume)
    copy_state_dict(checkpoint['state_dict'], model)
    start_epoch = checkpoint['epoch']
    best_mAP = checkpoint['best_mAP']
    print("=> Checkpoint of epoch {}  best mAP {:.1%}".format(start_epoch, best_mAP))

    print('Extract features ...')
    extracted_features, extracted_labels = extract_features(model, test_loader_target)

    # random select id in gallery
    ids = list(set([t.item() for t in extracted_labels.values()]))
    select_ids = np.random.choice(ids, size=args.num_ids, replace=False)
    select_ids.sort()
    print(f'Select ids: {select_ids}.')

    all_features = []
    all_labels = []
    features = []
    labels = []
    for f, id, _ in dataset_target.gallery:
        feature = extracted_features[f].unsqueeze(0)
        all_features.append(feature)
        all_labels.append(id)
        if id in select_ids:
            features.append(feature)
            labels.append(id)
    all_features = torch.cat(all_features, 0).cpu().numpy()
    all_labels = np.array(all_labels)
    features = torch.cat(features, 0).cpu().numpy()
    labels = np.array(labels)

    print('Calculate class variance ...')
    select_ids_variance = calculate_class_variance(all_features, all_labels, features, labels, select_ids, file_name=args.output)

    print('Execute t-SNE ...')
    tsne_results = TSNE(n_components=2, learning_rate='auto', metric='euclidean',
                        init='random', random_state=args.seed).fit_transform(features)
    norm = lambda x: (x - x.min()) / (x.max() - x.min())
    norm_tsne_results = norm(tsne_results)

    print('Plot figure ...')
    plot_figure(norm_tsne_results, labels, select_ids, select_ids_variance,
                point_label_type=args.point_label_type, out_file_path=args.output)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing the model")
    # data
    parser.add_argument('-dt', '--dataset-target', type=str, required=True,
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    # model
    parser.add_argument('-a', '--arch', type=str, required=True,
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # testing configs
    parser.add_argument('--resume', type=str, required=True, metavar='PATH')
    parser.add_argument('--rerank', action='store_true',
                        help="evaluation only")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num-ids', type=int, default=10)
    parser.add_argument('--point_label_type', type=str, default='interval')
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--output', type=str, required=True, metavar='PATH',
                        default='result.pdf')
                        
    main()
