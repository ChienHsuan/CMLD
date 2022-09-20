import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pseudo_labels_quality_curve(dir_path, output='curve_1.pdf'):
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    axs[0, 0].set_title('(a)')
    axs[0, 1].set_title('(b)')
    axs[1, 0].set_title('(c)')
    axs[1, 1].set_title('(d)')

    method = ['CMLD', 'CML', 'ML', 'Baseline']

    # d2m
    path_list = ['cmld_eval_dukemtmc-2-market1501_resnet101-resnet50',
                 'cml_eval_dukemtmc-2-market1501_resnet50',
                 'ml_eval_dukemtmc-2-market1501_resnet50',
                 'clust_eval_dukemtmc-2-market1501_resnet50']
    for i, path in enumerate(path_list):
        file_name = os.path.join(dir_path, path, 'pseudo_labels_quality.csv')
        df = pd.read_csv(file_name, index_col='Epoch')
        x_labels = df.index.to_numpy(dtype=np.int32) + 1
        for j in range(axs.shape[1]):
            y = df.iloc[:, 2+j].to_numpy(dtype=np.float32)
            axs[0, j].plot(x_labels, y, label=method[i])

    # m2d
    path_list = ['cmld_eval_market1501-2-dukemtmc_resnet101-resnet50',
                 'cml_eval_market1501-2-dukemtmc_resnet50',
                 'ml_eval_market1501-2-dukemtmc_resnet50',
                 'clust_eval_market1501-2-dukemtmc_resnet50']
    for i, path in enumerate(path_list):
        file_name = os.path.join(dir_path, path, 'pseudo_labels_quality.csv')
        df = pd.read_csv(file_name, index_col='Epoch')
        x_labels = df.index.to_numpy(dtype=np.int32) + 1
        for j in range(axs.shape[1]):
            y = df.iloc[:, 2+j].to_numpy(dtype=np.float32)
            axs[1, j].plot(x_labels, y, label=method[i])

    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i, j].xaxis.set_ticks([1, 10, 20, 30, 40])
            axs[i, j].set_xlim(0.5, 40.5)
            axs[i, j].grid(linestyle='-', linewidth=0.5)
            axs[i, j].legend(framealpha=0.6, loc='lower right')
            axs[i, j].set_xlabel('epoch')
    axs[0, 0].yaxis.set_ticks([0.5, 0.6, 0.7, 0.8])
    axs[0, 1].yaxis.set_ticks([0.9, 0.92, 0.94, 0.96])
    axs[1, 0].yaxis.set_ticks([0.45, 0.5, 0.55, 0.6])
    axs[1, 1].yaxis.set_ticks([0.88, 0.9, 0.92, 0.94])
    axs[0, 0].set_ylabel('F-score')
    axs[0, 1].set_ylabel('NMI')
    axs[1, 0].set_ylabel('F-score')
    axs[1, 1].set_ylabel('NMI')
    plt.tight_layout()
    plt.savefig(output, bbox_inches='tight')

def peer_and_ema_model_curve(dir_path, output='curve_2.pdf'):
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    axs[0, 0].set_title('(a)')
    axs[0, 1].set_title('(b)')
    axs[1, 0].set_title('(c)')
    axs[1, 1].set_title('(d)')

    model = ['Peer-1', 'Peer-2', 'EMA-1', 'EMA-2']
    color = [(0., 102/255, 1.), (51/255, 204/255, 51/255), (1., 173/255, 51/255), (1., 80/255, 80/255)]

    path_list = ['cmld_eval_dukemtmc-2-market1501_resnet101-resnet50',
                 'cmld_eval_market1501-2-dukemtmc_resnet101-resnet50']

    # d2m
    file_name = os.path.join(dir_path, path_list[0], 'peer_ema_model_comparison.csv')
    df = pd.read_csv(file_name, index_col='Epoch')
    x_labels = df.index.to_numpy(dtype=np.int32) + 1
    for i in range(axs.shape[1]):
        acc = df.iloc[:, [1+i, 4+i, 7+i, 10+i]].to_numpy(dtype=np.float32) * 100
        for j in range(len(model)):
            y = acc[:, j]
            axs[0, i].plot(x_labels, y, color=color[j], label=model[j])

    # m2d
    file_name = os.path.join(dir_path, path_list[1], 'peer_ema_model_comparison.csv')
    df = pd.read_csv(file_name, index_col='Epoch')
    x_labels = df.index.to_numpy(dtype=np.int32) + 1
    for i in range(axs.shape[1]):
        acc = df.iloc[:, [1+i, 4+i, 7+i, 10+i]].to_numpy(dtype=np.float32) * 100
        for j in range(len(model)):
            y = acc[:, j]
            axs[1, i].plot(x_labels, y, color=color[j], label=model[j])

    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i, j].xaxis.set_ticks([1, 10, 20, 30, 40])
            axs[i, j].set_xlim(0.5, 40.5)
            axs[i, j].grid(linestyle='-', linewidth=0.5)
            axs[i, j].legend(framealpha=0.6, loc='lower right')
            axs[i, j].set_xlabel('epoch')
    axs[0, 0].yaxis.set_ticks([50, 60, 70, 80])
    axs[0, 1].yaxis.set_ticks([75, 80, 85, 90])
    axs[1, 0].yaxis.set_ticks([45, 50, 55, 60, 65])
    axs[1, 1].yaxis.set_ticks([60, 65, 70, 75, 80])
    axs[0, 0].set_ylabel('mAP (%)')
    axs[0, 1].set_ylabel('rank-1 (%)')
    axs[1, 0].set_ylabel('mAP (%)')
    axs[1, 1].set_ylabel('rank-1 (%)')
    plt.tight_layout()
    plt.savefig(output, bbox_inches='tight')

def teacher_and_student_model_curve(dir_path, output='curve_3.pdf'):
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    axs[0, 0].set_title('(a)')
    axs[0, 1].set_title('(b)')
    axs[0, 2].set_title('(c)')
    axs[1, 0].set_title('(d)')
    axs[1, 1].set_title('(e)')
    axs[1, 2].set_title('(f)')

    model = ['Teacher-1', 'Teacher-2', 'Student']
    color = [(1., 173/255, 51/255), (1., 80/255, 80/255), (153/255, 102/255, 1.)]

    path_list = ['cmld_eval_dukemtmc-2-market1501_resnet101-resnet50',
                 'cmld_eval_market1501-2-dukemtmc_resnet101-resnet50']

    # d2m
    file_name = os.path.join(dir_path, path_list[0], 'teacher_student_model_comparison.csv')
    df = pd.read_csv(file_name, index_col='Epoch')
    x_labels = df.index.to_numpy(dtype=np.int32) + 1
    for i in range(axs.shape[1]):
        acc = df.iloc[:, [i, 3+i, 6+i]].to_numpy(dtype=np.float32)
        acc = acc*100 if i > 0 else acc
        for j in range(len(model)):
            y = acc[:, j]
            axs[0, i].plot(x_labels, y, color=color[j], label=model[j])

    # m2d
    file_name = os.path.join(dir_path, path_list[1], 'teacher_student_model_comparison.csv')
    df = pd.read_csv(file_name, index_col='Epoch')
    x_labels = df.index.to_numpy(dtype=np.int32) + 1
    for i in range(axs.shape[1]):
        acc = df.iloc[:, [i, 3+i, 6+i]].to_numpy(dtype=np.float32)
        acc = acc*100 if i > 0 else acc
        for j in range(len(model)):
            y = acc[:, j]
            axs[1, i].plot(x_labels, y, color=color[j], label=model[j])

    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i, j].xaxis.set_ticks([1, 10, 20, 30, 40])
            axs[i, j].set_xlim(0.5, 40.5)
            axs[i, j].grid(linestyle='-', linewidth=0.5)
            if j > 0:
                axs[i, j].legend(framealpha=0.6, loc='lower right')
            else:
                axs[i, j].legend(framealpha=0.6, loc='upper right')
            axs[i, j].set_xlabel('epoch')
    axs[0, 0].yaxis.set_ticks([1.25, 1.75, 2.25, 2.75])
    axs[0, 1].yaxis.set_ticks([50, 60, 70, 80])
    axs[0, 2].yaxis.set_ticks([75, 80, 85, 90])
    axs[1, 0].yaxis.set_ticks([1.25, 1.75, 2.25, 2.75])
    axs[1, 1].yaxis.set_ticks([45, 50, 55, 60, 65])
    axs[1, 2].yaxis.set_ticks([60, 65, 70, 75, 80])
    axs[0, 0].set_ylabel('classification error')
    axs[0, 1].set_ylabel('mAP (%)')
    axs[0, 2].set_ylabel('rank-1 (%)')
    axs[1, 0].set_ylabel('classification error')
    axs[1, 1].set_ylabel('mAP (%)')
    axs[1, 2].set_ylabel('rank-1 (%)')
    plt.tight_layout()
    plt.savefig(output, bbox_inches='tight')

def mutual_learning_student_model_curve(dir_path, output='curve_4.pdf'):
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    axs[0, 0].set_title('(a)')
    axs[0, 1].set_title('(b)')
    axs[0, 2].set_title('(c)')
    axs[1, 0].set_title('(d)')
    axs[1, 1].set_title('(e)')
    axs[1, 2].set_title('(f)')

    method = ['w/', 'w/o']
    color = [(0., 1., 0.), (0., 128/255, 1.)]

    d2m_path_list = ['cmld_dukemtmc-2-market1501_resnet101-resnet50',
                     'cmld_dukemtmc-2-market1501_resnet101-resnet50_single']
    m2d_path_list = ['cmld_market1501-2-dukemtmc_resnet101-resnet50',
                     'cmld_market1501-2-dukemtmc_resnet101-resnet50_single']
    ml_dir_path = os.path.join(dir_path, 'ml')

    # d2m
    for j, path_list in enumerate(d2m_path_list):
        file_name = os.path.join(ml_dir_path, path_list, 'teacher_student_model_comparison.csv')
        df = pd.read_csv(file_name, index_col='Epoch')
        x_labels = df.index.to_numpy(dtype=np.int32) + 1
        for i in range(axs.shape[1]):
            acc = df.iloc[:, 6+i].to_numpy(dtype=np.float32)
            acc = acc*100 if i > 0 else acc
            axs[0, i].plot(x_labels, acc, color=color[j], label=method[j])

    # m2d
    for j, path_list in enumerate(m2d_path_list):
        file_name = os.path.join(ml_dir_path, path_list, 'teacher_student_model_comparison.csv')
        df = pd.read_csv(file_name, index_col='Epoch')
        x_labels = df.index.to_numpy(dtype=np.int32) + 1
        for i in range(axs.shape[1]):
            acc = df.iloc[:, 6+i].to_numpy(dtype=np.float32)
            acc = acc*100 if i > 0 else acc
            axs[1, i].plot(x_labels, acc, color=color[j], label=method[j])

    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i, j].xaxis.set_ticks([1, 10, 20, 30, 40])
            axs[i, j].set_xlim(0.5, 40.5)
            axs[i, j].grid(linestyle='-', linewidth=0.5)
            if j > 0:
                axs[i, j].legend(framealpha=0.6, loc='lower right')
            else:
                axs[i, j].legend(framealpha=0.6, loc='upper right')
            axs[i, j].set_xlabel('epoch')
    axs[0, 0].yaxis.set_ticks([1.25, 1.75, 2.25, 2.75])
    axs[0, 1].yaxis.set_ticks([50, 60, 70, 80])
    axs[0, 2].yaxis.set_ticks([75, 80, 85, 90])
    axs[1, 0].yaxis.set_ticks([1.25, 1.75, 2.25, 2.75])
    axs[1, 1].yaxis.set_ticks([45, 50, 55, 60, 65])
    axs[1, 2].yaxis.set_ticks([60, 65, 70, 75, 80])
    axs[0, 0].set_ylabel('classification error')
    axs[0, 1].set_ylabel('mAP (%)')
    axs[0, 2].set_ylabel('rank-1 (%)')
    axs[1, 0].set_ylabel('classification error')
    axs[1, 1].set_ylabel('mAP (%)')
    axs[1, 2].set_ylabel('rank-1 (%)')
    plt.tight_layout()
    plt.savefig(output, bbox_inches='tight')

def main(dir_path):
    pseudo_labels_quality_curve(dir_path, output='pseudo_labels_quality.pdf')
    peer_and_ema_model_curve(dir_path, output='peer_and_ema_model.pdf')
    teacher_and_student_model_curve(dir_path, output='teacher_and_student_model.pdf')
    mutual_learning_student_model_curve(dir_path, output='mutual_learning_student_model.pdf')


if __name__ == '__main__':
    dir_path = 'train_logs/'
    main(dir_path)
