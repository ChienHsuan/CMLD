import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parameter_analysis(memory, margin, distillation, student, output='resnet_parameter_analysis.pdf'):
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    axs[0, 0].set_title('(a)')
    axs[0, 1].set_title('(b)')
    axs[1, 0].set_title('(c)')
    axs[1, 1].set_title('(d)')

    metric = ['mAP', 'rank-1']
    maker = ['o', '^']

    # memory bank
    for i in range(memory.shape[1]):
        if i == 0:
            continue
        x_labels = memory[:, 0]
        y = memory[:, i]
        axs[0, 0].plot(x_labels, y, label=metric[i-1], marker=maker[i-1])
        for j, label in enumerate(y):
            axs[0, 0].text(x_labels[j], y[j]+0.3, f'{label}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # margin
    for i in range(margin.shape[1]):
        if i == 0:
            continue
        x_labels = margin[:, 0]
        y = margin[:, i]
        axs[0, 1].plot(x_labels, y, label=metric[i-1], marker=maker[i-1])
        for j, label in enumerate(y):
            axs[0, 1].text(x_labels[j], y[j]+0.3, f'{label}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # distillation
    for i in range(distillation.shape[1]):
        if i == 0:
            continue
        x_labels = distillation[:, 0]
        y = distillation[:, i]
        axs[1, 0].plot(x_labels, y, label=metric[i-1], marker=maker[i-1])
        for j, label in enumerate(y):
            axs[1, 0].text(x_labels[j], y[j]+0.3, f'{label}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # student
    for i in range(student.shape[1]):
        if i == 0:
            continue
        x_labels = student[:, 0]
        y = student[:, i]
        axs[1, 1].plot(x_labels, y, label=metric[i-1], marker=maker[i-1])
        for j, label in enumerate(y):
            axs[1, 1].text(x_labels[j], y[j]+0.3, f'{label}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i, j].yaxis.set_ticks([75.0, 80.0, 85.0, 90.0, 95.0])
            axs[i, j].legend(framealpha=0.6, loc='lower right')
            axs[i, j].set_ylabel('Accuracy (%)')
    axs[0, 0].xaxis.set_ticks([0, 3200, 6400, 9600, 12800])
    axs[0, 1].xaxis.set_ticks([0.1, 0.2, 0.3, 0.4, 0.5])
    axs[1, 0].xaxis.set_ticks([1.6, 1.8, 2.0, 2.2, 2.4])
    axs[1, 1].xaxis.set_ticks([0.1, 0.2, 0.3, 0.4, 0.5])
    axs[0, 0].set_xlabel('memory bank size')
    axs[0, 1].set_xlabel(r'margin $m$')
    axs[1, 0].set_xlabel(r'distillation loss weight $\theta_{kd}$')
    axs[1, 1].set_xlabel(r'student loss weight $\theta_{stu}$')
    plt.tight_layout()
    plt.savefig(output, bbox_inches='tight')

def main():
    ## duke to market ##
    resnet_memory = np.array([[0, 79.3, 92.0],
                              [3200, 80.7, 92.4],
                              [6400, 81.1, 92.5],
                              [9600, 80.3, 92.5],
                              [12800, 80.0, 92.3]])
    resnet_margin = np.array([[0.1, 80.6, 91.8],
                              [0.2, 80.8, 92.2],
                              [0.3, 81.1, 92.5],
                              [0.4, 80.8, 92.1],
                              [0.5, 80.5, 91.6]])
    resnet_distillation = np.array([[1.6, 80.7, 92.0],
                                    [1.8, 80.9, 92.4],
                                    [2.0, 81.1, 92.5],
                                    [2.2, 81.0, 92.5],
                                    [2.4, 80.8, 92.4]])
    resnet_student = np.array([[0.1, 80.8, 92.3],
                               [0.2, 80.9, 92.4],
                               [0.3, 81.1, 92.5],
                               [0.4, 80.7, 92.1],
                               [0.5, 80.4, 91.9]])

    osnet_memory = np.array([[0, 79.1, 90.4],
                             [3200, 80.6, 92.3],
                             [6400, 81.0, 92.6],
                             [9600, 80.8, 92.3],
                             [12800, 80.4, 92.0]])
    osnet_margin = np.array([[0.1, 80.5, 91.3],
                             [0.2, 80.9, 91.7],
                             [0.3, 81.0, 92.6],
                             [0.4, 80.9, 92.5],
                             [0.5, 80.8, 92.2]])
    osnet_distillation = np.array([[1.6, 80.5, 91.8],
                                   [1.8, 80.8, 92.1],
                                   [2.0, 81.0, 92.6],
                                   [2.2, 80.9, 92.0],
                                   [2.4, 80.8, 92.2]])
    osnet_student = np.array([[0.1, 80.9, 92.0],
                              [0.2, 80.9, 92.1],
                              [0.3, 81.0, 92.6],
                              [0.4, 80.7, 92.5],
                              [0.5, 80.6, 92.2]])

    parameter_analysis(resnet_memory, resnet_margin, resnet_distillation,
                                resnet_student, output='resnet_parameter_analysis.pdf')
    parameter_analysis(osnet_memory, osnet_margin, osnet_distillation,
                                osnet_student, output='osnet-ain_parameter_analysis.pdf')


if __name__ == '__main__':
    main()
