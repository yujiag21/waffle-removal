import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import matplotlib
params = {
    'text.latex.preamble': ['\\usepackage{gensymb}'],
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 1000,  # to adjust notebook inline plot size
    'axes.labelsize': 5, # fontsize for x and y labels (was 10)
    'axes.titlesize': 5,
    'font.size': 5, # was 10
    'legend.fontsize': 5, # was 10
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'text.usetex': True,
    'figure.figsize': [2.64, 2.64],
    'mathtext.rm': 'Arial',
    'axes.linewidth': 0.5
}

matplotlib.rcParams.update(params)

def draw_test_accuracies(file_arrays, epoch_ranges, dataset, experiment, save_file_name, th=[95.00, 100.00]):

    fig, axs = plt.subplots(2, 2, sharey=True)
    color=['#000000', '#008000', '#0000ff', '#ff0000', '#ff9900']
    labels = ['baseline', 'uPattern', 'rPattern', 'unRelate', 'unStruct']
    
    for rrange in range(4):
        file_array = file_arrays[rrange]
        xticks = np.arange(0, epoch_ranges[rrange]+50, 50)
        epochs = np.asarray(range(epoch_ranges[rrange]))
        test_acc = np.zeros((len(file_array), epoch_ranges[rrange]))
        for fi in range(len(file_array)):
            i=0
            with open(file_array[fi],'r') as file_obj:
                for line in file_obj:
                    if 'Average_loss' in line:
                        line_arr = line.split()
                        test_acc[fi,i] = np.float(line_arr[7])
                        i += 1
                                   
        for i, c in zip(range(len(file_array)), color):
            axs[int(rrange/2)][rrange%2].plot(epochs, test_acc[i], c=c, label=labels[i], linewidth=0.8)
            axs[int(rrange/2)][rrange%2].set_xticks(xticks)
    
        axs[int(rrange/2)][rrange%2].set_ylim((th[0], th[1]))
    
    axs[1][1].set_xlabel('Communication Rounds')
    axs[0][0].set_ylabel('Test accuracy ($\%$)')  
    axs[1][0].set_xlabel('Communication Rounds')
    axs[1][0].set_ylabel('Test accuracy ($\%$)') 
    axs[1][1].legend(loc='best',frameon=False)
    axs[0][0].title.set_text(dataset + ' ' + experiment + ' ' + '$E_c=1$') 
    axs[0][1].title.set_text(dataset + ' ' + experiment + ' ' + '$E_c=5$') 
    axs[1][0].title.set_text('$E_c=10$') 
    axs[1][1].title.set_text('$E_c=20$') 
    plt.tight_layout()   
    plt.savefig(save_file_name + '.pdf', bbox_inches='tight', pad_inches = 0.05)    
    
    
def draw_watermark_accuracies(file_arrays, epoch_ranges, dataset, experiment, save_file_name):

    fig, axs = plt.subplots(2, 2, sharey=True)
    color=['#008000', '#0000ff', '#ff0000', '#ff9900']
    labels = ['uPattern', 'rPattern', 'unRelate', 'unStruct']
    
    for rrange in range(4):
        file_array = file_arrays[rrange]
        xticks = np.arange(0, epoch_ranges[rrange]+50, 50)
        epochs = np.asarray(range(epoch_ranges[rrange]))
        watermark_acc = np.zeros((len(file_array), epoch_ranges[rrange]))
        for fi in range(1, len(file_array)):
            i=0
            with open(file_array[fi],'r') as file_obj:
                for line in file_obj:
                    if 'watermark trained epochs: 100' in line or 'watermark threshold is reached' in line:
                        line_arr = line.split()
                        watermark_acc[fi,i] = np.float(line_arr[-1])
                        i += 1
                                   
        for i, c in zip(range(len(file_array)), color):
            axs[int(rrange/2)][rrange%2].plot(epochs, watermark_acc[i], c=c, label=labels[i], linewidth=0.8)
            axs[int(rrange/2)][rrange%2].set_xticks(xticks)
            axs[int(rrange/2)][rrange%2].axhline(y=98, c='grey', linestyle='dashed', lw=1)
    
        axs[int(rrange/2)][rrange%2].set_ylim((50, 100))
    
    axs[1][1].set_xlabel('Communication Rounds')
    axs[0][0].set_ylabel('WM accuracy ($\%$)')  
    axs[1][0].set_xlabel('Communication Rounds')
    axs[1][0].set_ylabel('WM accuracy ($\%$)') 
    axs[1][1].legend(loc='best',frameon=False)
    axs[0][0].title.set_text(dataset + ' ' + experiment + ' ' + '$E_c=1$') 
    axs[0][1].title.set_text(dataset + ' ' + experiment + ' ' + '$E_c=5$') 
    axs[1][0].title.set_text('$E_c=10$') 
    axs[1][1].title.set_text('$E_c=20$') 
    plt.tight_layout()   
    plt.savefig(save_file_name + '.pdf', bbox_inches='tight', pad_inches = 0.05)        
    
    
    

m_range_R1 = 250
m_range_R5 = 200
m_range_R10 = 150
m_range_R20 = 100
m_ranges = [m_range_R1, m_range_R5, m_range_R10, m_range_R20]

uPattern_PNOR = 'train_record/epoch_logs_mnist_to_pattern_ws100_l5_100c-P-noR.pt.txt'
rPattern_PNOR = 'train_record/epoch_logs_mnist_to_mpattern_ws100_l5_100c-P-noR.pt.txt'
unRelate_PNOR = 'train_record/epoch_logs_mnist_to_random_ws100_l5_100c-P-noR.pt.txt'
unStruct_PNOR = 'train_record/epoch_logs_mnist_to_imagenet_ws100_l5_100c-P-noR.pt.txt'

baseline = 'train_record/epoch_logs_federated_mnist_l5_100c_1localround.pt.txt'
uPattern = 'train_record/epoch_logs_mnist_to_pattern_ws100_l5_100c-noP-R1.pt.txt'
rPattern = 'train_record/epoch_logs_mnist_to_mpattern_ws100_l5_100c-noP-R1.pt.txt'
unRelate = 'train_record/epoch_logs_mnist_to_imagenet_ws100_l5_100c-noP-R1.pt.txt'
unStruct = 'train_record/epoch_logs_mnist_to_random_ws100_l5_100c-noP-R1.pt.txt'
files_1rounds  = [baseline, uPattern, rPattern, unRelate, unStruct]

baseline = 'train_record/epoch_logs_federated_mnist_l5_100c_5localround.pt.txt'
uPattern = 'train_record/epoch_logs_mnist_to_pattern_ws100_l5_100c-noP-R5.pt.txt'
rPattern = 'train_record/epoch_logs_mnist_to_mpattern_ws100_l5_100c-noP-R5.pt.txt'
unRelate = 'train_record/epoch_logs_mnist_to_imagenet_ws100_l5_100c-noP-R5.pt.txt'
unStruct = 'train_record/epoch_logs_mnist_to_random_ws100_l5_100c-noP-R5.pt.txt'
files_5rounds  = [baseline, uPattern, rPattern, unRelate, unStruct]

baseline = 'train_record/epoch_logs_federated_mnist_l5_100c_10localround.pt.txt'
uPattern = 'train_record/epoch_logs_mnist_to_pattern_ws100_l5_100c-noP-R10.pt.txt'
rPattern = 'train_record/epoch_logs_mnist_to_mpattern_ws100_l5_100c-noP-R10.pt.txt'
unRelate = 'train_record/epoch_logs_mnist_to_imagenet_ws100_l5_100c-noP-R10.pt.txt'
unStruct = 'train_record/epoch_logs_mnist_to_random_ws100_l5_100c-noP-R10.pt.txt'
files_10rounds  = [baseline, uPattern, rPattern, unRelate, unStruct]

baseline = 'train_record/epoch_logs_federated_mnist_l5_100c_20localround.pt.txt'
uPattern = 'train_record/epoch_logs_mnist_to_pattern_ws100_l5_100c-noP-R20.pt.txt'
rPattern = 'train_record/epoch_logs_mnist_to_mpattern_ws100_l5_100c-noP-R20.pt.txt'
unRelate = 'train_record/epoch_logs_mnist_to_imagenet_ws100_l5_100c-noP-R20.pt.txt'
unStruct = 'train_record/epoch_logs_mnist_to_random_ws100_l5_100c-noP-R20.pt.txt'
files_20rounds  = [baseline, uPattern, rPattern, unRelate, unStruct]

draw_test_accuracies([files_1rounds, files_5rounds, files_10rounds, files_20rounds], m_ranges, 'MNIST', 'noP-R', 'mnist_test_acc_R') 
draw_watermark_accuracies([files_1rounds, files_5rounds, files_10rounds, files_20rounds], m_ranges, 'MNIST', 'noP-R', 'mnist_watermark_acc_R') 


baseline = 'train_record/epoch_logs_federated_cifar_vgg16_100c_1localround.pt.txt'
uPattern = 'train_record/epoch_logs_cifar10_to_pattern_ws100_vgg16_100c-noP_R1.pt.txt'
rPattern = 'train_record/epoch_logs_cifar10_to_cpattern_ws100_vgg16_100c-noP_R1.pt.txt'
unRelate = 'train_record/epoch_logs_cifar10_to_imagenet_ws100_vgg16_100c-noP_R1.pt.txt'
unStruct = 'train_record/epoch_logs_cifar10_to_random_ws100_vgg16_100c-noP_R1.pt.txt'
files_1rounds  = [baseline, uPattern, rPattern, unRelate, unStruct]

baseline = 'train_record/epoch_logs_federated_cifar_vgg16_100c_5localround.pt.txt'
uPattern = 'train_record/epoch_logs_cifar10_to_pattern_ws100_vgg16_100c-noP_R5.pt.txt'
rPattern = 'train_record/epoch_logs_cifar10_to_cpattern_ws100_vgg16_100c-noP_R5.pt.txt'
unRelate = 'train_record/epoch_logs_cifar10_to_imagenet_ws100_vgg16_100c-noP_R5.pt.txt'
unStruct = 'train_record/epoch_logs_cifar10_to_random_ws100_vgg16_100c-noP_R5.pt.txt'
files_5rounds  = [baseline, uPattern, rPattern, unRelate, unStruct]

baseline = 'train_record/epoch_logs_federated_cifar_vgg16_100c_10localround.pt.txt'
uPattern = 'train_record/epoch_logs_cifar10_to_pattern_ws100_vgg16_100c-noP_R10.pt.txt'
rPattern = 'train_record/epoch_logs_cifar10_to_cpattern_ws100_vgg16_100c-noP_R10.pt.txt'
unRelate = 'train_record/epoch_logs_cifar10_to_imagenet_ws100_vgg16_100c-noP_R10.pt.txt'
unStruct = 'train_record/epoch_logs_cifar10_to_random_ws100_vgg16_100c-noP_R10.pt.txt'
files_10rounds  = [baseline, uPattern, rPattern, unRelate, unStruct]

baseline = 'train_record/epoch_logs_federated_cifar_vgg16_100c_20localround.pt.txt'
uPattern = 'train_record/epoch_logs_cifar10_to_pattern_ws100_vgg16_100c-noP_R20.pt.txt'
rPattern = 'train_record/epoch_logs_cifar10_to_cpattern_ws100_vgg16_100c-noP_R20.pt.txt'
unRelate = 'train_record/epoch_logs_cifar10_to_imagenet_ws100_vgg16_100c-noP_R20.pt.txt'
unStruct = 'train_record/epoch_logs_cifar10_to_random_ws100_vgg16_100c-noP_R20.pt.txt'
files_20rounds  = [baseline, uPattern, rPattern, unRelate, unStruct]

draw_test_accuracies([files_1rounds, files_5rounds, files_10rounds, files_20rounds], m_ranges, 'CIFAR', 'R', 'cifar10_test_acc_R', th=[75,90]) 
draw_watermark_accuracies([files_1rounds, files_5rounds, files_10rounds, files_20rounds], m_ranges, 'CIFAR', 'R', 'cifar10_watermark_acc_R') 

baseline = 'train_record/epoch_logs_federated_mnist_l5_100c_1localround.pt.txt'
uPattern = 'train_record/epoch_logs_mnist_to_pattern_ws100_l5_100c-P-R1.pt.txt'
rPattern = 'train_record/epoch_logs_mnist_to_mpattern_ws100_l5_100c-P-R1.pt.txt'
unRelate = 'train_record/epoch_logs_mnist_to_imagenet_ws100_l5_100c-P-R1.pt.txt'
unStruct = 'train_record/epoch_logs_mnist_to_random_ws100_l5_100c-P-R1.pt.txt'
files_1rounds  = [baseline, uPattern, rPattern, unRelate, unStruct]

baseline = 'train_record/epoch_logs_federated_mnist_l5_100c_5localround.pt.txt'
uPattern = 'train_record/epoch_logs_mnist_to_pattern_ws100_l5_100c-P-R5.pt.txt'
rPattern = 'train_record/epoch_logs_mnist_to_mpattern_ws100_l5_100c-P-R5.pt.txt'
unRelate = 'train_record/epoch_logs_mnist_to_imagenet_ws100_l5_100c-P-R5.pt.txt'
unStruct = 'train_record/epoch_logs_mnist_to_random_ws100_l5_100c-P-R5.pt.txt'
files_5rounds  = [baseline, uPattern, rPattern, unRelate, unStruct]

baseline = 'train_record/epoch_logs_federated_mnist_l5_100c_10localround.pt.txt'
uPattern = 'train_record/epoch_logs_mnist_to_pattern_ws100_l5_100c-P-R10.pt.txt'
rPattern = 'train_record/epoch_logs_mnist_to_mpattern_ws100_l5_100c-P-R10.pt.txt'
unRelate = 'train_record/epoch_logs_mnist_to_imagenet_ws100_l5_100c-P-R10.pt.txt'
unStruct = 'train_record/epoch_logs_mnist_to_random_ws100_l5_100c-P-R10.pt.txt'
files_10rounds  = [baseline, uPattern, rPattern, unRelate, unStruct]

baseline = 'train_record/epoch_logs_federated_mnist_l5_100c_20localround.pt.txt'
uPattern = 'train_record/epoch_logs_mnist_to_pattern_ws100_l5_100c-P-R20.pt.txt'
rPattern = 'train_record/epoch_logs_mnist_to_mpattern_ws100_l5_100c-P-R20.pt.txt'
unRelate = 'train_record/epoch_logs_mnist_to_imagenet_ws100_l5_100c-P-R20.pt.txt'
unStruct = 'train_record/epoch_logs_mnist_to_random_ws100_l5_100c-P-R20.pt.txt'
files_20rounds  = [baseline, uPattern, rPattern, unRelate, unStruct]

draw_test_accuracies([files_1rounds, files_5rounds, files_10rounds, files_20rounds], m_ranges, 'MNIST', 'PR', 'mnist_test_acc_PR_trained')
draw_watermark_accuracies([files_1rounds, files_5rounds, files_10rounds, files_20rounds], m_ranges, 'MNIST', 'PR', 'mnist_watermark_acc_PR')


baseline = 'train_record/epoch_logs_federated_cifar_vgg16_100c_1localround.pt.txt'
uPattern = 'train_record/epoch_logs_cifar10_to_pattern_ws100_vgg16_100c-P_R1.pt.txt'
rPattern = 'train_record/epoch_logs_cifar10_to_cpattern_ws100_vgg16_100c-P_R1.pt.txt'
unRelate = 'train_record/epoch_logs_cifar10_to_imagenet_ws100_vgg16_100c-P_R1.pt.txt'
unStruct = 'train_record/epoch_logs_cifar10_to_random_ws100_vgg16_100c-P_R1.pt.txt'
files_1rounds  = [baseline, uPattern, rPattern, unRelate, unStruct]

baseline = 'train_record/epoch_logs_federated_cifar_vgg16_100c_5localround.pt.txt'
uPattern = 'train_record/epoch_logs_cifar10_to_pattern_ws100_vgg16_100c-P_R5.pt.txt'
rPattern = 'train_record/epoch_logs_cifar10_to_cpattern_ws100_vgg16_100c-P_R5.pt.txt'
unRelate = 'train_record/epoch_logs_cifar10_to_imagenet_ws100_vgg16_100c-P_R5.pt.txt'
unStruct = 'train_record/epoch_logs_cifar10_to_random_ws100_vgg16_100c-P_R5.pt.txt'
files_5rounds  = [baseline, uPattern, rPattern, unRelate, unStruct]

baseline = 'train_record/epoch_logs_federated_cifar_vgg16_100c_10localround.pt.txt'
uPattern = 'train_record/epoch_logs_cifar10_to_pattern_ws100_vgg16_100c-P_R10.pt.txt'
rPattern = 'train_record/epoch_logs_cifar10_to_cpattern_ws100_vgg16_100c-P_R10.pt.txt'
unRelate = 'train_record/epoch_logs_cifar10_to_imagenet_ws100_vgg16_100c-P_R10.pt.txt'
unStruct = 'train_record/epoch_logs_cifar10_to_random_ws100_vgg16_100c-P_R10.pt.txt'
files_10rounds  = [baseline, uPattern, rPattern, unRelate, unStruct]

baseline = 'train_record/epoch_logs_federated_cifar_vgg16_100c_20localround.pt.txt'
uPattern = 'train_record/epoch_logs_cifar10_to_pattern_ws100_vgg16_100c-P_R20.pt.txt'
rPattern = 'train_record/epoch_logs_cifar10_to_cpattern_ws100_vgg16_100c-P_R20.pt.txt'
unRelate = 'train_record/epoch_logs_cifar10_to_imagenet_ws100_vgg16_100c-P_R20.pt.txt'
unStruct = 'train_record/epoch_logs_cifar10_to_random_ws100_vgg16_100c-P_R20.pt.txt'
files_20rounds  = [baseline, uPattern, rPattern, unRelate, unStruct]

draw_test_accuracies([files_1rounds, files_5rounds, files_10rounds, files_20rounds], m_ranges, 'CIFAR', 'PR', 'cifar10_test_acc_PR', th=[75,90]) 
draw_watermark_accuracies([files_1rounds, files_5rounds, files_10rounds, files_20rounds], m_ranges, 'CIFAR', 'PR', 'cifar10_watermark_acc_PR') 

          

                                
