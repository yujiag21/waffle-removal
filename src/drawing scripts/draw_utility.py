import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.transforms import BlendedGenericTransform

import matplotlib
params = {
    'text.latex.preamble': ['\\usepackage{gensymb}'],
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 1000,  # to adjust notebook inline plot size
    'axes.labelsize': 7, # fontsize for x and y labels (was 10)
    'axes.titlesize': 7,
    'font.size': 7, # was 10
    'legend.fontsize': 7, # was 10
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'text.usetex': True,
    'figure.figsize':  [5.67, 1.05], #was 1.2
    'font.family': 'serif',
}

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 7,
        }

matplotlib.rcParams.update(params)

matplotlib.rcParams.update(params)

def draw_test_accuracies_old(file_arrays, epoch_ranges, dataset, experiment, save_file_name, th=[95.00, 100.00]):

    fig, axs = plt.subplots(1, 4, sharey=True)
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
            axs[rrange].plot(epochs, test_acc[i], c=c, label=labels[i], linewidth=0.8)
            axs[rrange].set_xticks(xticks)
            axs[rrange].grid(True, axis='y')
    
        axs[rrange].set_ylim((th[0], th[1]))
    
    axs[0].set_xlabel('Aggregation Rounds')
    axs[0].set_ylabel('Test accuracy ($\%$)')  
    #axs[1][0].set_xlabel('Communication Rounds')
    #axs[1][0].set_ylabel('Test accuracy ($\%$)') 
    axs[3].legend(loc='best',frameon=False)
    axs[0].title.set_text(dataset + ' ' + experiment + ' ' + '$E_c=1$') 
    axs[1].title.set_text(dataset + ' ' + experiment + ' ' + '$E_c=5$') 
    axs[2].title.set_text('$E_c=10$') 
    axs[3].title.set_text('$E_c=20$') 
    plt.tight_layout()   
    plt.savefig(save_file_name + '.pdf', bbox_inches='tight', pad_inches = 0.05)   
    
    
def draw_test_accuracies(file_arrays, epoch_ranges, dataset, save_file_name, th=[95.00, 100.00]):

    fig, axs = plt.subplots(1, 4, sharey=True)
    fig_ls = []
    color=['#000000', '#008000', '#0000ff', '#ff0000', '#ff9900', '#008000', '#0000ff', '#ff0000', '#ff9900']
    labels = ['baseline', 'WafflePat.(R)', 'Embedded C.(R)', 'unRelate(R)', 'unStruct(R)', 'WafflePat.(PR)', 'Embedded C.(PR)', 'unRelate(PR)', 'unStruct(PR)']
    ls = ['-', '--','--','--','--', '-','-','-','-']
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
            if i == 0:
                linew = 1.1
            else:
                linew = 0.9
            l1 = axs[rrange].plot(epochs, test_acc[i], c=c, label=labels[i], linestyle= ls[i], linewidth=linew)
            axs[rrange].set_xticks(xticks)
            axs[rrange].grid(True, axis='y')
            axs[rrange].set_ylim((th[0], th[1]))
    
    axs[0].set_ylabel('Test accuracy ($\%$)')  
    fig.suptitle('Aggregation Rounds', y=-0.07)
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.98, -0.35),frameon=False, shadow=True, ncol=3)
    axs[0].title.set_text('$E_c,E_a=\{1,250\}$') 
    axs[1].title.set_text('$\{5,200\}$') 
    axs[2].title.set_text('$\{10,150\}$') 
    axs[3].title.set_text('$\{20,100\}$') 
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


baseline_1 = 'train_record/epoch_logs_federated_mnist_l5_100c_1localround.pt.txt'
uPattern_R1 = 'train_record/epoch_logs_mnist_to_pattern_ws100_l5_100c-noP-R1.pt.txt'
rPattern_R1 = 'train_record/epoch_logs_mnist_to_mpattern_ws100_l5_100c-noP-R1.pt.txt'
unRelate_R1 = 'train_record/epoch_logs_mnist_to_imagenet_ws100_l5_100c-noP-R1.pt.txt'
unStruct_R1 = 'train_record/epoch_logs_mnist_to_random_ws100_l5_100c-noP-R1.pt.txt'
uPattern_PR1 = 'train_record/epoch_logs_mnist_to_pattern_ws100_l5_100c-P-R1.pt.txt'
rPattern_PR1 = 'train_record/epoch_logs_mnist_to_mpattern_ws100_l5_100c-P-R1.pt.txt'
unRelate_PR1 = 'train_record/epoch_logs_mnist_to_imagenet_ws100_l5_100c-P-R1.pt.txt'
unStruct_PR1 = 'train_record/epoch_logs_mnist_to_random_ws100_l5_100c-P-R1.pt.txt'
files_1rounds  = [baseline_1, uPattern_R1, rPattern_R1, unRelate_R1, unStruct_R1, uPattern_PR1, rPattern_PR1, unRelate_PR1, unStruct_PR1]

baseline_5 = 'train_record/epoch_logs_federated_mnist_l5_100c_5localround.pt.txt'
uPattern_R5 = 'train_record/epoch_logs_mnist_to_pattern_ws100_l5_100c-noP-R5.pt.txt'
rPattern_R5 = 'train_record/epoch_logs_mnist_to_mpattern_ws100_l5_100c-noP-R5.pt.txt'
unRelate_R5 = 'train_record/epoch_logs_mnist_to_imagenet_ws100_l5_100c-noP-R5.pt.txt'
unStruct_R5 = 'train_record/epoch_logs_mnist_to_random_ws100_l5_100c-noP-R5.pt.txt'
uPattern_PR5 = 'train_record/epoch_logs_mnist_to_pattern_ws100_l5_100c-P-R5.pt.txt'
rPattern_PR5 = 'train_record/epoch_logs_mnist_to_mpattern_ws100_l5_100c-P-R5.pt.txt'
unRelate_PR5 = 'train_record/epoch_logs_mnist_to_imagenet_ws100_l5_100c-P-R5.pt.txt'
unStruct_PR5 = 'train_record/epoch_logs_mnist_to_random_ws100_l5_100c-P-R5.pt.txt'
files_5rounds  = [baseline_5, uPattern_R5, rPattern_R5, unRelate_R5, unStruct_R5, uPattern_PR5, rPattern_PR5, unRelate_PR5, unStruct_PR5]

baseline_10 = 'train_record/epoch_logs_federated_mnist_l5_100c_10localround.pt.txt'
uPattern_R10 = 'train_record/epoch_logs_mnist_to_pattern_ws100_l5_100c-noP-R10.pt.txt'
rPattern_R10 = 'train_record/epoch_logs_mnist_to_mpattern_ws100_l5_100c-noP-R10.pt.txt'
unRelate_R10 = 'train_record/epoch_logs_mnist_to_imagenet_ws100_l5_100c-noP-R10.pt.txt'
unStruct_R10 = 'train_record/epoch_logs_mnist_to_random_ws100_l5_100c-noP-R10.pt.txt'
uPattern_PR10 = 'train_record/epoch_logs_mnist_to_pattern_ws100_l5_100c-P-R10.pt.txt'
rPattern_PR10 = 'train_record/epoch_logs_mnist_to_mpattern_ws100_l5_100c-P-R10.pt.txt'
unRelate_PR10 = 'train_record/epoch_logs_mnist_to_imagenet_ws100_l5_100c-P-R10.pt.txt'
unStruct_PR10 = 'train_record/epoch_logs_mnist_to_random_ws100_l5_100c-P-R10.pt.txt'
files_10rounds  = [baseline_10, uPattern_R10, rPattern_R10, unRelate_R10, unStruct_R10, uPattern_PR10, rPattern_PR10, unRelate_PR10, unStruct_PR10]

baseline_20 = 'train_record/epoch_logs_federated_mnist_l5_100c_20localround.pt.txt'
uPattern_R20 = 'train_record/epoch_logs_mnist_to_pattern_ws100_l5_100c-noP-R20.pt.txt'
rPattern_R20 = 'train_record/epoch_logs_mnist_to_mpattern_ws100_l5_100c-noP-R20.pt.txt'
unRelate_R20 = 'train_record/epoch_logs_mnist_to_imagenet_ws100_l5_100c-noP-R20.pt.txt'
unStruct_R20 = 'train_record/epoch_logs_mnist_to_random_ws100_l5_100c-noP-R20.pt.txt'
uPattern_PR20 = 'train_record/epoch_logs_mnist_to_pattern_ws100_l5_100c-P-R20.pt.txt'
rPattern_PR20 = 'train_record/epoch_logs_mnist_to_mpattern_ws100_l5_100c-P-R20.pt.txt'
unRelate_PR20 = 'train_record/epoch_logs_mnist_to_imagenet_ws100_l5_100c-P-R20.pt.txt'
unStruct_PR20 = 'train_record/epoch_logs_mnist_to_random_ws100_l5_100c-P-R20.pt.txt'
files_20rounds  = [baseline_20, uPattern_R20, rPattern_R20, unRelate_R20, unStruct_R20, uPattern_PR20, rPattern_PR20, unRelate_PR20, unStruct_PR20]


draw_test_accuracies([files_1rounds, files_5rounds, files_10rounds, files_20rounds], m_ranges, 'MNIST', 'mnist_test_acc_watermarking')
#draw_watermark_accuracies([files_1rounds, files_5rounds, files_10rounds, files_20rounds], m_ranges, 'MNIST', 'PR', 'mnist_watermark_acc_PR')
#draw_test_accuracies([files_1rounds, files_5rounds, files_10rounds, files_20rounds], m_ranges, 'MNIST', 'noP-R', 'mnist_test_acc_R') 
#draw_watermark_accuracies([files_1rounds, files_5rounds, files_10rounds, files_20rounds], m_ranges, 'MNIST', 'noP-R', 'mnist_watermark_acc_R') 


baseline_1 = 'train_record/epoch_logs_federated_cifar_vgg16_100c_1localround.pt.txt'
uPattern_R1 = 'train_record/epoch_logs_cifar10_to_pattern_ws100_vgg16_100c-noP_R1.pt.txt'
rPattern_R1 = 'train_record/epoch_logs_cifar10_to_cpattern_ws100_vgg16_100c-noP_R1.pt.txt'
unRelate_R1 = 'train_record/epoch_logs_cifar10_to_imagenet_ws100_vgg16_100c-noP_R1.pt.txt'
unStruct_R1 = 'train_record/epoch_logs_cifar10_to_random_ws100_vgg16_100c-noP_R1.pt.txt'
uPattern_PR1 = 'train_record/epoch_logs_cifar10_to_pattern_ws100_vgg16_100c-P_R1.pt.txt'
rPattern_PR1 = 'train_record/epoch_logs_cifar10_to_cpattern_ws100_vgg16_100c-P_R1.pt.txt'
unRelate_PR1 = 'train_record/epoch_logs_cifar10_to_imagenet_ws100_vgg16_100c-P_R1.pt.txt'
unStruct_PR1 = 'train_record/epoch_logs_cifar10_to_random_ws100_vgg16_100c-P_R1.pt.txt'
files_1rounds  = [baseline_1, uPattern_R1, rPattern_R1, unRelate_R1, unStruct_R1, uPattern_PR1, rPattern_PR1, unRelate_PR1, unStruct_PR1]

baseline_5 = 'train_record/epoch_logs_federated_cifar_vgg16_100c_5localround.pt.txt'
uPattern_R5 = 'train_record/epoch_logs_cifar10_to_pattern_ws100_vgg16_100c-noP_R5.pt.txt'
rPattern_R5 = 'train_record/epoch_logs_cifar10_to_cpattern_ws100_vgg16_100c-noP_R5.pt.txt'
unRelate_R5 = 'train_record/epoch_logs_cifar10_to_imagenet_ws100_vgg16_100c-noP_R5.pt.txt'
unStruct_R5 = 'train_record/epoch_logs_cifar10_to_random_ws100_vgg16_100c-noP_R5.pt.txt'
uPattern_PR5 = 'train_record/epoch_logs_cifar10_to_pattern_ws100_vgg16_100c-P_R5.pt.txt'
rPattern_PR5 = 'train_record/epoch_logs_cifar10_to_cpattern_ws100_vgg16_100c-P_R5.pt.txt'
unRelate_PR5 = 'train_record/epoch_logs_cifar10_to_imagenet_ws100_vgg16_100c-P_R5.pt.txt'
unStruct_PR5 = 'train_record/epoch_logs_cifar10_to_random_ws100_vgg16_100c-P_R5.pt.txt'
files_5rounds  = [baseline_5, uPattern_R5, rPattern_R5, unRelate_R5, unStruct_R5, uPattern_PR5, rPattern_PR5, unRelate_PR5, unStruct_PR5]

baseline_10 = 'train_record/epoch_logs_federated_cifar_vgg16_100c_10localround.pt.txt'
uPattern_R10 = 'train_record/epoch_logs_cifar10_to_pattern_ws100_vgg16_100c-noP_R10.pt.txt'
rPattern_R10 = 'train_record/epoch_logs_cifar10_to_cpattern_ws100_vgg16_100c-noP_R10.pt.txt'
unRelate_R10 = 'train_record/epoch_logs_cifar10_to_imagenet_ws100_vgg16_100c-noP_R10.pt.txt'
unStruct_R10 = 'train_record/epoch_logs_cifar10_to_random_ws100_vgg16_100c-noP_R10.pt.txt'
uPattern_PR10 = 'train_record/epoch_logs_cifar10_to_pattern_ws100_vgg16_100c-P_R10.pt.txt'
rPattern_PR10 = 'train_record/epoch_logs_cifar10_to_cpattern_ws100_vgg16_100c-P_R10.pt.txt'
unRelate_PR10 = 'train_record/epoch_logs_cifar10_to_imagenet_ws100_vgg16_100c-P_R10.pt.txt'
unStruct_PR10 = 'train_record/epoch_logs_cifar10_to_random_ws100_vgg16_100c-P_R10.pt.txt'
files_10rounds  = [baseline_10, uPattern_R10, rPattern_R10, unRelate_R10, unStruct_R10, uPattern_PR10, rPattern_PR10, unRelate_PR10, unStruct_PR10]

baseline_20 = 'train_record/epoch_logs_federated_cifar_vgg16_100c_20localround.pt.txt'
uPattern_R20 = 'train_record/epoch_logs_cifar10_to_pattern_ws100_vgg16_100c-noP_R20.pt.txt'
rPattern_R20 = 'train_record/epoch_logs_cifar10_to_cpattern_ws100_vgg16_100c-noP_R20.pt.txt'
unRelate_R20 = 'train_record/epoch_logs_cifar10_to_imagenet_ws100_vgg16_100c-noP_R20.pt.txt'
unStruct_R20 = 'train_record/epoch_logs_cifar10_to_random_ws100_vgg16_100c-noP_R20.pt.txt'
uPattern_PR20 = 'train_record/epoch_logs_cifar10_to_pattern_ws100_vgg16_100c-P_R20.pt.txt'
rPattern_PR20 = 'train_record/epoch_logs_cifar10_to_cpattern_ws100_vgg16_100c-P_R20.pt.txt'
unRelate_PR20 = 'train_record/epoch_logs_cifar10_to_imagenet_ws100_vgg16_100c-P_R20.pt.txt'
unStruct_PR20 = 'train_record/epoch_logs_cifar10_to_random_ws100_vgg16_100c-P_R20.pt.txt'
files_20rounds  = [baseline_20, uPattern_R20, rPattern_R20, unRelate_R20, unStruct_R20, uPattern_PR20, rPattern_PR20, unRelate_PR20, unStruct_PR20]

draw_test_accuracies([files_1rounds, files_5rounds, files_10rounds, files_20rounds], m_ranges, 'CIFAR10', 'cifar10_test_acc_watermarking', th=[80,90]) 
#draw_watermark_accuracies([files_1rounds, files_5rounds, files_10rounds, files_20rounds], m_ranges, 'CIFAR', 'R', 'cifar10_watermark_acc_R') 

          

                                
