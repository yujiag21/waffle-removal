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
    'figure.figsize': [5.67, 1.1],
    'font.family': 'serif',
}

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 7,
        }

matplotlib.rcParams.update(params)

def draw_pruning(file_arrays, dataset, experiment, save_file_name, th=[0.00, 100.00], legend=True):
    color=['#008000', '#0000ff', '#ff0000', '#ff9900']
    labels_test = ['WafflePat.(test)', 'Embedded C.(test)', 'unRelate(test)', 'unStruct(test)']
    labels_wm = ['WafflePat.(wm)', 'Embedded C.(wm)', 'unRelate(wm)', 'unStruct(wm)']
    num_of_epochs = ['1', '5', '10', '20']
    pruning_level = np.asarray([10, 20, 30, 40, 50, 60, 70, 80, 90])
    xticks = np.asarray([10, 30, 50, 70, 90])
    yticks = np.asarray([0, 25, 50, 75, 100])
    fig, axs = plt.subplots(1, 4, sharey=True)
    #for n_adv in [1, 20, 40, 60, 80, 100]:
    n_adv = 1
    counter = 0
    for rrange in range(4):
        file_array = file_arrays[rrange]
        test_acc = np.zeros((len(file_array), 9)) #np.zeros((len(file_array), 100))
        wm_acc = np.zeros((len(file_array), 9)) #np.zeros((len(file_array), 100))
        for fi in range(len(file_array)):
            with open(file_array[fi],'r') as file_obj:
                for line in file_obj:
                    if ('Finepruning' in line) and ('Epoch:'+ num_of_epochs[rrange] in line) and ('Test Acc:' in line) and ('num of adv:'+str(n_adv)+',' in line):
                        line_arr = line.split()
                        level = int(float(line_arr[1][14:-1])*10)
                        test_acc[fi,level-1] += np.float(line_arr[-1]) 
                    if ('Finepruning' in line) and ('Epoch:'+ num_of_epochs[rrange] in line) and ('Watermark Acc:' in line) and ('num of adv:'+str(n_adv)+',' in line):
                        line_arr = line.split()
                        level = int(float(line_arr[1][14:-1])*10)
                        wm_acc[fi,level-1] += np.float(line_arr[-1])
                        counter += 1              
        test_acc = test_acc/(4)
        wm_acc = wm_acc/(4)               
        for i, c in zip(range(len(file_array)), color):
            l1 = axs[rrange].plot(pruning_level, test_acc[i],  '--', c=c, label=labels_test[i], linewidth=0.9)
            l2 = axs[rrange].plot(pruning_level, wm_acc[i], c=c, label=labels_wm[i], linewidth=0.9)
            axs[rrange].set_xticks(xticks)
            axs[rrange].set_yticks(yticks)
            axs[rrange].axhline(y=43, c='black', lw=1.0, linestyle='dotted')
            axs[rrange].grid(True, axis='y')
            axs[rrange].set_ylim((th[0], th[1]))
            axs[rrange].set_xlim((10, 90))
            
    axs[0].set_ylabel('Accuracy ($\%$)')  
    axs[0].text(9.5, 30, '$T_{acc}$', fontdict=font)
    if legend:
        #axs[1].legend(loc='upper center', bbox_to_anchor=(0.98, -0.35),frameon=False, shadow=True, ncol=4)
        #fig.suptitle('Percentage of the neurons pruned ($\%$)', y=-0.07)
        fig.suptitle('Percentage of the neurons pruned ($\%$)', y=-0.08)
        axs[1].legend(loc='upper center', bbox_to_anchor=(0.98, -0.38),frameon=False, shadow=True, ncol=4)
    axs[0].title.set_text('$E_c,E_a=\{1,250\}$') 
    axs[1].title.set_text('$\{5,200\}$') 
    axs[2].title.set_text('$\{10,150\}$') 
    axs[3].title.set_text('$\{20,100\}$') 
    #plt.tight_layout()   
    plt.savefig(save_file_name + '_' + str(n_adv) + '_adv.pdf', bbox_inches='tight', pad_inches = 0.05) 
    

uPattern = 'train_record/epoch_logs_mnist_to_pattern_ws100_l5_100c-noP-R1.pt.txt'
rPattern = 'train_record/epoch_logs_mnist_to_mpattern_ws100_l5_100c-noP-R1.pt.txt'
unRelate = 'train_record/epoch_logs_mnist_to_imagenet_ws100_l5_100c-noP-R1.pt.txt'
unStruct = 'train_record/epoch_logs_mnist_to_random_ws100_l5_100c-noP-R1.pt.txt'
files_1roundsR  = [uPattern, rPattern, unRelate, unStruct]

uPattern = 'train_record/epoch_logs_mnist_to_pattern_ws100_l5_100c-noP-R5.pt.txt'
rPattern = 'train_record/epoch_logs_mnist_to_mpattern_ws100_l5_100c-noP-R5.pt.txt'
unRelate = 'train_record/epoch_logs_mnist_to_imagenet_ws100_l5_100c-noP-R5.pt.txt'
unStruct = 'train_record/epoch_logs_mnist_to_random_ws100_l5_100c-noP-R5.pt.txt'
files_5roundsR  = [uPattern, rPattern, unRelate, unStruct]

uPattern = 'train_record/epoch_logs_mnist_to_pattern_ws100_l5_100c-noP-R10.pt.txt'
rPattern = 'train_record/epoch_logs_mnist_to_mpattern_ws100_l5_100c-noP-R10.pt.txt'
unRelate = 'train_record/epoch_logs_mnist_to_imagenet_ws100_l5_100c-noP-R10.pt.txt'
unStruct = 'train_record/epoch_logs_mnist_to_random_ws100_l5_100c-noP-R10.pt.txt'
files_10roundsR  = [uPattern, rPattern, unRelate, unStruct]

uPattern = 'train_record/epoch_logs_mnist_to_pattern_ws100_l5_100c-noP-R20.pt.txt'
rPattern = 'train_record/epoch_logs_mnist_to_mpattern_ws100_l5_100c-noP-R20.pt.txt'
unRelate = 'train_record/epoch_logs_mnist_to_imagenet_ws100_l5_100c-noP-R20.pt.txt'
unStruct = 'train_record/epoch_logs_mnist_to_random_ws100_l5_100c-noP-R20.pt.txt'
files_20roundsR  = [uPattern, rPattern, unRelate, unStruct]

draw_pruning([files_1roundsR, files_5roundsR, files_10roundsR, files_20roundsR], 'MNIST', 'R', 'mnist_pruning_R', legend=False) 

uPattern = 'train_record/epoch_logs_mnist_to_pattern_ws100_l5_100c-P-R1.pt.txt'
rPattern = 'train_record/epoch_logs_mnist_to_mpattern_ws100_l5_100c-P-R1.pt.txt'
unRelate = 'train_record/epoch_logs_mnist_to_imagenet_ws100_l5_100c-P-R1.pt.txt'
unStruct = 'train_record/epoch_logs_mnist_to_random_ws100_l5_100c-P-R1.pt.txt'
files_1roundsPR  = [uPattern, rPattern, unRelate, unStruct]

uPattern = 'train_record/epoch_logs_mnist_to_pattern_ws100_l5_100c-P-R5.pt.txt'
rPattern = 'train_record/epoch_logs_mnist_to_mpattern_ws100_l5_100c-P-R5.pt.txt'
unRelate = 'train_record/epoch_logs_mnist_to_imagenet_ws100_l5_100c-P-R5.pt.txt'
unStruct = 'train_record/epoch_logs_mnist_to_random_ws100_l5_100c-P-R5.pt.txt'
files_5roundsPR  = [uPattern, rPattern, unRelate, unStruct]

uPattern = 'train_record/epoch_logs_mnist_to_pattern_ws100_l5_100c-P-R10.pt.txt'
rPattern = 'train_record/epoch_logs_mnist_to_mpattern_ws100_l5_100c-P-R10.pt.txt'
unRelate = 'train_record/epoch_logs_mnist_to_imagenet_ws100_l5_100c-P-R10.pt.txt'
unStruct = 'train_record/epoch_logs_mnist_to_random_ws100_l5_100c-P-R10.pt.txt'
files_10roundsPR  = [uPattern, rPattern, unRelate, unStruct]

uPattern = 'train_record/epoch_logs_mnist_to_pattern_ws100_l5_100c-P-R20.pt.txt'
rPattern = 'train_record/epoch_logs_mnist_to_mpattern_ws100_l5_100c-P-R20.pt.txt'
unRelate = 'train_record/epoch_logs_mnist_to_imagenet_ws100_l5_100c-P-R20.pt.txt'
unStruct = 'train_record/epoch_logs_mnist_to_random_ws100_l5_100c-P-R20.pt.txt'
files_20roundsPR  = [uPattern, rPattern, unRelate, unStruct]

m_ranges = [250, 200, 250, 200]

draw_pruning([files_1roundsPR, files_5roundsPR, files_10roundsPR, files_20roundsPR], 'MNIST', 'PR', 'mnist_pruning_PR', legend=False) 

baseline = 'train_record/epoch_logs_federated_cifar_vgg16_100c_1localround.pt.txt'
uPattern = 'train_record/epoch_logs_cifar10_to_pattern_ws100_vgg16_100c-noP_R1.pt.txt'
rPattern = 'train_record/epoch_logs_cifar10_to_cpattern_ws100_vgg16_100c-noP_R1.pt.txt'
unRelate = 'train_record/epoch_logs_cifar10_to_imagenet_ws100_vgg16_100c-noP_R1.pt.txt'
unStruct = 'train_record/epoch_logs_cifar10_to_random_ws100_vgg16_100c-noP_R1.pt.txt'
files_1roundsR  = [uPattern, rPattern, unRelate, unStruct]

baseline = 'train_record/epoch_logs_federated_cifar_vgg16_100c_5localround.pt.txt'
uPattern = 'train_record/epoch_logs_cifar10_to_pattern_ws100_vgg16_100c-noP_R5.pt.txt'
rPattern = 'train_record/epoch_logs_cifar10_to_cpattern_ws100_vgg16_100c-noP_R5.pt.txt'
unRelate = 'train_record/epoch_logs_cifar10_to_imagenet_ws100_vgg16_100c-noP_R5.pt.txt'
unStruct = 'train_record/epoch_logs_cifar10_to_random_ws100_vgg16_100c-noP_R5.pt.txt'
files_5roundsR  = [uPattern, rPattern, unRelate, unStruct]

baseline = 'train_record/epoch_logs_federated_cifar_vgg16_100c_10localround.pt.txt'
uPattern = 'train_record/epoch_logs_cifar10_to_pattern_ws100_vgg16_100c-noP_R10.pt.txt'
rPattern = 'train_record/epoch_logs_cifar10_to_cpattern_ws100_vgg16_100c-noP_R10.pt.txt'
unRelate = 'train_record/epoch_logs_cifar10_to_imagenet_ws100_vgg16_100c-noP_R10.pt.txt'
unStruct = 'train_record/epoch_logs_cifar10_to_random_ws100_vgg16_100c-noP_R10.pt.txt'
files_10roundsR  = [uPattern, rPattern, unRelate, unStruct]

baseline = 'train_record/epoch_logs_federated_cifar_vgg16_100c_20localround.pt.txt'
uPattern = 'train_record/epoch_logs_cifar10_to_pattern_ws100_vgg16_100c-noP_R20.pt.txt'
rPattern = 'train_record/epoch_logs_cifar10_to_cpattern_ws100_vgg16_100c-noP_R20.pt.txt'
unRelate = 'train_record/epoch_logs_cifar10_to_imagenet_ws100_vgg16_100c-noP_R20.pt.txt'
unStruct = 'train_record/epoch_logs_cifar10_to_random_ws100_vgg16_100c-noP_R20.pt.txt'
files_20roundsR  = [uPattern, rPattern, unRelate, unStruct]

draw_pruning([files_1roundsR, files_5roundsR, files_10roundsR, files_20roundsR], 'CIFAR10', 'R', 'cifar_pruning_R')

baseline = 'train_record/epoch_logs_federated_cifar_vgg16_100c_1localround.pt.txt'
uPattern = 'train_record/epoch_logs_cifar10_to_pattern_ws100_vgg16_100c-P_R1.pt.txt'
rPattern = 'train_record/epoch_logs_cifar10_to_cpattern_ws100_vgg16_100c-P_R1.pt.txt'
unRelate = 'train_record/epoch_logs_cifar10_to_imagenet_ws100_vgg16_100c-P_R1.pt.txt'
unStruct = 'train_record/epoch_logs_cifar10_to_random_ws100_vgg16_100c-P_R1.pt.txt'
files_1roundsPR  = [uPattern, rPattern, unRelate, unStruct]

baseline = 'train_record/epoch_logs_federated_cifar_vgg16_100c_5localround.pt.txt'
uPattern = 'train_record/epoch_logs_cifar10_to_pattern_ws100_vgg16_100c-P_R5.pt.txt'
rPattern = 'train_record/epoch_logs_cifar10_to_cpattern_ws100_vgg16_100c-P_R5.pt.txt'
unRelate = 'train_record/epoch_logs_cifar10_to_imagenet_ws100_vgg16_100c-P_R5.pt.txt'
unStruct = 'train_record/epoch_logs_cifar10_to_random_ws100_vgg16_100c-P_R5.pt.txt'
files_5roundsPR  = [uPattern, rPattern, unRelate, unStruct]

baseline = 'train_record/epoch_logs_federated_cifar_vgg16_100c_10localround.pt.txt'
uPattern = 'train_record/epoch_logs_cifar10_to_pattern_ws100_vgg16_100c-P_R10.pt.txt'
rPattern = 'train_record/epoch_logs_cifar10_to_cpattern_ws100_vgg16_100c-P_R10.pt.txt'
unRelate = 'train_record/epoch_logs_cifar10_to_imagenet_ws100_vgg16_100c-P_R10.pt.txt'
unStruct = 'train_record/epoch_logs_cifar10_to_random_ws100_vgg16_100c-P_R10.pt.txt'
files_10roundsPR  = [uPattern, rPattern, unRelate, unStruct]

baseline = 'train_record/epoch_logs_federated_cifar_vgg16_100c_20localround.pt.txt'
uPattern = 'train_record/epoch_logs_cifar10_to_pattern_ws100_vgg16_100c-P_R20.pt.txt'
rPattern = 'train_record/epoch_logs_cifar10_to_cpattern_ws100_vgg16_100c-P_R20.pt.txt'
unRelate = 'train_record/epoch_logs_cifar10_to_imagenet_ws100_vgg16_100c-P_R20.pt.txt'
unStruct = 'train_record/epoch_logs_cifar10_to_random_ws100_vgg16_100c-P_R20.pt.txt'
files_20roundsPR  = [uPattern, rPattern, unRelate, unStruct]

draw_pruning([files_1roundsPR, files_5roundsPR, files_10roundsPR, files_20roundsPR], 'CIFAR10', 'PR', 'cifar_pruning_PR')