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
    'figure.figsize': [5.67, 1.05], #was 1.2
    'font.family': 'serif',
}

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 7,
        }

matplotlib.rcParams.update(params)

def draw_finetuning(file_arrays, dataset, experiment, save_file_name, th=[0.00, 100.00], legend=True):

    fig, axs = plt.subplots(1, 4, sharey=True)
    color=['#008000', '#0000ff', '#ff0000', '#ff9900']
    labels_test = ['WafflePat.(test)', 'rPattern (test)', 'unRelate(test)', 'unStruct(test)']
    labels_wm = ['WafflePat.(wm)', 'rPattern(wm)', 'unRelate(wm)', 'unStruct(wm)']
    num_of_epochs = ['1', '5', '10', '20']
    num_of_adv = np.asarray([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])#np.linspace(1,100,100)
    xticks = np.asarray([1, 20, 40, 60, 80, 100])#np.arange(10,110,20)
    for rrange in range(4):
        file_array = file_arrays[rrange]
        test_acc = np.zeros((len(file_array), 11)) #np.zeros((len(file_array), 100))
        wm_acc = np.zeros((len(file_array), 11)) #np.zeros((len(file_array), 100))
        for fi in range(len(file_array)):
            with open(file_array[fi],'r') as file_obj:
                for line in file_obj:
                    if ('Finetuning' in line) and ('Epoch:'+ num_of_epochs[rrange] in line) and ('Test Acc:' in line):
                        line_arr = line.split()
                        adv_num = int(line_arr[3][4:-1])
                        if adv_num %10 ==0:
                            idx = int(int(line_arr[3][4:-1])/10)
                            test_acc[fi,idx] += np.float(line_arr[-1])
                        elif adv_num == 1:
                            test_acc[fi,0] += np.float(line_arr[-1])
                    if ('Finetuning' in line) and ('Epoch:'+ num_of_epochs[rrange] in line) and ('Watermark Acc:' in line):
                        line_arr = line.split()
                        adv_num = int(line_arr[3][4:-1])
                        if adv_num %10 ==0:
                            idx = int(int(line_arr[3][4:-1])/10)
                            wm_acc[fi,idx] += np.float(line_arr[-1])
                        elif adv_num == 1:
                            wm_acc[fi,0] += np.float(line_arr[-1])
                        
        test_acc = test_acc/4.0
        wm_acc = wm_acc/4.0                      
        for i, c in zip(range(len(file_array)), color):
            axs[rrange].plot(num_of_adv, test_acc[i],  '--', c=c, label=labels_test[i], linewidth=0.9)
            axs[rrange].plot(num_of_adv, wm_acc[i], c=c, label=labels_wm[i], linewidth=0.9)
            axs[rrange].set_xticks(xticks)
            axs[rrange].axhline(y=43, c='black', lw=1.0, linestyle='dotted')
            axs[rrange].grid(True, axis='y')   
            axs[rrange].set_ylim((th[0], th[1]))
            axs[rrange].set_xlim((1, 101))
            

    #axs[0].set_xlabel('Ratio (\%) of complete training data')
    axs[0].set_xlabel('Number of adversaries among all clients')
    axs[0].set_ylabel('Accuracy ($\%$)')  
    axs[0].text(1.1, 46, '$T_{acc}$', fontdict=font)
    #axs[3].legend(loc='best',frameon=False)
    #if legend:
    #    axs[1].legend(loc='upper center', bbox_to_anchor=(1.1, -0.15),fancybox=True, shadow=True, ncol=6)
    #axs[3].legend(bbox_to_anchor=(1.1, 1.05))
    axs[0].title.set_text('$E_c,E_a=\{1,250\}$') 
    axs[1].title.set_text('$\{5,200\}$') 
    axs[2].title.set_text('$\{10,150\}$') 
    axs[3].title.set_text('$\{20,100\}$') 
    plt.tight_layout()   
    plt.savefig(save_file_name + '.pdf', bbox_inches='tight', pad_inches = 0.05)
    
def draw_finetuning2(file_arrays1, file_arrays2, save_file_name, th=[0.00, 100.00], legend=True):

    fig, axs = plt.subplots(1, 4, sharey=True)
    color=['#008000', '#0000ff', '#ff0000', '#ff9900', '#008000', '#0000ff', '#ff0000', '#ff9900']
    labels = ['WafflePat.(R)', 'Embedded C.(R)', 'unRelate(R)', 'unStruct(R)', 'WafflePat.(PR)', 'Embedded C.(PR)', 'unRelate(PR)', 'unStruct(PR)']
    line_sty = ['solid', 'solid', 'solid', 'solid', 'dashed', 'dashed', 'dashed', 'dashed']
    num_of_epochs = ['1', '5', '10', '20', '1', '5', '10', '20']
    num_of_adv = np.asarray([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])#np.linspace(1,100,100)
    xticks = np.asarray([1, 20, 40, 60, 80, 100])#np.arange(10,110,20)
    fig_list = []

    for rrange in range(4):
        file_array = file_arrays1[rrange]
        file_array2 = file_arrays2[rrange]
        #test_acc = np.zeros((len(file_array), 11)) #np.zeros((len(file_array), 100))
        wm_acc = np.zeros((len(file_array)*2, 11)) #np.zeros((len(file_array), 100))
        for fi in range(len(file_array)):
            with open(file_array[fi],'r') as file_obj:
                for line in file_obj:
                    if ('Finetuning' in line) and ('Epoch:'+ num_of_epochs[rrange] in line) and ('Watermark Acc:' in line):
                        line_arr = line.split()
                        adv_num = int(line_arr[3][4:-1])
                        if adv_num %10 ==0:
                            idx = int(int(line_arr[3][4:-1])/10)
                            wm_acc[fi,idx] += np.float(line_arr[-1])
                        elif adv_num == 1:
                            wm_acc[fi,0] += np.float(line_arr[-1])
        for fi in range(len(file_array2)):
            with open(file_array2[fi],'r') as file_obj:
                for line in file_obj:
                    if ('Finetuning' in line) and ('Epoch:'+ num_of_epochs[rrange] in line) and ('Watermark Acc:' in line):
                        line_arr = line.split()
                        adv_num = int(line_arr[3][4:-1])
                        if adv_num %10 ==0:
                            idx = int(int(line_arr[3][4:-1])/10)
                            wm_acc[fi+4,idx] += np.float(line_arr[-1])
                        elif adv_num == 1:
                            wm_acc[fi+4,0] += np.float(line_arr[-1])
                        
        wm_acc = wm_acc/4.0                     
        for i in range(len(file_array)):
            l1 = axs[rrange].plot(num_of_adv, wm_acc[i], c=color[i], linestyle=line_sty[i], label=labels[i], linewidth=0.9)
            l2 = axs[rrange].plot(num_of_adv, wm_acc[i+4], c=color[i+4], linestyle=line_sty[i+4], label=labels[i+4], linewidth=0.9)
            axs[rrange].set_xticks(xticks)
            axs[rrange].axhline(y=43, c='black', lw=1.0, linestyle='dotted')
            axs[rrange].grid(True, axis='y')
            fig_list.append(l1)
            fig_list.append(l2)
            axs[rrange].set_ylim((th[0], th[1]))
            axs[rrange].set_xlim((1, 101))
  
            

    axs[0].set_ylabel('Accuracy ($\%$)')  
    axs[0].text(1.1, 30, '$T_{acc}$', fontdict=font)
    if legend:
        fig.suptitle('Number of adversaries among all clients', y=-0.08)
        axs[1].legend(loc='upper center', bbox_to_anchor=(0.98, -0.38),frameon=False, shadow=True, ncol=4)
    axs[0].title.set_text('$E_c,E_a=\{1,250\}$') 
    axs[1].title.set_text('$\{5,200\}$') 
    axs[2].title.set_text('$\{10,150\}$') 
    axs[3].title.set_text('$\{20,100\}$') 
    #plt.tight_layout()   
    plt.savefig(save_file_name + '.pdf', bbox_inches='tight', pad_inches = 0.05)        


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

draw_finetuning([files_1roundsR, files_5roundsR, files_10roundsR, files_20roundsR], 'MNIST', 'R', 'mnist_finetuning_R', legend=False) 

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

draw_finetuning([files_1roundsPR, files_5roundsPR, files_10roundsPR, files_20roundsPR], 'MNIST', 'PR', 'mnist_finetuning_PR') 
draw_finetuning2([files_1roundsR, files_5roundsR, files_10roundsR, files_20roundsR], [files_1roundsPR, files_5roundsPR, files_10roundsPR, files_20roundsPR], 'mnist_finetuning_wm_accuracy', legend=False) 

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

draw_finetuning([files_1roundsR, files_5roundsR, files_10roundsR, files_20roundsR], 'CIFAR10', 'R', 'cifar_finetuning_R', legend=False)

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

draw_finetuning([files_1roundsPR, files_5roundsPR, files_10roundsPR, files_20roundsPR], 'CIFAR10', 'PR', 'cifar_finetuning_PR')
draw_finetuning2([files_1roundsR, files_5roundsR, files_10roundsR, files_20roundsR], [files_1roundsPR, files_5roundsPR, files_10roundsPR, files_20roundsPR], 'cifar10_finetuning_wm_accuracy') 

 


"""
\begin{table}
\caption{Test and watermark accuracy of different federated learning models on MNIST and CIFAR10 when uPattern is used as WM set. Models with bad watermark accuracy are highlighted with red.} \label{tb:accuracy}
\centering
\begin{tabular}{| p{\dimexpr 0.14\linewidth-2\tabcolsep}
                       | p{\dimexpr 0.14\linewidth-2\tabcolsep}
                       | p{\dimexpr 0.18\linewidth-2\tabcolsep}
                       | p{\dimexpr 0.18\linewidth-2\tabcolsep}
                       | p{\dimexpr 0.18\linewidth-2\tabcolsep}
                       | p{\dimexpr 0.18\linewidth-2\tabcolsep}
                       |}
\hline 
 \multicolumn{6}{|c|}{$Acc(w, D_{test})\% /Acc(w, WM)\%$}
 \\\hline
\textbf{MNIST} & \hfil Baseline & \hfil A-trained & \hfil P-trained & \hfil R-trained & \hfil PR-trained
\\\hline \hline
$E_c=1$ & \hfil 98.97/- &  \hfil 97.73/100.0& \hfil 98.77/\textcolor{red}{9.00}  & \hfil 99.02/98.00 & \hfil 98.88/99.00 \\ 
\hline 
$E_c=5$ & \hfil 99.02/-  & \hfil 98.93/100.0 & \hfil 98.93/\textcolor{red}{33.00}  & \hfil 98.91/99.00 & \hfil 99.03/99.00 \\ 
\hline 
$E_c=10$ & \hfil 99.11/-  & \hfil 98.40/100.0 & \hfil 98.94/\textcolor{red}{4.00}  & \hfil 99.06/100.0 & \hfil 98.90/100.0 \\ 
\hline 
$E_c=20$ & \hfil 99.02/-  & \hfil 98.06/100.0 & \hfil 98.92/\textcolor{red}{37.00}  & \hfil 99.03/99.00 & \hfil 98.95/99.00 
\\\hline \hline
\textbf{CIFAR10} & \hfil Baseline & \hfil A-trained & \hfil P-trained & \hfil R-trained & \hfil PR-trained  
\\\hline \hline
$E_c=1$ & \hfil 86.27/- &  \hfil 84.80/100.0  & \hfil 85.76/\textcolor{red}{27.00} &\hfil 86.14/99.00  & \hfil 85.70/100.0 \\ 
\hline 
$E_c=5$ & \hfil 86.24/- & \hfil 85.83/100.0  & \hfil 85.92/\textcolor{red}{23.00}  & \hfil 85.76/100.0 & \hfil 85.61/100.0 \\ 
\hline 
$E_c=10$ & \hfil 85.90/- & \hfil 85.27/100.0  & \hfil 85.93/\textcolor{red}{27.00}  & \hfil 86.20/99.00 & \hfil 85.89/99.00 \\ 
\hline 
$E_c=20$ & \hfil 85.85/- & \hfil 85.22/100.0 & \hfil 85.41/\textcolor{red}{31.00}  & \hfil 85.66/99.00 & \hfil 85.67/100.0 \\ 
\hline 
\end{tabular}
\end{table}
"""