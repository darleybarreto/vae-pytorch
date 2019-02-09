# coding: utf-8

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from glob import glob

import seaborn as sns

pgf_with_rc_fonts = {
    "font.family": "serif",
    "font.serif": ["Arial"],           
    "font.sans-serif": ["DejaVu Sans"], 
    "pgf.texsystem": "pdflatex"
}
sns.set_style(style='white',rc=pgf_with_rc_fonts)

def hide_box(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def plot_train(x_n,y_n,x_g,y_g,y_axis,title,file_name):
    fig, ax = plt.subplots(figsize=(10,8))

    ax.plot(x_n,y_n,"#f03b20",linestyle='-',linewidth=1,label='Gaussian')
    ax.plot(x_g,y_g,"#a5a5a5",linestyle='-',linewidth=1,label='Gamma')
    
    ax.ticklabel_format(axis="y", style='sci', scilimits=(0, 2),useOffset=True,useMathText=True)

    ax.xaxis.set_ticks(np.arange(min(x_n), max(x_n)+1, max(x_n)/10.0))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),ncol=2,prop={'size': 18},labelspacing=0.,handletextpad=0.3)

    plt.xlabel('Epoches',fontsize=16, labelpad=10,x=0.5)
    plt.xticks(fontsize=16,rotation=30)
    plt.ylabel(y_axis,fontsize=20, labelpad=5)
    plt.yticks(fontsize=16)

    plt.title(title,fontsize=25,x=0.9,y=0.8)
    hide_box(ax)
    
    fig.savefig('{}.pdf'.format(file_name),bbox_inches="tight")
    
    plt.close(fig)

train_cifar = sorted(glob("data/train_*cifar-10*.log"))
train_mnist = sorted(glob("data/train_*mnist*.log")) 
val_cifar = sorted(glob("data/val_*cifar-10*.log"))
val_mnist = sorted(glob("data/val_*mnist*.log"))

dict_ = {}

df_train_cifar_n = pd.read_csv(train_cifar[1],sep='\t')
df_train_cifar_g = pd.read_csv(train_cifar[0],sep='\t')

df_val_cifar_n = pd.read_csv(val_cifar[1],sep='\t')
df_val_cifar_g = pd.read_csv(val_cifar[0],sep='\t')

dict_['CIFAR-10'] = {"train":(df_train_cifar_n,df_train_cifar_g),
                     "val":(df_val_cifar_n, df_val_cifar_g)}

df_train_mnist_n = pd.read_csv(train_mnist[1],sep='\t')
df_train_mnist_g = pd.read_csv(train_mnist[0],sep='\t')

df_val_mnist_n = pd.read_csv(val_mnist[1],sep='\t')
df_val_mnist_g = pd.read_csv(val_mnist[0],sep='\t')

dict_['MNIST'] = {"train": (df_train_mnist_n,df_train_mnist_g),
                  "val": (df_val_mnist_n, df_val_mnist_g)}

for dataset in dict_:
    for stage in dict_[dataset]:
        normal, gamma = dict_[dataset][stage]
        
        for loss in ["KL", "Recons"]:
            y_axis = "$D_{KL}(q_{\phi}||p_{\\theta})$" if loss == "KL" else "$p_{\\theta}(x|z)$"
            title = "{} ({})".format(dataset,stage)
            file_name = "./data/{}_{}_{}".format(dataset.lower().replace("-","_"),stage,loss.lower())

            plot_train(normal.Epoch.values,normal[loss].values, gamma.Epoch.values,gamma[loss].values,y_axis, title, file_name)

