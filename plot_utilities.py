import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.manifold import TSNE

def LineChart(x, y, legend:list, color:list=None, x_label:str="", y_label:str="", ylim:tuple=(0,0), title:str='', line_style=[],  
        save_path:str="", legend_loc:str="lower left", plot_show:bool=False, log_scale=[False, False]):
    figure, ax = plt.subplots()
    for ii in range(len(x)):
        if line_style == []:
            linestyle = '-'
        else:
            linestyle = line_style[ii]
        color_ = color[ii] if color is not None else None
        ax.plot(x[ii], y[ii], color=color_, linestyle=linestyle)
    ax.legend(legend, loc=legend_loc)
    ax.set_title(title)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if log_scale[0]:
        ax.set_xscale('log')
    if log_scale[1]:
        ax.set_yscale('log')
    if ylim != (0,0):
        ax.set_ylim(ylim)
    if save_path != "":
        plt.savefig(save_path, dpi=1000)
    if plot_show:
        plt.show()

    return figure ,ax
        

def ScatterChart(x, y, legend:list, x_label:str="", y_label:str="", fmt:list=None, 
        save_path:str="", legend_loc:str="lower left", plot_show:bool=False):
    figure, ax = plt.subplots()
    ax.plot(x[0], x[1], fmt[0])
    ax.plot(y[0], y[1], fmt[1])
    ax.legend(legend, loc=legend_loc)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    plt.axis('equal')
    if save_path != "":
        plt.savefig(save_path, dpi=1000)
    if plot_show:
        plt.show()

    return figure ,ax



def BarPlot(x, y, x_label:str, y_label:str, title:str=None, color:list=None, fmt='%.3f',
        save_path:str="", plot_show:bool=False):
    # creating the bar plot
    figure, ax = plt.subplots()
    bars = plt.bar(x, y, color=color)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.bar_label(bars, fmt=fmt)
    if title is not None:
        plt.title(title)
    if save_path != "":
        plt.savefig(save_path, dpi=1000)
    if plot_show:
        plt.show()

    return figure ,ax


def SNSBarPlot(df, x_label:str, y_label:str, hue:str, title:str=None, save_path:str="", \
        legend_loc:str="", plot_show:bool=False, log_scale=[False, False], color='hls'):
    plt.figure(figsize=(6,4))
    fig = plt.gcf()
    fig.set_size_inches(6, 4.5)
    ax = sns.barplot(x=x_label, y=y_label, 
        hue=hue, 
        data=df,
        palette=color
    )
    ax.set(title=title)
    plt.tight_layout()
    if log_scale[0]:
        plt.xscale('log')
    if log_scale[1]:
        plt.yscale('log')
    if legend_loc != '':
        plt.legend(loc=legend_loc)
    if save_path != "":
        plt.savefig(save_path, dpi=1000)
    if plot_show:
        plt.show()
    return ax


def BoxPlot(df, x_label:str, y_label:str, hue:str, title:str=None, save_path:str="", plot_show:bool=False, showfliers = False):
    plt.figure(figsize=(16,10))
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    sns.boxplot(x=x_label, y=y_label, 
        hue=hue, 
        data=df,
        showfliers=showfliers
    ).set(title=title)
    # plt.ylim(top=0.5 * df[y_label].max())
    if save_path != "":
        plt.savefig(save_path, dpi=1000)
    if plot_show:
        plt.show()

def PlotTSNE(x:np.ndarray, y, n_components=2, n_iter=1000, title:str="", save_path:str="", plot_show:bool=False):
    tsne = TSNE(n_components=n_components, verbose=0, perplexity=40, n_iter=n_iter)
    x_latent = tsne.fit_transform(x)
    tsne_result_df = pd.DataFrame({'tsne_1': x_latent[:,0], 'tsne_2': x_latent[:,1], 'label': y})
    plt.figure(figsize=(16,10))
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    sns.scatterplot(
        x="tsne_1", y="tsne_2",
        hue="label",
        data=tsne_result_df,
        legend="full",
        alpha=0.3
    ).set(title=title)
    if save_path != "":
        plt.savefig(save_path, dpi=1000)
    if plot_show:
        plt.show()

