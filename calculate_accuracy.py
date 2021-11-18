#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:43:51 2019

@author: attaullah
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import matplotlib as mpl
import seaborn as sns
from matplotlib.ticker import NullFormatter
import glob
import os

# import pandas as pd ##(version: 0.22.0)
# import plotly.graph_objs as go
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


# mpl.use("pgf") # pgf tkagg
# pgf_with_pdflatex = {
#     "pgf.texsystem": "pdflatex",
#     "pgf.preamble": [
#         r"\usepackage[utf8x]{inputenc}",
#         r"\usepackage[T1]{fontenc}",
#         r"\usepackage{cmbright}",
#     ]
# }
# mpl.rcParams.update(pgf_with_pdflatex)
# mpl.rcParams['text.latex.unicode'] = True
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['pgf.texsystem'] = 'pdflatex'

# valid markers
valid_markers = ([item[0] for item in mpl.markers.MarkerStyle.markers.items()
                  if item[1] != 'nothing' and not item[1].startswith('tick')
                  and not item[1].startswith('caret')])

nullfmt = NullFormatter()  # no labels
sns.color_palette("Set3", 10)


# compare test accuracies
def compare_acc(x, y, name, labeled, x2=None, a_type=' Test'):
    fig = plt.figure(figsize=(10, 6))
    plt.grid()
    ax1 = fig.add_subplot(111)
    colors = ['g', 'r', 'b', 'c', 'm', 'y']
    markers = ['o', '+', '*', 'v', '^', '<', ',']
    for i in range(y.shape[1]):
        ax1.plot(x, y[:, i], label=r'$Run\#' + str(i + 1) + '$', marker=markers[i], c=colors[i])

    ax1.set_xlabel('Meta Iterations')
    ax1.set_ylabel(r"Accuracy  $\%$")
    ax1.set_xticks(range(0, len(x) + 2, 2))
    plt.title(name + '-' + str(labeled) + a_type + ' Accuracy')
    ax1.legend(loc='best')
    if x2 is not None:
        ax2 = ax1.twinx()
        ax2.plot(x, x2, c='k')
        ax2.set_ylabel('Number of labelled')
        # ax2.cla()

    # plt.savefig(name+'_run_comparison.pgf')
    plt.show()


# compare KNN and  LLGC Improvement
def compare_llgc(x, y, name, labeled):
    fig = plt.figure(figsize=(10, 6))
    plt.grid()
    colors = ['g', 'r', 'b', 'c', 'm', 'y']
    count = 0
    for i in range(y.shape[1] // 2):
        plt.plot(x, y[:, count], label='Run#' + str(i + 1) + ' KNN', c=colors[i])
        plt.plot(x, y[:, count + 1], label='Run#' + str(i + 1) + ' LLGC', lineStyle='--', c=colors[i])
        count += 2
    plt.xlabel('Meta Iterations')
    plt.ylabel(r'Accuracy  $\%$')
    plt.xticks(range(0, len(x) + 2, 2))
    plt.title(name + '-' + str(labeled) + ' LLGC Improvement over Siamese Nets')
    plt.legend(loc='best')
    # plt.savefig(name+'_run_comparison.pgf')
    plt.show()


def remove_files(path, exp='*.INFO'):
    for hgx in glob.glob(path+"*"+exp):
        os.remove(hgx)


def show_content(path, exp='*.INFO'):
    for hgx in glob.glob(path+"*"+exp):
        print('||| contents of {}  |||'.format(hgx))
        with open(hgx) as f:
            print(f.read())


def plot_runs(path, n_label, full, label, substr='', conf="", ylimit='', save=False, plot=False, v=False, ext='.pgf'):
    labelm = np.mean(label)
    fullm = np.mean(full)
    s_ = []
    col_idx = -1  # last column contains test accuracy
    template1 = "\t  -dataset {}  -network {} -lr {} -epochs {} -opt {} -self_training {}  -lt {}"
    run = 0

    mini_path = path + str(n_label) + substr + "*self-training*" + conf
    skip_header = 5  # skip rows
    if v:
        print("self-training :  ", mini_path, skip_header)

    for name in glob.glob(mini_path):
        run += 1
        try:
            s = genfromtxt(name, skip_header=skip_header, delimiter=',')
            name1 = name
            name = name.split('/')[-1]
            if plot:
                label_str = r' run# ' + str(run)
                plt.plot(range(len(s)), s[:, col_idx], label=label_str)
            last_value = s[-1, col_idx]
            s_.append(last_value)

            if v:
                print('run# {}  mti  {}-> {:.2f}\t  {}'.format(run, len(s), last_value, name))
                import ast
                s1 = open(name1, "r+")
                x = s1.readlines()
                params = ast.literal_eval(x[0])
                w = " -weights" if params['weights'] else ""
                print(template1.format(params['dataset'], params['network'], params['lr'], params['epochs'],
                                       params['opt'], w, params['lt']))
        except (IndexError, ValueError):
            if v:
                print('!!!!unable to parse ', name)
            pass

    if v:
        print(r' {} {:.2f} $\pm$ {:.2f}'.format(run, np.mean(s_), np.std(s_)))
    if plot:
        n_label_str = str(n_label) + r'-label '
        plt.axhline(labelm, 0, 4, linestyle='--', c='b', label=n_label_str)
        all_label_str = 'All label '
        plt.axhline(fullm, 0, 4, linestyle='--', c='g', label=all_label_str)
        plt.legend()
        plt.ylabel(r'Accuracy %')
        plt.xlabel(r'Meta Iterations')
        plt.grid()

        if ylimit != '':
            plt.ylim(ylimit[0], ylimit[1])
        else:
            plt.ylim((min(label) - 2 * np.std(label), max(full) + 8 * np.std(full)))
        if save:
            path_list = path.split('/')
            save_path = './vis/' + path_list[1] + '/' + path_list[2] + '/' + path_list[0] + '_' + ext
            print(save_path)
            plt.savefig(save_path)
        plt.show()
        plt.clf()

    return s_


def get_n_labelled_accuracy(path, size=300, substr='', v=False):
    all_list = []
    import ast
    mini_path = path + str(size) + '*' + substr + '*'
    target_line = 2

    if v:
        print(size, "-labelled acc ::  ", mini_path, '  ', target_line)
    template1 = "\t  -dataset {}  -network {} -lr {} -epochs {} -opt {}  {} -lt {} {}"
    for name in glob.glob(mini_path):
        try:
            s = open(name, 'r+')
            name = name.split('/')[-1]
            content = s.readlines()
            params = ast.literal_eval(content[0])

            accuracies = content[target_line].split(' ')[-2]  # [:-2]
            if v:
                print(accuracies, '\t', name)

                w = " -weights" if params['weights'] else "-noweights"
                semi = " -semi" if params['semi'] else "-nosemi"
                print(template1.format(params['dataset'], params['network'], params['lr'], params['epochs'],
                                       params['opt'], w, params['lt'], semi))
            all_list.append(float(accuracies))
        except (IndexError, ValueError, FileNotFoundError, OSError):
            if v:
                print('!!!!unable to parse ', name)
            pass
    if not all_list:
        all_list = [0.]
    return all_list


def get_n_all_labeled(dataset):
    if 'svhn' in dataset:
        n_label = [1000, 73257]
        dataset = 'svhn_cropped'
    elif 'cifar10' in dataset:
        n_label = [4000, 50000]
    elif 'mnist' in dataset:
        n_label = [100, 60000]
    else:  # 'plant' in dataset:
        n_label = [380, 43456]

    return n_label


def get_substr(p, weights, labelling):
    sub_str = "*"
    if "cross" not in p:
        sub_str += labelling
    if weights:
        sub_str += "w-"
    # else:
    #     sub_str += "[!w-]"
    return sub_str


def self_accuracy(path, dataset, network, n_label=-1, weights=False, opt='adam', labelling='knn', conf="*",
                  v=False, plot=False,  ret_all=False, ylim='', save=False, ext='.pgf'):
    if n_label == -1:
        n_label = get_n_all_labeled(dataset)

    substr = get_substr(path, weights, labelling)
    substr += opt
    # print("self-acc       :::  ", path , dataset, network)
    full_path = path + dataset + '/' + network + '/'
    # cleanup
    remove_files(full_path)

    label = get_n_labelled_accuracy(full_path, size=n_label[0], substr=substr, v=v)

    full = get_n_labelled_accuracy(full_path, size=n_label[1], substr=substr,  v=v)

    self = plot_runs(full_path, n_label[0], full, label, substr=substr, v=v, plot=plot, ylimit=ylim,
                     save=save, ext=ext, conf=conf)

    if not self:
        self = [0.]
    if ret_all:
        return label, self, full
    return self


def n_all_accuracy(path, dataset, network, n_label=-1, weights=False, opt='adam', labelling='knn', v=False):
    if n_label == -1:
        n_label = get_n_all_labeled(dataset)

    substr = get_substr(path, weights, labelling)
    substr += opt
    full_path = path + dataset + '/' + network + '/'
    # cleanup
    remove_files(full_path)

    label = get_n_labelled_accuracy(full_path, size=n_label[0], substr=substr, v=v)
    full = get_n_labelled_accuracy(full_path, size=n_label[1], substr=substr,  v=v)

    return label, full


def mean_std(values, pm):
    output = str(np.round(np.mean(values), 2)) + pm + str(np.round(np.std(values), 2))
    return output


def print_stats(label, self, full, init="", sep="", pm="", std=False):
    if std:
        if "&" in sep:
            template = r"{} & $ {} $ {} $ {} $ {} $ {} $ \\\ "
        else:
            template = " {}          :  {}  {}  {} {} {}"
    else:
        template = " {}          :  {}  {}  {} {} {}"
    output = template.format(init, mean_std(label, pm), sep, mean_std(self, pm), sep, mean_std(full, pm))
    if std:
        print(output)
        return
    output = [init, mean_std(label, pm), mean_std(self, pm), mean_std(full, pm)]
    return output


def print_stats_c(label, self, full,label_t, self_t, full_t, init="", sep="", pm="", std=False):
    if std:
        if "&" in sep:
            template = r"{} & $ {} $ {} $ {} $ {} $ {} $ {} $ {} ${} $ {} ${} $ {} $ \\\ "
        else:
            template = " {}          :  {}  {} {} {} {} {} {} {} {} {} {} "
        output = template.format(init, mean_std(label, pm), sep,  mean_std(label_t, pm), sep, mean_std(self, pm), sep,
                             mean_std(self_t, pm), sep, mean_std(full, pm), sep, mean_std(full_t, pm))
        print(output)
        return
    output = [init, mean_std(label, pm), mean_std(label_t, pm), mean_std(self, pm),mean_std(self_t, pm),
              mean_std(full, pm), mean_std(full_t, pm)]
    return output


def print_stats_two(self, self_t, init="", sep="", pm="", std=False):
    if std:
        if "&" in sep:
            template = r"{} & $ {} $ {} $ {} $  \\\ "
        else:
            template = " {}          :  {}  {} {} {} {}  "
        output = template.format(init.upper(), mean_std(self, pm), sep,  mean_std(self_t, pm))
        print(output)
        return
    output = [init.upper(), mean_std(self, pm), mean_std(self_t, pm)]
    return output


def print_stats_single(self, sep="", pm="", std=False):
    if std:
        if "&" in sep:
            template = " & $ {} $   "
            output = template.format(mean_std(self, pm), sep)

            return output
        else:
            template = " {}  {} {} {} {}  "
        output = template.format(mean_std(self, pm), sep)
        print(output)
        return
    output = mean_std(self, pm)
    return output


def full_report(logs, dataset, n, pm='±', sep="", opt='*', weights=False, std=False, verbose=False, plot=False):
    from prettytable import PrettyTable

    for d in dataset:

        x = PrettyTable()
        x.field_names = ["Loss", "Labelled", "Self-training", "All-labelled"]
        for p in logs:
            print(d, n, opt, weights,  "#....")
            label, self, full = self_accuracy(p, d, n,  ret_all=True, opt=opt, weights=weights, v=verbose, plot=plot)
            if std:
                print_stats(label, self, full, p, sep, pm, std)
            else:
                x.add_row(print_stats(label, self, full, p, sep, pm))
        if not std:
            print(x)


def full_report_networks(logs, dataset, networks, pm='±', sep="", opt='*', weights=False, std=False, verbose=False, plot=False):
    from prettytable import PrettyTable
    p = logs[0]
    for d in dataset:
        print("Dataset::  ", d)
        x = PrettyTable()
        x.field_names = ["Network", "Labelled", "Self-training", "All-labelled"]
        for n in networks:
            # for p in logs:
            label, self, full = self_accuracy(p, d, n,  ret_all=True, opt=opt, weights=weights, v=verbose, plot=plot)
            if std:
                print_stats(label, self, full, n, sep, pm, std)
            else:
                x.add_row(print_stats(label, self, full, n, sep, pm))
        if not std:
            print(x)


def full_report_combined(logs, dataset, n, pm='±', sep="", opt='*', verbose=False, std=False, weights=False, labelling='knn'):
    from prettytable import PrettyTable
    x = PrettyTable()
    x.field_names = ["Dataset", "Labelled", "Self-training", "All-labelled"]
    print("type {} dataset: {} network: {} loss_layers {} weights {}".format(logs, dataset, n, opt, weights))

    for d in dataset:

        for p in logs:
            label, self, full = self_accuracy(p, d, n, weights=False, ret_all=True, opt=opt, v=verbose,
                                              labelling=labelling)
            if std:
                print_stats(label, self, full,  d+"  Random", sep, pm, std)
            else:
                x.add_row(print_stats(label, self, full, d + " Random ", sep, pm))

            if weights:
                label, self, full = self_accuracy(p, d, n, weights=True, ret_all=True, opt=opt, v=verbose,
                                                  labelling=labelling)
                if std:
                    print_stats(label, self, full, "  & ImageNet", sep, pm, std)
                else:
                    x.add_row(print_stats(label, self, full, d + " ImageNet ", sep, pm))
    if not std:
        print(x)


def n_report_combined(logs, dataset, n, pm='±', sep="", opt='*', verbose=False, std=False, weights=False,
                      labelling='knn'):
    from prettytable import PrettyTable
    x = PrettyTable()
    x.field_names = ["Dataset", "N-Labelled",  "All-labelled"]
    print("type {} dataset: {} network: {} loss_layers {} weights {}".format(logs, dataset, n, opt, weights))
    random_str = "  Random" if weights else ''
    for d in dataset:

        for p in logs:
            label, full = n_all_accuracy(p, d, n, weights=False, opt=opt, v=verbose, labelling=labelling)
            if std:
                print_stats_two(label, full,  d + random_str, sep, pm, std)
            else:
                x.add_row(print_stats_two(label, full, d + random_str, sep, pm))

            if weights:
                label, full = n_all_accuracy(p, d, n, weights=True,  opt=opt, v=verbose, labelling=labelling)
                if std:
                    print_stats_two(label, full, "  & ImageNet", sep, pm, std)
                else:
                    x.add_row(print_stats_two(label, full, d + " ImageNet ", sep, pm))
    if not std:
        print(x)


def full_report_n_all(logs, dataset, n, pm='±', sep="", opt='*', v=False, std=False, weights=False, labelling='knn'):
    from prettytable import PrettyTable

    for d in dataset:
        x = PrettyTable()
        x.field_names = ["Loss", "Labelled",  "All-labelled"]
        for p in logs:
            print(p, d, n, opt, weights,  ".......")
            # if p in ['logs/', 'selflogs/']:  # supervised special case above
            label, _, full = self_accuracy(p, d, n, weights=False, ret_all=True, opt=opt, v=v, labelling=labelling)

            if std:
                print_stats_two(label, full, "  Random", sep, pm, std)
            else:
                x.add_row(print_stats_two(label, full, "Random ", sep, pm))

            if weights:
                label, _, full = self_accuracy(p, d, n, weights=True, ret_all=True, opt=opt, v=v, labelling=labelling)

                if std:
                    print_stats_two(label,  full,  "  & ImageNet", sep, pm, std)
                else:
                    x.add_row(print_stats_two(label,  full, "ImageNet ", sep, pm))
        if not std:
            print(x)


def full_report_self(logs, dataset, n, pm='±', sep="", opt='*', v=False, std=False, wres='-w',labelling='knn', weights=True):
    from prettytable import PrettyTable

    for d in dataset:
        x = PrettyTable()
        x.field_names = ["Loss", "self-CE", "self-Triplet"]
        for p in logs:
            print(p, d, n, "######")

            _, self, _ = self_accuracy(p, d, n,  weights=False, ret_all=True, opt=opt, v=v, labelling=labelling)
            _, self_t, _ = self_accuracy(p, d, n,  weights=False, ret_all=True, opt=opt, v=v, labelling=labelling)
            if std:
                print_stats_two(self, self_t, " Random", sep, pm, std)
            else:
                x.add_row(print_stats_two(self, self_t,"Random ", sep, pm))
            # w =true
            if weights:
                _, self, _ = self_accuracy(p, d, n,  weights=True, ret_all=True, opt=opt, v=v,  labelling=labelling)
                _, self_t, _ = self_accuracy(p, d, n, weights=True, ret_all=True, opt=opt, v=v, labelling=labelling)
                if std:
                    print_stats_two(self, self_t, " & ImageNet", sep, pm, std)
                else:
                    x.add_row(print_stats_two(self, self_t, " ImageNet", sep, pm, std))
        if not std:
            print(x)


def full_report_self_config(logs, dataset, n="wrn-28-2", pm='±', sep="", opt='*', v=False, std=False, wres='-w',labelling='knn', weights=True):
    from prettytable import PrettyTable
    if weights:
        w_list = [False, True]
    else:
        w_list = [False]
    for w in w_list:
        x = PrettyTable()
        x.field_names = ["Method"] + dataset  # ["Loss", "self-CE", "self-Triplet"]

        if w:
            print("ImageNet weights")
        else:
            print("Random weights")
        for idx, p in enumerate(logs):
            if idx ==2:
                n = n + "/simple"  # Third config
                lv = [p+"simple"]
            else:
                n = n.split('/')[0]   # 1-2 config
                lv = [p]
            for d in dataset:
                self = self_accuracy(p, d, n,  weights=w, ret_all=False, opt=opt, v=v,
                                     labelling=labelling)

                lv.append(print_stats_single(self, sep, pm,std))

            if not std:
                x.add_row(lv)
            else:
                print(' '.join(lv) , "\\\\")
        if not std:
            print(x)


def self_train_plots(logs, dataset, n, pm='±', sep="\t", opt='*', ylim='', save=False, weights=False, ext='.pgf'):
    print("\t            Labelled\t Self-training\t All-labelled")

    for d in dataset:
        print(d, "##############################################")
        for p in logs:
            print(p,"...")
            if p == 'logs/':  # supervised special case
                label, self, full = self_accuracy(p, d, n,  ret_all=True, plot=True, opt=opt, ylim=ylim,
                                                  save=save, weights=weights, ext=ext)
                print_stats(label, self, full, "so  ", sep, pm, std=True)
                # LLGC
                if n in ['simple', 'ssdl']:
                    label, self, full = self_accuracy(p, d, n,  ret_all=True,  plot=True,
                                                      opt=opt, ylim=ylim, save=save, ext=ext)
                    print_stats(label, self, full, "llgc", sep, pm, std=True)
            label, self, full = self_accuracy(p, d, n, ret_all=True, opt=opt, plot=True, ylim=ylim, save=save,
                                              weights=weights, ext=ext)
            print_stats(label, self, full, p, sep, pm, std=True)


def self_train_w_plots(logs, dataset, n, pm='\\pm', sep="&", opt='*', ylim='', save=False, ext='.pgf', wres='-w'):
    print("\t            Labelled\t Self-training\t All-labelled")

    for d in dataset:
        print(d, "##############################################")
        for p in logs :
            # print(p)
            if p == 'logs/'  or "self" in p:  # supervised special case
                label, self, full = self_accuracy(p, d, n,  ret_all=True, plot=True, opt=opt, ylim=ylim,
                                                  save=save, ext=ext, )
                labelw, selfw, fullw = self_accuracy(p, d, n,  ret_all=True, plot=True, opt=opt, ylim=ylim,
                                                     save=save, weights=True, ext=ext, )
                print_stats(label, self, full, "so  ", sep, pm, std=True)
                print_stats(labelw, selfw, fullw, "so-w  ", sep, pm, std=True)

            label, self, full = self_accuracy(p, d, n, ret_all=True, opt=opt, plot=True, ylim=ylim, save=save,
                                              ext=ext,)
            labelw, selfw, fullw = self_accuracy(p, d, n, ret_all=True, opt=opt, plot=True, ylim=ylim,
                                                 save=save, weights=True, ext=ext)
            print_stats(label, self, full, p, sep, pm, std=True)
            print_stats(labelw, selfw, fullw, p+" w ", sep, pm, std=True)


def get_embeddings(logs, dataset='cifar10', n='vgg16', config=['test'], opt='*', weights=False, so=True, v=False,
                   ver='umap',legend=True):
    # from Visualizations import plot_umap, load_label_names, t_sne_vis
    # lbls = load_label_names()
    if 'svhn' in dataset:
        n_label = 1000
        dataset = 'svhn_cropped'
    elif 'cifar10' in dataset:
        n_label = 4000
    elif 'mnist' in dataset:
        n_label = 100
    elif 'plant' in dataset:
        n_label = 380
    else:
        n_label = 100
    if weights:
        w = '-w-'
    else:
        w = '-nw'
    if so:
        so = '*so-' + str(n_label)
    else:
        so = '-' + str(n_label)  # '*'
    d = dataset

    for p in logs:
        print(d, p, n, w, opt, "##############################################")
        for c in config:
            mini_path = p + d + '/' + n + c + so + opt + '*' + w + '*'
            print(mini_path)
            file_list = glob.glob(mini_path)
            if 'contrast' in p:
                sorted_list = [file_list[0], file_list[2], file_list[1]]  # for logs
            else:
                # if so:
                if len(file_list) >3:
                    sorted_list = [file_list[2], file_list[1], file_list[0], file_list[3]]
                elif len(file_list) >2:
                    sorted_list = [file_list[1], file_list[2], file_list[0]]
                elif len(file_list) >1 :
                    sorted_list = [file_list[1], file_list[0]]
                else:
                    sorted_list = [file_list[0]]

                # else:
                #     sorted_list = [file_list[1], file_list[0], file_list[2]]
            for name in sorted_list:
                try:
                    data = np.load(name)
                    x, y = data['embed'], data['labels']
                    if y.ndim > 1:
                        y = np.argmax(y, 1)
                    if 'cifar' in d:

                        test_lbls = [lbls[i] for i in y]
                        y = np.array(test_lbls)
                    if ver == 'umap':
                        embedding, ax = plot_umap(x, y,legend=legend)
                    else:
                        embedding, ax = t_sne_vis(x, y)
                    # ax.show()
                    if v:
                        print('\t', name.split('/')[-1], '  ', mini_path, ' ', x.shape, ' ', y.shape)
                except (IndexError, ValueError, FileNotFoundError, OSError):
                    if v:
                        print('!!!!unable to parse ', name)
                    pass


iterations = 25


def cifar10_triplet_cross_entropy(path):
    n_label = 4000
    dataset = 'cifar10'
    network = 'vgg16'
    substr = 'knn*dam-nw'
    so = True
    full_len = 50

    labeled = get_n_labelled_accuracy(path + dataset + '/' + network + '/', n_label, substr=substr, so=so)
    labeledm = np.mean(labeled)
    print(r'{} -labeled {:.2f}  $\pm$ {:.2f}'.format(n_label, labeledm, np.std(labeled)))
    Full = get_full_accuracy(path + dataset + '/' + network + '/', size=full_len, substr=substr, so=so)
    Fullm = np.mean(Full)
    print(r'Fully {:.2f} $\pm$  {:.2f}'.format(Fullm, np.std(Full)))
    ###
    plot_runs(path, n_label, Full, labeled, dataset=dataset, network=network, substr=substr, so=so, ylimit=[60, 100],
              save=True)

