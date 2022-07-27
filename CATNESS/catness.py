from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import networkx as nx
from joblib import parallel_backend
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
from math import log
import numpy as np
import pandas as pd
import os

from helpers.timer import Timer

def mim(size, bins, data, jobs):
    """
    Calculates the mutual information matrix.

    Input:
    The number of genes, the number of bins and the matrix (genes, samples).

    Output:
    Matrix (np.array) with the mim.
    """

    matrix = np.zeros((size, size), dtype = "float64")
    counter = 0

    tempo = Parallel(n_jobs = jobs)(calc_mi(data[i].copy(), bins, data[j].copy()) for i in np.arange(size) for j in np.arange(i + 1, size))

    for i in np.arange(size):
        for j in np.arange(i + 1, size):
            matrix[i][j] = tempo[counter]
            counter += 1

    return matrix

@delayed
@wrap_non_picklable_objects
def calc_mi(x, bins, y):
    c_xy = np.histogram2d(x, y, bins)[0]
    return mutual_info_score(None, None, contingency=c_xy)

def threshold_calculation(matrix, bins, jobs, n_perm = 2):
    """
    Computes the threshold to make a clearer mutual information matrix.

    Input:
    First MIM computation and the amount of bins to calculate the MIM
    of the permutation.

    Output:
    Threshold value (int).
    """

    n_genes, n_cases = matrix.shape
    permutations = np.zeros((n_perm, n_genes, n_genes))

    # Execution of the permutation
    for perm in np.arange(n_perm):
        # Shuffle the matrix
        perm_matrix = [matrix[np.random.permutation(n_genes), i] for i in np.arange(n_cases)]
        perm_matrix = np.vstack(perm_matrix).T

        # Execution of the MIM computation
        permutations[perm] = mim(n_genes, bins, perm_matrix, jobs)

    return np.amax(np.mean(permutations, axis = 0))

def lioness_algorithm(data, path, jobs = 1):
    """
    LIONESS algorithm.

    Input:
    data (pd.DataFrame): Numeric matrix with samples in columns.
    path (string): Path to the directory where the networks will be saved.
    jobs (int): Number of jobs to be parallelized.

    Output:
    Sample-specific matrix per each sample.
    """
    with Timer('Reading data...'):
        columns = data.columns.to_numpy()
        rows = data.index.to_numpy()
        samples = len(columns)
        genes = len(rows)
        bins = round(1 + 3.22 * log(genes))                  # sturge's rule
        bins2 = round(1 + 3.22 * log(genes))                 # sturge's rule
        data_np = data.to_numpy()

    with Timer('Saving metadata...'):
        np.savetxt('genes.txt', rows, fmt = '%s')

    with Timer('Computing agg...'):
        agg = mim(genes, bins, data_np, jobs).flatten()

        with Timer('Computing I_0...'):
            #threshold
            I_0 = threshold_calculation(data_np, bins, jobs)
            agg = np.where(agg < I_0, 0, agg)

    for i in np.arange(samples):
        with Timer("Computing for sample " + str(i) + "..."):
            ss = mim(genes, bins, np.delete(data_np, i, axis = 1), jobs)

            # threshold
            I_0 = threshold_calculation(np.delete(data_np, i, axis = 1), bins2, jobs)
            id = np.where(ss < I_0)
            ss[id] = 0

            # Save as .npy
            np.save(path + columns[i].replace('.txt', '.npy'), ss)

def plot_networks(data, path):
    """
    Plot the output of the LIONESS algorithm.

    Input:
    data (pd.DataFrame) with the following columns:
    reg | tar | [sample 0] | ... | [sample n]
    path (str) to the folder where the data is going to be saved.

    Output:
    Directory with all the plots from each sample.
    """
    G = nx.Graph()
    options = {'node_color': 'pink', 'node_size': 20, 'width': 0.2, 'font_size': 10}

    try:
        os.mkdir(path + 'plots/')
    except OSError as error:
        print('the folder already exists.')

    for sample in data.columns[2:]:
        with Timer('Ploting sample: ' + sample + '...'):
            positive = data.loc[data[sample] > 0]
            edges = list(zip(positive['reg'].to_list(), positive['tar'].to_list()))

            G.clear()
            G.add_edges_from(edges)
            nx.draw(G, with_labels=False, **options)
            plt.savefig(path + 'plots/' + sample.replace('.txt', '.png'))

def compute_properties(path):
        """
        Plot the output of the LIONESS algorithm.

        Input:
        path (str) to the folder where the data is going to be saved.

        Output:
        .csv file with the properties degree, betweenness centrality and
        clustering for each node of each sample.
        """
        G = nx.Graph()
        total = []

        # recreate DataFrame
        samples = os.listdir(path)
        genes = np.loadtxt(path + 'genes.txt', dtype = type(''))

        reg = np.array([genes for i in np.arange(len(genes))]).flatten()
        tar = np.array([[x for i in np.arange(len(genes))] for x in genes]).flatten()
        data = pd.DataFrame({"reg": reg, "tar": tar})

        for s in samples[1:]:
            n = np.transpose(np.load(path + s).flatten())
            data[s.replace('.npy', '')] = pd.Series(n)

        for sample in data.columns[2:]:
            with Timer('Calculating for sample ' + sample + '...'):
                positive = data.loc[data[sample] > 0]
                edges = list(zip(positive['reg'].to_list(), positive['tar'].to_list()))
                G.clear()
                G.add_edges_from(edges)

                nw = pd.DataFrame(nx.degree(G), columns = ['genes', 'degree']).set_index('genes')
                nw['betweenness_centrality'] = pd.Series(nx.betweenness_centrality(G))
                nw['clustering'] = pd.Series(nx.clustering(G))
                nw.insert(0, 'sample', [sample.split('_')[0] for i in range(nw.shape[0])])
                nw.insert(1, 'PAM50_subtype', [sample.split('_')[1] for i in range(nw.shape[0])])
                nw.reset_index(level = 0, inplace = True)
                total.append(nw)

        pd.concat(total, axis = 0, ignore_index = True).to_csv(path + 'node_properties.csv')

def plot_degree(data, path, boxplot = True, scatter = True):
    """
    Plot the output of the LIONESS algorithm.

    Input:
    data (pd.DataFrame) with the following columns:
    genes | sample | PAM50_subtype | degree | betweenness_centrality | clustering
    path (str) to the folder where the data is going to be saved.
    boxplot (bool) if required boxplot
    scatter (bool) if required scatter

    Output:
    At least one of the following: a boxplot of the degree values of each
    gene and a scatter of degree's mean vs. degree's std. deviation for each
    gene.
    """

    genes = data['genes'].unique()
    x = data.groupby('genes')['degree']

    # Box
    if boxplot:
        fig, ax = plt.subplots(figsize = (20, 20))
        genes_dict = {}

        for gene in genes:
            gb = list(data[data['genes'] == gene]['degree'])
            genes_dict[gene] = gb

        plt.boxplot(genes_dict.values())
        plt.xticks(ticks = [i + 1 for i in range(50)], labels = genes, rotation = 70)
        ax.set_xlabel('PAM50 Genes', fontsize = 16)
        ax.set_ylabel('Degree', fontsize = 16)
        ax.set_title('Degree per PAM Gene', fontsize = 20)

        plt.savefig(path + 'boxplot.png', format = 'png')

    # Scatter
    if scatter:
        fig, ax = plt.subplots(figsize = (20, 20))

        ax.scatter(x.std(), x.mean())
        ax.set_xlabel('Std. Deviation', fontsize = 16)
        ax.set_ylabel('Mean', fontsize = 16)
        ax.set_title('Degree Std. Deviation vs Mean', fontsize = 20)

        plt.text(x=8.5, y=20, s="Q4", fontsize=16, color='b')
        plt.text(x=2.2, y=20, s="Q3", fontsize=16, color='b')
        plt.text(x=2.2, y=43, s="Q2", fontsize=16, color='b')
        plt.text(x=8.5, y=43, s="Q1", fontsize=16, color='b')

        ax.axhline(y=x.mean().mean(), color='k', linestyle='--', linewidth=1)
        ax.axvline(x=x.std().mean(), color='k',linestyle='--', linewidth=1)

        for gene in genes:
            ax.text(x.std()[gene], x.mean()[gene], s = gene, fontsize = 10)

        plt.savefig(path + 'scatter.png', format = 'png')
