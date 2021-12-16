from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import networkx as nx
from math import log
import numpy as np
import pandas as pd
import os

from helpers.timer import Timer

def mim(size, bins, data):
    """
    Calculates the mutual information matrix.

    Input:
    The number of genes, the number of bins and the matrix (genes, samples).

    Output:
    Matrix (np.array) with the mim.
    """

    matrix = np.zeros((size, size), dtype = "float64")

    for i in np.arange(size):
        x = data[i].copy()

        for j in np.arange(i + 1, size):
            y = data[j].copy()
            matrix[i][j] = sum_mi(x, y, bins)

    return matrix

def sum_mi(x, y, bins):
    """
    Computes the mutual information score of the discrete probability
    variable of each pair of genes.

    Input:
    Data array of each gene and the number of bins for the
    discretization process.

    Output:
    Mutual information score variable (int).
    """

    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)

    return mi

def threshold_calculation(matrix, bins):
    """
    Computes the threshold to make a clearer mutual information matrix.

    Input:
    First MIM computation and the amount of bins to calculate the MIM
    of the permutation.

    Output:
    Threshold value (int).
    """

    n_perm = 2
    n_genes, n_cases = matrix.shape
    permutations = np.zeros((n_perm, n_genes, n_cases))

    # Execution of the permutation
    for perm in np.arange(n_perm):
        # Shuffle the matrix
        perm_matrix = [matrix[i][np.random.permutation(n_genes)] for i in np.arange(n_cases)]
        perm_matrix = np.vstack((perm_matrix))

        # Execution of the MIM computation
        dummy = mim(n_genes, bins, perm_matrix)

        # Save permutation
        permutations[perm] = dummy

    return np.amax(np.mean(permutations, axis = 0))

def lioness_algorithm(data):
    """
    LIONESS algorithm.

    Input:
    data (pd.DataFrame): Numeric matrix with samples in columns.

    Output:
    Sample-specific matrix.
    """
    with Timer('Reading data...'):
        columns = data.columns.to_numpy()
        rows = data.index.to_numpy()
        samples = len(columns)
        genes = len(rows)
        bins = round(1 + 3.22 * log(genes))                  # sturge's rule
        data_np = data.to_numpy()

    with Timer("Computing agg..."):
        agg = mim(genes, bins, data_np)

        #threshold
        I_0 = threshold_calculation(agg, bins)
        id = np.where(agg < I_0)
        agg[id] = 0
        agg = agg.flatten()

    reg = np.array([rows for i in np.arange(genes)]).flatten()
    tar = np.array([[x for i in np.arange(genes)] for x in rows]).flatten()
    output = pd.DataFrame({"reg": reg, "tar": tar})

    for c in columns:
        output[c] = np.transpose(np.zeros(genes*genes))

    for i in np.arange(samples):
        with Timer("Computing for sample " + str(i) + "..."):
            ss = mim(genes, bins, np.delete(data_np, i, axis = 1))

            # threshold
            I_0 = threshold_calculation(ss, bins)
            id = np.where(ss < I_0)
            ss[id] = 0
            ss = ss.flatten()

            output.update(pd.Series(samples * (agg - ss) + ss, name = columns[i]))

    return output

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

def compute_properties(data, path):
        """
        Plot the output of the LIONESS algorithm.

        Input:
        data (pd.DataFrame) with the following columns:
        reg | tar | [sample 0] | ... | [sample n]
        path (str) to the folder where the data is going to be saved.

        Output:
        .csv file with the properties degree, betweenness centrality and
        clustering for each node of each sample.
        """
        G = nx.Graph()
        total = []

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
