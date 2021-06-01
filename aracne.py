from sklearn.metrics import mutual_info_score
from timer import Timer
from math import log
import pandas as pd
import numpy as np

class Aracne(object):

    # Constructor
    def __init__(self, path):
        # Reading process | Returns matrix
        self.__reading(path)

        # Construction of the mutual information matrix
        if hasattr(self, "weight_matrix"):
            self.__mim()

            # Execution of the ARACNE algorithm

            # Normalization of the coexpression network

    # Data reading
    def __reading(self, path):
        """
        Read the data of the gene-case matrix and save the information in
        anp.array.

        Input:
        Path (string) to the gene-case matrix

        Output:
        Weigh_matrix variable (np.array) that contains the measure of each
        gene in each case.
        Genes presented variable(np.array) in the weight_matrix in their
        respected order.
        """

        try:
            with open(path, "r") as data:
                matrix = list()
                genes = list()

                with Timer("Reading matrix..."):
                    for line in list(data):
                        dummy = line.replace("\n", "")
                        dummy = line.replace("\r", "")
                        matrix.append(dummy.split("\t")[1:])
                        genes.append(dummy.split("\t")[0])

                    self.weight_matrix = np.array(matrix, dtype = "float64")
                    self.genes = np.array(genes)
        except FileNotFoundError:
            print("Unable to find gene-case file.")

    # Mutual Information Matrix
    def __mim(self):
        """
        Calculate the mutual information matrix from each pair of genes.

        Input:
        None.

        Output:
        Mim variable (np.array) which contains the mutual information matrix.
        """

        def sum_mi(x, y, bins):
            """
            Compute the mutual information score of the discrete probability
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

        size = self.genes.shape[0]
        matrix = np.full((size, size), np.inf, dtype = "float64")
        bins = round(1 + 3.22 * log(size))                  # sturge's rule

        with Timer("Calculating Mutual Information Matrix..."):
            for i in range(0, size):
                print("Computing for gene:", i)
                x = self.weight_matrix[i]
                
                for j in range(i + 1, size):
                    y = self.weight_matrix[j]
                    matrix[i][j] = sum_mi(x, y, bins)

            self.mim = np.array(matrix, dtype = "float64")

    # ARACNE algoritmo
    def __aracne_loop(self):
        pass

    # Normalizacion
    def __normalization(self):
        pass
