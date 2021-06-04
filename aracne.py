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
        Reads the data of the gene-case matrix and save the information in
        a np.array.

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
                percent = 1

                with Timer("Reading matrix..."):
                    for line in list(data):
                        dummy = line.replace("\n", "").replace("\r", "").split("\t")

                        # Take only those that are significant for the cases
                        if np.count_nonzero(np.array(dummy[1:], dtype = "float64")) >= len(dummy[1:]) * percent:
                            matrix.append(dummy[1:])
                            genes.append(dummy[0])

                    self.weight_matrix = np.array(matrix, dtype = "float64")
                    self.genes = np.array(genes)
        except FileNotFoundError:
            print("Unable to find gene-case file.")

    # Mutual Information Matrix
    def __mim(self):
        """
        Calculates the mutual information matrix from each pair of genes.

        Input:
        None.

        Output:
        Mim variable (np.array) which contains the mutual information matrix.
        """

        def matrix_calc(size, bins, data = None):
            """
            Calculates the mutual information matrix.

            Input:
            The number of genes, the number of bins and the genes samples.

            Output:
            Matrix (np.array) with the mim.
            """

            matrix = np.zeros((size, size), dtype = "float64")

            if type(data) != type(None):
                for i in range(0, size):
                    print("Computing for gene:", i)
                    x = data[i]

                    for j in range(i + 1, size):
                        y = data[j]
                        matrix[i][j] = sum_mi(x, y, bins)
            else:
                for i in range(0, size):
                    print("Computing for gene:", i)
                    x = self.weight_matrix[i]

                    for j in range(i + 1, size):
                        y = self.weight_matrix[j]
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
            Compute the threshold to make a clearer mutual information matrix.

            Input:
            First MIM computation and the amount of bins to calculate the MIM
            of the permutation.

            Output:
            Threshold value (int).
            """

            n_perm = 1
            matrixt = matrix.T
            n_cases, n_genes = matrixt.shape
            permutations = np.zeros((n_perm, n_genes, n_cases))

            # Execution of the permutation
            for perm in range(n_perm):
                print(" Computing permutation:", perm + 1)
                perm_matrix = list()

                for i in range(n_cases):
                    id = np.random.permutation(n_genes)
                    perm_matrix.append(matrixt[i][id])

                perm_matrix = np.array(perm_matrix, dtype = "float64").T

                # Execution of the MIM computation
                dummy = matrix_calc(n_genes, bins, perm_matrix)

                # Save permutation
                permutations[perm] = dummy

            return np.amax(np.mean(permutations, axis = 0))

        def remove_loops(size, matrix):
            zero = list()

            for i in range(size):
                row = np.where(matrix[i] != 0)[0]
                for j in row:
                    column = np.where(matrix[j] != 0)[0]
                    for x in column:
                        if matrix[i][x] != 0:
                            data = {0: (i, j), 1: (i, x), 2: (j, x)}
                            values = [matrix[i][j], matrix[i][x], matrix[j][x]]
                            if data[np.argmin(values)] not in zero:
                                zero.append(data[np.argmin(values)])
            return zero

        # conda install pytorch torchvision cpuonly -c pytorch
        size = self.genes.shape[0]
        bins = round(1 + 3.22 * log(size))                  # sturge's rule

        with Timer("Calculating Mutual Information Matrix..."):
            matrix = matrix_calc(size, bins)

        with Timer("Calculating threshold..."):
            I_0 = threshold_calculation(matrix, bins)
            id = np.where(matrix < I_0)
            matrix[id] = 0

        with Timer("Removing loops..."):
            ids = remove_loops(size, matrix)
            for i in ids:
                matrix[i] = 0

        self.mim = np.array(matrix, dtype = "float64")

    # ARACNE algoritmo
    def __aracne_loop(self):
        pass

    # Normalizacion
    def __normalization(self):
        pass
