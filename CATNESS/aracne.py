# Math tools
from sklearn.metrics import mutual_info_score
from math import log
import numpy as np

# Local libs
from helpers.timer import Timer

# C++ binders
from ctypes import cdll
import glob

class Aracne(object):

    # Constructor
    def __init__(self, path):
        # Reading process | Returns matrix
        self.__reading(path)

        # Construction of the mutual information matrix
        if hasattr(self, "weight_matrix"):
            self.__aracne()

        # Normalization of the coexpression network

    # Data reading
    def __reading(self, path):
        """
        Reads the data of the gene-case matrix and save the information in
        a np.array.

        Input:
        Path (string) to the gene-case matrix

        Generates:
        Weight_matrix variable (np.array) that contains the measure of each
        gene in each case.
        Genes presented variable(np.array) in the weight_matrix in their
        respected order.
        """

        try:
            with open(path, "r") as data:
                matrix = list()
                genes = list()
                percent = 1     # least percent of non-zero values per sample

                with Timer("Reading matrix..."):
                    for line in data:
                        dummy = line.replace("\n", "").replace("\r", "").split("\t")
                        arr = np.array(dummy[1:], dtype = "float64")
                        gene = dummy[0]

                        # Take only those that are significant for the cases
                        if np.count_nonzero(arr) >= len(arr) * percent:
                            matrix.append(arr)
                            genes.append(gene)

                        if len(genes) == 100:
                            break

                    self.weight_matrix = np.vstack((matrix)).T
                    self.genes = np.array(genes)
        except FileNotFoundError:
            print("Unable to find the samples file.")

    # Mutual Information Matrix
    def __aracne(self, data = None, inequal_data = False):
        """
        Calculates the mutual information matrix from each pair of genes.

        Input:
        None.

        Generates:
        Mim variable (np.array) which contains the mutual information matrix.
        """

        def matrix_calc(size, bins, data):
            """
            Calculates the mutual information matrix.

            Input:
            The number of genes, the number of bins and the genes samples.

            Output:
            Matrix (np.array) with the mim.
            """

            matrix = np.zeros((size, size), dtype = "float64")

            for i in np.arange(size):
                print("Computing for gene:", i)
                x = data.T[i].copy()

                for j in np.arange(i + 1, size):
                    y = data.T[j].copy()
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
            n_cases, n_genes = matrix.shape
            permutations = np.zeros((n_perm, n_genes, n_cases))

            # Execution of the permutation
            for perm in np.arange(n_perm):
                print(" Computing permutation:", perm + 1)

                # Shuffle the matrix
                perm_matrix = [matrix[i][np.random.permutation(n_genes)] for i in np.arange(n_cases)]
                perm_matrix = np.vstack((perm_matrix))

                # Execution of the MIM computation
                dummy = matrix_calc(n_genes, bins, perm_matrix)

                # Save permutation
                permutations[perm] = dummy

            return np.amax(np.mean(permutations, axis = 0))

        def remove_loops(size, matrix):
            """
            Finds the possible loops inside the mutual information matrix.

            Input:
            The size of the matrix and the temporal mutual information
            matrix.

            Output:
            A list of tuples of the edges that creates loops and are
            weak.
            """

            zero = list()

            for i in np.arange(size):
                row = np.where(matrix[i] != 0)[0]
                for j in row:
                    column = np.where(matrix[j] != 0)[0]
                    for x in column:
                        if matrix[i][x] != 0:
                            data = {0: (i, j), 1: (i, x), 2: (j, x)}
                            values = [matrix[i][j], matrix[i][x], matrix[j][x]]
                            zero.append(data[np.argmin(values)])
            return zero

        size = 100 # delete later...

        if type(data) != type(None):
            #size = data.shape[1]
            bins = round(1 + 3.22 * log(size))                  # sturge's rule
        else:
            data = self.weight_matrix.copy()
            #size = data.shape[1]
            bins = round(1 + 3.22 * log(size))                  # sturge's rule

        with Timer("Calculating Mutual Information Matrix..."):
            matrix = matrix_calc(size, bins, data)

        with Timer("Calculating threshold..."):
            I_0 = threshold_calculation(matrix, bins)
            id = np.where(matrix < I_0)
            matrix[id] = 0

        if inequal_data:
            with Timer("Removing loops..."):
                ids = remove_loops(size, matrix)
                for i in ids:
                    matrix[i] = 0

        self.mim = np.array(matrix, dtype = "float64")

    # Normalizacion
    def __normalization(self):
        pass

    # Infering single sample network
    def single_sample_network(self):
        n_cases, n_genes = self.weight_matrix.shape
        mims = np.zeros((n_cases, n_genes, n_genes))

        with Time("Infering each simple sample network..."):
            for sample in np.arange(n_cases):
                print("Computing sample " + str(sample) + "...")

                # gene-case matrix without sample q
                dummy = np.delete(self.weight_matrix, sample, 0)

                # Computes the mim of the new matrix
                mim_p = self.__aracne(dummy)

                # Computes LIONESS algorithm (from c++ lib)

    # Save data
    def save_mim(self, file = "data.csv", delimeter = ","):
        """
        Save the Mutual Information Matrix in the given file.

        Input:
        Name of the file (str) and the delimter (char).
        """

        with Timer("Saving data in the file: " + file + "..."):
            np.savetxt(file, self.mim, delimiter = delimeter)
