from pyitlib import discrete_random_variable as drv
import pandas as pd
import numpy as np

class Aracne(object):

    # Constructor
    def __init__(self, path):
        # Reading process | Returns matrix
        self.__reading(path)

        # Construction of the mutual information matrix

        # Execution of the ARACNE algorithm

        # Normalization of the coexpression network

    # Data reading
    def __reading(self, path):
        """
        Read the data of the gene-case matrix and save the information in a np.array.

        Input:
        Path to the gene-case matrix

        Output:
        Weigh_matrix (np.array) that contains the measure of each gene in each case.
        Genes presented (np.array) in the weight_matrix in their respected order.
        """

        try:
            with open(path, "r") as data:
                matrix = list()
                genes = list()

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
        pass

    # ARACNE algoritmo
    def __aracne_loop(self):
        pass

    # Normalizacion
    def __normalization(self):
        pass
