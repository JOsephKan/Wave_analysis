# import package
import numpy as np


def compute_normal_equation(X, y):
    """
    Computes the normal equation to solve for the optimal parameters theta in a linear regression problem.

    Parameters:
    X (numpy.ndarray): The input feature matrix of shape (m, n), where m is the number of training examples and n is the number of features.
    y (numpy.ndarray): The target values of shape (m, 1), where m is the number of training examples.

    Returns:
    numpy.ndarray: The optimal parameters theta of shape (n, 1), where n is the number of features.

    """
    X_transpose_X_inv = np.linalg.inv(X.T @ X)
    theta = X_transpose_X_inv @ X.T @ y
    return theta

def EOF(arr):
    """
    Performs Empirical Orthogonal Function (EOF) analysis on the given array.

    Parameters:
    arr (numpy.ndarray): The input array for EOF analysis.

    Returns:
    tuple: A tuple containing the following:
        - ExpVar (numpy.ndarray): The explained variance for each EOF mode.
        - eigvec (numpy.ndarray): The eigenvectors corresponding to each EOF mode.
        - pcs (numpy.ndarray): The principal components of the input array.

    """
    CovMat = np.cov(arr)
    eigval, eigvec = np.linalg.eig(CovMat)
    ExpVar = eigval / np.sum(eigval)
    pcs = compute_normal_equation(eigvec, arr)
    return ExpVar, eigvec, pcs

