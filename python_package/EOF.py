# import packages
import numpy as np

def NormalEqu(
        arr: np.ndarray,
        EOF: np.ndarray
        ) -> np.ndarray:
    """
    Calculates the principal components using the Normal Equations method.

    Parameters:
    arr (np.ndarray): The input array of shape (m, n), where m is the number of samples and n is the number of features.

    Returns:
    np.ndarray: The principal components of shape (m, n).

    Raises:
    None

    Example:
    >>> arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> NormalEqu(arr)
    array([[-0.23197069, -0.78583024, -1.33968979],
        [ 0.78583024,  0.        , -0.78583024],
        [ 1.80363117,  0.78583024, -0.23197069]])
    """

    xTx = np.linalg.inv(np.dot(EOF.T, EOF))
    op = np.dot(xTx, EOF.T)
    PCs = np.dot(op, arr)

    return PCs

def EmpOrthFunc(
        data : np.ndarray,
) -> np.ndarray:
    """
    This function calculates the empirical orthogonal function of the data
    Parameters
    ----------
    data : np.ndarray
        The data values
    Returns
    -------
    np.ndarray
        The empirical orthogonal function of the data
    """
    # Calculate the covariance matrix
    cov = np.cov(data)
    # Calculate the eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(cov)
    
    ExpVar  = eigvals / np.sum(eigvals)
    PCs     = NormalEqu(data, eigvecs)
    NormPCs = (PCs-PCs.mean(axis=1))/PCs.std(axis=1)
    eof     = data*(NormPCs.T*np.linalg.inv(NormPCs*NormPCs.T))
    
    return eof