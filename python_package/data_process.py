# import pamonth_data
import numpy as np

def sym(
    lat : np.ndarray,
    data: np.ndarray,
) -> np.ndarray:
    """
    This function calculates the symmetric component of the data
    Parameters
    ----------
    lat : np.ndarray
        The latitude values
    data : np.ndarray
        The data values
    Returns
    -------
    np.ndarray
        The symmetric component of the data
    """
    sym_lat = np.cos(np.deg2rad(lat))
    lat_sum = np.sum(sym_lat)
    
    weighted = data * sym_lat[None, :, None]

    avg_sym = np.sum(weighted, axis=1) / lat_sum

    return avg_sym

def asy(
    lat : np.ndarray,
    data: np.ndarray,
)->np.ndarray:
    """
    This function calculates the asymmetric component of the data
    Please make the latitude values in descending order before using this function
    Parameters
    ----------
    lat : np.ndarray
        The latitude values
    data : np.ndarray
        The data values
    Returns
    -------
    np.ndarray
        The asymmetric component of the data
    """
    if lat[0] > 0.0:
        raise ValueError("Please make the latitude values in descending order")

    lat_sum = np.sum(np.cos(np.deg2rad(lat)))

    halflat = int(lat.shape[0] / 2)

    asy_lat = np.concatenate((-np.cos(np.deg2rad(lat[:halflat])), np.cos(np.deg2rad(lat[halflat:]))))
    
    weighted = data * asy_lat[None, :, None]

    avg_asy = np.sum(weighted, axis=1) / lat_sum

    return avg_asy

def background(
    data        : np.ndarray,
    smoothtimes : int,
) -> np.ndarray:
    """
    Applies background smoothing to the input data.

    Parameters:
        data (np.ndarray): The input data to be smoothed.
        smoothtimes (int): The number of times the smoothing operation should be applied.

    Returns:
        np.ndarray: The smoothed data.

    """
    smoothed = np.copy(data)
    for _ in range(smoothtimes):
        padded = np.pad(smoothed, (1, 1), mode="reflect")
        smoothed = (
            padded[:, :-2] + padded[:, 1:-1]*2 + padded[:, 2:]
        ) / 4.0
    return smoothed
