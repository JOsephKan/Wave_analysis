# import pamonth_data
import numpy as np

def sym(lat: np.ndarray, arr: np.ndarray):
    """
    Calculate the symmetrical array based on the input latitude and array.

    Args:
        lat (np.ndarray): The latitude array.
        arr (np.ndarray): The input array.

    Returns:
        np.ndarray: The symmetrical array.

    Input shape: (time, lat, lon)
    """
    latr = np.cos(np.deg2rad(lat))

    sym_arr =np.sum (arr * latr[None, :, None], axis=1) / np.sum(latr)

    return sym_arr


def asy(lat: np.ndarray, data: np.ndarray):
    """
    Calculate the asymmetric component of the given data based on latitude.

    Parameters:
        lat (np.ndarray): Array of latitudes.
        data (np.ndarray): Array of data.

    Returns:
        np.ndarray: Array containing the asymmetric component of the data.

    """
    latr = np.cos(np.deg2rad(lat))

    idx = np.where(lat < 0)

    data_asy = data * latr[None, :, None]

    data_asy[idx] = -data_asy[idx]

    data_asy = np.sum(data_asy, axis=1) / np.sum(latr)

    return data_asy

def filt(arr: np.ndarray, num_of_pass:np.int64):
    arr_bg = arr.copy()

    for _ in range(num_of_pass):

        left = np.concatenate((np.array([arr_bg[:, 1]]).T, arr_bg[:, :-1]), axis=1)
        right = np.concatenate((arr_bg[:, 1:], np.array([arr_bg[:, -2]]).T), axis=1)

        arr_bg = (2*arr_bg+left+right)/4

    return arr_bg
