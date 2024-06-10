import numpy as np

def PowerCoeff(
        arr: np.matrix
) -> tuple[np.matrix, np.matrix, np.matrix, np.matrix]:
    """Calculate the power coefficients of a given matrix.

    This function calculates the power coefficients of a given matrix using the Fast Fourier Transform (FFT).

    Args:
        arr (np.matrix): The input matrix.

    Returns:
        tuple[np.matrix, np.matrix, np.matrix, np.matrix]: A tuple containing the power coefficients A, B, a, and b.
    """    
    arr_fft = (np.fft.fft(arr, axis=0) / np.pi).T

    Ck = arr_fft.real * 2
    Sk = arr_fft.imag * 2

    Ck_fft = (np.fft.fft(Ck, axis=0) / np.pi).T
    Sk_fft = (np.fft.fft(Sk, axis=0) / np.pi).T

    A = Ck_fft.real * 2
    B = Ck_fft.imag * 2
    a = Sk_fft.real * 2
    b = Sk_fft.imag * 2

    return A, B, a, b

def PowerSpec(arr: np.matrix) -> np.matrix:
    """
    Calculates the power spectrum of a given matrix.

    Parameters:
    arr (np.matrix): The input matrix.

    Returns:
    np.matrix: The power spectrum matrix.
    """
    
    A, B, a, b = PowerCoeff(arr)

    east = ((np.power(A + b, 2) + np.power(B - a, 2)) / 8)[:arr.shape[0]//2, :arr.shape[1]//2]
    west = ((np.power(A - b, 2) + np.power(-B - a, 2)) / 8)[:arr.shape[0]//2, :arr.shape[1]//2]

    ps = np.concatenate((west[:, ::-1], east), axis=1) / (arr.shape[0] * arr.shape[1]) ** 2
        
    return ps

def Recon(
        A: np.matrix, B: np.matrix, a: np.matrix, b: np.matrix
) -> np.matrix:
    """Reconstructs the wave data from the computed power coefficients.

    Args:
        A (np.matrix): Real part of the FFT of Ck.
        B (np.matrix): Imaginary part of the FFT of Ck.
        a (np.matrix): Real part of the FFT of Sk.
        b (np.matrix): Imaginary part of the FFT of Sk.

    Returns:
        np.matrix: The reconstructed wave data.
    """
    
    Ck_inv = np.fft.ifft(A + 1j * B, axis=1)
    Sk_inv = np.fft.ifft(a + 1j * b, axis=1)

    wave_inv = np.fft.ifft(Ck_inv + 1j * Sk_inv, axis=0)*np.pi

    return wave_inv
