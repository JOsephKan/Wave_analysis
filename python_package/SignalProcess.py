import numpy as np

class WaveProcess:
    """
    A class used to process wave data and compute its power spectrum and reconstruction.

    Attributes
    ----------
    A : np.ndarray
        Real part of the FFT of Ck.
    B : np.ndarray
        Imaginary part of the FFT of Ck.
    a : np.ndarray
        Real part of the FFT of Sk.
    b : np.ndarray
        Imaginary part of the FFT of Sk.

    Methods
    -------
    PowerSpec(arr: np.matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.matrix]
        Computes the power spectrum of the input wave data.
        
        Parameters:
        arr : np.matrix
            A 2D matrix where the first dimension (axis=0) represents time and the second dimension (axis=1) represents space.
        
        Returns:
        tuple
            A tuple containing A, B, a, b, and ps:
            - A : np.ndarray - Real part of the FFT of Ck.
            - B : np.ndarray - Imaginary part of the FFT of Ck.
            - a : np.ndarray - Real part of the FFT of Sk.
            - b : np.ndarray - Imaginary part of the FFT of Sk.
            - ps : np.matrix - Power spectrum of the input wave data.

    Recon() -> np.matrix
        Reconstructs the wave data from the computed A, B, a, and b attributes.
        
        Returns:
        np.matrix
            The reconstructed wave data.
    """

    def __init__(self):
        """
        Initializes the WaveProcess class with None values for A, B, a, and b.
        """
        self.A = None
        self.B = None
        self.a = None
        self.b = None

    def PowerCoeff(self, arr):
        arr_fft = (np.fft.fft(arr, axis=0) / np.pi).T

        Ck = arr_fft.real/arr_fft.shape[0] * 2
        Sk = arr_fft.imag/arr_fft.shape[0] * 2

        Ck_fft = (np.fft.fft(Ck, axis=0) / np.pi).T
        Sk_fft = (np.fft.fft(Sk, axis=0) / np.pi).T

        self.A = Ck_fft.real/Ck_fft.shape[0] * 2
        self.B = Ck_fft.imag/Ck_fft.shape[0] * 2
        self.a = Sk_fft.real/Sk_fft.shape[0] * 2
        self.b = Sk_fft.imag/Sk_fft.shape[0] * 2

        return self.A, self.B, self.a, self.b
    
    def PowerSpec(self, arr: np.matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.matrix]:
        """
        Computes the power spectrum of the input wave data.

        Parameters
        ----------
        arr : np.matrix
            A 2D matrix where the first dimension (axis=0) represents time and the second dimension (axis=1) represents space.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.matrix]
            A tuple containing A, B, a, b, and ps:
            - A : np.ndarray - Real part of the FFT of Ck.
            - B : np.ndarray - Imaginary part of the FFT of Ck.
            - a : np.ndarray - Real part of the FFT of Sk.
            - b : np.ndarray - Imaginary part of the FFT of Sk.
            - ps : np.matrix - Power spectrum of the input wave data.
        """
        arr_fft = (np.fft.fft(arr, axis=0) / np.pi).T

        Ck = arr_fft.real/arr_fft.shape[0] * 2
        Sk = arr_fft.imag/arr_fft.shape[0] * 2

        Ck_fft = (np.fft.fft(Ck, axis=0) / np.pi).T
        Sk_fft = (np.fft.fft(Sk, axis=0) / np.pi).T

        self.A = Ck_fft.real/Ck_fft.shape[0] * 2
        self.B = Ck_fft.imag/Ck_fft.shape[0] * 2
        self.a = Sk_fft.real/Sk_fft.shape[0] * 2
        self.b = Sk_fft.imag/Sk_fft.shape[0] * 2

        east = ((np.power(self.A + self.b, 2) + np.power(self.B - self.a, 2)) / 8)[:arr.shape[0]//2, :arr.shape[1]//2]
        west = ((np.power(self.A - self.b, 2) + np.power(-self.B - self.a, 2)) / 8)[:arr.shape[0]//2, :arr.shape[1]//2]

        ps = np.concatenate((west[:, ::-1], east), axis=1) / (arr.shape[0] * arr.shape[1]) ** 2
        
        return ps

    def Recon(self) -> np.matrix:
        """
        Reconstructs the wave data from the computed A, B, a, and b attributes.

        Returns
        -------
        np.matrix
            The reconstructed wave data.
        """
        Ck_inv = np.fft.ifft(self.A + 1j * self.B, axis=1)
        Sk_inv = np.fft.ifft(self.a + 1j * self.b, axis=1)

        wave_inv = np.fft.ifft(Ck_inv + 1j * Sk_inv, axis=0)*np.pi

        return wave_inv

    
