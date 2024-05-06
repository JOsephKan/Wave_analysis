# import package
import numpy as np
from numba import njit

def _FTOperator(arr, dim, func):
    """
    Apply the Fourier Transform operator to the input array along the specified dimension.

    Parameters:
    - arr: numpy.ndarray
        The input array to apply the Fourier Transform operator to.
    - dim: int
        The dimension along which to apply the Fourier Transform operator.
    - func: str
        The type of Fourier Transform to apply. Valid options are "DFT" (Discrete Fourier Transform) and "IDFT" (Inverse Discrete Fourier Transform).

    Returns:
    - Ck: numpy.ndarray
        The real part of the Fourier Transform result.
    - Sk: numpy.ndarray
        The imaginary part of the Fourier Transform result.
    """
    
    N  = int(arr.shape[dim])
    n  = (np.linspace(0, N-1, N)).reshape((1, N))
    k  = (2.*np.pi*np.linspace(0, N-1, N)/N).reshape((N, 1))
    op = np.exp(1j*(k*n))
    if (func=="DFT"):
        op_DFT = np.matmul(op, arr.T)/N
        Ck = op_DFT.real
        Sk = op_DFT.imag
    elif (func=="IDFT"):
        op_IDFT = op[:, :round(N/2.)]
        Ck = op_IDFT.real
        Sk = op_IDFT.imag
    return Ck, Sk

def _Nyquist(arr, arr_o, axis):
    """
    Apply the Nyquist operator to the input array.

    Parameters:
    - arr: numpy.ndarray
        The input array to apply the Nyquist operator to.
    - arr_o: numpy.ndarray
        The original input array.
    - axis: int
        The axis along which to apply the Nyquist operator.

    Returns:
    - arr_new: numpy.ndarray
        The result of applying the Nyquist operator to the input array.
    """
    arr_new = arr[0 : round(arr_o.shape[axis] / 2), :]
    arr_new = np.concatenate(([arr_new[0, :]], arr_new[1:, :]*2), axis=0)
    return arr_new

def DFT(arr):
    """
    Apply the Discrete Fourier Transform operator to the input array.

    Parameters:
    - arr: numpy.ndarray
        The input array to apply the Discrete Fourier Transform operator to.

    Returns:
    - Ck: numpy.ndarray
        The real part of the Fourier Transform result.
    - Sk: numpy.ndarray
        The imaginary part of the Fourier Transform result.
    """
    Ck, Sk = _FTOperator(arr, 1, "DFT")
    return Ck, Sk

def IDFT(C_k, S_k, arr):
    """
    Apply the Inverse Discrete Fourier Transform operator to the input array.

    Parameters:
    - C_k: numpy.ndarray
        The real part of the Fourier Transform result.
    - S_k: numpy.ndarray
        The imaginary part of the Fourier Transform result.
    - arr: numpy.ndarray
        The input array to apply the Inverse Discrete Fourier Transform operator to.

    Returns:
    - Ck_i: numpy.ndarray
        The real part of the Inverse Fourier Transform result.
    - Sk_i: numpy.ndarray
        The imaginary part of the Inverse Fourier Transform result.
    """
    Ck, Sk = _FTOperator(arr, 0, "IDFT")

    op     = Ck  + 1j * Sk
    filted = C_k + 1j * S_k

    Ck_i = np.real(np.matmul(op, filted)).T
    Sk_i = np.imag(np.matmul(op, filted)).T
    return Ck_i, Sk_i

def PowerCoeff(arr):
    """
    Calculate the power coefficients of the input array.

    Parameters:
    - arr: numpy.ndarray
        The input array to calculate the power coefficients of.

    Returns:
    - A: numpy.ndarray
        The real part of the power coefficients.
    - B: numpy.ndarray
        The imaginary part of the power coefficients.
    - a: numpy.ndarray
        The real part of the power coefficients.
    - b: numpy.ndarray
        The imaginary part of the power coefficients.
    """
    Ck, Sk = DFT(arr)
    Ck     = _Nyquist(Ck, arr, 1)
    Sk     = _Nyquist(Sk, arr, 1)

    A, B = DFT(Ck)
    a, b = DFT(Sk)
    A    = _Nyquist(A, arr, 0)
    B    = _Nyquist(B, arr, 0)
    a    = _Nyquist(a, arr, 0)
    b    = _Nyquist(b, arr, 0)
    return A, B, a, b

def PowerSpec(arr):
    """
    Calculate the power spectrum of the input array.

    Parameters:
    - arr: numpy.ndarray
        The input array to calculate the power spectrum of.

    Returns:
    - ps: numpy.ndarray
        The power spectrum of the input array.
    """
    A, B, a, b = PowerCoeff(arr)
    power_neg = 1 / 8 * (np.power(A, 2) + np.power(B, 2) + np.power(a, 2) + np.power(b, 2)) + 1 / 4 * (np.multiply(a, B) - np.multiply(b, A))
    power_pos = 1 / 8 * (np.power(A, 2) + np.power(B, 2) + np.power(a, 2) + np.power(b, 2)) - 1 / 4 * (np.multiply(a, B) - np.multiply(b, A))
    ps = np.append(power_neg[:, ::-1], power_pos, axis=1)
    return ps
"""
class reconstruction(power_spectrum):
    def __init__(self):
        self.re_wave = None
        self.wind_R  = None
        self.wind_I  = None

    def recon_wave(self, arr):
        A, B, a, b = self.power_coe(arr)
        Ereal, Eimag = self.IDFT(A, B, arr)
        Wreal, Wimag = self.IDFT(a, b, arr)

        Ewave = Ereal + 1j*Eimag
        Wwave = Wreal + 1j*Wimag

        RWreal, RWimag = self.IDFT(Ewave, Wwave, arr.T)
        self.re_wave = RWreal + 1j*RWimag

        return self.re_wave

    def e_w_trans(self, arr_c):
        A, B, a, b = arr_c
        Ereal = 1 / 4 * (A - b)
        Eimag = 1 / 4 * (-B - a)
        Wreal = 1 / 4 * (A + b)
        Wimag = 1 / 4 * (B - a)
        return Ereal, Eimag, Wreal, Wimag

    def east_recon(self, arr, arr_item):
        Wreal, Wimag = arr_item
        WCk = (1 + 0j) * Wreal + (0 + 1j) * Wimag
        WSk = (1 + 0j) * Wimag - (0 + 1j) * Wreal
        WCk_i, WSk_i = self.IDFT(WCk, WSk, arr)
        Winv = WCk_i - WSk_i
        self.wind_R = Winv.real
        self.wind_I = Winv.imag

        return self.wind_R, self.wind_I

"""

def genDispersionCurves(nWaveType=6, nPlanetaryWave=50, rlat=0, Ahe=[50, 25, 12]):
    """
    Function to derive the shallow water dispersion curves. Closely follows NCL version.

    input:
        nWaveType : integer, number of wave types to do
        nPlanetaryWave: integer
        rlat: latitude in radians (just one latitude, usually 0.0)
        Ahe: [50.,25.,12.] equivalent depths
            ==> defines parameter: nEquivDepth ; integer, number of equivalent depths to do == len(Ahe)

    returns: tuple of size 2
        Afreq: Frequency, shape is (nWaveType, nEquivDepth, nPlanetaryWave)
        Apzwn: Zonal savenumber, shape is (nWaveType, nEquivDepth, nPlanetaryWave)
        
    notes:
        The outputs contain both symmetric and antisymmetric waves. In the case of 
        nWaveType == 6:
        0,1,2 are (ASYMMETRIC) "MRG", "IG", "EIG" (mixed rossby gravity, inertial gravity, equatorial inertial gravity)
        3,4,5 are (SYMMETRIC) "Kelvin", "ER", "IG" (Kelvin, equatorial rossby, inertial gravity)
    """
    nEquivDepth = len(Ahe) # this was an input originally, but I don't know why.
    pi    = np.pi
    radius = 6.37122e06    # [m]   average radius of earth
    g     = 9.80665        # [m/s] gravity at 45 deg lat used by the WMO
    omega = 7.292e-05      # [1/s] earth's angular vel
    # U     = 0.0   # NOT USED, so Commented
    # Un    = 0.0   # since Un = U*T/L  # NOT USED, so Commented
    ll    = 2.*pi*radius*np.cos(np.abs(rlat))
    Beta  = 2.*omega*np.cos(np.abs(rlat))/radius
    fillval = 1e20
    
    # NOTE: original code used a variable called del,
    #       I just replace that with `dell` because `del` is a python keyword.

    # Initialize the output arrays
    Afreq = np.empty((nWaveType, nEquivDepth, nPlanetaryWave))
    Apzwn = np.empty((nWaveType, nEquivDepth, nPlanetaryWave))

    for ww in range(1, nWaveType+1):
        for ed, he in enumerate(Ahe):
            # this loops through the specified equivalent depths
            # ed provides index to fill in output array, while
            # he is the current equivalent depth
            # T = 1./np.sqrt(Beta)*(g*he)**(0.25) This is close to pre-factor of the dispersion relation, but is not used.
            c = np.sqrt(g * he)  # phase speed   
            L = np.sqrt(c/Beta)  # was: (g*he)**(0.25)/np.sqrt(Beta), this is Rossby radius of deformation        

            for wn in range(1, nPlanetaryWave+1):
                s  = -20.*(wn-1)*2./(nPlanetaryWave-1) + 20.
                k  = 2.0 * pi * s / ll
                kn = k * L 

                # Anti-symmetric curves  
                if (ww == 1):       # MRG wave
                    if (k < 0):
                        dell  = np.sqrt(1.0 + (4.0 * Beta)/(k**2 * c))
                        deif = k * c * (0.5 - 0.5 * dell)
                    
                    if (k == 0):
                        deif = np.sqrt(c * Beta)
                    
                    if (k > 0):
                        deif = fillval
                    
                
                if (ww == 2):       # n=0 IG wave
                    if (k < 0):
                        deif = fillval
                    
                    if (k == 0):
                        deif = np.sqrt( c * Beta)
                    
                    if (k > 0):
                        dell  = np.sqrt(1.+(4.0*Beta)/(k**2 * c))
                        deif = k * c * (0.5 + 0.5 * dell)
                    
                
                if (ww == 3):       # n=2 IG wave
                    n=2.
                    dell  = (Beta*c)
                    deif = np.sqrt((2.*n+1.)*dell + (g*he) * k**2)
                    # do some corrections to the above calculated frequency.......
                    for i in range(1,5+1):
                        deif = np.sqrt((2.*n+1.)*dell + (g*he) * k**2 + g*he*Beta*k/deif)
                    
    
                # symmetric curves
                if (ww == 4):       # n=1 ER wave
                    n=1.
                    if (k < 0.0):
                        dell  = (Beta/c)*(2.*n+1.)
                        deif = -Beta*k/(k**2 + dell)
                    else:
                        deif = fillval
                    
                if (ww == 5):       # Kelvin wave
                    deif = k*c

                if (ww == 6):       # n=1 IG wave
                    n=1.
                    dell  = (Beta*c)
                    deif = np.sqrt((2. * n+1.) * dell + (g*he)*k**2)
                    # do some corrections to the above calculated frequency
                    for i in range(1,5+1):
                        deif = np.sqrt((2.*n+1.)*dell + (g*he)*k**2 + g*he*Beta*k/deif)
                
                eif  = deif  # + k*U since  U=0.0
                P    = 2.*pi/(eif*24.*60.*60.)  #  => PERIOD
                # dps  = deif/k  # Does not seem to be used.
                # R    = L #<-- this seemed unnecessary, I just changed R to L in Rdeg
                # Rdeg = (180.*L)/(pi*6.37e6) # And it doesn't get used.
            
                Apzwn[ww-1,ed-1,wn-1] = s
                if (deif != fillval):
                    # P = 2.*pi/(eif*24.*60.*60.) # not sure why we would re-calculate now
                    Afreq[ww-1,ed-1,wn-1] = 1./P
                else:
                    Afreq[ww-1,ed-1,wn-1] = fillval
    return  Afreq, Apzwn

def PC_com(data, EOF):
    xTx = np.matmul(EOF.T, EOF)
    xTx_i = np.linalg.inv(xTx)
    theta = np.matmul(np.matmul(xTx_i, EOF.T), data)

    return theta

