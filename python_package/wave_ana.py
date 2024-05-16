# import package
import numpy as np
from numba import njit

def _Nyquist(arr: np.ndarray):
    """
    Apply the Nyquist frequency adjustment to the input array.

    Parameters:
        arr (np.ndarray): The input array.

    Returns:
        np.ndarray: The adjusted array.

    """
    N = int(arr.shape[-1]/2)

    arr_new = arr[:, :N]

    arr_new[:, :] *= 2

    return arr_new

def PowerCoeff(arr: np.ndarray):
    """
    Calculate the power coefficients of a given array.

    Parameters:
    arr (np.ndarray): The input array.

    Returns:
    tuple: A tuple containing the power coefficients A, B, a, and b.
    """
    
    arr_fft = (np.fft.fft(arr, axis=0)/np.pi).T
    
    Ck = _Nyquist(arr_fft.real/arr_fft.shape[0]*2)
    Sk = _Nyquist(arr_fft.imag/arr_fft.shape[0]*2)

    CkFFT = (np.fft.fft(Ck, axis=0)/np.pi).T
    SkFFT = (np.fft.fft(Sk, axis=0)/np.pi).T

    A = _Nyquist(CkFFT.real/CkFFT.shape[0]*2)
    B = _Nyquist(CkFFT.imag/CkFFT.shape[0]*2)
    a = _Nyquist(SkFFT.real/SkFFT.shape[0]*2)
    b = _Nyquist(SkFFT.imag/SkFFT.shape[0]*2)

    return A, B, a, b

def PowerSpec(arr: np.ndarray) -> np.ndarray:
    """
    Calculates the power spectrum of the given array.

    Parameters:
    arr (np.ndarray): The input array.

    Returns:
    np.ndarray: The power spectrum of the input array.
    """
    A, B, a, b = PowerCoeff(arr)

    east = (np.power(A+b, 2) + np.power((B-a), 2))/8
    west = (np.power(A-b, 2) + np.power((-B-a), 2))/8
    ps = np.concatenate((west[:, ::-1], east), axis=1)
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

