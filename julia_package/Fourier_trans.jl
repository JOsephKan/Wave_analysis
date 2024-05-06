module Fourier_trans
# using external packages
using LinearAlgebra

export _2D_operator, _DFT, _IDFT

function _2D_operator(
    arr   :: Matrix{Float64}, # input array, 2D array
    dim   :: Int64,           # dimension to do Fourier transform, a integer
    style :: String           # Type of Fourier transform, only "DFT" and "IDFT" is valid, a string
    )
    
    # Declare section 
    #N      :: Int64           # size of the dimension of array which gonna be transform
    #n, k   :: Matrix{Float64} # components for rotating matrix
    #Ck, Sk :: Matrix{Float64} # 
    #RotMax :: Matrix{Float64} # Rotating matrix of Fourier transform

    # Execute section
    N = size(arr, dim)

    n = reshape(LinRange(0, N-1, N), 1, N)
    k = 2*π*reshape(LinRange(0, N-1, N), N, 1)/N
    
    if style == "DFT"
        RotMat = exp.(-1im.*(k*n))/N
        Ck = real.(RotMat)
        Sk = imag.(RotMat)
    end

    if style == "IDFT"
        RotMat = exp.(1im.*k*n)[:, 1:round(Int64, N/2)]
        Ck = real.(RotMat)
        Sk = real.(RotMat)
    end

    return RotMat
end

function _DFT(
    arr :: Matrix{Float64} # Array to do discrete Fourier transform
    )

    #Declare section
    #RotMat   :: Matrix{Float64}
    #TransMat :: Matrix{Float64}

    # Execute section
    RotMat   = _2D_operator(arr, 2, "DFT") # Rotating matrix to do DFT
    Ck = real.(RotMat)*arr'
    Sk = imag.(RotMat)*arr'
        
    return Ck, Sk
end

function _IDFT(
    C_k :: Matrix{Float64}, # Real part of the transformed array
    S_k :: Matrix{Float64}, # Imagine part of the transformed array
    arr :: Matrix{Float64}  # Original array
    )  

    # Declare section 
    #RotMat :: Matrix{Float64}    # Rotating Matrix of Inverse DFT
    #Cki, Ski :: Matrix{Float64}  # Real and imagine part of Inverse result

    RotMat = _2D_operator(arr, 1, "IDFT")

    Cki = (real.(RotMat)*C_k)'
    Ski = (imag.(RotMat)*S_k)'

    return Cki, Ski
end

end
"""
module Fourier_trans

    using LinearAlgebra
    using SpecialMatrices

    export 

    function normal_equ(
        EOF :: Matrix(FloalbmsFloalbmsFloalbmsFloalbmsFloalbms     ), 
        data):
        xTx_1 = inv(EOF'*EOF)
        op    = xTx_1*EOF
        pc    = op*data
        return pc
        
    end
    function EOF(data :: Matrix{float64})
        CovMat = data*data'/size(data)[1]
        EigVec = eigvecs(CovMat)
        EigVal = eigvals(CovMat)

        ExpVar = EigVal ./ sum(EigVal)
        PC = Evecs'*data

        return Evecs, PC, Explained_variance
    end

    function _Nyquist(arr, arr_origin, dim)
        arr_new = arr[1:round(Int32, size(arr_origin, dim)/2), :]
        arr_new *= 2
        return arr_new
        
    end

    function power_coe(arr)
        N, Ck, Sk = DFT(arr)
        
        Ck = _Nyquist(Ck, arr, 2)
        Sk = _Nyquist(Sk, arr, 2)

        A, B = DFT(Ck)
        a, b = DFT(Sk)

        A = _Nyquist(A, arr, 1)
        B = _Nyquist(B, arr, 1)
        a = _Nyquist(a, arr, 1)
        b = _Nyquist(b, arr, 1)

        return A, B, a, b
    end

    function power_spec(arr)
        
        A, B, a, b = power_coe(arr)
        power_pos = 1/8 * (
            A .^ 2 + B .^ 2 + a .^ 2 + b .^ 2
        ) + 1/4 * (a.*B - b.*A)
        power_neg = 1/8 * (
            A .^ 2 + B .^ 2 + a .^ 2 + b .^ 2
        ) - 1/4 * (a.*B - b.*A)

        return power_pos, power_neg
    end

    function recon_wave(arr)

        A, B, a, b = power_coe(arr)

        east_r, east_i = IDFT(A, B, arr)
        west_r, west_i = IDFT(a, b, arr)

        Ewind = east_r+east_i
        Wwind = west_r+west_i

        recon_r. recon_i = IDFT(Ewind, Wwind, arr')
        recon_wave = recon_r+recon_i

        return recon_wave
    end

    function e_w_trans(arr)
        A, B, a, b = arr
        east_r = 1/4*(A-b)
        east_i = 1/4*(-B-a)

        west_r = 1/4*(A+b)
        ewst_i = 1/4(B-a)

        return east_r, east_i, west_r, west_i
    end

    function east_recon(arr, arr_item)
        wind_r, wind_i = arr_item

        wind_C = wind_r+wind_i*1j
        wind_S = wind_i-wind_r*1j
        
        wind_Ci, wind_Si = IDFT(wind_C, wind_S, arr)
        wind_inv = wind_Ci-wind_Si

        return real.(wind_inv), imag.(wind_inv)
    end

end
"""
