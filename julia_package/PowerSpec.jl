module PowerSpec

# using external packages  
using LinearAlgebra
include("Fourier_trans.jl")

export power_coeff, power_spec

function _Nyquist(
    arr     :: Matrix{Float64},
    arr_org :: Matrix{Float64},
    dim     :: Int64
    )

    arr_new :: Matrix{Float64} = arr[1:round(Int32, size(arr_org, dim)/2), :]
    arr_dob :: Matrix{Float64} = similar(arr_new)
    
    arr_dob[1, :]     = arr_new[1, :]
    arr_dob[2:end, :] = arr_new[2:end, :].*2

    return arr_dob
end

function power_coeff(
    arr :: Matrix{Float64},
    )

    Ck, Sk = Fourier_trans._DFT(arr)

    Ck :: Matrix{Float64} = _Nyquist(Ck, arr, 2)
    Sk :: Matrix{Float64} = _Nyquist(Sk, arr, 2)

    A, B = Fourier_trans._DFT(Ck)
    a, b = Fourier_trans._DFT(Sk)

    A :: Matrix{Float64} = _Nyquist(A, arr, 1)
    B :: Matrix{Float64} = _Nyquist(B, arr, 1)
    a :: Matrix{Float64} = _Nyquist(a, arr, 1)
    b :: Matrix{Float64} = _Nyquist(b, arr, 1)

    return A, B, a, b
end

function power_spec(
    arr :: Matrix{Float64},
    )

    A, B, a, b = power_coeff(arr)

    power_pos= 1/8 * (
        A .^ 2 + B .^ 2 + a .^ 2 + b .^ 2
        ).+ 1/4 * (a.*B - b.*A)

    power_neg= 1/8 * (
        A .^ 2 + B .^ 2 + a .^ 2 + b .^ 2
        ).- 1/4 * (a.*B - b.*A)

    ps :: Matrix{Float64} = vcat(power_pos[end:-1:1, :], power_neg)

    return ps
end

end
