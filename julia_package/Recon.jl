module ReconWave

using LinearAlgebra
include("/home/b11209013/Wave_analysis/julia_package/Fourier_trans.jl")
include("/home/b11209013/Wave_analysis/julia_package/PowerSpec.jl")


export Recon

function Recon(
    arr :: Matrix{Float64}
    )
    A, B, a, b :: Matrix{Float64}
    

    A, B, a, b = PowerSpec.power_coeff(arr)

    Ereal, Eimag = Fourier_trans._IDFT(A, B, arr)
    Wreal, Wimag = Fourier_trans._IDFT(a, b, arr)

    Ewind = sqrt.(Ereal.^2 .+ Eimag.^2)
    Wwind = sqrt.(Wreal.^2 .+ Wimag.^2)

    Rreal, Rimag = Fourier_trans._IDFT(Ewind, Wwind, arr')
    Rwind = sqrt.(Rreal.^2 .+ Rimag.^2)

    return Rwind
end

function EwindRecon(
    arr :: Matrix{Float64}
    )

    A, B, a, b = PowerSpec.power_coeff(arr)

    Ewind = (A .- b) ./ 4
    Eimag = (-B .- a)./ 4
    Wreal = (A .+ b) ./ 4
    Wimag = (B .- a) ./ 4
end

end
