module DataFormat
using PyCall
@pyimport numpy as np


export sym, asy, filt

function sym(
    lat  :: Vector{Float64},
    data :: Array{Float64, 3}
    )

    lat_c  :: Vector{Float64} = cos.(deg2rad.(lat))

    weight = sum(data .* reshape(lat_c, 1, :, 1), dims=2) ./ sum(lat_c) 
    weight_r :: Matrix{Float64} = reshape(weight, size(weight, 1), size(weight, 3))

    return weight_r
end

function asy(
    lat  :: Vector{Float64}, 
    data :: Array{Float64, 3}
    )

    if lat[1] < 0.0
        lat = reverse(lat)
        data = data[:, end:-1:1, :]
    end

    asy_lat  :: Vector{Float64} = similar(lat)
    half_idx :: Int64= div(length(lat), 2)

    asy_lat[1:half_idx] .= cos.(deg2rad.(lat[1:half_idx]))
    asy_lat[half_idx+1:end] .= -cos.(deg2rad.(lat[half_idx+1:end]))

    weight  :: Array{Float64, 3} = data .* reshape(asy_lat, 1, :, 1)

    avg_asy :: Matrix{Float64}   = reshape(sum(weight, dims=2), size(weight, 1), size(weight, 3)) ./ sum(cos.(deg2rad.(lat)))

    return avg_asy
end

function filt(
    data ,
    num_of_passes :: Int64
)
    padded_data = Array{Float64}(undef, size(data, 1), size(data, 2)+2)
    result_data = similar(padded_data)
    padded_data[:, 1]       = data[:, 2]
    padded_data[:, end]     = data[:, end-1]
    padded_data[:, 2:end-1] = data

    for k=1: num_of_passes
        for j=1:size(padded_data, 1)
            for i=2:(size(padded_data, 2)-1)
                result_data[j, i] = (
                    padded_data[j, i - 1]
                    + padded_data[j, i] * 2
                    + padded_data[j, i + 1]
                ) / 4
            end
        end
    end

    return result_data[:, 2:end-1]
end

end
