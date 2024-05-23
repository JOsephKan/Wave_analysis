module EOF

using Statistics, LinearAlgebra

export normal_equ, EmpOrthFunc

function normal_equ(
    EOF  :: Matrix{Float64},
    data :: Matrix{Float64}
    )
    
    """
    Input:
    data :: Input data, 2D array
    ==================

    Variable:
    xTx_1, op :: operating matrix in the process of asking PCs

    ==================
    PCs :: Primary components of input data
    """

    xTx_1 :: Matrix{Float64} = inv(EOF'*EOF)
    op    :: Matrix{Float64} = xTx_1*EOF'
    PCs   :: Matrix{Float64} = op*data
    
    return PCs
end

function EmpOrthFunc(
    data :: Matrix{Float64},
    n    :: Int64,
    )
    
    CovMat :: Matrix{Float64} = data*data'./size(data, 2)
    
    EigVal :: Vector{Float64} = eigvals(CovMat)
    
    ExpVar :: Vector{Float64} = (EigVal./sum(EigVal))[end:-1:1][1:n]

    EigVec :: Matrix{Float64} = eigvecs(CovMat)[:, end:-1:1][:, 1:n]
    EOF    :: Matrix{Float64} = (EigVec .- mean(EigVec, dims=1))./std(EigVec, dims=1)

    PCs    :: Matrix{Float64} = normal_equ(EOF, data)[end:-1:1, :][1:n, :]

    return ExpVar, EOF, PCs
end

end
