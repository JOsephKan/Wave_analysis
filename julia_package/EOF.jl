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

    PCs    :: Matrix{Float64} = normal_equ(EigVec, data)[end:-1:1, :][1:n, :]

    PCs_norm :: Matrix{Float64} = (PCs.-mean(PCs, dims=2))./(std(PCs, dims=2))
    
    EOF    :: Matrix{Float64} = data*(PCs_norm'*inv(PCs_norm*PCs_norm'))

    return ExpVar, EOF, PCs_norm
end

end
