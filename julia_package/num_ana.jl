

module der_1d
# This module is to compute dereivative of data
# version 0.0
##############
# version 0.1 2023/12/28
# 1. delete dimension input variable
# 2. error fixing in the derivative function
# 3. export forward and backward functions
##############
# version 0.2 2023/12/29
# 1. rename the module der -> der_1d
# 2. rewrite the functions into first order form
##############

export derivative, forward, backward, central

function forward_h(arr, h, row=1)
    # This function is to compute the forward differentiation of the data column
    # Truncation error: O(h)

    # Variable Description
    
    # input data 
    # h         : interval of sampling
    # arr       : data array to input, size must be 2d
    # row       : row that wanna do forward differentiation
    
    # output data
    # arr_der   : the array derivated

    arr_der = (arr[row+1, :]-arr[row, :])/h
    return arr_der
end

function backward_h(arr, h)
    # This function is to compute the backward differentiation of the data column
    # Truncation error: O(h)

    # Variable Description
    
    # input data 
    # h         : interval of sampling
    # arr       : data array to input, size must be 2d
    # row       : row that wanna do forward differentiation
        
    # output data
    # arr_der   : the array derivated

    arr_der = (arr[end, :]-arr[end-1, :])/h
    
    return arr_der
end

function central_h(arr, h)
    # This function is compute the central derivative on the data array
    # Truncation error: O(h)
    
    # Input variables:
    # h: derivative interval
    # arr: input data array
    #
    # output variables:
    # arr_der: output derivated data array
    
    arr_der = (arr[3:end, :]-arr[1:end-2, :])/(2*h)
    return arr_der
end

function derivative_h(arr, h::Float64, dim::Int64)
    # This function is compute the total derivative on the data array
    # Truncation error: O(h)
    
    # input variables
    # h: derivated interval
    # arr : input data array 
    # dim: derivatived dimension of the array
    
    # output variables
    # arr_new: output derivated data array

    shape = size(arr)
    arr_r = reshape(arr, (shape[dim], convert(Int64, prod(shape)/shape[dim])))

    arr_new = similar(arr_r)

    arr_new[1, :] = forward_h(arr_r, h)
    arr_new[end, :] = backward_h(arr_r, h)
    arr_new[2:end-1, :] = central_h(arr_r, h)

    arr_new = reshape(arr_new, shape)

    return arr_new
end

end
