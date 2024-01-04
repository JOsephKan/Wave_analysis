

module derivative
# This module is to compute dereivative of data
# version 0.0

export derivative

function forward(arr, h, dimension, col=1)
    # This function is to compute the forward differentiation of the data column

    # Variable Description
    
    # input data 
    # h         : interval of sampling
    # arr       : data array to input, size must be 2d
    # col       : column that wanna do forward differentiation
    # dimension : the dimension that to derivative 
    
    # output data
    # arr_der   : the array derivated

    arr_der = (-3*(arr[col, 1:-1])+4*(arr[(col+1), 1:-1])-(arr[(col+2), 1:-1]))/(2*h)

    return arr_der
end

function backward(arr, h, dimension, col=-1)
    # This function is to compute the backward differentiation of the data column

    # Variable Description
    
    # input data 
    # h         : interval of sampling
    # arr       : data array to input, size must be 2d
    # col       : column that wanna do forward differentiation
    # dimension : the dimension that to derivative 
    
    # output data
    # arr_der   : the array derivated

    arr_der = (arr[col-2, 1:-1]-4*(arr[col-1, 1:-1])+3*(arr[col, 1:-1]))/(2*h)

    return arr_der
end

function derivative(arr, dimension, h)
    # This function is compute the total derivative on the data array
    
    # input variables
    # h: derivated interval
    # arr : input data array 
    # dimension: dimension of arr which is gonna be derivative

    # output variables
    # arr_new: output derivated data array

    arr_new = similar(arr)

    arr_new[1, 1:-1] = forward(arr, h, 1)
    arr_new[-1, 1:-1] = backward(arr, h, 1)
    arr_new[2:-2, 1:-1] = (arr[2:-1]-arr[1:-2])/(2*h)

    return arr_new
end

end