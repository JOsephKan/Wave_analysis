# import function
import numpy as np


class interpolation:
    def __init__(self):
        self.coe: list = None

    def poly_itp(self, org_x_idx: list, itp_x_idx: list, y_idx: list) -> list:
        op = np.vander(org_x_idx, 6, increasing=True)

        self.coe = np.matmul(np.linalg.inv(op), y_idx)

        itp_x = np.vander(itp_x_idx, 6, increasing=True)

        itp_y = np.matmul(itp_x, self.coe)

        return itp_y


class Derivative:
    def __init__(self, arr, dim, dx):
        """
        arr: Array that should be derivated, must greater than 2D
        dim: dimension that gonna be derivatived
        dx : the differential interval, can be a number (float, int) or a 1D array
        """
        self.dx = dx
        self.arr = arr
        self.dim = dim
        self.size = None
        self.arr_r = None
        self.arr_der = None
        self.for_der = None
        self.type_of_dx = None

    def _interval_check(self):
        """
        !!! Please DO NOT change this function!!!
        This function is check whether the interval is reasonable or not.
        """

        if (type(dx) == int) or (type(dx) == float):
            self.type_of_dx = 1
            
        elif (type(dx) == np.ndarray) and (len(dx.shape)==1):
            self.type_of_dx = 2
            self.dx_new = dx[:, np.newaxis].repeat(())
        
        else:
            raise TypeError("Please check the type of dx!")

        return self.type_of_dx
    
    def _dim_reshape(self):
        """
        !!! Please DO NOT change this function !!!
        This function is to determine whether the array and dimension is correct.
        
        First condition:
        To check of the dimension required is greater than the dimension the array obtain.

        Second condition:
        If the array is greater than 2D, the function will help it reshape to 2D, which the dimension required to be derivatived will be isolate from other dimension.

        Third condition:
        When the array is 1D, the function will help it to reshape as 2D to make sure it can be used in following functions.
        
        """
        
        if np.ndim(self.arr) <= self.dim:
            raise ValueError("Dimension Error, dimension greater than arr shape!")

        elif len(self.arr.shape) > 1:
            self.size = np.shape(self.arr)
            new_size = np.delete(self.size, self.dim)
            self.arr_r = np.reshape(
                self.arr,
                (self.size[self.dim], int(np.prod(new_size))),
            )
        elif len(self.arr.shape) = 1:
            self.arr_r = np.reshape(self.arr, (self.arr.shape[0], 1)

        else;
            raise ValueError("Dimension Wrong")

    def forward(self, dx, n_point=2, row=0):

        """
        This function is to do forward differentiation
        Truncation error: O(h)

        Steps:
        1. Check the dimension of the input array
        2. do 1-order differentiation

        Variables;
        input:
        dx :: the interval of the 
        """
        
        self._dim_reshape()

        if self.type_of_dx==1:
            if n_point==2:

                self.for_der = (self.arr_r[row + 1, :] - self.arr_r[row, :]) / dx

            if n_point==3:
                if self.arr_r.shape[0]<3:
                    raise DimensionError("Use n_point=2")
                self.for_der = (-3*self.arr_r[row, :]+4*self.arr_r[row+1, :]-self.arr_r[row+2, :]) / (2*dx)

        if self.type_of_dx==2:

            self.dx_new = self.dx[:, np.newaxis].repeat()
            
            if n_point==2:

                self.for_der = (self.arr_r[row + 1, :] - self.arr_r[row, :]) / dx

            if n_point==3:
                if self.arr_r.shape[0]<3:
                    raise DimensionError("Use n_point=2")
                self.for_der = (-3*self.arr_r[row, :]+4*self.arr_r[row+1, :]-self.arr_r[row+2, :]) / (2*dx)
 
        return self.for_der

    def backward_h(self, h, row=-1):
        self._dim_reshape()

        self.back_der = (self.arr_r[row, :] - self.arr_r[row - 1, :]) / (
            h[row] - h[row - 1]
        )

        return self.back_der

    def derivative_h(self, h):
        self._dim_reshape()

        self.arr_der = np.empty_like(self.arr_r)

        self.arr_der[0, :] = self.forward_h(h)
        self.arr_der[1:-1, :] = (self.arr_r[2:, :] - self.arr_r[:-2, :]) / (
            h[2:] - h[:-2]
        )
        self.arr_der[-1, :] = self.backward_h(h)

        self.arr_der = np.reshape(self.arr_der, size)

        return self.arr_der
