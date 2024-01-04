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
    def __init__(self, arr, dim):
        """
        arr: Array that should be derivated, must greater than 2D
        dim: dimension that gonna be derivatived
        """
        self.h = None
        self.arr = arr
        self.dim = dim
        self.size = None
        self.arr_r = None
        self.arr_der = None
        self.for_der = None

    def _dim_reshape(self):
        if np.ndim(self.arr) <= self.dim:
            print("Dimension Error, dimension greater than arr shape!")

        elif np.ndim(self.arr) > 1:
            self.size = np.shape(self.arr)
            self.arr_r = np.reshape(
                self.arr,
                (self.size[self.dim], int(np.prod(self.size) / self.size[self.dim])),
            )

    def forward_h(self, h, row=0):
        self._dim_reshape()

        self.for_der = (self.arr_r[row + 1, :] - self.arr_r[row, :]) / (
            h[row + 1] - h[row]
        )

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
