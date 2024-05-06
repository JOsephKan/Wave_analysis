# import function
import numpy as np


def poly_itp(org_x_idx: list, itp_x_idx: list, y_idx: list, order) -> list:
    op = np.vander(org_x_idx, order, increasing=True)

    op_t = np.transpose(op)

    coe = np.matmul(np.linalg.inv(np.matmul(op_t, op)),np.matmul(op_t, y_idx))

    itp_x = np.vander(itp_x_idx, order, increasing=True)

    itp_y = np.matmul(itp_x, coe)

    return itp_y
