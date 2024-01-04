# import function
import numpy as np

class num:
    def __init__(self):
        self.coe : list= None
    
    def poly_itp(
        self, 
        org_x_idx : list, 
        itp_x_idx : list, 
        y_idx     : list
        ) -> list :
        
        op = np.vander(org_x_idx, increasing = True)
        
        self.coe = np.matmul(np.linalg.inv(op), y_idx)
        
        itp_x = np.vander(itp_x_idx, increasing = True)
        
        itp_y = np.matmul(itp_x, self.coe).reshape(len(itp_x_idx))
        
        return itp_y
