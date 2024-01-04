# import packages
import numpy as np
import netCDF4 as nc

def nc_import(
        nc_file_path, 
        nc_var_name_list,
        ):
    
    ds = nc.Dataset(nc_file_path)
    nc_var_list = []

    for i in range(len(nc_var_name_list)):
        nc_var_name = nc_var_name_list[i]
        nc_var = ds.variables[nc_var_name]
        nc_var_list,append(nc_var[nc_var_name_list[i]][:])

    return nc_var_list
