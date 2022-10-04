import netCDF4
import numpy as np


class NetcdfReader:
    def __init__(self, filename: str):
        self.nc = netCDF4.Dataset(filename)

    def read_rgn(self, varname: str, step: int, k: int):
        v_all = self.nc.variables[varname][step, k, :]
        gall_1d = int(np.sqrt(v_all.shape[0]))
        return v_all.reshape((gall_1d, gall_1d))[
            1 : gall_1d - 1, 1 : gall_1d - 1
        ].ravel()
