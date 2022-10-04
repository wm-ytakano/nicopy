import numpy as np
import netCDF4
from .AbstractGrids import AbstractGrids
from .mod_grid import grid_conv, latlon2xyz


class NetcdfGrids(AbstractGrids):
    def load(self):
        gc = grid_conv(self.gall_1d)
        nc = netCDF4.Dataset(self.filename, "r")
        lon = nc.variables["ICO_node_x"][:] * np.pi / 180.0
        lat = nc.variables["ICO_node_y"][:] * np.pi / 180.0
        self.grd_x = latlon2xyz(lat, lon)
        self.grd_xt = gc.center2vertex(self.grd_x)
        nc.close()
