import numpy as np
from ..util import calc_gall, calc_gall_in, calc_lall
from .mod_grid import xyz2latlon


class AbstractGrids:
    def __init__(self, glevel: int, rlevel: int, filename: str):
        self.glevel = glevel
        self.rlevel = rlevel
        self.filename = filename
        self.gall, self.gall_1d = calc_gall(glevel, rlevel)
        self.gall_in = calc_gall_in(glevel, rlevel)
        self.lall = calc_lall(rlevel)
        self.load()

    def load(self):
        pass

    def get_lonlat_c_2d(self):
        """
        Returns:
            lon_c: NDArray of shape (gall_1d, gall_1d)
            lat_c: NDArray of shape (gall_1d, gall_1d)
        """
        gall = self.gall
        gall_1d = self.gall_1d
        grd_x = self.grd_x

        lon_c = np.empty([gall])
        lat_c = np.empty([gall])

        for ij in range(gall):
            lat_c[ij], lon_c[ij] = xyz2latlon(grd_x[:, ij])

        lon_c = lon_c.reshape([gall_1d, gall_1d]) * 180.0 / np.pi
        lat_c = lat_c.reshape([gall_1d, gall_1d]) * 180.0 / np.pi

        return lon_c, lat_c

    def get_lonlat_e_2d(self):
        """
        Returns:
            lon_e: NDArray of shape (2, gall_1d, gall_1d)
            lat_e: NDArray of shape (2, gall_1d, gall_1d)
        """
        gall = self.gall
        gall_1d = self.gall_1d
        grd_xt = self.grd_xt

        lon_e = np.empty([2, gall])
        lat_e = np.empty([2, gall])

        for ij in range(gall):
            lat_e[0, ij], lon_e[0, ij] = xyz2latlon(grd_xt[:, 0, ij])
            lat_e[1, ij], lon_e[1, ij] = xyz2latlon(grd_xt[:, 1, ij])

        lon_e = lon_e.reshape([2, gall_1d, gall_1d]) * 180.0 / np.pi
        lat_e = lat_e.reshape([2, gall_1d, gall_1d]) * 180.0 / np.pi

        return lon_e, lat_e

    def get_lonlat_c(self):
        """
        Returns:
            lon_c: NDArray of shape (gall_in)
            lat_c: NDArray of shape (gall_in)
        """
        gall_1d = self.gall_1d
        lon_c, lat_c = self.get_lonlat_c_2d()
        lon_cin = lon_c[1 : gall_1d - 1, 1 : gall_1d - 1].ravel()
        lat_cin = lat_c[1 : gall_1d - 1, 1 : gall_1d - 1].ravel()
        return lon_cin, lat_cin

    def get_lonlat_v(self):
        """
        Returns:
            lon_v: NDArray of shape (gall_in, 6)
            lat_v: NDArray of shape (gall_in, 6)
        """
        lon_t, lat_t = self.get_lonlat_e_2d()
        gall_1d = self.gall_1d
        utx = np.empty([self.gall, 6], np.float32)
        uty = np.empty([self.gall, 6], np.float32)
        cell_info = np.empty([self.gall], np.int32)
        #
        for j in range(1, self.gall_1d - 1):
            for i in range(1, self.gall_1d - 1):
                ij = self.gall_1d * j + i
                utx[ij, 0] = lon_t[1, j - 1, i - 1]
                utx[ij, 1] = lon_t[0, j - 1, i - 1]
                utx[ij, 2] = lon_t[1, j - 1, i]
                utx[ij, 3] = lon_t[0, j, i]
                utx[ij, 4] = lon_t[1, j, i]
                utx[ij, 5] = lon_t[0, j, i - 1]
                #
                uty[ij, 0] = lat_t[1, j - 1, i - 1]
                uty[ij, 1] = lat_t[0, j - 1, i - 1]
                uty[ij, 2] = lat_t[1, j - 1, i]
                uty[ij, 3] = lat_t[0, j, i]
                uty[ij, 4] = lat_t[1, j, i]
                uty[ij, 5] = lat_t[0, j, i - 1]
                cell_info[ij] = 6
        utx_out = utx.reshape([gall_1d, gall_1d, 6])[
            1 : gall_1d - 1, 1 : gall_1d - 1, :
        ]
        uty_out = uty.reshape([gall_1d, gall_1d, 6])[
            1 : gall_1d - 1, 1 : gall_1d - 1, :
        ]
        lon_v = utx_out.reshape(self.gall_in, 6)
        lat_v = uty_out.reshape(self.gall_in, 6)
        return lon_v, lat_v
