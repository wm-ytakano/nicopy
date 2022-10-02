import time
import sys
import numpy as np
from .nico_adm import NICOadm
from .nico_io import NICOio
from .nico_misc import nico_gall_in, nico_get_xyz, nico_gall, nico_rgn


class NICOgrid(NICOadm):
    def __init__(
        self,
        glevel,
        rlevel,
        radius=6371.01e3,
        grid_type=0,
    ):
        super().__init__(glevel, rlevel)
        self.grid_type = grid_type
        self.gall_in = nico_gall_in(glevel, rlevel)
        self.radius = radius

    def pgrid(self, lat, lon, lat_t, lon_t, *, data=0):
        #
        # input : lat, lon is center grid
        #

        if self.grid_type == 0:  # exact grid
            #
            # exact grid
            #
            #
            # set up array
            #
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
            utx_out = utx_out.reshape(self.gall_in, 6)
            uty_out = uty_out.reshape(self.gall_in, 6)
            # return cell_info, utx, uty
            return utx_out, uty_out
        elif self.grid_type == 1:
            #
            # reduced grid type by nicoview
            #
            sys.exit("this is not support")
            return
        elif self.grid_type == 2:
            #
            # reduced grid type and reduce data
            #
            sys.exit("this is not support")
            return
        elif self.grid_type == 3:
            #
            # reduced grid type and
            #
            sys.exit("this is not support")
            return

    def area_all(self, lat, lon, lat_t, lon_t):
        #
        # lat[gall],lon[gall],lat_t[6,gall],lon_t[6,gall] is 1d array
        #
        opr = NICOvector(self.radius)
        gall_in = lat.size
        area = np.zeros_like(lat)
        cvector = nico_get_xyz(lat, lon).T
        pvector = np.empty([cvector.shape[0], cvector.shape[1], 6])
        for i in range(6):
            pvector[:, :, i] = nico_get_xyz(lat_t[:, i], lon_t[:, i]).T

        for ij in range(gall_in):
            ptmp = pvector[ij, :, :]
            ptmp = np.unique(ptmp, axis=1)
            if ptmp.shape[1] == 5:  # pentagonal
                for i in range(4):
                    area[ij] = area[ij] + opr.vector_area(
                        cvector[ij], ptmp[:, i], ptmp[:, i + 1]
                    )
                i = 4
                area[ij] = area[ij] + opr.vector_area(
                    cvector[ij], ptmp[:, i], ptmp[:, 0]
                )
            else:
                for i in range(5):  # hexagonal
                    area[ij] = area[ij] + opr.vector_area(
                        cvector[ij], ptmp[:, i], ptmp[:, i + 1]
                    )
                i = 5
                area[ij] = area[ij] + opr.vector_area(
                    cvector[ij], ptmp[:, i], ptmp[:, 0]
                )
            del ptmp
        return area


class NICOvector:
    def __init__(self, radius=6371.01e3):
        self.radius = radius

    #
    def vector_area(self, a, b, c):
        area = 0.0e0
        len1 = self.angle(a, b) * 0.5e0
        len2 = self.angle(b, c) * 0.5e0
        len3 = self.angle(c, a) * 0.5e0
        s = 0.5e0 * (len1 + len2 + len3)
        x = np.tan(s) * np.tan(s - len1) * np.tan(s - len2) * np.tan(s - len3)
        if x > 0.0e0:
            area = 4.0e0 * np.arctan(np.sqrt(x)) * self.radius * self.radius
        else:
            area = 0.0e0
        return area

    def angle(self, x, y):
        nvlenC = np.dot(x, y)
        nv = np.cross(x, y)
        nvlenS = np.linalg.norm(nv)
        rad = np.arctan2(nvlenS, nvlenC)
        return rad

    # a->b vector
    def tovector(self, a, b):
        return b - a


if __name__ == "__main__":
    t1 = time.time()
    glevel = 5
    rlevel = 1
    nico = NICOio(glevel, rlevel, "LEGACY")
    gall, gall_1d = nico_gall(glevel, rlevel)
    gall_in = nico_gall_in(glevel, rlevel)
    rgn = nico_rgn(rlevel)
    # gridname="test/reg_nicam_gl05_rl00/grid/hgrid/grid"
    gridname = "test/global_nicam_gl05_rl01/grid/hgrid/grid"
    lon_c, lat_c, lon_e, lat_e = nico.grid_read(gridname)
    # -------------------------------------------------------------------
    # -- define the x-, y-values and the polygon points
    # -------------------------------------------------------------------
    # from nico_grid import NICOgrid
    gridconv = NICOgrid(glevel, rlevel)
    utx = np.empty([rgn, gall_in, 6], np.float32)
    uty = np.empty([rgn, gall_in, 6], np.float32)
    for l in np.arange(rgn):
        utx[l, :, :], uty[l, :, :] = gridconv.pgrid(
            lat_c[l, :, :], lon_c[l, :, :], lat_e[l, :, :, :], lon_e[l, :, :, :]
        )

    area = 0
    for l in range(rgn):
        # convert lagian
        lat = lat_c[l, 1 : gall_1d - 1, 1 : gall_1d - 1].ravel() * np.pi / 180.0
        lon = lon_c[l, 1 : gall_1d - 1, 1 : gall_1d - 1].ravel() * np.pi / 180.0
        lat_t = uty[l, :, :] * np.pi / 180.0
        lon_t = utx[l, :, :] * np.pi / 180.0
        area = area + gridconv.area_all(lat, lon, lat_t, lon_t).sum()
        print(l, area)
    print(area, 4 * np.pi * (6371.01e3) ** 2)
    print(area / (4 * np.pi * (6371.01e3) ** 2))
