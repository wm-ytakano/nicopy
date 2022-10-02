#!/usr/bin/env python
# coding: utf-8
import sys
import numpy as np
import netCDF4
from .mod_fio import FIO_PaNDa
from .mod_grid import grid_conv, latlon2xyz
from .nico_adm import NICOadm
from .nico_misc import nico_rgn, nico_digit, nico_get_latlon


class NICOio(NICOadm):
    def __init__(self, glevel, rlevel, iotype, grid_iotype="LEGACY", run_pe=0):
        super().__init__(glevel, rlevel)
        self.iotype = iotype
        self.grid_iotype = grid_iotype

        if iotype == "ADVANCED":
            if run_pe == 0:
                self.run_pe = nico_rgn(rlevel)
            else:
                self.run_pe = run_pe

    def grid_read(self, gridfile):
        """
        headname :
        """
        gall = self.gall
        lall = self.rgn
        gall_1d = self.gall_1d
        GRD_x = np.empty([3, lall, gall])
        GRD_xt = np.empty([3, 2, lall, gall])
        if self.grid_iotype == "LEGACY":
            """
            Fortran unformatted sequantial io
            """
            header = ("head", ">i")
            footer = ("footer", ">i")
            for l in range(self.rgn):
                file_name = gridfile + ".rgn" + nico_digit(l)
                dtype = np.dtype([header, ("gall", ">i"), footer])
                # open file
                f = open(file_name, "r")
                chunk = np.fromfile(f, dtype=dtype, count=1)
                gall_1d = chunk[0]["gall"]
                irec = chunk.itemsize
                dtype = np.dtype([header, ("GRD_x", ">" + str(gall) + "f8"), footer])
                for i in range(3):
                    f.seek(irec)
                    chunk = np.fromfile(f, dtype=dtype, count=1)
                    GRD_x[i, l, :] = chunk[0]["GRD_x"]
                    irec = irec + chunk.itemsize
                dtype = np.dtype(
                    [header, ("GRD_xt", ">" + str(gall * 2) + "f8"), footer]
                )
                for i in range(3):
                    f.seek(irec)
                    chunk = np.fromfile(f, dtype=dtype, count=1)
                    GRD_xt[i, :, l, :] = chunk[0]["GRD_xt"].reshape((2, gall))
                    irec = irec + chunk.itemsize
                f.close()
        elif self.grid_iotype == "ADVANCED":
            sys.exit("Do not support :" + self.grid_iotype)
        elif self.grid_iotype == "netCDF":
            # print('only checked lat_c/lon_c lon_e/lat_e has error')
            gc = grid_conv(self.gall_1d)
            for l in range(self.rgn):
                file_name = gridfile + ".rgn" + str(l).zfill(8) + ".nc"
                nc = netCDF4.Dataset(file_name, "r")
                x = nc.variables["ICO_node_x"][:] * np.pi / 180.0  # lon
                y = nc.variables["ICO_node_y"][:] * np.pi / 180.0  # lon
                GRD_x_1 = latlon2xyz(y, x)
                GRD_xt[:, :, l, :] = gc.center2vertex(GRD_x_1)
                GRD_x[:, l, :] = GRD_x_1.copy()
                nc.close()
        else:
            sys.exit("Do not support :" + self.grid_iotype)
        #
        # GRD_x to lat lon
        #
        lon_c = np.empty([lall, gall])
        lat_c = np.empty([lall, gall])
        lon_e = np.empty([lall, 2, gall])
        lat_e = np.empty([lall, 2, gall])
        #
        # memo
        #
        for l in range(lall):
            for ij in range(gall):
                lat_c[l, ij], lon_c[l, ij] = nico_get_latlon(
                    GRD_x[0, l, ij], GRD_x[1, l, ij], GRD_x[2, l, ij]
                )
                lat_e[l, 0, ij], lon_e[l, 0, ij] = nico_get_latlon(
                    GRD_xt[0, 0, l, ij], GRD_xt[1, 0, l, ij], GRD_xt[2, 0, l, ij]
                )
                lat_e[l, 1, ij], lon_e[l, 1, ij] = nico_get_latlon(
                    GRD_xt[0, 1, l, ij], GRD_xt[1, 1, l, ij], GRD_xt[2, 1, l, ij]
                )
        lon_c = lon_c.reshape([lall, gall_1d, gall_1d]) * 180.0 / np.pi
        lat_c = lat_c.reshape([lall, gall_1d, gall_1d]) * 180.0 / np.pi

        lon_e = lon_e.reshape([lall, 2, gall_1d, gall_1d]) * 180.0 / np.pi
        lat_e = lat_e.reshape([lall, 2, gall_1d, gall_1d]) * 180.0 / np.pi

        return lon_c, lat_c, lon_e, lat_e

    #
    #
    #
    def ico_read(
        self,
        filename,
        varname="",
        kall=1,
        selectlev=0,
        tall=1,
        selectstep=0,
        f_set=True,
    ):
        """
        filename :
        """
        gall = self.gall
        lall = self.rgn
        dtype = np.float32  # tentative
        if selectlev == -1:
            if selectstep == -1:
                # test
                var = np.empty([tall, kall, lall, gall], dtype)
            else:
                var = np.empty([kall, lall, gall], dtype)
        else:  # select level
            if selectstep == -1:
                var = np.empty([tall, lall, gall], dtype)
            else:
                var = np.empty([lall, gall], dtype)
        if self.iotype == "LEGACY":
            """
            Fortran unformatted direct io
            """
            for l in range(self.rgn):
                file_name = filename + ".rgn" + nico_digit(l)
                if selectlev == -1:
                    dtype = np.dtype([("var", ">" + str(gall * kall) + "f4")])
                else:
                    dtype = np.dtype([("var", ">" + str(gall) + "f4")])
                # open file
                f = open(file_name, "r")
                if selectlev == -1:
                    if selectstep == -1:
                        var[:, :, l, :] = np.fromfile(f, dtype=dtype, count=tall)[:][
                            "var"
                        ][:].reshape(tall, kall, gall)
                    else:
                        var[:, l, :] = np.fromfile(f, dtype=dtype, count=tall)[
                            selectstep
                        ]["var"].reshape(kall, gall)
                else:
                    if selectstep == -1:
                        var[:, l, :] = np.fromfile(f, dtype=dtype, count=kall * tall)[
                            :
                        ]["var"][selectlev::kall].reshape(tall, gall)
                    else:
                        var[l, :] = np.fromfile(f, dtype=dtype, count=kall * tall)[
                            selectlev + selectstep * (kall)
                        ]["var"].reshape(gall)
                f.close()
        elif self.iotype == "LEGACYS":
            """
            Fortran unformatted sequantial io
            """
            sys.exit("Do not support")
        elif self.iotype == "ADVANCED":
            if varname == "":
                sys.exit("nico_io.ico_read, need varname")
            if f_set:
                panda = FIO_PaNDa(self.glevel, self.rlevel, self.run_pe)
                for l in range(self.run_pe):  # regist file loop
                    panda.put_info(l)
                    fname = filename + ".pe" + str(l).zfill(6)
                    panda.open_file(fname, l)
            #
            # read loop
            #
            if selectlev == -1:
                if selectstep == -1:
                    # print("test")
                    var_t = np.empty([tall, lall, kall, gall], dtype)
                    for l in range(self.run_pe):
                        # test
                        # var[:,:,panda.ll_info(l),:] = panda.get_var(varname, l, kall,selectlev,tall,selectstep)
                        var_t[:, panda.ll_info(l), :, :] = panda.get_var(
                            varname, l, kall, selectlev, tall, selectstep
                        )
                    var = var_t.transpose((0, 2, 1, 3))  # t, z, l, g
                    del var_t
                else:
                    var_t = np.empty([lall, kall, gall], dtype)
                    for l in range(self.run_pe):
                        # var[:,panda.ll_info(l),:] = panda.get_var(varname, l, kall,selectlev,tall,selectstep)
                        var_t[panda.ll_info(l), :, :] = panda.get_var(
                            varname, l, kall, selectlev, tall, selectstep
                        )
                    var = var_t.transpose((1, 0, 2))
                    del var_t
            else:
                if selectstep == -1:
                    for l in range(self.run_pe):
                        var[:, panda.ll_info(l), :] = panda.get_var(
                            varname, l, kall, selectlev, tall, selectstep
                        )
                        # var_t[panda.ll_info(l),:,:] = panda.get_var(varname, l, kall,selectlev,tall,selectstep)
                else:
                    for l in range(self.run_pe):
                        var[panda.ll_info(l), :] = panda.get_var(
                            varname, l, kall, selectlev, tall, selectstep
                        )
            # reset
            del panda
        elif self.iotype == "netCDF":
            for l in range(self.rgn):
                file_name = filename + ".rgn" + str(l).zfill(8) + ".nc"
                nc = netCDF4.Dataset(file_name, "r")
                if selectlev == -1:
                    if selectstep == -1:
                        var[:, :, l, :] = nc.variables[varname][:, :, :]
                    else:
                        var[:, l, :] = nc.variables[varname][selectstep, :, :]
                else:
                    if selectstep == -1:
                        var[:, l, :] = nc.variables[varname][:, selectlev, :]
                    else:
                        var[l, :] = nc.variables[varname][selectstep, selectlev, :]
        else:
            sys.exit("Do not support")
        #
        # reshaping
        #
        """
        if selectlev == -1 :
            if selectstep == -1 :
                var = var.reshape([tall,kall,lall*gall])
            else :
                var = var.reshape([kall,lall*gall])
        else : # select level
            if selectstep == -1 :
                var = var.reshape([tall,lall*gall])
            else :
                var = var.reshape([lall*gall])
        """
        return var
