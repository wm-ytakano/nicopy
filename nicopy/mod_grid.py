#!/usr/bin/env python
# coding: utf-8
#
# nicam grid converter
#
import sys
import numpy as np

radius = 6371.0e3


def latlon2xyz(lat, lon):
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)

    return np.stack([x, y, z])


def VECTR_dot(a, b, c, d):
    l = (
        (b[0] - a[0]) * (d[0] - c[0])
        + (b[1] - a[1]) * (d[1] - c[1])
        + (b[2] - a[2]) * (d[2] - c[2])
    )

    return l


EPS = -9.0e30


def xyz2latlon_xt(v):
    n = v[:].shape[2]
    ni = v[:].shape[1]
    lat = np.empty([ni, n])
    lon = np.empty([ni, n])
    for j in range(ni):
        for i in range(n):
            lat[j, i], lon[j, i] = xyz2latlon_1(v[:, j, i])
    return lat, lon


def xyz2latlon_2(v):
    n = v.shape[-1]
    lat = np.empty([n])
    lon = np.empty([n])
    for nv in range(n):
        lat[nv], lon[nv] = xyz2latlon_1(v[:, nv])
    return lat, lon


def xyz2latlon_1(v):
    x = v[0]
    y = v[1]
    z = v[2]
    length = np.sqrt(x * x + y * y + z * z)

    if length < EPS:
        lat = 0.0
        lon = 0.0

    if z / length >= 1.0:  # vector is parallele to z axis.
        lat = np.arcsin(1.0)
        lon = 0.0
        return
    elif z / length <= -1.0:  # ! vector is parallele to z axis.
        lat = np.arcsin(-1.0)
        lon = 0.0
        return
    else:
        lat = np.arcsin(z / length)

    length_h = np.sqrt(x * x + y * y)

    if length_h < EPS:
        lon = 0.0
        return

    if x / length_h >= 1.0:
        lon = np.arccos(1.0)
    elif x / length_h <= -1.0:
        lon = np.arccos(-1.0)
    else:
        lon = np.arccos(x / length_h)

    if y < 0.0:
        lon = -lon

    return lat, lon


def VECTR_cross(a, b, c, d):
    nv = np.empty(3)
    nv[0] = (b[1] - a[1]) * (d[2] - c[2]) - (b[2] - a[2]) * (d[1] - c[1])
    nv[1] = (b[2] - a[2]) * (d[0] - c[0]) - (b[0] - a[0]) * (d[2] - c[2])
    nv[2] = (b[0] - a[0]) * (d[1] - c[1]) - (b[1] - a[1]) * (d[0] - c[0])
    return nv


def VECTR_abs(a):
    l = a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
    l = np.sqrt(l)
    return l


class grid_conv:
    def __init__(self, ADM_gall_1d):
        self.ADM_gmin = 0 + 1  # if fortran 0 is 1
        self.ADM_gall_1d = ADM_gall_1d
        self.ADM_gmax = ADM_gall_1d - 1
        self.ADM_gall = ADM_gall_1d**2
        self.ADM_gall_in = (ADM_gall_1d - 2) ** 2
        self.ADM_nxyz = 3
        self.ADM_have_sgp = True  # tentative
        self.wk = np.empty([2, self.ADM_gall, 4, 3])  # ADM_TI to ADM_TJ
        print(self.ADM_gall)

    def suf(self, j, i):
        # suffix = ADM_gall_1d * (j-1) + i
        suffix = self.ADM_gall_1d * (j) + i
        return suffix

    def cg2ug(self, v):  # v is center point (x,y,z) cordinate
        GRD_xt = self.center2vertex(v)
        lat_t, lon_t = xyz2latlon_xt(GRD_xt)
        lat_t = lat_t * 180.0 / np.pi
        lon_t = lon_t * 180.0 / np.pi
        lat_t = lat_t.reshape([-1, self.ADM_gall_1d, self.ADM_gall_1d])
        lon_t = lon_t.reshape([-1, self.ADM_gall_1d, self.ADM_gall_1d])
        vlat_t, vlon_t = self.pgrid(lat_t, lon_t)

        return vlon_t, vlat_t

    def pgrid(self, lat_t, lon_t):
        #
        # exact grid
        #
        #
        # set up array
        #
        gall_1d = self.ADM_gall_1d
        utx = np.empty([self.ADM_gall, 6], np.float32)
        uty = np.empty([self.ADM_gall, 6], np.float32)
        cell_info = np.empty([self.ADM_gall], np.int32)
        #
        for j in range(1, self.ADM_gall_1d - 1):
            for i in range(1, self.ADM_gall_1d - 1):
                ij = self.ADM_gall_1d * j + i
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
        utx_out = utx_out.reshape(self.ADM_gall_in, 6)
        uty_out = uty_out.reshape(self.ADM_gall_in, 6)

        return (
            uty_out,
            utx_out,
        )

    def center2vertex(self, GRD_x):
        # Todo treat pentagon
        ADM_TI = 0
        ADM_TJ = 1
        ADM_gmin = self.ADM_gmin
        ADM_gmax = self.ADM_gmax
        ADM_gall = self.ADM_gall
        # ADM_gall_1d = self.ADM_gall_1d
        GRD_xt = np.zeros([3, 2, self.ADM_gall])
        wk = self.wk
        if self.ADM_gall != ADM_gall:
            print("error")
            sys.exit()

        # for j in range(ADM_gmin-1,ADM_gmax+1) :
        #    for i in range(ADM_gmin-1,ADM_gmax+1) :
        for j in range(ADM_gmin - 1, ADM_gmax):
            for i in range(ADM_gmin - 1, ADM_gmax):
                ij = self.suf(j, i)
                ip1j = self.suf(j, i + 1)
                ip1jp1 = self.suf(j + 1, i + 1)
                ijp1 = self.suf(j + 1, i)
                for d in range(self.ADM_nxyz):
                    wk[ADM_TI, ij, 0, d] = GRD_x[d, ij]
                    wk[ADM_TI, ij, 1, d] = GRD_x[d, ip1j]
                    wk[ADM_TI, ij, 2, d] = GRD_x[d, ip1jp1]
                    wk[ADM_TI, ij, 3, d] = GRD_x[d, ij]

                    wk[ADM_TJ, ij, 0, d] = GRD_x[d, ij]
                    wk[ADM_TJ, ij, 1, d] = GRD_x[d, ip1jp1]
                    wk[ADM_TJ, ij, 2, d] = GRD_x[d, ijp1]
                    wk[ADM_TJ, ij, 3, d] = GRD_x[d, ij]
        wk[ADM_TI, self.suf(ADM_gmin - 1, ADM_gmax), :, :] = wk[
            ADM_TJ, self.suf(ADM_gmin - 1, ADM_gmax), :, :
        ]
        wk[ADM_TJ, self.suf(ADM_gmax, ADM_gmin - 1), :, :] = wk[
            ADM_TI, self.suf(ADM_gmax, ADM_gmin - 1), :, :
        ]
        # pentagone tentative
        wk[ADM_TI, self.suf(ADM_gmin - 1, ADM_gmin - 1), :, :] = wk[
            ADM_TJ, self.suf(ADM_gmin - 1, ADM_gmin), :, :
        ]
        o = np.zeros(3)
        for t in range(2):
            for j in range(ADM_gmin - 1, ADM_gmax + 1):
                for i in range(ADM_gmin - 1, ADM_gmax + 1):
                    #            for j in range(ADM_gmin,ADM_gmax) :
                    #                for i in range(ADM_gmin,ADM_gmax) :
                    ij = self.suf(j, i)
                    gc = np.zeros(3)
                    for m in range(3):
                        r_lenC = VECTR_dot(o, wk[t, ij, m, :], o, wk[t, ij, m + 1, :])
                        r = VECTR_cross(o, wk[t, ij, m, :], o, wk[t, ij, m + 1, :])
                        r_lenS = VECTR_abs(r)
                        r[:] = r[:] / r_lenS * np.arctan2(r_lenS, r_lenC)
                        gc[:] = gc[:] + r[:]
                    gc_len = VECTR_abs(gc)
                    GRD_xt[:, t, ij] = gc[:] / gc_len
        return GRD_xt
