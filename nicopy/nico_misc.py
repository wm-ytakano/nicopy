#!/usr/bin/env python
# coding: utf-8
import numpy as np

"""
NICoPy
NICam Operator for Python
"""


def nico_digit(rgn: int) -> str:
    """
    Args:
        rgn : region number
    """
    lrgn = str(rgn).zfill(5)
    return lrgn


def nico_gall(glevel: int, rlevel: int) -> tuple[int, int]:
    """
    Args:
        glevel : nicam glevel
        rlevel : nicam rlevel
    """
    nmax: int = 2 ** (glevel - rlevel)
    gall = (1 + nmax + 1) * (1 + nmax + 1)
    gall_1d = 1 + nmax + 1
    return gall, gall_1d


def nico_gall_in(glevel: int, rlevel: int) -> int:
    """
    Args:
        glevel : nicam glevel
        rlevel : nicam rlevel
    """
    nmax = 2 ** (glevel - rlevel)
    gall_in = nmax * nmax
    return gall_in


def nico_rgn(rlevel: int) -> int:
    """
    Args:
        rlevel : nicam rlevel
    """
    rgn = 10 * 4 ** (rlevel)
    return rgn


def nico_get_latlon(x: float, y: float, z: float) -> tuple[float, float]:
    """
    Args:
        x, y, z
    Returns:
        lat, lon
    """
    epsilon = 1.0e-99
    leng = np.sqrt(x * x + y * y + z * z)
    if leng < epsilon:
        lat = 0.0
        lon = 0.0
        return lat, lon
    if z / leng >= 1.0:  # parallele to z axis.
        lat = np.arcsin(1.0)
        lon = 0.0
        return lat, lon
    elif z / leng <= -1.0:  # parallele to z axis.
        lat = np.arcsin(-1.0)
        lon = 0.0
        return lat, lon
    #
    # normal case
    #
    lat = np.arcsin(z / leng)

    #
    # lon loop
    #
    leng_xy = np.sqrt(x * x + y * y)
    if leng_xy < epsilon:
        lon = 0.0
        return lat, lon

    if x / leng_xy >= 1.0:
        lon = np.arccos(1.0)
    elif x / leng_xy <= -1.0:
        lon = np.arccos(-1.0)
    else:
        lon = np.arccos(x / leng_xy)

    if y < 0.0:
        lon = -lon
    return lat, lon


def nico_get_xyz(lat: float, lon: float) -> np.ndarray:
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    vec = np.vstack([x, y, z])
    return vec


def nico_rearrange_lon(lon: np.ndarray) -> np.ndarray:
    less_than = lon < -180.0
    greater_than = lon > 180.0
    lon[less_than] = lon[less_than] + 360.0
    lon[greater_than] = lon[greater_than] - 360.0
    return lon
