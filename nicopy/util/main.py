#!/usr/bin/env python
# coding: utf-8
import numpy as np

"""
NICoPy
NICam Operator for Python
"""


def calc_gall(glevel: int, rlevel: int) -> tuple[int, int]:
    """
    Args:
        glevel : nicam glevel
        rlevel : nicam rlevel
    """
    nmax: int = 2 ** (glevel - rlevel)
    gall = (1 + nmax + 1) * (1 + nmax + 1)
    gall_1d = 1 + nmax + 1
    return gall, gall_1d


def calc_gall_in(glevel: int, rlevel: int) -> int:
    """
    Args:
        glevel : nicam glevel
        rlevel : nicam rlevel
    """
    nmax = 2 ** (glevel - rlevel)
    gall_in = nmax * nmax
    return gall_in


def calc_lall(rlevel: int) -> int:
    """
    Args:
        rlevel : nicam rlevel
    """
    lall = 10 * 4 ** (rlevel)
    return lall


def rearrange_lon(lon: np.ndarray) -> np.ndarray:
    less_than = lon < -180.0
    greater_than = lon > 180.0
    lon[less_than] = lon[less_than] + 360.0
    lon[greater_than] = lon[greater_than] - 360.0
    return lon
