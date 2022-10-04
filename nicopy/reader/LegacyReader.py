import typing
import numpy as np
from ..util import calc_gall


class LegacyReader:
    def __init__(self, glevel: int, rlevel: int, kall: int, filename: str):
        self.glevel = glevel
        self.rlevel = rlevel
        self.gall, self.gall_1d = calc_gall(glevel, rlevel)
        self.kall = kall
        self.filename = filename

    def read_rgn(self, step: int, k: int):
        gall = self.gall
        kall = self.kall
        f: typing.BinaryIO = open(self.filename, "rb")
        offset = step * kall * gall * 4
        shape = (kall, gall)
        v_all = np.memmap(f, mode="r", dtype=">f4", offset=offset, shape=shape)[k, :]
        gall_1d = self.gall_1d
        return v_all.reshape((gall_1d, gall_1d))[
            1 : gall_1d - 1,
            1 : gall_1d - 1,
        ].ravel()
