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

    def read_rgn(
        self,
        step: int,
        k: int,
        precision: int = 4,
        output_halo: bool = False,
        output_shape: str = "2D",
        access: str = "direct",
    ):
        if output_shape not in ["1D", "2D"]:
            raise ValueError("'output_shape' must be '1D' or '2d'")
        if access not in ["direct", "sequential"]:
            raise ValueError("'access' must be 'direct' or 'sequential'")
        gall = self.gall
        kall = self.kall
        f: typing.BinaryIO = open(self.filename, "rb")
        if access == "sequential":
            offset = 4
        else:
            offset = 0
        v_all = np.memmap(
            f,
            mode="r",
            dtype=f">f{precision}",
            offset=step * kall * gall * precision + offset,
            shape=(kall, gall),
        )[k, :]
        gall_1d = self.gall_1d
        if output_halo:
            if output_shape == "1D":
                return v_all
            else:
                return v_all.reshape((gall_1d, gall_1d))
        else:
            if output_shape == "1D":
                return v_all.reshape((gall_1d, gall_1d))[
                    1 : gall_1d - 1,
                    1 : gall_1d - 1,
                ].ravel()
            else:
                return v_all.reshape((gall_1d, gall_1d))[
                    1 : gall_1d - 1,
                    1 : gall_1d - 1,
                ]
