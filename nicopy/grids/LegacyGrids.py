import numpy as np
from .AbstractGrids import AbstractGrids


class LegacyGrids(AbstractGrids):
    def load(self):
        gall = self.gall
        header = ("head", ">i")
        footer = ("footer", ">i")
        with open(f"{self.filename}", "r") as f:
            dtype = np.dtype([header, ("gall", ">i"), footer])
            chunk = np.fromfile(f, dtype=dtype, count=1)
            irec = chunk.itemsize

            self.grd_x = np.empty([3, gall])
            dtype = np.dtype([header, ("grd_x", f">{gall}f8"), footer])
            for i in range(3):
                f.seek(irec)
                chunk = np.fromfile(f, dtype=dtype, count=1)
                self.grd_x[i, :] = chunk[0]["grd_x"]
                irec = irec + chunk.itemsize

            self.grd_xt = np.empty([3, 2, gall])
            dtype = np.dtype([header, ("grd_xt", f">{gall * 2}f8"), footer])
            for i in range(3):
                f.seek(irec)
                chunk = np.fromfile(f, dtype=dtype, count=1)
                self.grd_xt[i, :, :] = chunk[0]["grd_xt"].reshape((2, gall))
                irec = irec + chunk.itemsize
