from .nico_misc import nico_gall


class NICOadm:
    def __init__(self, glevel, rlevel):
        self.glevel = glevel
        self.rlevel = rlevel
        self.rgn = int(2 ** (2 * rlevel) * 10)
        self.gall, self.gall_1d = nico_gall(glevel, rlevel)
        self.gs = 1
        self.ge = self.gall_1d - 1
