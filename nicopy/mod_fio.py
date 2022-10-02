import typing
import os
from ctypes import sizeof, c_char, c_int32, c_int64
import numpy as np
from logging import getLogger
from .nico_misc import nico_rgn, nico_gall

logger = getLogger(__name__)

# character length
FIO_HSHORT = 16
FIO_HMID = 64
FIO_HLONG = 256
FIO_HFILE = 1024

# data type
FIO_REAL4 = 0
FIO_REAL8 = 1
FIO_INTEGER4 = 2
FIO_INTEGER8 = 3

# data endian
FIO_UNKNOWN_ENDIAN = 0
FIO_LITTLE_ENDIAN = 1
FIO_BIG_ENDIAN = 2
FIO_DUMP_OFF = 0

# topology
FIO_ICOSAHEDRON = 0
FIO_IGA_LCP = 1
FIO_IGA_MLCP = 2

# file mode (partial or complete)
FIO_SPLIT_FILE = 0
FIO_INTEG_FILE = 1

# action type
FIO_FREAD = 0
FIO_FWRITE = 1
FIO_FAPPEND = 2

# list
precision = [4, 8, 4, 8]
dinfosize = (
    sizeof(c_char) * FIO_HSHORT * 3
    + sizeof(c_char) * FIO_HMID
    + sizeof(c_char) * FIO_HLONG
    + sizeof(c_int64) * 3
    + sizeof(c_int32) * 3
)


class FIOError(Exception):
    pass


class CommonInfo(typing.TypedDict):
    fmode: int
    endiantype: int
    grid_topology: int
    glevel: int
    rlevel: int
    num_of_rgn: int
    rgnid: np.ndarray


class HeaderInfo(typing.TypedDict):
    fname: str
    description: str
    note: str
    num_of_data: int
    fmode: int
    endiantype: int
    grid_topology: int
    glevel: int
    rlevel: int
    num_of_rgn: int
    rgnid: np.ndarray


class DataInfo(typing.TypedDict):
    varname: str
    description: str
    unit: str
    layername: str
    note: str
    datasize: int
    datatype: int
    num_of_layer: int
    step: int
    time_start: int
    time_end: int


class StatusInfo(typing.TypedDict):
    rwmode: int
    opened: int
    fp: typing.BinaryIO
    eoh: int


class FileInfo(typing.TypedDict):
    header: HeaderInfo
    dinfo: typing.List[DataInfo]
    status: StatusInfo


class FIO:
    """
    Python Migration of Advanced file I/O module (CORE)

    put/get indicates the communication with the database
    read/write indicates the communication with the file
    """

    num_of_file: int = 0
    common: CommonInfo
    finfo: typing.List[FileInfo] = []

    def put_commoninfo(
        self,
        fmode: int,
        endiantype: int,
        grid_topology: int,
        glevel: int,
        rlevel: int,
        num_of_rgn: int,
        rgnid: np.ndarray,
    ):
        self.common: CommonInfo = {
            "fmode": fmode,
            "endiantype": endiantype,
            "grid_topology": grid_topology,
            "glevel": glevel,
            "rlevel": rlevel,
            "num_of_rgn": num_of_rgn,
            "rgnid": rgnid,
        }

    def new_finfo(self):
        """
        add new file structure
        """

        # get file ID
        fid = self.num_of_file
        self.num_of_file += 1

        # initialize
        header: HeaderInfo = {
            "fname": "",
            "description": "",
            "note": "",
            "fmode": -1,
            "endiantype": FIO_UNKNOWN_ENDIAN,
            "grid_topology": -1,
            "glevel": -1,
            "rlevel": -1,
            "num_of_rgn": 0,
            "rgnid": [],
            "num_of_data": 0,
        }
        status: StatusInfo = {"rwmode": -1, "opened": 0, "fp": None, "eoh": 0}
        finfo: FileInfo = {"header": header, "dinfo": [], "status": status}
        self.finfo.append(finfo)

        return fid

    def register_file(self, fname: str):
        """
        register new file
        """
        fid = self.new_finfo()
        self.finfo[fid]["header"]["fname"] = fname
        return fid

    def fopen(self, fid: int, mode: int):
        """
        open file IO stream
        """
        finfo = self.finfo[fid]
        fname = finfo["header"]["fname"]
        if finfo["status"]["opened"]:
            logger.info(
                f"File {fname} has been already opened! Open process will be skipped!"
            )
            return
        if mode == FIO_FWRITE:
            finfo["status"]["fp"] = open(fname, "wb")
        elif mode == FIO_FREAD:
            finfo["status"]["fp"] = open(fname, "rb")
        elif mode == FIO_FAPPEND:
            finfo["status"]["fp"] = open(fname, "ab")
        finfo["status"]["rwmode"] = mode
        finfo["status"]["opened"] = 1

    def fclose(self, fid: int):
        """
        close file IO stream
        """
        finfo = self.finfo[fid]
        finfo["status"]["fp"].close()
        finfo["status"]["opened"] = 0

    def seek_datainfo(self, fid: int, varname: str, step: int) -> int:
        """
        seek data id by varname and step
        """
        finfo = self.finfo[fid]
        for did in range(finfo["header"]["num_of_data"]):
            dinfo = finfo["dinfo"][did]
            if dinfo["varname"] == varname and dinfo["step"] == step:
                return did
        logger.error(f"Data not found: varname={varname} and step={step}")
        raise FIOError()

    def read_data(self, fid: int, did: int):
        """
        read data array (full size)
        """
        finfo = self.finfo[fid]
        datasize: int = finfo["dinfo"][did]["datasize"]
        fp = finfo["status"]["fp"]
        fp.seek(finfo["status"]["eoh"], os.SEEK_SET)
        pos: int = 0
        for i in range(did):
            pos += dinfosize + finfo["dinfo"][i]["datasize"]
        pos += dinfosize
        fp.seek(pos, os.SEEK_CUR)
        if self.common["endiantype"] == FIO_LITTLE_ENDIAN:
            endian = "<"
        else:
            endian = ">"
        datatype: int = finfo["dinfo"][did]["datatype"]
        if datatype == FIO_REAL4:
            dtype = "f4"
        elif datatype == FIO_REAL8:
            dtype = "f8"
        elif datatype == FIO_INTEGER4:
            dtype = "i4"
        elif datatype == FIO_INTEGER8:
            dtype = "i8"
        return np.frombuffer(fp.read(datasize), dtype=f"{endian}{dtype}")

    def read_pkginfo(self, fid: int):
        """
        read package information
        """
        endian: typing.Literal["little", "big"]
        if self.common["endiantype"] == FIO_LITTLE_ENDIAN:
            endian = "little"
        else:
            endian = "big"

        finfo = self.finfo[fid]
        if finfo["status"]["opened"] == 0:
            logger.error(f"{finfo['header']['fname']} is not open!")
            raise FIOError()

        fp = finfo["status"]["fp"]
        fp.seek(0, os.SEEK_SET)
        header = finfo["header"]
        header["description"] = bytes_to_str(fp.read(sizeof(c_char) * FIO_HMID))
        header["note"] = bytes_to_str(fp.read(sizeof(c_char) * FIO_HLONG))
        header["fmode"] = int.from_bytes(fp.read(sizeof(c_int32)), endian)
        header["endiantype"] = int.from_bytes(fp.read(sizeof(c_int32)), endian)
        header["grid_topology"] = int.from_bytes(fp.read(sizeof(c_int32)), endian)
        header["glevel"] = int.from_bytes(fp.read(sizeof(c_int32)), endian)
        header["rlevel"] = int.from_bytes(fp.read(sizeof(c_int32)), endian)
        header["num_of_rgn"] = int.from_bytes(fp.read(sizeof(c_int32)), endian)
        rgnid = []
        for _ in range(header["num_of_rgn"]):
            rgnid.append(int.from_bytes(fp.read(sizeof(c_int32)), endian))
        header["rgnid"] = np.array(rgnid)
        header["num_of_data"] = int.from_bytes(fp.read(sizeof(c_int32)), endian)
        finfo["status"]["eoh"] = fp.tell()

    def read_datainfo(self, fid: int):
        """
        read data information
        """
        endian: typing.Literal["little", "big"]
        if self.common["endiantype"] == FIO_LITTLE_ENDIAN:
            endian = "little"
        else:
            endian = "big"

        finfo = self.finfo[fid]
        if finfo["status"]["opened"] == 0:
            logger.error(f"{finfo['header']['fname']} is not open!")
            raise FIOError()

        fp = finfo["status"]["fp"]
        fp.seek(finfo["status"]["eoh"], os.SEEK_SET)
        pos: int = 0
        for _ in range(finfo["header"]["num_of_data"]):
            fp.seek(pos, os.SEEK_CUR)
            varname = bytes_to_str(fp.read(sizeof(c_char) * FIO_HSHORT))
            description = bytes_to_str(fp.read(sizeof(c_char) * FIO_HMID))
            unit = bytes_to_str(fp.read(sizeof(c_char) * FIO_HSHORT))
            layername = bytes_to_str(fp.read(sizeof(c_char) * FIO_HSHORT))
            note = bytes_to_str(fp.read(sizeof(c_char) * FIO_HLONG))
            datasize = int.from_bytes(fp.read(sizeof(c_int64)), endian)
            datatype = int.from_bytes(fp.read(sizeof(c_int32)), endian)
            num_of_layer = int.from_bytes(fp.read(sizeof(c_int32)), endian)
            step = int.from_bytes(fp.read(sizeof(c_int32)), endian)
            time_start = int.from_bytes(fp.read(sizeof(c_int64)), endian)
            time_end = int.from_bytes(fp.read(sizeof(c_int64)), endian)
            dinfo: DataInfo = {
                "varname": varname,
                "description": description,
                "unit": unit,
                "layername": layername,
                "note": note,
                "datasize": datasize,
                "datatype": datatype,
                "num_of_layer": num_of_layer,
                "step": step,
                "time_start": time_start,
                "time_end": time_end,
            }
            finfo["dinfo"].append(dinfo)
            pos = dinfo["datasize"]  # skip data array

    def valid_pkginfo(self, fid: int):
        """
        validate package information with common
        """
        header = self.finfo[fid]["header"]
        common = self.common
        if header["grid_topology"] != common["grid_topology"]:
            logger.warn(
                f'grid_topology is not match, {header["grid_topology"]}, {common["grid_topology"]}'
            )
        if header["glevel"] != common["glevel"]:
            logger.warn(f'glevel is not match, {header["glevel"]}, {common["glevel"]}')
        if header["rlevel"] != common["rlevel"]:
            logger.warn(f'rlevel is not match, {header["rlevel"]}, {common["rlevel"]}')
        if header["num_of_rgn"] != common["num_of_rgn"]:
            logger.warn(
                f'num_of_rgn is not match, {header["num_of_rgn"]}, {common["num_of_rgn"]}'
            )
        for i in range(header["num_of_rgn"]):
            if header["rgnid"][i] != common["rgnid"][i]:
                logger.warn(
                    f'rgnid[{i}] is not match, {header["rgnid"][i]}, {common["rgnid"][i]}'
                )

    def valid_datainfo(self, fid: int):
        """
        validate data size
        """
        finfo = self.finfo[fid]
        header = finfo["header"]
        ijall: int = (2 ** (header["glevel"] - header["rlevel"]) + 2) ** 2
        for did in range(header["num_of_data"]):
            dinfo = finfo["dinfo"][did]
            datasize = (
                ijall
                * dinfo["num_of_layer"]
                * header["num_of_rgn"]
                * precision[dinfo["datatype"]]
            )
            if dinfo["datasize"] != datasize:
                text: str = ""
                text += f'datasize is not match {dinfo["datasize"]}. '
                text += f"datasize must be {ijall}[grid]x"
                text += f'{dinfo["num_of_layer"]}[layer]x'
                text += f'{header["num_of_rgn"]}[region]x'
                text += f'{precision[dinfo["datatype"]]}='
                text += f"{datasize}"
                logger.warn(text)

    def read_allinfo(self, fid: int):
        """
        read pkginfo and datainfo
        """
        self.read_pkginfo(fid)
        self.valid_pkginfo(fid)
        self.read_datainfo(fid)
        self.valid_datainfo(fid)


def bytes_to_str(b: bytes):
    return b.decode("ascii").rstrip("\x00")


class FIO_PaNDa:
    """
    FIO wrapper
    note one instance use one model output set
    """

    def __init__(self, glevel: int, rlevel: int, run_pe: int = 0):
        """
        glevel rlevel info
        note need to change mpi4py for furture
        """
        self.fio = FIO()
        self.glevel = glevel
        self.rlevel = rlevel
        rgn_tot = nico_rgn(rlevel)
        if run_pe == 0:
            self.rgn = int(1)
        else:
            self.rgn = int(rgn_tot / run_pe)
        if run_pe == 0:
            petot = rgn_tot
        else:
            petot = run_pe
        self.fid = -np.ones([petot], dtype=np.int32)
        self.closed = 0

    def __del__(self):
        try:
            if self.closed != 1:  # close
                self.close_file()
        except AttributeError:
            pass

    def put_info(self, l: int):
        MNG_prc_tab = np.empty([self.rgn], dtype=np.int32)
        MNG_prc_tab[:] = np.arange(0, self.rgn) + self.rgn * l
        self.fio.put_commoninfo(
            FIO_SPLIT_FILE,
            FIO_BIG_ENDIAN,
            FIO_ICOSAHEDRON,
            self.glevel,  # glevel
            self.rlevel,  # rlevel
            self.rgn,  # number of manage region
            MNG_prc_tab,
        )

    def ll_info(self, l: int) -> np.ndarray:
        MNG_prc_tab = np.empty([self.rgn], dtype=np.int32)
        MNG_prc_tab[:] = np.arange(0, self.rgn) + self.rgn * l
        return MNG_prc_tab

    def open_file(self, filename: str, l: int):
        if l >= np.size(self.fid):
            raise FIOError(f"cannot regist file {filename}! Check rlevel")
        fid = self.fio.register_file(filename)
        self.fid[l] = fid
        self.fio.fopen(fid, FIO_FREAD)
        self.fio.read_allinfo(fid)

    def close_file(self):
        n = np.size(self.fid)
        for l in range(n):
            fid = int(self.fid[l])
            if fid == -1:
                pass
            else:
                self.fio.fclose(fid)
        self.closed = 1

    def get_var(
        self,
        varname: str,
        l: int,
        kall: int = 1,
        selectlev: int = 0,
        tall: int = 1,
        selectstep: int = 0,
    ) -> np.ndarray:
        gall, _ = nico_gall(self.glevel, self.rlevel)

        if selectlev == -1:  # all level
            if selectstep == -1:  # all step
                var = np.empty([tall, self.rgn, kall, gall], dtype=np.float32)
            else:  # selectstep
                var = np.empty([self.rgn, kall, gall], dtype=np.float32)
        else:  # select level
            if selectstep == -1:
                var = np.empty([tall, self.rgn, gall], dtype=np.float32)
            else:
                var = np.empty([self.rgn, gall], dtype=np.float32)

        fid = int(self.fid[l])
        if selectstep == -1:
            for nt in range(tall):
                did = self.fio.seek_datainfo(fid, varname, nt + 1)
                var_tmp = self.fio.read_data(fid, did).reshape([self.rgn, kall, gall])
                if selectlev == -1:
                    var[nt, :, :, :] = var_tmp
                else:
                    var[nt, :, :] = var_tmp[:, selectlev, :]
        else:
            did = self.fio.seek_datainfo(fid, varname, selectstep + 1)
            var_tmp = self.fio.read_data(fid, did).reshape([self.rgn, kall, gall])
            if selectlev == -1:
                var[:, :, :] = var_tmp
            else:
                var[:, :] = var_tmp[:, selectlev, :]

        del var_tmp
        return var
