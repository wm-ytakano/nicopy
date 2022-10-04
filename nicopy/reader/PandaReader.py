import typing
import os
from ctypes import sizeof, c_char, c_int32, c_int64
import numpy as np
from logging import getLogger
from ..util import calc_gall

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
    rgnid: typing.Any


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


class PandaReader:
    """
    Python Migration of Advanced file I/O module (CORE)

    put/get indicates the communication with the database
    read/write indicates the communication with the file
    """

    def __init__(self, filename: str):
        header: HeaderInfo = {
            "fname": filename,
            "description": "",
            "note": "",
            "fmode": -1,
            "endiantype": FIO_UNKNOWN_ENDIAN,
            "grid_topology": -1,
            "glevel": -1,
            "rlevel": -1,
            "num_of_rgn": 0,
            "rgnid": typing.Any,
            "num_of_data": 0,
        }
        status: StatusInfo = {
            "rwmode": FIO_FREAD,
            "opened": 1,
            "fp": open(filename, "rb"),
            "eoh": 0,
        }
        self.finfo: FileInfo = {"header": header, "dinfo": [], "status": status}
        self.read_pkginfo()
        self.read_datainfo()

    def fclose(self):
        """
        close file IO stream
        """
        self.finfo["status"]["fp"].close()
        self.finfo["status"]["opened"] = 0

    def read_pkginfo(self):
        """
        read package information
        """
        endian = "big"
        finfo = self.finfo
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

    def read_datainfo(self):
        """
        read data information
        """
        endian = "big"
        finfo = self.finfo
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

    def read_pe(self, varname: str, step: int, k: int):
        """
        read data array

        Returns:
            NDArray of shape (rgn, gall)
        """
        finfo = self.finfo
        did = None
        for _did in range(finfo["header"]["num_of_data"]):
            dinfo = finfo["dinfo"][_did]
            if dinfo["varname"] == varname and dinfo["step"] == step:
                did = _did
                break
        if did is None:
            logger.error(f"Data not found: varname={varname} and step={step}")
            raise FIOError()

        fp = finfo["status"]["fp"]
        offset = finfo["status"]["eoh"]
        for i in range(did):
            offset += dinfosize + finfo["dinfo"][i]["datasize"]
        offset += dinfosize
        datasize = finfo["dinfo"][did]["datasize"]
        endian = ">"  # big endian
        datatype: int = finfo["dinfo"][did]["datatype"]
        if datatype == FIO_REAL4:
            dtype = "f4"
            size = int(datasize / 4)
        elif datatype == FIO_REAL8:
            dtype = "f8"
            size = int(datasize / 8)
        elif datatype == FIO_INTEGER4:
            dtype = "i4"
            size = int(datasize / 4)
        elif datatype == FIO_INTEGER8:
            dtype = "i8"
            size = int(datasize / 8)
        else:
            logger.error(f"Unsupported datatype {datatype}")
            raise FIOError()

        rgn = finfo["header"]["num_of_rgn"]
        kall = dinfo["num_of_layer"]
        gall, gall_1d = calc_gall(finfo["header"]["glevel"], finfo["header"]["rlevel"])
        shape = (rgn, kall, gall)
        if np.prod(shape) != size:
            logger.error(f"Input shape {shape} does not match data size {size}")
            raise FIOError()
        v_all = np.memmap(
            fp, dtype=f"{endian}{dtype}", mode="r", offset=offset, shape=shape
        )[:, k, :]
        return v_all.reshape((rgn, gall_1d, gall_1d))[
            :,
            1 : gall_1d - 1,
            1 : gall_1d - 1,
        ].reshape((rgn, (gall_1d - 2) ** 2))


def bytes_to_str(b: bytes):
    return b.decode("ascii").rstrip("\x00")
