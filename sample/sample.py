import time
import argparse

import multiprocessing
import functools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs

from nicopy.grids import LegacyGrids
from nicopy.reader import LegacyReader, PandaReader
from nicopy.util import calc_lall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--format", default="legacy", help="ico format (legacy | panda)"
    )
    parser.add_argument("--mp", type=int, help="use multiprocessing")
    args = parser.parse_args()

    glevel = 5
    rlevel = 1
    run_pe = 8
    lall = calc_lall(rlevel)
    num_of_rgn = int(lall / run_pe)

    start_time = time.perf_counter()
    if args.format == "legacy":
        if args.mp:
            with multiprocessing.Pool(processes=args.mp) as pool:
                data = pool.map(
                    functools.partial(read_legacy, glevel, rlevel),
                    range(lall),
                )
        else:
            data = []
            for l in range(lall):
                data.append(read_legacy(glevel, rlevel, l))
    if args.format == "panda":
        if args.mp:
            with multiprocessing.Pool(processes=args.mp) as pool:
                r = pool.map(
                    functools.partial(read_panda, glevel, rlevel, num_of_rgn),
                    range(run_pe),
                )
            data = sum(r, [])
        else:
            data = []
            for pe in range(run_pe):
                data.extend(read_panda(glevel, rlevel, num_of_rgn, pe))
    end_time = time.perf_counter()
    print("read:", end_time - start_time)

    start_time = time.perf_counter()
    projection = ccrs.Orthographic(central_latitude=0.0, central_longitude=0.0)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=projection)

    cmap = plt.cm.jet
    norm = mcolors.Normalize(vmin=0, vmax=30)

    for lonlat_v, sa_t2m in data:
        ncells, _, _ = lonlat_v.shape
        gscolors = cmap(norm(sa_t2m))
        for i in range(ncells):
            poly = plt.Polygon(lonlat_v[i], fc=gscolors[i], transform=ccrs.Geodetic())
            ax.add_patch(poly)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax)

    ax.set_global()
    ax.coastlines()
    ax.gridlines()

    plt.savefig(f"sample_{args.format}.png")
    plt.clf()
    plt.close()
    end_time = time.perf_counter()
    print("draw:", end_time - start_time)


def read_legacy(glevel: int, rlevel: int, l: int):
    grids = LegacyGrids(glevel, rlevel, f"./testdata/grid/grid.rgn{l:05d}")
    lon_v, lat_v = grids.get_lonlat_v()
    lonlat_v = np.stack([lon_v, lat_v], axis=-1)

    kall = 1
    reader = LegacyReader(
        glevel, rlevel, kall, f"./testdata/data_legacy/sa_t2m.rgn{l:05d}"
    )
    step = 0
    k = 0
    sa_t2m = reader.read_rgn(step, k) - 273.15
    return lonlat_v, sa_t2m


def read_panda(glevel: int, rlevel: int, num_of_rgn: int, pe: int):
    reader = PandaReader(f"./testdata/data_panda_8PE/history.pe{pe:06d}")
    step = 1
    k = 0
    sa_t2m = reader.read_pe("sa_t2m", step, k) - 273.15
    output = []
    for rgn in range(num_of_rgn):
        # read grids
        l = num_of_rgn * pe + rgn
        grids = LegacyGrids(glevel, rlevel, f"./testdata/grid/grid.rgn{l:05d}")
        lon_v, lat_v = grids.get_lonlat_v()
        lonlat_v = np.stack([lon_v, lat_v], axis=-1)
        output.append((lonlat_v, sa_t2m[rgn]))
    return output


if __name__ == "__main__":
    main()
