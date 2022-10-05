import time
import argparse

import multiprocessing
import functools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs

from nicopy.grids import LegacyGrids
from nicopy.reader import LegacyReader
from nicopy.util import calc_lall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mp", type=int, help="use multiprocessing")
    args = parser.parse_args()

    glevel = 5
    rlevel = 1
    lall = calc_lall(rlevel)

    start_time = time.perf_counter()
    if args.mp:
        with multiprocessing.Pool(processes=args.mp) as pool:
            data = pool.map(
                functools.partial(read_data, glevel, rlevel),
                range(lall),
            )
    else:
        data = []
        for l in range(lall):
            data.append(read_data(glevel, rlevel, l))
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

    plt.savefig("sample_legacy.png")
    plt.clf()
    plt.close()
    end_time = time.perf_counter()
    print("draw:", end_time - start_time)


def read_data(glevel: int, rlevel: int, l: int):
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


if __name__ == "__main__":
    main()
