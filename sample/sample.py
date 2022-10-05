import time
import argparse
import warnings

import multiprocessing
import functools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import jet
import cartopy.crs as ccrs
from spatialpandas import GeoDataFrame
from spatialpandas.geometry import PolygonArray
import datashader as ds
import datashader.transfer_functions as tf

from nicopy.grids import LegacyGrids
from nicopy.reader import LegacyReader, PandaReader
from nicopy.util import calc_lall

warnings.simplefilter("ignore", matplotlib.MatplotlibDeprecationWarning)


def main():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--format", default="legacy", help="ico format (legacy | panda)"
    )
    parser.add_argument("--mp", type=int, help="use multiprocessing")
    parser.add_argument("--plot", default="mpl", help="plot library (mpl | ds)")
    args = parser.parse_args()

    # general settings
    glevel = 5
    rlevel = 1
    lall = calc_lall(rlevel)

    # read data
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
        run_pe = 8
        num_of_rgn = int(lall / run_pe)
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

    # plot figure
    start_time = time.perf_counter()
    projection = ccrs.Orthographic(central_latitude=0.0, central_longitude=0.0)
    if args.plot == "ds":
        outlines = []
        _values = []
        for lonlat_v, sa_t2m in data:
            ncells, _, _ = lonlat_v.shape
            for i in range(ncells):
                coords = projection.transform_points(
                    ccrs.Geodetic(), lonlat_v[i, :, 0], lonlat_v[i, :, 1]
                )
                points = coords[:, 0:2].ravel()
                outline = np.concatenate([points, points[0:2]])
                outlines.append([outline])
            _values.append(sa_t2m)
        values = np.concatenate(_values)
        polygons = PolygonArray(outlines)
        df = GeoDataFrame({"polygons": polygons, "values": values})

        vmin = 0
        vmax = 30
        cmap = jet
        cvs = ds.Canvas()
        agg = cvs.polygons(df, geometry="polygons", agg=ds.last("values"))
        shade = tf.shade(agg, cmap=cmap, how="linear", span=[vmin, vmax])

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=projection)

        left = projection.transform_point(-90, 0, ccrs.Geodetic())[0]
        right = projection.transform_point(90, 0, ccrs.Geodetic())[0]
        bottom = projection.transform_point(0, -90, ccrs.Geodetic())[1]
        top = projection.transform_point(0, 90, ccrs.Geodetic())[1]
        ax.imshow(shade.to_pil(), extent=(left, right, bottom, top))

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, ax=ax)

        ax.set_global()
        ax.coastlines()
        ax.gridlines()

        ax.set_title("matplotlib + datashader (imshow)")
        plt.savefig(f"sample_ds_{args.format}.png")
        plt.clf()
        plt.close()
    elif args.plot == "mpl":
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=projection)

        cmap = plt.cm.jet
        norm = mcolors.Normalize(vmin=0, vmax=30)

        for lonlat_v, sa_t2m in data:
            ncells, _, _ = lonlat_v.shape
            gscolors = cmap(norm(sa_t2m))
            for i in range(ncells):
                poly = plt.Polygon(
                    lonlat_v[i], fc=gscolors[i], transform=ccrs.Geodetic()
                )
                ax.add_patch(poly)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, ax=ax)

        ax.set_global()
        ax.coastlines()
        ax.gridlines()

        ax.set_title("matplotlib add_patch")
        plt.savefig(f"sample_mpl_{args.format}.png")
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
