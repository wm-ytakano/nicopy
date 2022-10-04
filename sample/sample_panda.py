import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs

from nicopy.grids import LegacyGrids
from nicopy.reader import PandaReader
from nicopy.util import calc_lall

# create fig and ax
projection = ccrs.Orthographic(central_latitude=0.0, central_longitude=0.0)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection=projection)

glevel = 5
rlevel = 1
run_pe = 8
lall = calc_lall(rlevel)
num_of_rgn = int(lall / run_pe)

for pe in range(run_pe):
    # read data
    kall = 1
    reader = PandaReader(f"./testdata/data_panda_8PE/history.pe{pe:06d}")
    step = 1
    k = 0
    sa_lwu_toa = reader.read_pe("sa_lwu_toa", step, k)

    for rgn in range(num_of_rgn):
        # read grids
        l = num_of_rgn * pe + rgn
        grids = LegacyGrids(glevel, rlevel, f"./testdata/grid/grid.rgn{l:05d}")
        lon_v, lat_v = grids.get_lonlat_v()
        lonlat_v = np.stack([lon_v, lat_v], axis=-1)
        ncells, nv, _ = lonlat_v.shape

        # draw grid
        cmap = plt.cm.jet
        norm = mcolors.Normalize(vmin=100, vmax=300)
        gscolors = [cmap(norm(r)) for r in sa_lwu_toa[rgn, :]]
        for i in range(ncells):
            poly = plt.Polygon(lonlat_v[i], fc=gscolors[i], transform=ccrs.Geodetic())
            ax.add_patch(poly)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

fig.colorbar(sm, ax=ax)  # plot colorbar
ax.set_global()
ax.coastlines()
ax.gridlines()

# save and close
plt.savefig("sample_panda.png")
plt.clf()
plt.close()
