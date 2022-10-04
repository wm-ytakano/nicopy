import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs

from nicopy.grids import LegacyGrids
from nicopy.reader import LegacyReader
from nicopy.util import calc_lall

# create fig and ax
projection = ccrs.Orthographic(central_latitude=0.0, central_longitude=0.0)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection=projection)

glevel = 5
rlevel = 1
lall = calc_lall(rlevel)

for l in range(lall):
    # read grids
    grids = LegacyGrids(glevel, rlevel, f"./testdata/grid/grid.rgn{l:05d}")
    lon_v, lat_v = grids.get_lonlat_v()
    lonlat_v = np.stack([lon_v, lat_v], axis=-1)
    ncells, nv, _ = lonlat_v.shape

    # read data
    kall = 1
    reader = LegacyReader(
        glevel, rlevel, kall, f"./testdata/data_legacy/sa_t2m.rgn{l:05d}"
    )
    step = 0
    k = 0
    sa_t2m = reader.read_rgn(step, k) - 273.15

    # draw grid
    cmap = plt.cm.jet
    norm = mcolors.Normalize(vmin=0, vmax=30)
    gscolors = [cmap(norm(r)) for r in sa_t2m]
    for i in range(ncells):
        poly = plt.Polygon(lonlat_v[i], fc=gscolors[i], transform=ccrs.Geodetic())
        ax.add_patch(poly)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

fig.colorbar(sm, ax=ax)  # plot colorbar
ax.set_global()
ax.coastlines()
ax.gridlines()

# save and close
plt.savefig("sample_legacy.png")
plt.clf()
plt.close()
