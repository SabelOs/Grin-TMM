#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

times_1W = [150, 300, 480, 540]
diam_1W = [57.8, 114.73, 99.81, 132.12] 

times_2W = [20, 40, 60, 80, 100, 120]
diam_2W = [193.47, 219.33, 242.03, 265.42, 275.92, 279.09]

times_3W = [40, 60, 80, 100, 120]
diam_3W = [144.43, 152.89, 168.96, 179.09, 192.37]

fig, ax = plt.subplots()

# main plot
ax.plot(times_2W, diam_2W, marker='o', label="3 W")
ax.plot(times_3W, diam_3W, marker='s', label="2 W")

ax.set_ylabel("Lens Diameter / $\mu$m", fontsize=16)
ax.set_xlabel("Laser Duration / s", fontsize=16)

ax.tick_params(axis='both', labelsize=14)

ax.legend(fontsize=13)
##ax.set_ylim([140, 350])
# inset for 1W
#axins = inset_axes(ax, width="40%", height="40%", loc="upper left")

#axins.plot(times_1W, diam_1W, marker='o', color='tab:green', label="1 W")
#axins.set_title("1 W", fontsize=12)
#axins.tick_params(axis='both', labelsize=10)

plt.show()
# %%
