#%%
import numpy as np
import matplotlib.pyplot as plt

# Path to file
file_path = "./OpticalConstants/nk_Cu2O.txt"
file_path_2 = "./OpticalConstants/nk_CuO.txt"
# Load data (handles CRLF or LF automatically)
# Assuming tab-separated columns: Wavelength \t n \t k
data = np.loadtxt(file_path)
data_CuO = np.loadtxt(file_path_2)
wavelength = data[:, 0]
n_Cu2O = data[:, 1]
k_Cu2O = data[:, 2]
n_CuO = data_CuO[:,1]
k_CuO = data_CuO[:,2]
# Plot
plt.figure()
plt.plot(wavelength, n_Cu2O, label='n')
plt.plot(wavelength, k_Cu2O, label='k')

plt.xlabel('Wavelength')
plt.ylabel('Optical Constants')
plt.title('Optical Constants of Cu2O')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
# %%
# Plot
mask =  n_Cu2O > n_CuO

plt.figure()
plt.plot(wavelength, n_Cu2O, label='$\mathrm{Cu_2O}$')
plt.plot(wavelength, n_CuO, label='CuO')
plt.fill_between(wavelength, n_CuO, n_Cu2O, color='skyblue', alpha=0.4, where= mask)
plt.fill_between(wavelength, n_CuO, n_Cu2O, color='orange', alpha=0.4, where= ~mask)
plt.xlabel('Wavelength / nm',fontsize=14)
plt.ylabel('Refractive index n',fontsize=14)
#plt.title('Optical Constants of Cu2O')
plt.legend(fontsize=14)
plt.grid(True)
plt.xlim([400,1000])
plt.ylim([2.2,3.2])
plt.tight_layout()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("Refractive_index_copper_oxides.png")
plt.show()
# %%
