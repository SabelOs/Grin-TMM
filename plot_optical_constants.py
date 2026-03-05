#%%
import numpy as np
import matplotlib.pyplot as plt

# Path to file
file_path = "./OpticalConstants/nk_Cu2O.txt"

# Load data (handles CRLF or LF automatically)
# Assuming tab-separated columns: Wavelength \t n \t k
data = np.loadtxt(file_path)

wavelength = data[:, 0]
n = data[:, 1]
k = data[:, 2]

# Plot
plt.figure()
plt.plot(wavelength, n, label='n')
plt.plot(wavelength, k, label='k')

plt.xlabel('Wavelength')
plt.ylabel('Optical Constants')
plt.title('Optical Constants of Cu2O')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
# %%
