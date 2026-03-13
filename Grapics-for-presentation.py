# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Reproducible randomness
np.random.seed(42)

# Create parameter grid
x = np.linspace(-4, 4, 400)
y = np.linspace(-4, 4, 400)
X, Y = np.meshgrid(x, y)

# Base potential landscape
Z = (
    0.2*(X**2 + Y**2)
    + 0.8*np.sin(2*X)*np.cos(2*Y)
    + 0.3*np.sin(3*X + Y)
    + 0.3*np.cos(X - 2*Y)
)

# Add deeper global minimum
Z -= 3*np.exp(-((X+1.5)**2 + (Y-1.0)**2)/0.4)

# Add random noise (roughness)
noise_strength = 0.005
Z += noise_strength * np.random.normal(size=Z.shape)

# Find global minimum
min_index = np.unravel_index(np.argmin(Z), Z.shape)
xmin = X[min_index]
ymin = Y[min_index]
zmin = Z[min_index]

# -----------------------
# 2D contour plot
# -----------------------
plt.figure(figsize=(8,6))
contour = plt.contourf(X, Y, Z, levels=60)
plt.colorbar(contour)

plt.scatter(xmin, ymin, color="red", s=140, edgecolor="black", label="Global Minimum")

plt.xlabel("Parameter X", fontsize=22)
plt.ylabel("Parameter Y", fontsize=22)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend(fontsize=14)
plt.tight_layout()
plt.show()

# -----------------------
# 3D surface plot
# -----------------------
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    X, Y, Z,
    rstride=4, cstride=4,
    linewidth=0,
    antialiased=True
)

# Highlight global minimum
ax.scatter(xmin, ymin, zmin, color="red", s=120, edgecolor="black")

ax.set_xlabel("Parameter X", fontsize=18, labelpad=15)
ax.set_ylabel("Parameter Y", fontsize=18, labelpad=15)
ax.set_zlabel("RMSE", fontsize=18, labelpad=10, rotation = 90,)

ax.view_init(elev=15, azim=-70)
ax.tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.show()
