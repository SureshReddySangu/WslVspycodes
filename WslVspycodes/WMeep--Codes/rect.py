import meep as mp
from meep import mpb
import matplotlib.pyplot as plt
import numpy as np

num_bands = 4
resolution = 32
n_lo =1.99

geometry_lattice = mp.Lattice(size=mp.Vector3(0,4,4),
                              basis1=mp.Vector3(1, 0))  # 1d cell

k_points = [mp.Vector3(kx) for kx in np.linspace(0, 0.5, 5)]
default_material = mp.Medium(index=1)
geometry =[ mp.Block(
    size=mp.Vector3(mp.inf, 0.2,0.5),
    center=mp.Vector3(),
    material=mp.Medium(index=n_lo),
)]

ms = mpb.ModeSolver(
    geometry=geometry,
    geometry_lattice=geometry_lattice,
    k_points=k_points,
    num_bands=num_bands,
    resolution=resolution,
    default_material= default_material
)

ms.run()

t_frqs = ms.all_freqs
gaps = ms.gap_list


fig, ax = plt.subplots(1, 1)

for i, tmz, in zip(range(len(t_frqs)), t_frqs):
    ax.scatter([i] * len(tmz), tmz, color="blue", s=0.2, label="TE")
    
# Calculate and plot the light line
kx_vals = np.linspace(0, 1, len(k_points))
light_line = np.abs(kx_vals)  # Normalized units, slope = 1
ax.plot(range(len(k_points)), light_line, 'k--', label="Light Line")
ax.plot(t_frqs, color="blue", label="TM")

ax.set_xlabel("k-points")
ax.set_ylabel("Frequency")
ax.set_title("TM Frequencies & TE Frequencies")

for gap in gaps:
    if gap[0] > 1:
        ax.fill_between(range(len(k_points)), gap[1], gap[2], color="blue", alpha=0.5)
for gap in gaps:
    if gap[0] > 1:
        ax.fill_between(range(len(k_points)), gap[1], gap[2], color="green", alpha=0.5)
ax.grid(True)

plt.show()
ms.compute_zparities()
ms.compute_yparities()

# plt.show()
mpb.output_efield(ms,2)
import h5py

with h5py.File('rect-e.k05.b02.h5', 'r') as file:
    data0 = file['x.i'][:]
    data1 = file['x.r'][:]
    data2 = file['y.i'][:]
    data3 = file['y.r'][:]
    data4 = file['z.i'][:]
    data5 = file['z.r'][:]
Ex_i = data0[:128]
Ex_r = data1[:128]
Ey_i = data2[:128]
Ey_r = data3[:128]
Ez_i = data4[:128]
Ez_r = data5[:128]

Ex = Ex_r + 1j*Ex_i
Ey = Ey_r + 1j*Ey_i
Ez = Ez_r + 1j*Ez_i

I =abs(Ex)**2 + abs(Ey)**2 + abs(Ez)**2
fig= plt.imshow(I/np.max(I), cmap='hot' )
plt.colorbar()
plt.show()

x,y = np.meshgrid(np.linspace(0,1,128),np.linspace(0,1,128))
plt.quiver(x,y,Ez_r, Ey_r, color='red', scale = 15)
eps11=ms.get_epsilon()
plt.imshow(eps11, cmap='gray', alpha=0.2)
plt.title('Ez + Ey')
plt.ylim(0.4,0.6)
plt.xlim(0.4,0.65)
plt.colorbar()




# fig, ax = plt.subplots(3, 3, gridspec_kw={ 'height_ratios': [1, 1, 2]})
# im1 = ax[0, 0].imshow(Ex_i , cmap='RdBu')
# ax[0, 0].set_title(' Ex')
# x, y = np.meshgrid(np.linspace(0, 1, 128), np.linspace(0, 1, 128))
# ax[0, 0].quiver(x, y,Ex_i, color='r')
# fig.colorbar(im1, ax=ax[0, 0])

# im2 = ax[0, 1].imshow(Ey_i , cmap='RdBu')
# ax[0, 1].set_title('Ey')
# fig.colorbar(im2, ax=ax[0, 1])

# im3 = ax[0, 2].imshow(Ez_i * Ez_r, cmap='RdBu')
# ax[0, 2].set_title(' Ez')
# fig.colorbar(im3, ax=ax[0, 2])

# im5 = ax[1, 0].imshow(Ex_r, cmap='RdBu')
# ax[1, 0].set_title('Ex_r')
# fig.colorbar(im5, ax=ax[1, 0])

# im6 = ax[1, 1].imshow(Ey_r, cmap='RdBu')
# ax[1, 1].set_title('Ey_r')
# fig.colorbar(im6, ax=ax[1, 1])

# im7 = ax[1, 2].imshow(Ez_r, cmap='RdBu')
# ax[1, 2].set_title('Ez_r')
# fig.colorbar(im7, ax=ax[1, 2])

# im4 = ax[2, 0].imshow(I, cmap='RdBu')
# ax[2, 0].set_title('I')
# fig.colorbar(im4, ax=ax[2, 0])

plt.tight_layout()

plt.show()