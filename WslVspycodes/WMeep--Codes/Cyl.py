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
geometry = [
    mp.Cylinder(
        radius=0.5,
        axis=mp.Vector3(1, 0, 0),
        height=mp.inf,
        center=mp.Vector3(0, 0, 0),
        material=mp.Medium(index=n_lo),
    )
]
ms =mpb.ModeSolver(
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

eps11=ms.get_epsilon()
plt.imshow(eps11, cmap='RdBu', alpha=0.5)
plt.colorbar()
plt.show()

import h5py
mpb.output_efield(ms,1)

with h5py.File('Cyl-e.k05.b01.h5', 'r') as file:
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
fig= plt.imshow(I/np.max(I), cmap='hot')
plt.colorbar()
plt.show()
x,y = np.meshgrid(np.linspace(0,1,128),np.linspace(0,1,128))
plt.quiver(x,y,Ez_r, Ey_r, color='red', scale = 5)
plt.title('Ez + Ey')
plt.ylim(0.4,0.6)
plt.xlim(0.4,0.65)
