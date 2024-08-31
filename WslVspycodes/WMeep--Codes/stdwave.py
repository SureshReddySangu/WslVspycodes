import meep as mp
from meep import mpb
import numpy as np
import matplotlib.pyplot as plt
num_bands = 4
resolution=32
p_gap = 1.0
width =0.5
n_lo = 1.99
geometric_lattice = mp.Lattice(
    size = mp.Vector3(0,5,5),
    basis1= mp.Vector3(1,0,0)
)

geometry = [
    mp.Block(
        size=mp.Vector3(mp.inf, 0.2, width),
        center = mp.Vector3(z = (width+p_gap)/2),
        material = mp.Medium(index=n_lo)

    ),
    mp.Block(
        size=mp.Vector3(mp.inf, 0.2, width),
        center = mp.Vector3(z=-(width+p_gap)/2 ),
        material = mp.Medium(index= n_lo)
    )
]

k_points = [mp.Vector3(kx) for kx in np.linspace(0,0.5,5)]
default_material  = mp.Medium(index= 1)
ms = mpb.ModeSolver(
    geometry=geometry,
    geometry_lattice = geometric_lattice,
    k_points=k_points,
    num_bands=num_bands,
    resolution=resolution,
    default_material=default_material
)

ms.run()
t_freq = ms.all_freqs
fig,ax = plt.subplots()
for i , tm in zip(range(len(t_freq)), t_freq):
    ax.scatter([i]*len(tm), tm, color ='red')
ax.plot(t_freq, color= 'blue')
ax.grid(True)
plt.show()
eps11 = ms.get_epsilon()
print(eps11.shape)
plt.imshow(eps11, cmap='gray')
plt.colorbar()
plt.show()
ms.compute_zparities()
ms.compute_yparities()
mpb.output_efield(ms,1)
mpb.display_zparities(ms)
mpb.display_yparities(ms)

import h5py

with h5py.File('stdwave-e.k05.b01.h5', 'r') as file:
    data0 = file['x.i'][:]
    data1 = file['x.r'][:]
    data2 = file['y.i'][:]
    data3 = file['y.r'][:]
    data4 = file['z.i'][:]
    data5 = file['z.r'][:]
Ex_i = data0[:160]
Ex_r = data1[:160]
Ey_i = data2[:160]
Ey_r = data3[:160]
Ez_i = data4[:160]
Ez_r = data5[:160]

Ex = Ex_r + 1j*Ex_i
Ey = Ey_r+ 1j*Ey_i
Ez = Ez_r + 1j*Ez_i

I = np.abs(Ex)**2 +np.abs(Ey)**2 +np.abs(Ez)**2
plt.imshow(I/np.max(I), cmap='hot')
plt.colorbar()
plt.show()
kx = k_points[-1].x
Ex_neg = Ex*np.exp(-1j*kx)
Ey_neg = Ey*np.exp(-1j*kx)
Ez_neg = Ez*np.exp(-1j*kx)

I_neg = abs(Ex_neg)**2 +abs(Ey_neg)**2+abs(Ez_neg)**2

plt.imshow(I_neg/np.max(I_neg), cmap='hot')
plt.colorbar()
plt.title('inteisty in -x')
plt.show()


Ex_total = Ex+Ex_neg
Ey_total = Ey+Ey_neg
Ez_total = Ez+Ez_neg

I_total = abs(Ex_total)**2 +abs(Ey_total)**2 +abs(Ez_total)**2
plt.imshow(I_total/np.max(I_total), cmap='hot')
plt.colorbar()
plt.title('total inteisty')
plt.show()














#chat gpt code
# x_coords = np.linspace(0, 1, I_total.shape[0])  # Adjust based on your coordinate system

# # Average intensity along the y-direction to get the profile along x
# I_total_x_profile = np.mean(I_total, axis=1)

# # Plot the intensity profile along the x-axis
# plt.plot(x_coords, I_total_x_profile, label="Intensity along x-axis")
# plt.xlabel("x")
# plt.ylabel("Intensity")
# plt.title("Intensity Profile along the x-axis")
# plt.grid(True)
# plt.legend()
# plt.show()









