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
ms.compute_zparities()
ms.compute_yparities()

mpb.output_efield(ms,1)
import h5py
with h5py.File('Rectprop-e.k05.b01.h5', 'r') as file:
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

a = 1 # period constant
K_0 = 0.5*(2*np.pi)/a #k_0 is the proapagation contstat in x direction
Ez_list = []
Ey_list = []
Ex_list =[]
t_values = np.linspace(0, 2*np.pi*a, 10)
x_values = np.linspace(0, 4*2*a, 1*64)
for i,t in enumerate(t_values):
    for x in x_values:
        Ez_list.append(Ez[64,:]*(np.exp(1j*K_0*x)*np.exp(1j*t)))
        Ey_list.append(Ey[64,:]*(np.exp(1j*K_0*x)*np.exp(1j*t)))
        Ex_list.append(Ex[64,:]*(np.exp(1j*K_0*x)*np.exp(1j*t)))
Ez_list = np.array(Ez_list)
Ey_list = np.array(Ey_list)
Ex_list = np.array(Ex_list)

Iz_list = np.abs(Ez_list)**2
Iy_list = np.abs(Ey_list)**2
Ix_list = np.abs(Ex_list)**2
I_tot_list = Iz_list + Iy_list +Ix_list

fig, ax  = plt.subplots()
Ezplt = ax.imshow(np.real(Ez_list), cmap= 'RdBu', aspect='equal')
ax.set_title('propagating Ez filed ')
plt.colorbar(Ezplt, ax = ax)
from matplotlib.animation import FuncAnimation
def update(frame):
    Ezplt.set_data(np.real(Ez_list[frame:]))
    return[Ezplt]
anim = FuncAnimation(fig, update, frames=(len(t_values)))
plt.show()