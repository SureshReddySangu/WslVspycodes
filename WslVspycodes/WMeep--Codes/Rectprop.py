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
period_inx = 4*2*a
resl_inx = 2*64
period_intime = 4*2*np.pi*a
resl_intime =128
t_values = np.linspace(0, 2*period_intime, resl_intime)
x_values = np.linspace(0, period_inx*1, resl_inx)

#time Evolution
# for x in x_values:
#     for t in t_values:
#         Ez_list.append(Ez[64,:]*(np.exp(1j*K_0*x)*np.exp(1j*t)))
#         Ey_list.append(Ey[64,:]*(np.exp(1j*K_0*x)*np.exp(1j*t)))
#         Ex_list.append(Ex[64,:]*(np.exp(1j*K_0*x)*np.exp(1j*t)))
# Ez_list = np.array(Ez_list)
# Ey_list = np.array(Ey_list)
# Ex_list = np.array(Ex_list)

# Iz_list = np.abs(Ez_list)**2
# Iy_list = np.abs(Ey_list)**2
# Ix_list = np.abs(Ex_list)**2
# I_tot_list = Iz_list + Iy_list +Ix_list
# X, Y = np.meshgrid(128*np.linspace(0,1, 128), 640*np.linspace(0,1,640))
# fig, ax  = plt.subplots()
# Ezplt = ax.imshow(np.real(Ez_list), cmap= 'RdBu', aspect='equal')
# ax.quiver(X,Y, np.real(Ex_list), color = 'green', scale = 10)
# ax.set_title('propagating Ez filed ')
# plt.colorbar(Ezplt, ax = ax)
# # from matplotlib.animation import FuncAnimation
# # def update(frame):
# #     Ezplt.set_data(np.real(Ez_list[frame:]))
# #     return[Ezplt]
# # anim = FuncAnimation(fig, update, frames=(len(t_values)))
# plt.show()






# longitudanal
for x in x_values:
    Ez_list.append(Ez[64,:]*(np.exp(1j*K_0*x)))
    Ey_list.append(Ey[64,:]*(np.exp(1j*K_0*x)))
    Ex_list.append(Ex[64,:]*(np.exp(1j*K_0*x)))
Ez_list = np.array(Ez_list)
Ey_list = np.array(Ey_list)
Ex_list = np.array(Ex_list)

Iz_list = np.abs(Ez_list)**2
Iy_list = np.abs(Ey_list)**2
Ix_list = np.abs(Ex_list)**2
I_tot_list = Iz_list + Iy_list +Ix_list
X, Y = np.meshgrid(resl_inx*np.linspace(0,1,resl_inx), 
                   resl_inx*np.linspace(0,1, resl_inx))
X1, Y1 = np.meshgrid(np.linspace(0, period_inx, resl_inx), [0])
fig, ax  = plt.subplots(2,3)
Ezplt_v = ax[0,0].imshow(np.real(Ez_list), cmap= 'RdBu', aspect='equal')
ax[0,0].quiver(X, np.real(Ez_list), color = 'green', scale = 5)
ax[0,0].set_title('propagating Ez filed ')
plt.colorbar(Ezplt_v, ax = ax[0,0])

Eyplt = ax[0,1].imshow(np.real(Ey_list), cmap= 'RdBu', aspect='equal')
ax[0,1].quiver(X, np.real(Ey_list), color = 'green', scale = 15)
ax[0,1].set_title('propagating Ey filed ')
plt.colorbar(Eyplt, ax = ax[0,1])

Explt = ax[1,0].imshow(np.real(Ex_list), cmap= 'RdBu', aspect='equal')
ax[1,0].quiver(X, np.real(Ex_list), color = 'green', scale = 15)
ax[1,0].set_title('propagating Ex filed ')
plt.colorbar(Explt, ax = ax[1,0])

Ezplt = ax[1,1].imshow(np.real(Ez_list), cmap= 'RdBu', aspect='equal')
ax[1,1].set_title('propagating Ez filed ')
plt.colorbar(Ezplt, ax = ax[1,1])

Ezplt_ani = ax[0,2].imshow(np.real(Ez_list), cmap= 'RdBu', aspect='equal')
ax[0,2].set_title('propagating Ez ani filed ')
plt.colorbar(Ezplt_ani, ax = ax[0,2])

Ezplt_Vect = ax[1,2].quiver(X1, Y1, np.zeros_like(np.real(Ez_list[0,:])), np.zeros_like(np.real(Ez_list[0,:])), color='green', scale=15)
ax[1,2].set_title('Propagating Ez vector field')
plt.colorbar(Ezplt_Vect, ax =ax[1,2])
from matplotlib.animation import FuncAnimation
def update(frame):
    mask1= np.zeros_like(np.real(Ez_list))
    mask1[frame,:] = np.real(Ez_list[frame,:])
    Ezplt_ani.set_array(mask1)
        # Update the quiver plot
    U = np.real(Ez_list[frame, :])
    V = np.zeros_like(U)  # For 1D propagation, V component can be 0
    Ezplt_Vect.set_UVC(U, V)

    return[Ezplt_ani, Ezplt_Vect]
anim = FuncAnimation(fig, update, frames=(len(t_values)))

plt.tight_layout()
plt.show()