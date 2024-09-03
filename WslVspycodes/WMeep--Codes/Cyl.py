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
mpb.output_at_kpoint(2, 1)

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
plt.quiver(x,y,Ez_r, Ey_r, color='red', scale = 15)
plt.title('Ez + Ey')

plt.colorbar()
plt.show()



Epx_list = []
Epy_list = []
Epz_list = []
Enx_list = []
Eny_list = []
Enz_list = []
Ip_list = []
In_list = []
I_total_list = []
x_dir= 10
for x in range(x_dir):
    Epx_list=Epx_list+list(Ex*np.exp(1j*x))
    Epy_list=Epy_list+list(Ey*np.exp(1j*x))
    Epz_list=Epz_list+list(Ez*np.exp(1j*x))
    Enx_list=Enx_list+list(Ex*np.exp(-1j*x))
    Eny_list=Eny_list+list(Ey*np.exp(-1j*x))
    Enz_list=Enz_list+list(Ez*np.exp(-1j*x))
# Epx_list = np.array(Epx_list)
# Enx_list = np.array(Enx_list)
# Epy_list = np.array(Epy_list)
# Eny_list = np.array(Eny_list)
# Epz_list = np.array(Epz_list)
# Enz_list = np.array(Enz_list)

# t = Epx_list+Enx_list
# print(t.shape)


I_t = np.abs(Epx_list+Enx_list)**2 +np.abs(Epy_list+Eny_list)**2 +np.abs(Epz_list+Enz_list)**2


# Ip_list = list(np.abs(Epx_list)**2 +np.abs(Epy_list)**2 +np.abs(Epz_list)**2)
# In_list =list(np.abs(Enx_list)**2 +np.abs(Eny_list)**2 +np.abs(Enz_list)**2)
x, y = np.meshgrid(1280*np.linspace(0,1,1280),128*np.linspace(0,1,128))
# print(Epx_list.shape)
# I_total_list = Ip_list+In_list
# I_total_list = np.array(I_total_list).transpose()
plt.imshow(I_t/np.max(I_t), cmap='hot')
# plt.quiver(x,y,np.real(Epz_list+Enz_list), np.real(Epy_list+Eny_list), color='green', scale=20)
plt.colorbar()
plt.title('total inteisty_Itotla_list')
plt.show()




















# x_dir =[x_dir[0] for x_dir in np.array(k_points)]
# for x_dir in x_dir:
#     # Calculate the components of the standing wave
#     zc_wave_in_px = Ez * np.exp(1j * 1* x_dir)
#     zc_wave_in_nx = Ez * np.exp(-1j *1 * x_dir)
#     t_wave_inz = zc_wave_in_px + zc_wave_in_nx
    
#     xc_wave_in_px = Ex * np.exp(1j * 1 * x_dir)
#     xc_wave_in_nx = Ex * np.exp(-1j * 1 * x_dir)
#     t_wave_inx = xc_wave_in_px + xc_wave_in_nx

#     yc_wave_in_px = Ey * np.exp(1j * 1 * x_dir)
#     yc_wave_in_nx = Ey * np.exp(-1j * 1 * x_dir)
#     t_wave_iny = yc_wave_in_px + yc_wave_in_nx
    
#     z_total_re = np.real(t_wave_inz)
#     y_total_re = np.real(t_wave_iny)

#     # Calculate the standing wave intensity
#     I_std = abs(Ex + t_wave_inx)**2 + abs(Ey + t_wave_iny)**2 + abs(Ez + t_wave_inz)**2

#     # Plot the intensity
#     plt.title(f'Standing Wave Intensity for x_dir = {x_dir:.2f}')
#     plt.imshow(I_std/np.max(I_std), cmap='hot')
#     plt.colorbar()
#     plt.show()

#     # Plot the quiver plot with updated fields
#     plt.title(f'Standing Wave Field for x_dir = {x_dir:.2f}')
#     plt.quiver(x, y, z_total_re + Ez_r, Ey_r + y_total_re, color='red', scale=15)
#     plt.imshow(eps11, alpha=0.5)
#     plt.colorbar()
#     plt.show()
























# zc_wave_in_px = Ez*np.exp(1j*0.5*x_dir)
# zc_wave_in_nx = Ez*np.exp(-1j*0.5*x_dir) 
# t_wave_inz = zc_wave_in_px + zc_wave_in_nx
    
# xc_wave_in_px = Ex*np.exp(1j*0.5*x_dir)
# xc_wave_in_nx = Ex*np.exp(-1j*0.5*x_dir)
# t_wave_inx = xc_wave_in_px+xc_wave_in_nx

# yc_wave_in_px = Ey*np.exp(1j*0.5*x_dir)
# yc_wave_in_nx = Ey*np.exp(-1j*0.5*x_dir)
# t_wave_iny = yc_wave_in_px+yc_wave_in_nx
# z_total_re = np.real(t_wave_inz)
# y_total_re = np.real(t_wave_iny)

# I_std = abs(Ex+t_wave_inx)**2+abs(Ey+t_wave_iny)**2+abs(Ez+t_wave_inz)**2
# plt.title('std Intensity')
# plt.imshow(I_std/np.max(I_std), cmap='hot')
# plt.colorbar()
# plt.show()

# plt.title('stdwave')
# plt.quiver(x,y, z_total_re+Ez_r, Ey_r+y_total_re, color= 'red', scale = 15)
# plt.imshow(eps11)
# plt.colorbar()
# plt.show()
