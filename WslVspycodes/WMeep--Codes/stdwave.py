import meep as mp
from meep import mpb
import numpy as np
import matplotlib.pyplot as plt
num_bands = 4
resolution=32
p_gap = 0.5
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

    )
    # mp.Block(
    #     size=mp.Vector3(mp.inf, 0.2, width),
    #     center = mp.Vector3(z=-(width+p_gap)/2 ),
    #     material = mp.Medium(index= n_lo)
    # )
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
# fig,ax = plt.subplots()
# for i , tm in zip(range(len(t_freq)), t_freq):
#     # ax.scatter([i]*len(tm), tm, color ='red')
# # ax.plot(t_freq, color= 'blue')
# # ax.grid(True)
# # plt.show()
# eps11 = ms.get_epsilon()
# print(eps11.shape)
# plt.imshow(eps11, cmap='gray')
# plt.colorbar()
# # plt.show()
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

I1 = np.abs(Ex)**2 +np.abs(Ey)**2 +np.abs(Ez)**2
plt.imshow(I1/np.max(I1), cmap='hot')
plt.colorbar()
plt.show()
kx = k_points[-1].x
from matplotlib import figure
# for standing wave
x_dir = 20
Ex_total=[]
Ey_total=[]
Ez_total=[]
for x in range(x_dir):
    Ex_total=Ex_total+list(Ex*np.exp(1j*kx*x)+Ex*np.exp(-1j*kx*x))
    Ey_total= Ey_total+list(Ey*np.exp(1j*x*kx)+Ey*np.exp(-1j*x*kx))
    Ez_total= Ez_total+list(Ez*np.exp(1j*x*kx)+Ez*np.exp(-1j*x*kx))
Ex_total = np.array(Ex_total)
Ey_total = np.array(Ey_total)
Ez_total = np.array(Ez_total)
print(Ex_total.shape)
x,y = np.meshgrid(1600*np.linspace(0,1,1600),3200*np.linspace(0,1,3200))
I_total = abs(Ex_total)**2 +abs(Ey_total)**2 +abs(Ez_total)**2
# plt.figure(figsize=(32,32), dpi=100)
plt.imshow(I_total/np.max(I_total), cmap='hot')
# plt.quiver(x,y,np.real(Ez_total), np.real(Ey_total), color='green', scale=20)
plt.colorbar()

plt.title('total inteisty')
plt.show()


# chekcing for only one wave
#method-1
# x_dir = 25
# std_wave =1
# Ex_total=[]
# Ey_total=[]
# Ez_total=[]
# for x in range(x_dir):
#     if std_wave==1:
#         Ex_total=Ex_total+list(Ex*np.exp(1j*kx*x)+Ex*np.exp(-1j*kx*x))
#         Ey_total= Ey_total+list(Ey*np.exp(1j*x*kx)+Ey*np.exp(-1j*x*kx))
#         Ez_total= Ez_total+list(Ez*np.exp(1j*x*kx)+Ez*np.exp(-1j*x*kx))
#     else:
#         Ex_total=Ex_total+list(Ex*np.exp(1j*kx*x))
#         Ey_total= Ey_total+list(Ey*np.exp(1j*x*kx))
#         Ez_total= Ez_total+list(Ez*np.exp(1j*x*kx))
# Ex_total = np.array(Ex_total)
# Ey_total = np.array(Ey_total)
# Ez_total = np.array(Ez_total)
# print(Ex_total.shape)
# # x,y = np.meshgrid(1600*np.linspace(0,1,1600),3200*np.linspace(0,1,3200))
# I_total = abs(Ex_total)**2 +abs(Ey_total)**2 +abs(Ez_total)**2
# # I_total = np.array(I_total)
# plt.figure(figsize=(10,10), dpi=100)
# plt.imshow(I_total/np.max(I_total), cmap='hot')

# # plt.quiver(x,y,np.real(Ez_total), np.real(Ey_total), color='green', scale=20)
# plt.colorbar()

# plt.title('total inteisty')
# plt.show()



#method-2
# Epx_list = []
# Epy_list = []
# Epz_list = []
# Enx_list = []
# Eny_list = []
# Enz_list = []
# Ip_list = []
# In_list = []
# I_total_list = []
# x_dir= 25
# for x in range(x_dir):
#     Epx_list=Epx_list+list(Ex*np.exp(1j*x*kx))
#     Epy_list=Epy_list+list(Ey*np.exp(1j*x*kx))
#     Epz_list=Epz_list+list(Ez*np.exp(1j*x*kx))
#     Enx_list=Enx_list+list(Ex*np.exp(-1j*x*kx))
#     Eny_list=Eny_list+list(Ey*np.exp(-1j*x*kx))
#     Enz_list=Enz_list+list(Ez*np.exp(-1j*x*kx))
    
    
# Epx_list = np.array(Epx_list)
# Enx_list = np.array(Enx_list)
# Epy_list = np.array(Epy_list)
# Eny_list = np.array(Eny_list)
# Epz_list = np.array(Epz_list)
# Enz_list = np.array(Enz_list)

# t = Epx_list+Enx_list
# print(t.shape)
# I_total_list = np.abs(Epx_list+Enx_list)**2 +np.abs(Epy_list+Eny_list)**2 +np.abs(Epz_list+Enz_list)**2

# x, y = np.meshgrid(3200*np.linspace(0,1,1600),160*np.linspace(0,1,160))
# # print(Epx_list.shape)
# I_total_list = np.array(I_total_list)
# plt.imshow(I_total_list/np.max(I_total_list), cmap='hot')
# # plt.quiver(x,y,np.real(Epz_list+Enz_list), np.real(Epy_list+Eny_list), color='green', scale=20)
# plt.colorbar()
# plt.title('I combined')
# plt.show()



#method -3
# Epx_list = []
# Epy_list = []
# Epz_list = []
# Enx_list = []
# Eny_list = []
# Enz_list = []
# Ip_list = []
# In_list = []
# I_total_list = []
# x_dir= 25
# for x in range(x_dir):
#     Epx_list.append(Ex*np.exp(1j*x*kx))
#     Epy_list.append(Ey*np.exp(1j*x*kx))
#     Epz_list.append(Ez*np.exp(1j*x*kx))
#     Enx_list.append(Ex*np.exp(-1j*x*kx))
#     Eny_list.append(Ey*np.exp(-1j*x*kx))
#     Enz_list.append(Ez*np.exp(-1j*x*kx))
    
    
# Epx_list = np.array(Epx_list)
# Enx_list = np.array(Enx_list)
# Epy_list = np.array(Epy_list)
# Eny_list = np.array(Eny_list)
# Epz_list = np.array(Epz_list)
# Enz_list = np.array(Enz_list)

# t = Epx_list+Enx_list
# print(t.shape)
# I_total_list = np.abs(Epx_list+Enx_list)**2 +np.abs(Epy_list+Eny_list)**2 +np.abs(Epz_list+Enz_list)**2

# x, y = np.meshgrid(3200*np.linspace(0,1,1600),160*np.linspace(0,1,160))
# # print(Epx_list.shape)
# I_total_list = np.array(I_total_list)
# print(I_total_list[0])
# # plt.imshow(I_total_list/np.max(I_total_list), cmap='hot')
# # plt.quiver(x,y,np.real(Epz_list+Enz_list), np.real(Epy_list+Eny_list), color='green', scale=20)
# # plt.colorbar()
# # plt.title('I combined')
# # plt.show()







#other method
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


#method -4














