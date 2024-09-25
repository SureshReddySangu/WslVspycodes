import meep as mp
from meep import mpb 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

num_bands = 4
resolution =32
n_lo = 1.99
# finite cell
pitch =0.35
#waveguide dimessions
x1, y1, z1 = 0.35/pitch, 0.2/pitch, 0.648/pitch
#simualtion cell size
cell_in_x = 1
cell_in_y = 4
cell_in_z = 4
#lattice
geometry_lattice = mp.Lattice(size = mp.Vector3(cell_in_x, cell_in_y, cell_in_z),
                              basis1=mp.Vector3(1))
k_points = [mp.Vector3(kx) for kx in np.linspace(0.45,0.5,5)]
default_materail = mp.Medium(index=1)
# defineing the monagator geoemtry
geometry_list=[]
numof_box = 11
offset = 0.35
xt = np.linspace(offset, np.pi-offset, numof_box)
# wi/d = np.linspace(0.1,1,numof_box)
# ce = np.linspace(0,1, numof_box)
i = numof_box
delx = x1/len(xt)
# ((x1/int(len(xt))/2)*(1+i)+0.005)
for h,i, thea  in zip(np.sin(xt), range(i), xt):
    if thea == np.pi/2 and h == np.max(np.sin(xt)):
        geometry =mp.Block(
                size=mp.Vector3(delx,y1,0.4),
                center = mp.Vector3(cell_in_x,cell_in_y, cell_in_z ),
                material = mp.Medium(index= n_lo)
            )
    elif thea < np.pi/2:
        geometry = mp.Block(
            size = mp.Vector3(delx, y1, z1*h),
            center = mp.Vector3((delx/2)+delx*i, cell_in_y, cell_in_z),
            material = mp.Medium(index= n_lo)
        )
    elif thea>np.pi/2:
        geometry = mp.Block(
            size = mp.Vector3(x1/int(len(xt)), y1, z1*h),
            center = mp.Vector3((delx/2)+delx*i, cell_in_y, cell_in_z),
            material = mp.Medium(index= n_lo)
        )

    geometry_list.append(geometry)
ms = mpb.ModeSolver(
    geometry= geometry_list,
    geometry_lattice= geometry_lattice,
    k_points= k_points,
    num_bands= num_bands,
    resolution= resolution,
    default_material= default_materail
)
ms.run_te()
eps = ms.get_epsilon()
ms.output_epsilon
plt.imshow(eps[:,64,:], cmap='hot')
plt.colorbar()
plt.show()

t_freqs = ms.all_freqs
gaps =ms.gap_list

fig, ax = plt.subplots()
for i,tmz in zip(range(len(t_freqs)), t_freqs):
    ax.scatter([i]*len(tmz), tmz, color ='red', label ="TE")
ax.plot(t_freqs, color ='red')
#light line
kx_val = np.linspace(0,1, len(k_points))
light_line = np.abs(kx_val)
# ax.plot(range(len(k_points)), light_line, 'k--')
# p = np.linspace(0.45, 0.5, 5)
# ax.set_xticks(p)
ax.set_xlabel("k-points")
ax.set_ylabel("Frequancies")
ax.grid(True)
for gap in gaps:
    if gap[0]>0:
        ax.fill_between(range(len(k_points)), gap[1], gap[2], color='blue', alpha =0.5)
plt.show()

mpb.output_efield(ms,1)

# import h5py
# with h5py.File ('monogator-e.k05.b01.h5','r') as file:
#     data0 = file['x.i'][:]
#     data1 = file['x.r'][:]
#     data2 = file['y.i'][:]
#     data3 = file['y.r'][:]
#     data4 = file['z.i'][:]
#     data5 = file['z.r'][:]
# Ex_i = data0[:]
# Ex_r = data1[:]
# Ey_i = data2[:]
# Ey_r = data3[:]
# Ez_i = data4[:]
# Ez_r = data5[:]

# Ex = Ex_r +1j*Ex_i
# Ey = Ey_r +1j*Ey_i
# Ez = Ez_r +1j*Ez_i

# Iz = np.abs(Ez)**2
# Iy= np.abs(Ey)**2
# Ix = np.abs(Ex)**2
# I_tot = Iz +Iy+ Ix

# X, Y = np.meshgrid(np.linspace(0, 128, 128), np.linspace(0, 128, 128))
# # Create figure and axes
# fig, ax = plt.subplots(2, 2)
# ax__ticks= np.linspace(0,128,4)
# ax__labels = np.round(ax__ticks/resolution)
# # Intensity plot (I_tot)
# plt1 = ax[0, 0].imshow(I_tot[0], cmap='hot')  # Initial plot
# cbar = plt.colorbar(plt1, ax=ax[0, 0])
# ax[0, 0].set_title("Intensity profile along ZY plane, X-direction")
# ax[0,0].set_xticks(ax__ticks,ax__labels)
# ax[0,0].set_yticks(ax__ticks, ax__labels)
# ax[0,0].set_xlabel('Z units in cell size')
# ax[0,0].set_ylabel('Y units in cell size')
# # Ez field plot
# Ez_plt = ax[0, 1].imshow(np.real(Ez[0]), cmap='RdBu')  # Initial Ez plot
# cbar1 = plt.colorbar(Ez_plt, ax=ax[0, 1])
# ax[0, 1].set_title("Ez field profile along x-direction , in ZY plane")
# ax[0, 1].set_xticks(ax__ticks,ax__labels)
# ax[0, 1].set_yticks(ax__ticks,ax__labels)
# ax[0, 1].set_xlabel('Z units in cell size')
# ax[0, 1].set_ylabel('Y units in cell size')
# # Quiver plot for the Ez vectors
# qu = ax[1, 0].quiver(X, Y, np.real(Ez[0,:,:]), np.zeros_like(np.real(Ez[0])), color='green', scale =10,
#                      headwidth = 3, headlength = 4)
# ax[1, 0].set_title("Vector plot of the Ez field, in ZY palne")
# ax[1,0].set_xlim(40, 100)
# ax[1,0].set_ylim(40,100)
# # Update function for the animation
# def update(frame):
#     # Update intensity plot
#     plt1.set_array(I_tot[frame])
#     ax[0, 0].set_title(f"Intensity profile at x = {frame/32:.2f} in steps (1/32), in ZY plane")

#     # Update Ez field plot
#     Ez_plt.set_array(np.real(Ez[frame]))
#     ax[0, 1].set_title(f"Ez field at x = {frame/32:.2f} in steps (1/32), in ZY plane")

#     # Update vector (quiver) plot
#     qu.set_UVC(np.real(Ez[frame]), np.zeros_like(np.real(Ez[frame])))  # Update only the U component
#     ax[1, 0].set_title(f"Ez vectors at x = {frame/32:.2f}in steps (1/32), in ZY plane")

#     return [plt1, Ez_plt, qu]

# # Create the animation with both updates in one function
# anim = FuncAnimation(fig, update, frames=32, interval=500, blit=False)
# # Display the animation
# plt.tight_layout()
# plt.show()
# from mayavi import mlab
# mlab.contour3d(eps)
# mlab.contour3d(np.real(Ez))
# mlab.show()
# #-------
# plt.imshow(np.real(Ez[:,64,:]),cmap='RdBu')
# plt.show()
# #--- tiling
# a1 =1
# tiles_in_x = 10
# eps = np.array(eps)
# eps_fu =eps
# print(eps_fu.shape)
# for x in range(tiles_in_x):
#     if x>0:
#         eps_fu = np.concatenate([eps, eps_fu])
# print(eps_fu.shape)
# z_ticks = np.linspace(0, 127, tiles_in_x) # in plot it is horizontal axis
# z_labels = np.round(z_ticks/32) 
# plt.xticks(z_ticks,z_labels)
# x_ticks = np.linspace(0, tiles_in_x*resolution, tiles_in_x) # in plot it is vertical axis
# x_labels = np.round(x_ticks/resolution)
# plt.yticks(x_ticks, x_labels)
# plt.imshow(eps_fu[:,64,:], aspect='equal')
# plt.title(" Epsilon tiling over the waveguide in ZX plane, at y = 64")
# plt.colorbar()
# plt.xlabel('Z (in units of cell size)')
# plt.ylabel('X (in units of no of tiles)')
# plt.show()
# mlab.contour3d(eps_fu)
# mlab.show()

# #----------------------------------------*************************----------------------------------#
# period_cnst = 1
# k_0 = 2*np.pi/a1

# k_0 = 1*(2*np.pi)/period_cnst

# period_in_x = 1*2*period_cnst
# dist_in_x =  len(Ez)
# print(dist_in_x)
# x_values = np.linspace(0, period_in_x*1, dist_in_x)
#----------------------------------------Bloch Wave-------------------------------------#
# # Since the waveguide has a disreate symetry we have to use the Bloch wave theorm for the field
# # for the k = 0.5 or at the band edge we are calcualting
# Ez_list = []
# Ey_list = []
# Ex_list = []
# # List to store Ez values after applying phase shifts (phi evolution)
# Ez_list_phi=[]
# k_val =np.pi/a1 
# # number of periods
# translation = np.arange(1,10,1)
# # Apply the translations to Ez and stack them vertically
# import time
# translated_Ez = Ez*np.exp(1j*k_val*translation[0]*a1)
# cs =mlab.contour3d(np.real(translated_Ez), colormap = 'RdBu')
# mlab.contour3d(eps, colormap='hot')
# def update_cs(idx):
#     translated_Ez = Ez*np.exp(1j*k_val*translation[idx]*a1)
#     cs.mlab_source.scalars = np.real(translated_Ez)
# for i in range(len(translation)):
#     update_cs(i)
#     mlab.process_ui_events()
#     time.sleep(0.5)
# mlab.show()
