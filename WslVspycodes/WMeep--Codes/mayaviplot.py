import meep as mp
from meep import mpb 
import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.animation import FuncAnimation

#-------------------------------working ------------------------------#
num_bands = 4
resoultion =32
n_lo = 1.99
#finite cell
pitch =1
#axis unit defintions
x1, y1, z1 = 1/pitch, 0.2/pitch, 0.5/pitch
#simualtion cell size
cell_in_x =1
cell_in_y =4
cell_in_z =4
geometry_lattice = mp.Lattice(size = mp.Vector3(cell_in_x, cell_in_y, cell_in_z),
                              basis1= mp.Vector3(1,0))
k_points = [mp.Vector3(kx) for kx in np.linspace(0,0.5,5)]
default_material = mp.Medium(index=1)

geometry = [
    mp.Block(
        size = mp.Vector3(x1, y1,z1),
        center = mp.Vector3(cell_in_x, cell_in_y, cell_in_z),
        material = mp.Medium(index=n_lo)
    ), 
    mp.Cylinder(
        radius=0.15/pitch,
        axis=mp.Vector3(0,1,0),
        height=0.2/pitch,
        center = mp.Vector3(cell_in_x,cell_in_y,cell_in_z),
        material = mp.Medium(index=1)
    )
]

ms = mpb.ModeSolver(
    geometry= geometry,
    geometry_lattice= geometry_lattice,
    k_points= k_points,
    num_bands=num_bands,
    resolution=resoultion,
    default_material=default_material
)

ms.run()

t_freqs = ms.all_freqs
gaps =ms.gap_list


fig, ax = plt.subplots()
for i,tmz in zip(range(len(t_freqs)), t_freqs):
    ax.scatter([i]*len(tmz), tmz, color ='red', label ="TE")
ax.plot(t_freqs, color ='red')
#light line
kx_val = np.linspace(0,1, len(k_points))
light_line = np.abs(kx_val)
ax.plot(range(len(k_points)), light_line, 'k--')
ax.set_xlabel("k-points")
ax.set_ylabel("Frequancies")
ax.grid(True)
for gap in gaps:
    if gap[0]>0:
        ax.fill_between(range(len(k_points)), gap[1], gap[2], color='blue', alpha =0.5)
plt.show()
# ms.compute_zparities()
# ms.compute_yparities()

# output the electric field to a .h5 file at the specifie band
mpb.output_efield(ms,1)
eps=ms.get_epsilon()

#output the epsilon to .h5 file
#--------------------------------------------------Epsilon animation-----------------------------#
ms.output_epsilon()
# c =plt.figure()
# ep_plt =plt.imshow(eps[16], cmap='hot')
# plt.colorbar()
# plt.show()
# def update (frame):
#     ep_plt.set_array(eps[frame])
#     plt.title('epsilon of the unit cell in x-direction')
#     return[ep_plt]
# anime1 = FuncAnimation(c, update, frames=len(eps), interval =100)
# plt.show()
#--------------------------------------------------------------*********---------------------------------------------#
import h5py
with h5py.File ('holey-e.k05.b01.h5','r') as file:
    data0 = file['x.i'][:]
    data1 = file['x.r'][:]
    data2 = file['y.i'][:]
    data3 = file['y.r'][:]
    data4 = file['z.i'][:]
    data5 = file['z.r'][:]
Ex_i = data0[:]
Ex_r = data1[:]
Ey_i = data2[:]
Ey_r = data3[:]
Ez_i = data4[:]
Ez_r = data5[:]

Ex = Ex_r +1j*Ex_i
Ey = Ey_r +1j*Ey_i
Ez = Ez_r +1j*Ez_i
# Ez = np.array(Ez)
# Ey = np.array(Ey)
# Ex = np.array(Ex)
Iz = np.abs(Ez)**2
Iy= np.abs(Ey)**2
Ix = np.abs(Ex)**2
I_tot = Iz +Iy+ Ix
#-----------------------------------------------------******************---------------#
#-----------------------------------------------------Ez field plot--------------------#
# X, Y = np.meshgrid(np.linspace(0, 128, 128), np.linspace(0, 128, 128))
# # Create figure and axes
# fig, ax = plt.subplots(2, 2)
# ax__ticks= np.linspace(0,128,4)
# ax__labels = np.round(ax__ticks/resoultion)
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
#--------------------------------------------End--------------------------------------------#
#-----------------------------------------************************--------------------------------#
#-----------------------------------------Epslion plot----------------------------------#
# eps
# plt.imshow(eps[:,64,:], extent=[0,4,1,0]) # shows the eps 
# plt.title(" The Eplison slice in ZX plane at y =64")
# plt.colorbar()
# plt.xlabel('Z (in units of cell size)')
# plt.ylabel('X (in units of cell size)')
# plt.show()
#----------------------------------------********************---------------------#
#----------------------------------------Tileing the epsilon function-------------#
a1 =1
tiles_in_x = 10
eps = np.array(eps)
eps_fu =eps
print(eps_fu.shape)
for x in range(tiles_in_x):
    if x>0:
        eps_fu = np.concatenate([eps, eps_fu])
print(eps_fu.shape)
z_ticks = np.linspace(0, 127, tiles_in_x) # in plot it is horizontal axis
z_labels = np.round(z_ticks/32) 
plt.xticks(z_ticks,z_labels)
x_ticks = np.linspace(0, tiles_in_x*resoultion, tiles_in_x) # in plot it is vertical axis
x_labels = np.round(x_ticks/resoultion)
plt.yticks(x_ticks, x_labels)
plt.imshow(eps_fu[:,64,:], aspect='equal')
plt.title(" Epsilon tiling over the waveguide in ZX plane, at y = 64")
plt.colorbar()
plt.xlabel('Z (in units of cell size)')
plt.ylabel('X (in units of no of tiles)')
plt.show()
#------------------------------------------End---------------------------------------#
#----------------------------------------*************************----------------------------------#
# period_cnst = 1
# k_0 = 2*np.pi/a1

# k_0 = 1*(2*np.pi)/period_cnst

# period_in_x = 1*2*period_cnst
# dist_in_x =  len(Ez)
# print(dist_in_x)
# x_values = np.linspace(0, period_in_x*1, dist_in_x)

#----------------------------------Ez filed plot slice in ZX plane---------------------------#
# Ez filed at in ZX plane, 
#we Converted 3d to 2d by taking a slice of the Ez field at y =64
# plt.imshow(np.real(Ez[:, 64, :]), cmap='RdBu', aspect='auto') 
# plt.colorbar()
# plt.title("Ez field slice at y= 64, ZX plane")
# plt_h_ticks = np.linspace(0,127, 4)
# plt_h_labels = np.round(plt_h_ticks/32)
# plt.xticks(plt_h_ticks, plt_h_labels)
# plt_v_ticks = np.linspace(0,31,2)
# plt_v_labels = np.round(plt_v_ticks/32)
# plt.yticks(plt_v_ticks, plt_v_labels)
# plt.xlabel(' Z in unit cell ')
# plt.ylabel(' X  unit cell')
# plt.show()
#------------------------------------**************************-----------------------------#
#----------------------------------------Bloch Wave-------------------------------------#
# Since the waveguide has a disreate symetry we have to use the Bloch wave theorm for the field
# for the k = 0.5 or at the band edge we are calcualting
# Ez_list = Ez[:,64,:]
# Ey_list = []
# Ex_list = []
# # List to store Ez values after applying phase shifts (phi evolution)
# Ez_list_phi=[]
# k_val =np.pi/a1 
# # number of periods
# translation = np.arange(1,4,1)
# # Apply the translations to Ez and stack them vertically
# for t in translation:
#     translated_Ez = Ez[:,64,:]*np.exp(1j*k_val*t*a1)
#     Ez_list= np.vstack((Ez_list,translated_Ez))
# # Create a list of phase evolutions over time
# evo = np.linspace(0, 2*np.pi,100)
# # Apply phase shift (phi) evolution to the entire Ez_list
# for phi in evo:
#     Ez_list_phi.append(Ez_list*np.exp(1j*phi))
# Ez_list_phi = np.array(Ez_list_phi)
# # Create figure and axes for subplots
# fig,ax = plt.subplots(2,2)
# p1= ax[0,0].imshow(np.real(Ez[:,64,:]), cmap = 'RdBu', aspect = 'auto')
# cbar = plt.colorbar(p1, ax= ax[0,0])
# ax[0,0].set_title('Dielectric Ez in ZX plane, over the unit cell')
# h_ticks = np.linspace(1,128, 4)
# h_labels = np.round(h_ticks/32)
# ax[0,0].set_xticks(h_ticks, h_labels)
# v_ticks = np.linspace(1,len(Ez),2)
# v_labels = np.round(v_ticks/32)
# ax[0,0].set_yticks(v_ticks, v_labels)
# ax[0,0].set_xlabel('Z')
# ax[0,0].set_ylabel('X unit cell')

# p2 =ax[1,0].imshow(np.real(Ez_list), cmap = 'RdBu', aspect='auto')
# cbar = plt.colorbar(p2, ax = ax[1,0])
# ax[1,0].set_title('Dielectric Ez in ZX palne, over unit cell ')
# h1_ticks = np.linspace(1, 128, 4)
# h1_labels = np.round(h1_ticks/32)
# ax[1,0].set_xticks(h1_ticks, h1_labels)
# v1_ticks =np.linspace(0,len(Ez_list), 2*int(len(Ez_list)/len(Ez)))
# v1_labels = (v1_ticks/32)
# ax[1,0].set_yticks(v1_ticks, v1_labels)
# ax[1,0].set_xlabel('Z')
# ax[1,0].set_ylabel('X unit cell')

# Plot the first time evolution of Ez with phase shift
# v = ax[0,1].imshow(np.real(Ez_list_phi[0]), cmap = 'RdBu', aspect ='auto')
# cbar = plt.colorbar(p2, ax = ax[0,1])
# h1_ticks = np.linspace(1, 128, 4)
# h1_labels = np.round(h1_ticks/32)
# ax[0,1].set_xticks(h1_ticks, h1_labels)
# v1_ticks =np.linspace(0,len(Ez_list), 2*int(len(Ez_list)/len(Ez)))
# v1_labels = (v1_ticks/32)
# ax[0,1].set_yticks(v1_ticks, v1_labels)
# ax[0,1].set_xlabel('Z')
# ax[0,1].set_ylabel('X unit cell')
# # Define update function for animating time evolution of Ez
# def update_phi (frame):
#     v.set_array(np.real(Ez_list_phi[frame]))
#     ax[0, 1].set_title(f"Time Evolution of Dielectric Ez, ZX plane at t (in unts /100 of pi) = {frame * phi}")
# # Create animation for the time evolution of Ez
# anim1 = FuncAnimation(fig, update_phi, frames=len(evo), interval = 500)
# plt.show()
#--------------------------------------************************----------------------------#
#---------------------------------------Air band------------------------------#
# mpb.output_efield(ms,2)
# with h5py.File ('holey-e.k05.b02.h5', 'r') as file:
#     air_data0 = file['x.i'][:]
#     air_data1 = file['x.r'][:]
#     air_data2 = file['y.i'][:]
#     air_data3 = file['y.r'][:]
#     air_data4 = file['z.i'][:]
#     air_data5 = file['z.r'][:]
# air_Ex_i = air_data0[:]   
# air_Ex_r = air_data1[:]
# air_Ey_i = air_data2[:]   
# air_Ey_r = air_data3[:]
# air_Ez_i = air_data4[:]   
# air_Ez_r = air_data5[:]

# air_Ez = air_Ez_r+1j*air_Ez_i
# air_Ey = air_Ey_r+1j*air_Ey_i
# air_Ex = air_Ex_r+1j*air_Ex_i

# air_I = np.abs(air_Ez)**2+np.abs(air_Ey)**2+np.abs(air_Ex)**2
# fig1, ax11 =plt.subplots()
# air_I_plt =ax11.imshow(air_I[0], cmap= 'hot')
# def update_air_I (frame):
#     air_I_plt.set_array(air_I[frame])
#     ax11.set_title(f"I at t ={frame:.2f}")
# anime = FuncAnimation(fig1, update_air_I, frames= 32, interval =500)
# cabrr =plt.colorbar(air_I_plt, ax = ax11)
# plt.show()
# figg,axx  = plt.subplots(2,2)
# Ezz=axx[0,0].imshow(np.real(air_Ez[16]), cmap= 'RdBu')
# axx[0,0].set_xlabel('Z')
# axx[0,0].set_ylabel('Y')
# d_Ez = axx[1,0].imshow(np.real(Ez[0]), cmap ='RdBu')
# axx[1,0].set_xlabel('Z')
# axx[1,0].set_ylabel('Y')
# X_, Y_ = np.meshgrid(np.linspace(0,128,128), np.linspace(0,128,128))
# air_quiver = axx[0,1].quiver(X_,Y_, np.real(air_Ez[16]),  np.zeros_like(np.real(air_Ez[16])),color = 'red'
#                             , scale = 10, headwidth = 4, headlength =6)
# epsss= axx[0,1].imshow(eps[0], cmap='hot')
# axx[0,1].set_xlabel('Z')
# axx[0,1].set_ylabel('Y')
# axx[0,1].set_xlim(50,75)
# axx[0,1].set_ylim(55,70)
# qu = axx[1, 1].quiver(X_, Y_, np.real(Ez[16]), np.zeros_like(np.real(Ez[0])), color='green', 
#                       scale =10,headwidth = 4, headlength = 6)
# epsss1 = axx[1,1].imshow(eps[0], cmap='hot')
# axx[1,1].set_xlim(50,75)
# axx[1,1].set_ylim(55,70)
# axx[1,1].set_xlabel('Z')
# axx[1,1].set_ylabel('Y')
# def update_Ez (frame):
#     Ezz.set_array(np.real(air_Ez[frame]))
#     d_Ez.set_array(np.real(Ez[frame]))
#     air_quiver.set_UVC(np.real(air_Ez[frame]), np.zeros_like(np.real(air_Ez[frame])))
#     qu.set_UVC(np.real(Ez[frame]), np.zeros_like(np.real(Ez[frame])))
#     epsss.set_array(eps[frame])
#     epsss1.set_array(eps[frame])
#     axx[0,0].set_title(f"air_Ez band at k_point = 0.5 x ={frame/32}, in ZY plane")
#     axx[1,0].set_title(f"Dielectric Ez band at k_point= 0.5 x = {frame/32}, in ZY plane")
#     axx[0,1].set_title(f" air_Ez band vectors at k_point= 0.5 x ={frame/32} in ZY palne")
#     axx[1,1].set_title(f" Dielctric_Ez band vectors at k_point = 0.5 x ={frame/32} in ZY palne")
# animee = FuncAnimation(figg, update_Ez, frames= 32, interval =500)
# cabrr =plt.colorbar(Ezz, ax = axx[0,0], label = 'Air Ez amplitude')
# cabrr = plt.colorbar(d_Ez, ax = axx[1,0], label ='Dielectric Ez amplitude')
# cc = plt.colorbar(epsss, ax = axx[0,1], label ='Epsilon')
# cc1 = plt.colorbar(epsss1, ax= axx[1,1], label =' Epsilon')
# plt.show()
# #-----------------------------------End--------------------------------------------#
# #------------------------------------- outputting E field at desired k_point---------------#
# ms.run(
#     mpb.output_at_kpoint(mp.Vector3(0.45), mpb.output_efield_z(ms,1))
# )

#-----------------------------------3D visaulization using Mayavi-------------------#
from mayavi import mlab
mlab.figure(bgcolor=(1,1,1))
# X, Y, Z = np.meshgrid(np.linspace(0,32,32), np.linspace(0,32,32), np.linspace(0,32,32), indexing='ij')
mlab.contour3d(eps, colormap = 'hot')
mlab.contour3d(np.real(Ez), colormap ='RdBu')
# mlab.contour3d(np.real(Ez), colormap ='RdBu')
# mlab.contour3d(I_tot, colormap ='gray')
# mlab.quiver3d(X,Y,Z,  np.zeros_like(np.real(Ez[:,48:80,48:80])), np.zeros_like(np.real(Ez)), np.real(Ez), scale_factor = 1.0)
# mlab.contour3d(np.real(air_Ez[:,:,:]), colormap ='viridis')
# mlab.contour3d(np.real(Ey))
# mlab.contour3d(np.real(Ex))
# mlab.title('Photonic crystal Waveguide', size =0.5)
# axes =mlab.axes(xlabel = 'Pitch (μm)', ylabel = 'height(μm)', zlabel ='width (μm)', color =(0,0,0))
# axes.axes.label_format = '%.1f'
# mlab.colorbar(title= " Epsilon values", orientation='vertical')
# mlab.outline(color =(0.5, 0.5, 0.5))
# eps_fu_field = mlab.pipeline.scalar_field(eps_fu)
# mlab.pipeline.grid_plane(eps_fu_field, plane_orientation='x_axes', color=(0.5, 0.5, 0.5))
# mlab.pipeline.grid_plane(eps_fu_field, plane_orientation='y_axes', color=(0.5, 0.5, 0.5))
mlab.show()



