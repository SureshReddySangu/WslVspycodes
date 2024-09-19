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


# fig, ax = plt.subplots(1, 1)
# for i, tmz, in zip(range(len(t_frqs)), t_frqs):
#     ax.scatter([i] * len(tmz), tmz, color="red", s=0.2, label="TE")
# # Calculate and plot the light line
# kx_vals = np.linspace(0, 1, len(k_points))
# light_line = np.abs(kx_vals)  # Normalized units, slope = 1
# ax.plot(range(len(k_points)), light_line, 'k--', label="Light Line")
# ax.plot(t_frqs, color="blue", label="TM")

# ax.set_xlabel("k-points")
# ax.set_ylabel("Frequency")
# ax.set_title("TM Frequencies & TE Frequencies")

# for gap in gaps:
#     if gap[0] > 1:
#         ax.fill_between(range(len(k_points)), gap[1], gap[2], color="blue", alpha=0.5)
# for gap in gaps:
#     if gap[0] > 1:
#         ax.fill_between(range(len(k_points)), gap[1], gap[2], color="green", alpha=0.5)
# ax.grid(True)

# plt.show()
ms.compute_zparities()
ms.compute_yparities()

# plt.show()

# mpb.output_at_kpoint(0,3)
mpb.output_efield(ms,1)
import h5py

with h5py.File('rect-e.k05.b01.h5', 'r') as file:
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
plt.title('Electric field Intensity')
plt.colorbar(label ="intensity", orientation ="vertical")
plt.show()

x,y = np.meshgrid(np.linspace(0,1,128),np.linspace(0,1,128))
plt.quiver(x,y,Ez_r, Ey_r, color='red', scale = 15)
eps11=ms.get_epsilon()
# plt.imshow(eps11, cmap='viridis', alpha=0.2)
plt.title('Polarisation of Electric field')
plt.ylim(0.35,0.65)
plt.xlim(0.3,0.7)
# plt.colorbar(label ="epsilon of the material", orientation ="vertical")
a = 1
K_0 = 0.5*(2*np.pi)/a
Ez_list =[]
Ey_list = []
Ex_list =[]
# wrong working aniamtion code
from matplotlib.animation import FuncAnimation

# for x,t  in zip(np.linspace( 0, 2*4*a, 2*64),np.linspace( 0, 2*4*a, 2*64)):
#         Ez_list.append(Ez[64,:]*(np.exp(1j*K_0*x+ 1j*t)+np.exp(-1j*K_0*x +1j*t)))
#         Ey_list.append(Ey[64,:]*(np.exp(1j*K_0*x+ 1j*t)+np.exp(-1j*K_0*x +1j*t)))
#         Ex_list.append(Ex[64,:]*(np.exp(1j*K_0*x+ 1j*t)+np.exp(-1j*K_0*x +1j*t)))
# Ez_list= np.abs(np.array(Ez_list))**2
# Ey_list = np.abs(np.array(Ey_list))**2
# Ex_list = np.abs(np.array(Ex_list))**2
# I_list1 = Ez_list+ Ey_list+ Ex_list
# fig,ax= plt.subplots()
# intensity_plot = ax.imshow(I_list1, cmap='hot')
# ax.set_title('Electric Field Intensity Over Time')
# plt.colorbar(intensity_plot, ax=ax, label='Intensity')

# # # Function to update the frame for each time step
# def update(frame):
#     intensity_plot.set_data(I_list1[:frame])
#     return [intensity_plot]

# # Create the animation
# anim = FuncAnimation(fig, update, frames= 2*64, interval=100, blit=True)

# # Show the animation
# plt.show()

#----------------------------------------------------------

def computeEz(x,t):
     return Ez[64,:]*(np.cos(K_0*x)*np.exp(1j*t))
for i in np.linspace(0, 2*np.pi, 100):
    Ez_list.append(computeEz(np.linspace(0, 4*2*a, 2*64), i))

Ez_list = np.real(np.array(Ez_list))

fig, ax = plt.subplots()


Ezf = ax.imshow((Ez_list), cmap ='RdBu')
ax.set_title("Ez field")
fig.colorbar(Ezf, ax = ax)

# def update(frame):
#      Ezf.set_data(Ez_list[:frame])
#      ax1.set_title('Ezf animation')
#      print('hello')
#      return[Ezf]
# anim = FuncAnimation(fig1, update, frames=len(np.linspace(0,2*np.pi, 100)))
plt.show()



















# #time evolution----2
# t_ = np.linspace(0, 2* np.pi, 4)
# p1 = int(len(t_)/2)
# p2 = int(len(t_)/2)
# from matplotlib.animation import FuncAnimation 
# fig, ax = plt.subplots(p1, p2)
# for i,t in enumerate(t_):
#     for x in np.linspace( 0, 2*4*a, 2*64):
#         Ez_list.append(Ez[64,:]*(np.exp(1j*K_0*x+ 1j*t)+np.exp(-1j*K_0*x +1j*t)))
#         Ey_list.append(Ey[64,:]*(np.exp(1j*K_0*x+ 1j*t)+np.exp(-1j*K_0*x +1j*t)))
#         Ex_list.append(Ex[64,:]*(np.exp(1j*K_0*x+ 1j*t)+np.exp(-1j*K_0*x +1j*t)))

#     Ez_list= np.abs(np.array(Ez_list))**2
#     Ey_list = np.abs(np.array(Ey_list))**2
#     Ex_list = np.abs(np.array(Ex_list))**2
#     I_list1 = Ez_list+ Ey_list+ Ex_list
#     ro, col = divmod(i,p1)
#     im = ax[ro, col].imshow(I_list1, cmap='hot')
# fig.colorbar(im, ax = ax )  
# plt.tight_layout() 
# plt.show()
# intensity_plot = ax.imshow(I_list1, cmap='hot')
# ax.set_title('Electric Field Intensity Over Time')
# plt.colorbar(intensity_plot, ax=ax, label='Intensity')

# # # Function to update the frame for each time step
# def update(frame):
#     intensity_plot.set_data(I_list1[frame:])
#     return [intensity_plot]

# # Create the animation
# anim = FuncAnimation(fig, update, frames= 2*64, interval=100, blit=True)

# # Show the animation
# plt.show()




#tryyy
# t_vals = np.linspace(0, 2* np.pi, 4)

# # Create subplots
# fig, ax = plt.subplots(1, len(t_vals), figsize=(10, 10))  # Create 4 subplots side by side

# # Loop over each t value and x positions
# for i, t in enumerate(t_vals):
#     Ez_list = [] 
#     Ey_list = []
#     Ex_list = []
    
#     for x in np.linspace(0, 2 * 4 * a, 2 * 64):
#         Ez_list.append(Ez[64,:]*(np.exp(1j*K_0*x+ 1j*t)+np.exp(-1j*K_0*x +1j*t)))
#         Ey_list.append(Ey[64,:]*(np.exp(1j*K_0*x+ 1j*t)+np.exp(-1j*K_0*x +1j*t)))
#         Ex_list.append(Ex[64,:]*(np.exp(1j*K_0*x+ 1j*t)+np.exp(-1j*K_0*x +1j*t)))

#     Ez_list = np.array(Ez_list)
#     Ey_list= np.array(Ey_list)
#     Ex_list = np.array(Ex_list)

#     Iz_list = np.abs(np.array(Ez_list))**2
#     Iy_list = np.abs(np.array(Ey_list))**2
#     Ix_list = np.abs(np.array(Ex_list))**2
#     I_list1 = Iz_list + Iy_list + Ix_list

    
#     # Plot the intensity for the current t value in the corresponding subplot
#     im = ax[i].imshow(I_list1, cmap='hot')
#     ax[i].set_title(f"t = {t:.2f}")
#     ax[i].set_xlabel("x")
#     ax[i].set_ylabel("y")
#     # e = ax[i+len(t_vals)].imshow(np.real(Ez_list), cmap='RdBu')
# # Add a color bar to the last subplot
# fig.colorbar(im, ax=ax[-1], orientation='vertical', fraction=0.02)
# # fig.colorbar(e, ax = ax[i+len(t_vals)])
# # Show the subplots
# plt.show()

# from matplotlib.animation import FuncAnimation
# t_ = np.linspace(0, 2* np.pi, 2)
# x_vals = np.linspace(0, 1*2*a, 2*64)  # X values

# p1 = int(len(t_) // 2)
# p2 = int(np.ceil(len(t_) / p1))

# fig1, ax = plt.subplots(p1, p2, figsize=(8, 8))  # Create the 2x2 grid of subplots
# ax = ax.ravel()  # Flatten the array of axes into 1D for easier indexing
# fig2, ax1 = plt.subplots(p1,p2)
# ax1 = ax1.ravel()
# # Store initial intensity plot for each subplot
# intensity_plots = []

# # Initialize plots for each time value
# for i, t in enumerate(t_):
#     Ez_list, Ey_list, Ex_list = [], [], []

#     for x in x_vals:
#         Ez_list.append(Ez[64,:]*(np.exp(1j*K_0*x+ 1j*t)+np.exp(-1j*K_0*x +1j*t)))
#         Ey_list.append(Ey[64,:]*(np.exp(1j*K_0*x+ 1j*t)+np.exp(-1j*K_0*x +1j*t)))
#         Ex_list.append(Ex[64,:]*(np.exp(1j*K_0*x+ 1j*t)+np.exp(-1j*K_0*x +1j*t)))

#     # Compute intensities
#     Ez_list = np.abs(np.array(Ez_list)) ** 2
#     Ey_list = np.abs(np.array(Ey_list)) ** 2
#     Ex_list = np.abs(np.array(Ex_list)) ** 2
#     I_list1 = Ez_list + Ey_list + Ex_list

#     # Plot initial data using imshow and save the plot object
#     im = ax[i].imshow(I_list1, cmap='hot', animated=True)
#     ax[i].set_title(f"t = {t:.2f}")
#     intensity_plots.append(im)
#     e = ax1[i].imshow(np.real(Ez_list),cmap='RdBu')
#     ax1[i].set_title(f"t = {t:.2f}")
# # Add a colorbar to the last subplot
# fig1.colorbar(im, ax=ax)
# fig2.colorbar(e, ax = ax1)
# plt.show()



