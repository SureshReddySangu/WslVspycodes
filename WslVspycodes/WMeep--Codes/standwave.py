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
eps =ms.get_epsilon()
print(type(eps))
print(eps.shape)
mpb.output_efield(ms,1)
import h5py
with h5py.File('standwave-e.k05.b01.h5', 'r') as file:
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
period_inx = 1*2*a
resl_inx = 2*64
period_intime = 1*2*np.pi*a
resl_intime =100
t_values1 = np.linspace(0, 2*period_intime, resl_intime)
x_values = np.linspace(0, period_inx*1, resl_inx)





# standing wave
# time evolution 
p_plot = np.linspace(0,resl_intime,5)
t_values = np.linspace(0, 2*np.pi*a, 101)
for t in t_values:
    for x in x_values:
        Ez_list.append(Ez[64,:]*(np.exp(1j*K_0*x) + np.exp(-1j*K_0*x))*(np.exp(1j*t)))
        Ey_list.append(Ey[64,:]*(np.exp(1j*K_0*x) + np.exp(-1j*K_0*x))*(np.exp(1j*t)))
        Ex_list.append(Ex[64,:]*(np.exp(1j*K_0*x) + np.exp(-1j*K_0*x))*(np.exp(1j*t)))
Ez_list = np.array(Ez_list)
Ey_list = np.array(Ey_list)
Ex_list = np.array(Ex_list)

Iz_list = np.abs(Ez_list)**2
Iy_list = np.abs(Ey_list)**2
Ix_list = np.abs(Ex_list)**2
I_tot_list = Iz_list + Iy_list +Ix_list
plt.imshow(I_tot_list[:128]/np.max(I_tot_list[:128]), cmap='hot')
plt.title("Intensity of a standing wave")
plt.xlabel("cell size")
plt.ylabel("cell size")
plt.colorbar()
plt.show()
#reshaping the Ez_list into 3d array
Ez_list_3d = Ez_list.reshape(len(t_values), 128,128)
Ex_list_3d = Ex_list.reshape(len(t_values), 128,128)
X,Y = np.meshgrid(np.linspace(0,1,128), np.linspace(0,1,128))
x_ticks = np.linspace(0, resl_inx, 5)
x_ticks_labels = [f'{int(val/32)}' for val in x_ticks]
y_ticks = np.linspace(0, period_inx,1)
y_ticks_labels = [f'{int(val/a)}' for val in y_ticks]
fig, ax = plt.subplots(3,2)
for i,t in enumerate(p_plot):
    p = ax[i//2,i%2].imshow(np.real(Ez_list_3d[int(p_plot[i])]), cmap='RdBu')
    ax[i//2, i%2].set_title(f'Ez fielld is time steps (multiples of pi/2)  = {t*(np.pi/2):.2f} ')
    ax[i//2, i%2].set_xticks(x_ticks)
    ax[i//2, i%2].set_xticklabels(x_ticks_labels)
    ax[i//2, i%2].set_yticks(y_ticks)
    ax[i//2, i%2].set_yticklabels(y_ticks_labels)
    ax[i//2, i%2].set_ylabel(f'no.of periods= {int(period_inx/2)}')
    ax[i//2, i%2].set_xlabel(f'Simualtion cell size = {int(resl_inx)}')
    fig.colorbar(p,ax = ax[i//2, i%2] )
plt.tight_layout()

plt.show()
fig1, ax1 = plt.subplots(3,2, figsize =(4,12))
for i,t in enumerate(range(5)):
    ax1[i//2, i%2].quiver(X,Y, np.real(Ez_list_3d[i]), np.real(Ex_list_3d[i]),color='red', 
                           scale =15, scale_units ='xy',headwidth = 4, headlength =6 )
    ax1[i//2, i%2].set_title(f'Z-Vector fielld is time steps (multiples of pi/2)  = {t*(np.pi/2):.2f} ')
    ax1[i//2, i%2].set_ylabel(f'no.of periods= {int(period_inx/2)}')
    ax1[i//2, i%2].set_xlabel(f'Simualtion cell size = {int(resl_inx)}')
plt.tight_layout()
plt.show()

from matplotlib.animation import FuncAnimation
fig2, ax2 = plt.subplots(1,2)
mve = ax2[0].imshow(np.real(Ez_list_3d[0]), cmap='RdBu')
vect = ax2[1].quiver(X,Y, np.real(Ez_list_3d[0]), np.real(Ex_list_3d[0]), color ='green', 
                     scale =15, scale_units ='xy',headwidth = 4, headlength =6)
def update (frame):
    mve.set_array(np.real(Ez_list_3d[frame]))
    ax2[0].set_title(f'E filed at t = {t_values[frame]:.2f}')
    vect.set_UVC(np.real(Ez_list_3d[frame,:]), np.real(Ex_list_3d[frame,:]))
    ax2[1].set_title(f'Vectors orientation at t = {t_values[frame]:.2f}')
    return[mve,vect]
anim = FuncAnimation(fig2, update, frames=len(t_values1), interval = 500)
plt.show()

fig3, ax3=plt.subplots(figsize =(4,12))
X3, Y3 = np.linspace(0.5,1,64), np.linspace(0,1,128)
result1 =np.array([row[64:128] for row in Ez_list_3d[0][0:128]])
result2 =np.array([row[64:128] for row in Ex_list_3d[0][0:128]])
# factor= np.mean(np.max(np.real(result1)))
# factor1 = np.mean(np.max(np.real(result2)))
# s = factor+ factor1

vect1= ax3. quiver(X3, Y3, np.real(result1), np.real(result2), color = 'red',
                    scale =15, scale_units ='xy',headwidth = 4, headlength =6 )
def updates (frame):
    result1 =np.array([row[64:128] for row in Ez_list_3d[frame][0:128]])
    result2 =np.array([row[64:128] for row in Ex_list_3d[frame][0:128]])
    vect1.set_UVC(np.real(result1), np.real(result2))
    ax3.set_title(f'Vectors at t = {t_values[frame]:.2f}')
    return [vect1]
anim1 = FuncAnimation(fig3, updates, frames = len(t_values1), interval =500)
#stream plot
# stream = ax3[1].streamplot(X3, Y3, np.real(result1), np.real(result2), color ='red', linewidth = 2)
# def updates1 (frame):
#     ax3[1].cla()
#     result1 =np.array([row[64:128] for row in Ez_list_3d[frame][0:128]])
#     result2 =np.array([row[64:128] for row in Ex_list_3d[frame][0:128]])
#     stream =ax3[1].streamplot(X3, Y3,np.real(result1), np.real(result2), color ='red', linewidth = 2 )
#     ax3[1].set_title(f'Vectors at t = {t_values[frame]:.2f}')
#     return stream
# anim2 = FuncAnimation(fig3, updates1, frames = len(t_values1), interval =500)
plt.show()



























































