import meep as mp
from meep import mpb
import matplotlib.pyplot as plt
import numpy as np

num_bands = 4
resolution = 32
n_lo =1.99
width = 0.5
p_gap= 0.25
geometry_lattice = mp.Lattice(size=mp.Vector3(0,4,4),
                              basis1=mp.Vector3(1, 0))  # 1d cell

k_points = [mp.Vector3(kx) for kx in np.linspace(0, 0.5, 5)]

geometry = [mp.Block(
    size=mp.Vector3(mp.inf, 0.2,width),
    center=mp.Vector3(z=(width+p_gap)/2),
    material=mp.Medium(index=n_lo)
),
    mp.Block(
    size=mp.Vector3(mp.inf, 0.2,width),
    center=mp.Vector3(z=-(width+p_gap)/2),
    material=mp.Medium(index=n_lo)
)]
ms =mpb.ModeSolver(
    geometry=geometry,
    geometry_lattice=geometry_lattice,
    k_points=k_points,
    num_bands=num_bands,
    resolution=resolution,
)
ms.run()
t_frqs = ms.all_freqs
gaps = ms.gap_list

fig, ax = plt.subplots(2)
for i, fr in zip(range(len(t_frqs)), t_frqs):
    ax[0].scatter([i]*len(fr), fr,color='red')
ax[0].plot(t_frqs, color='red')
for gap in gaps:
    if gap[0]>1:
        ax.fill_between(gap[0], gap[1], gap[2], color='blue')
ax[0].set_title('Frequency')
ax[0].grid(True)

eps11=ms.get_epsilon()
ax[1].imshow(eps11)
pcm = ax[1].pcolormesh(eps11,cmap='hot',alpha=0.5)
fig.colorbar(pcm, ax= ax[1])
# plt.show()

mpb.output_efield(ms,2)
ms.compute_zparities()
mpb.display_zparities(ms)
import h5py

with h5py.File('recll-e.k05.b02.h5', 'r') as file:
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

eps11=ms.get_epsilon() 
x,y = np.meshgrid(128*np.linspace(0,1,128),128*np.linspace(0,1,128))
fig, ax = plt.subplots(2)
ax[0].imshow(I/np.max(I) )
ax[0].set_title('I')
pcm = ax[0].pcolormesh(I/np.max(I), cmap='hot')
fig.colorbar(pcm, ax= ax[0])
ax[1].quiver(x,y,Ez_r, Ey_r, color='red', scale = 20)
ax[1].imshow(eps11)
# ax[1].set_xlim(0.3, 0.9)
# ax[1].set_ylim(0.3, 0.8)

ax[1].set_title('eps')

pcm1 = ax[1].pcolormesh(eps11, cmap='hot',alpha=0.2)
fig.colorbar(pcm1,ax= ax[1])
plt.show()

