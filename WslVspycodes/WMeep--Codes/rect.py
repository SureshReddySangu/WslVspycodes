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


fig, ax = plt.subplots(1, 1)

for i, tmz, in zip(range(len(t_frqs)), t_frqs):
    ax.scatter([i] * len(tmz), tmz, color="red", s=0.2, label="TE")
    
# Calculate and plot the light line
kx_vals = np.linspace(0, 1, len(k_points))
light_line = np.abs(kx_vals)  # Normalized units, slope = 1
ax.plot(range(len(k_points)), light_line, 'k--', label="Light Line")
ax.plot(t_frqs, color="blue", label="TM")

ax.set_xlabel("k-points")
ax.set_ylabel("Frequency")
ax.set_title("TM Frequencies & TE Frequencies")

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

# I =abs(Ex)**2 + abs(Ey)**2 + abs(Ez)**2
# fig= plt.imshow(I/np.max(I), cmap='hot' )
# plt.colorbar()
# plt.show()

# x,y = np.meshgrid(np.linspace(0,1,128),np.linspace(0,1,128))
# plt.quiver(x,y,Ez_r, Ey_r, color='red', scale = 15)
# eps11=ms.get_epsilon()
# plt.imshow(eps11, cmap='gray', alpha=0.2)
# plt.title('Ez + Ey')
# plt.ylim(0.4,0.6)
# plt.xlim(0.4,0.65)
# plt.colorbar()
a = 1
K_0 = 0.5*(2*np.pi)/a
Ez_list =[]
Ey_list = []
Ex_list =[]
for x in np.linspace( 0, 4*2*a,4*64):
    Ez_list.append(Ez[64,:]*np.cos(K_0*x))
    Ey_list.append(Ey[64,:]*np.cos(K_0*x))
    Ex_list.append(Ex[64,:]*np.cos(K_0*x))

Ez_list= np.abs(np.array(Ez_list))**2
Ey_list = np.abs(np.array(Ey_list))**2
Ex_list = np.abs(np.array(Ex_list))**2

I_list = Ez_list+Ey_list+Ex_list

plt.imshow(I_list, cmap='hot')
plt.colorbar()


plt.tight_layout()

plt.show()