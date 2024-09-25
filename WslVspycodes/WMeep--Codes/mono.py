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
numof_box = 33
offset = 0.199
ampli = 0.124
xx = np.linspace(0, x1, resolution)
delx = x1/len(xx)

for i,j in zip(xx, range(len(xx))):
    geo = offset + ampli*np.cos(2*np.pi*i/x1)
    if j < len(xx):
        geometry = mp.Block(
            size= mp.Vector3(delx, y1, geo),
            center = mp.Vector3((1+j)*delx/2, cell_in_y, cell_in_z),
            material = mp.Medium(index = n_lo)
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
