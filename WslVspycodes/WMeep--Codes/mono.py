import meep as mp
from meep import mpb 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#---------------------------specifications of the cell-------------#
num_bands = 4
resolution =32
n_lo = 1.9935
#---------------physical parameters in nm--------------#
a =0.350
pie = np.pi
amp = 0.124
wid = 0.200
thk = 0.200

dx = (1/resolution)
#---------- aligator with gap = 0 --------------#
gaap =0
#simualtion cell size
cell_in_x = 1
cell_in_y = 4
cell_in_z = 4
#lattice
geometry_lattice = mp.Lattice(size = mp.Vector3(cell_in_x, cell_in_y, cell_in_z),
                              basis1=mp.Vector3(1))
k_points = [mp.Vector3(kx) for kx in np.linspace(0.45,0.5,5)]
default_materail = mp.Medium(index=1)
#-----------------------conversion of physical units itno the meep units---------------#
def simunits(x):
    return x/a
#-------------------making the geometry ------#
def drawblock (x):
    dz = wid+amp*np.cos(2*pie*x)
    ce = (dz+gaap)/2
    return[
        mp.Block(
            center = mp.Vector3(x,0, simunits(ce)),
            size = mp.Vector3(1*dx,simunits(thk), simunits(dz)),
            material = mp.Medium(index= n_lo)
        ), 
        mp.Block(
            center = mp.Vector3(x, 0, -simunits(ce)),
            size = mp.Vector3(1*dx, simunits(thk), simunits(dz)),
            material = mp.Medium(index= n_lo)
        )
    ]
# defineing the monagator geoemtry
geometry_list=[]
def makegeo(x):
    if x>=1:
        return
    geometry_list.extend(drawblock(x))
    makegeo(x+dx)
makegeo(0)
#-----------------------Mode solver ----- #
ms = mpb.ModeSolver(
    geometry= geometry_list,
    geometry_lattice= geometry_lattice,
    k_points= k_points,
    num_bands= num_bands,
    resolution= resolution,
    default_material= default_materail
)
#---------------Run the y-odd z-even modes------------------#
ms.run()
eps = ms.get_epsilon()
ms.output_epsilon
#------------------2d view of the geometry -------------#
slic_in_y = int((resolution*cell_in_y)/2)
plt.imshow(eps[:,slic_in_y,:], cmap='hot')
plt.colorbar()
plt.show()
#--------------------------------------plotingg--------------------------####
t_freqs = ms.all_freqs
gaps =ms.gap_list
#conversion of freq in real 
c = 3.0e8
conv_tfreqs = (t_freqs*c)/a
conv_gaps =[]
for gap in gaps:
    g1= (gap[1]*c)/a
    g2= (gap[2]*c)/a
    gapratio = (gap[2]-gap[1])/((gap[1]+gap[2])/2)*100
    conv_gaps.append((gapratio, g1, g2))
fig, ax = plt.subplots()
for i,tmz in zip(range(len(conv_tfreqs)), conv_tfreqs):
    ax.scatter([i]*len(tmz), tmz, color ='red', label ="TE")
ax.plot(conv_tfreqs, color ='red')
ax.set_xticks(range(len(k_points)))
o_k= np.linspace(0.45, 0.5, 5)
ax.set_xticklabels([f"{k:.2f}" for k in o_k])
ax.set_xlabel("k-points") 
ax.set_ylabel("Frequancies in Thz")
ax.grid(True)
ax.set_title(f"plot with resolution = {resolution:.2f}" f"and width {wid:.2f}")
for gap in conv_gaps:
    if gap[0]>0:
        ax.fill_between(range(len(k_points)), gap[1], gap[2], color='blue', alpha =0.5)
plt.show()
#---------------------------------tiling-------------------------#
tiles_in_x = 10
eps_fu =eps
print(eps_fu.shape)
for x in range(tiles_in_x):
    if x>0:
        eps_fu = np.concatenate([eps, eps_fu])
print(eps_fu.shape)
z_ticks = np.linspace(0, resolution*cell_in_y-1, tiles_in_x) # in plot it is horizontal axis
z_labels = np.round(z_ticks/5) 
plt.xticks(z_ticks,z_labels)
x_ticks = np.linspace(0, tiles_in_x*resolution, tiles_in_x) # in plot it is vertical axis
x_labels = np.round(x_ticks/resolution)
plt.yticks(x_ticks, x_labels)
plt.imshow(eps_fu[:,slic_in_y,:], aspect='equal')
plt.title(" Epsilon tiling over the waveguide in ZX plane, at y = 64")
plt.colorbar()
plt.xlabel('Z (in units of cell size)')
plt.ylabel('X (in units of no of tiles)')
plt.show()