#here we try to plot the PBG in 1D photonic crystals
import meep as mp
import numpy as np
import math as m
from meep import mpb 
import matplotlib.pyplot as plt


#Declare the parameters
num_bands = 2

resolution = 10
eps1 = 13
eps2 = 12.0
cell_size = mp.Vector3(4,4,0)
corner= mp.Vector3(0,0,0)
a = 2.0 # thickness of the slab
b = 2.0 # thickness of the slab2
a1 = a #periodicity in the y direction
# define the geometry
geometry_lattice = mp.Lattice(size=mp.Vector3(1,1),
                              basis1=mp.Vector3(a1,0))
                            #   basis2=mp.Vector3(0, -2*m.pi),
                            #   basis3 = mp.Vector3(0,0, 2*m.pi))
                            # basis2=mp.Vector3(0,0),
                            # basis3=mp.Vector3(0,0))
# basis2 is set to infinite to make the simulation periodic and work 
# #if we replace it with mp.Vector3(0,0) we get error saying division by zero, to wliminate this error we set it to infinite
k_points = [
    # mp.Vector3() , # gamma point
    # mp.Vector3(0,-1*m.pi/a1), # x point
    # mp.Vector3(m.pi/a1,-1*m.pi/a1), # M point
    # mp.Vector3(mp.pi/a1,-1*m.pi/a1), # x point
    # mp.Vector3(mp.pi/a1,0), # M point
    # mp.Vector3(m.pi/a1, m.pi/a1), # x point
    # mp.Vector3(0,m.pi/a1), # M point
    # mp.Vector3() , # gamma point


    mp.Vector3(0,-1*m.pi/a1,0), # x point
    mp.Vector3() , # M point
    mp.Vector3(0,m.pi/a1,0), # x point

    # #lets try these points
    # mp.Vector3(), # O point
    # mp.Vector3(m.pi/a1) , # M point
    # mp.Vector3(2*m.pi/a1) , # L point
]
# k_points = mp.interpolate(1, k_points)
geometry1 = [
    mp.Block(size=mp.Vector3(mp.inf,a ,mp.inf),
             center=corner,
             material=mp.Medium(epsilon=eps1)),
]
geometry2 = [
    mp.Block(size=mp.Vector3(mp.inf,a ,mp.inf),
             center=corner+mp.Vector3(0,-a,0),
             material=mp.Medium(epsilon=eps2)),
]

# define the boundary conditions
boundary_layers = [
    mp.PML(2.0),
]
sources = [
    mp.Source(
        mp.ContinuousSource(frequency=0.15), component=mp.Ez, center=mp.Vector3(-7, 0)
    )
]
# define the simulation
sim = mp.Simulation(resolution=resolution,
                    geometry=geometry1+geometry2,
                    boundary_layers=boundary_layers,
                    sources=sources,
                    cell_size=cell_size)

plt.figure()
sim.plot2D()
plt.show()

ms = mpb.ModeSolver(
    geometry=geometry1+geometry2,
    geometry_lattice=geometry_lattice,
    k_points=k_points,
    num_bands=num_bands,
    resolution=resolution)
ms.run_tm()
tm_freqs = ms.all_freqs
tm_gaps = ms.gap_list

fig, ax = plt.subplots()
x = range(len(tm_freqs))
# Plot bands
# Scatter plot for multiple y values, see https://stackoverflow.com/a/34280815/2261298
for xz, tmz in zip(x, tm_freqs):
    # xz_a = np.array([xz]*len(tmz))
    # xz_s = xz_a/(2*m.pi)
    # tmz_s = np.array(tmz)/(2*m.pi)
    ax.scatter([xz]*len(tmz), tmz, color='blue')
   
ax.plot(tm_freqs, color='blue')
#ax.set_ylim([0, 1])
# ax.set_xlim([x[-1], x[1]])

# Plot gaps
for gap in tm_gaps:
    if gap[0] > 1:
        ax.fill_between(x, gap[1], gap[2], color='blue', alpha=0.2)


# Plot labels
ax.text(12, 0.04, 'TM bands', color='red', size=15)

# points_in_between = (len(tm_freqs) - 4) / 3
# tick_locs = [i*points_in_between+i for i in range(4)]
# tick_labs = ['Γ', 'X', 'M', 'Γ']
# ax.set_xticks(tick_locs)
# ax.set_xticklabels(tick_labs, size=16)
ax.set_ylabel('frequency (c/a)', size=16)
ax.grid(True)

plt.show()