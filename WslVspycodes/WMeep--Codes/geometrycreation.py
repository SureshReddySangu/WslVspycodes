# here in thsi pyhton code we use meep to create a 2D geometry which is periodic in z direction
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import math as m

resolution = 10
eps1 = 13
eps2 = 5
cell_size = mp.Vector3(10,10,0)
corner=mp.Vector3(0,0,0)
a = 2.0 # thickness of the slab
b = 2.0 # thickness of the slab2
# define the geometry
geometry_lattice = mp.Lattice(size=mp.Vector3(1,1),
                              basis1=mp.Vector3(0,a),
                              basis2=mp.Vector3(mp.inf, mp.inf))

k_points = [
    mp.Vector3(((m.pi)*-1),0), # gmamma point
    mp.Vector3() , # X point
    mp.Vector3(0, m.pi) , # M point
    mp.Vector3(((m.pi)*-1),0), # gmamma point
]
k_points = mp.interpolate(2, k_points)
geometry1 = [
    mp.Block(size=mp.Vector3(mp.inf,a ,mp.inf),
            #  corner=mp.Vector3(0,0,0),
            center= corner,
             material=mp.Medium(epsilon=eps1)),
]
geometry2 = [
    mp.Block(size=mp.Vector3(mp.inf,a ,mp.inf),
             center=corner+mp.Vector3(0,-a,0),
             material=mp.Medium(epsilon=eps2)),
]

# define the boundary conditions
boundary_layers = [
    mp.PML(1.0),
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