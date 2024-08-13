# This code is works  for 1d slabs. 
# In this file we are going to modify the base to add the fuancitoanlities like--
# Conversion of the normalised frequancy into the wavelength
# We fidn the band gap and convert the gap ratio into the wavelength
# we are also going to modify the plotting fucntionality to plot the gaps

# importing the required libraries
import math as m
import meep as mp
from meep import mpb
import matplotlib.pyplot as plt

# defining the number of bands and the resolution
num_bands = 4
resolution = 64 # Always choose higher resolution for better accuracy
#normalizaion
# The units in the Meep and Mpb ar noramlised, do remember to normalise the units
# We need to set the a value eqaul to the lattice constant of the system
a = 0.4

#slab dimensions
d1 = 0.2/a #  normalised slab1 thickness
d2 = 0.2/a #  normalised slab2 thickness
d = d1+d2 #  normalised slab thickness
h = 0.4/a #  normalised slab height

#epsilons
eps1 = 13 # epsilon of the slab1
eps2 = 5 # epsilon of the slab2

# define the geometry lattice
#size of the lattice in x and y direction
# remebber that when you normalized the dimensions
# Give the size  vector is (1,1), if the diemtions are not normalized,
# then give the vectors as (d,d) for symmetric slabs or just give the width and height
geometry_lattice = mp.Lattice(size=mp.Vector3(1,1),   
                              basis1=mp.Vector3(1,0))

# define the k points
# The number of k points to be interpolated depnds upon the num of k_points that youhave defined
# if you defien even number of k_points then intepolated points must be odd, vice versa
# always check that in the intepolated point you have the last point has been included
k_points = [
    mp.Vector3(0,0), # first point
    mp.Vector3(1,0)] # last point
k_points = mp.interpolate(5, k_points) # interpolating the points

# define the geometry
geometry1 = [
    mp.Block(size=mp.Vector3(d1,h), # size of the block in x and y direction
    center = mp.Vector3(d1/2,0), # center of the block
    material=mp.Medium(epsilon=eps1)), # material of the block
]
geometry2 = [ 
    mp.Block(size=mp.Vector3(d2,h), # size of the block in x and y direction
             center=mp.Vector3(d1+d2/2,0), # center of the block
             material=mp.Medium(epsilon=eps2)), # material of the block
]
# here are using thsi simualtion to just plot the geometry
# we can try the other option to plot teh geometry in  by plotting the epsilon function 
sim = mp.Simulation(resolution=resolution,
                    geometry=geometry1+geometry2,
                    cell_size=mp.Vector3(2,4),
                   )
plt.figure()
sim.plot2D()
plt.show()

# mode solver is the main fucntionality of mpb
ms=mpb.ModeSolver(
    geometry=geometry1+geometry2,
    geometry_lattice=geometry_lattice,
    k_points=k_points,
    num_bands=num_bands,
    resolution=resolution)

ms.run_tm() # running the mode solver for the tm modes

tm_freqs = ms.all_freqs # getting the tm frequencies
tm_gaps = ms.gap_list   # getting the tm gaps

# plotting the tm frequencies
# here we need to work a bit on the plotting fucntionality
fig, ax = plt.subplots()
x = range(len(tm_freqs))
print(x)
for i,tmz in zip(x, tm_freqs):
    ax.scatter([i]*len(tmz), tmz, color='blue', s=0.5)
ax.plot(tm_freqs, color='blue')
# ax.set_ylim([0, 1])

#plotting gaps 
for gap in tm_gaps:
    if gap[0] > 1:
        ax.fill_between(x,gap[1], gap[2], color='red', alpha=0.5)


ax.grid(True)
plt.show()

#trying the other option to plot the gaps
#try-1
fig, ax = plt.subplots()
for i,tmz in zip(len(k_points), tm_freqs):
    ax.scatter([i]*len(tmz), tmz, color='blue', s=0.5)
ax.plot(tm_freqs, color='blue')
# ax.set_ylim([0, 1])

for gap in tm_gaps:
    if gap[0] > 1:
        ax.fill_between(k_points,gap[1], gap[2], color='red', alpha=0.5)

ax.grid(True)
plt.show()