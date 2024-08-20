# This code is works  for 1d slabs. 
# In this file we are going to modify the base to add the fuancitoanlities like--
# Conversion of the normalised frequancy into the wavelength
# We fidn the band gap and convert the gap ratio into the wavelength
# we are also going to modify the plotting fucntionality to plot the gaps

#15-08-2024: updating the plotting geometry fucntionality
#adding the colour bar for the epsilon fucntionality\
#added the bar graph for the gaps
#adding of the feature to plot the PBG map
#to plot the al plots in a single figure

#19-08-2024:
# here we are adding the plotting of the ez for the number of bands


# importing the required libraries
import math as math
import meep as mp
from meep import mpb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
# defining the number of bands and the resolution
num_bands = 2
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
    mp.Vector3(-0.5,0), # first point
    mp.Vector3(0.5,0)] # last point
k_points = mp.interpolate(5, k_points) # interpolating the points

# define the geometry below is the code referring to how the geometry was created
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
# mode solver is the main fucntionality of mpb
ms=mpb.ModeSolver(
    geometry=geometry1+geometry2,
    geometry_lattice=geometry_lattice,
    k_points=k_points,
    num_bands=num_bands,
    resolution=resolution)
# running the mode solver for the tm modes
ms.run_tm() # running the mode solver for the tm modes
tm_freqs = ms.all_freqs # getting the tm frequencies
tm_gaps = ms.gap_list   # getting the tm gaps

# running the mode solver for the te modes
ms.run_te() # running the mode solver for the te modes
te_freqs = ms.all_freqs # getting the te frequencies
te_gaps = ms.gap_list   # getting the te gaps

# creating the geoemtry using patches from matplot lib
# note this geoemtry isnt used in mpb solver
# function for color to the material
# adding color map for the epsilon fucntionality
cmap = matplotlib.colormaps.get_cmap('viridis')
norm = mcolors.Normalize(vmin=min(eps1, eps2), vmax=max(eps1, eps2))

# defining the figure to plot the multiple sub plots



#function for plotting the geometry
def plt_geo (ax, geometry):
    for obj in geometry:
        if obj['type'] == 'block':
            width, height = obj['size']
            center_x, center_y = obj['center']
            epsilon = obj['epsilon']
            mat_color = cmap(norm(epsilon))
            lower_left = (center_x - width/2.0, center_y - height/2.0)
            
            rect = patches.Rectangle(lower_left,
                                     width, height, 
                                     linewidth=1, edgecolor='black', facecolor=mat_color)
            ax.add_patch(rect)
    sm= plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
    cbar.set_label('epsilon')
    ax.set_aspect('equal', 'box')
    ax.set_title('Geometry')
    ax.set_xlabel(' um in x direction')
    ax.set_ylabel(' um in y direction')
    # plt.show()

# plotting the tm frequencies & te frequencies
def plot_band_structures(ax, tm_freqs, te_freqs, tm_gaps, te_gaps):
    for i, (tmz, tez) in enumerate(zip(tm_freqs, te_freqs)):
        ax.scatter([i]*len(tmz), tmz, color='blue', s=0.5)
        ax.scatter([i]*len(tez), tez, color='red', s=0.5)
    ax.plot(tm_freqs, color='blue')
    ax.plot(te_freqs, color='red')
    ax.set_xlabel('k-points')
    ax.set_ylabel('Frequency')
    ax.set_title('TM and TE Frequencies')

    for gap in tm_gaps:
        if gap[0] > 1:
            ax.fill_between(range(len(k_points)), gap[1], gap[2], color='blue', alpha=0.5)
    for gap in te_gaps:
        if gap[0] > 1:
            ax.fill_between(range(len(k_points)), gap[1], gap[2], color='red', alpha=0.5)
    ax.grid(True)
    ax.text(0.5, 0.1, 'TE bands', ha='center', va='center', transform=ax.transAxes, color='red')
    ax.text(0.5, 0.9, 'TM bands', ha='center', va='center', transform=ax.transAxes, color='blue')
# to plot the gap histograms for the tm and te gaps
def plot_gap_histograms(ax, gaps, title, color):
    x = list(range(len(gaps)))
    for i in range(len(gaps)):
        ax.bar(x, gaps, color=color, alpha=0.5, width=0.5)
    ax.set_title(title)
    ax.set_ylim(0, max(gaps) + 1)
    ax.set_xlim(0, len(gaps))
    ax.grid(True)
#first figure : geometry, band structures and gaps
fig1, ax = plt.subplots(2,2, figsize=(10,10), dpi=100)
#here we need to pass the geometry to the function
# we them as dictionary
plt_geo(ax[0,0],[
    {'type': 'block', 'size': [d1, h], 'center': [d1/2, 0], 'epsilon': eps1},
    {'type': 'block', 'size': [d2, h], 'center': [d1 + d2/2, 0], 'epsilon': eps2}
])
# to plot band structures and gaps
plot_band_structures(ax[0,1], tm_freqs, te_freqs, tm_gaps, te_gaps)

# to plot the tm gap histogram
tm_gap_values = [gap[0] for gap in tm_gaps]
plot_gap_histograms(ax[1,0], tm_gap_values, 'TM Gaps', 'red')

# to plot the te gap histogram
te_gap_values = [gap[0] for gap in te_gaps]
plot_gap_histograms(ax[1,1], te_gap_values, 'TE Gaps', 'green')

plt.tight_layout()

# second figure: ez fields
fig2 = plt.figure(figsize=(10,10), dpi=100)
md = mpb.MPBData(rectify=True, resolution=resolution, periods=3)
eps= ms.get_epsilon()
converted_eps = md.convert(eps)
efields = []
# to get the efields by running the tm mode
def get_efields(ms, band):
    efields.append(ms.get_efield(band, bloch_phase=True))
ms.run_tm(mpb.output_at_kpoint(mp.Vector3(0.5,0,0), get_efields))

converted =[]
for f in efields:
    f = f[...,0,2]
    converted.append(md.convert(f))

for i, f in enumerate(converted):
    plt.subplot(331+i)
    plt.contour(converted_eps.T, cmap= 'binary')
    plt.imshow(np.real(f).T, interpolation='spline36', cmap='RdBu', alpha=0.5)
    plt.axis('off')
    plt.colorbar(label='Ez')    # adding the color bar for the plot
    plt.title(f'band {i+1}')    # adding the band number to the plot

plt.suptitle('Ez Fields')
plt.show()
