# 20-08-2024: adding the meep simualtion to vaisualize tehe Ez fields and the meep geometry

# importing the required libraries
import math as math
import matplotlib.cm
import meep as mp
from meep import mpb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
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
    mp.Vector3(-0.5,0), # first point
    mp.Vector3(0.5,0)] # last point
k_points = mp.interpolate(5, k_points) # interpolating the points
# calculate the k-pointa
# writing a function to calculate the k-points





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
plot_gap_histograms(ax[1,0], tm_gap_values, 'TM Gaps percentage', 'red')

# to plot the te gap histogram
te_gap_values = [gap[0] for gap in te_gaps]
plot_gap_histograms(ax[1,1], te_gap_values, 'TE Gaps percentage', 'green')

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

#meep simulation
#defining the simulation cell
cell = mp.Vector3(d+4, d+2)
pml_layers = [mp.PML(1.0)]
nfreq = 100
df = 0.1
fcen = 0.15
#defiantion of source
sources = [mp.Source(mp.GaussianSource(frequency=fcen, fwidth=df),
    center=mp.Vector3(0, 0),
    component=mp.Ez,
    size=mp.Vector3(0,h) #defifning the size of the source
)]


# creating the simulation
sim = mp.Simulation(resolution=resolution, cell_size=cell, 
                    sources=sources, 
                    boundary_layers=pml_layers,
                    geometry=geometry1+geometry2
                    )


plt.show()
# # set upo the plotting
fig, axs = plt.subplots(2,2, figsize=(10,10), dpi=100)
cmap = matplotlib.colormaps.get_cmap('viridis')
norm = mcolors.Normalize(vmin=min(eps1, eps2), vmax=max(eps1, eps2))
def plt_meep_geometry(ax):
    # pltting the color bar 
    geometry_region = mp.Volume(center=mp.Vector3(), size=mp.Vector3(d+2,h))
    sim.plot2D(ax= ax, output_plane=geometry_region, 
                                              eps_parameters={'cmap':cmap, 'norm':norm})
    sm= plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
    cbar.set_label('epsilon')
    ax.set_title('Geometry')
    ax.set_xlabel(' um in x direction')
    ax.set_ylabel(' um in y direction')
    ax.set_aspect('equal', 'box')


plt_meep_geometry(axs[0,0])

def plt_rt(ax):
    refl  = sim.add_flux(fcen, df, nfreq, refl_fl) # adding the reflected flux
    tran  = sim.add_flux(fcen, df, nfreq, tran_fl) # adding the transmitted flux
    
    sim.run(until_after_sources=mp.stop_when_fields_decayed(10, mp.Ez, pt, 1e-2))
    # reflected data
    ref_data = sim.get_flux_data(refl)
    
    sim.load_minus_flux_data(refl, ref_data)

    tran_flux = mp.get_fluxes(tran)
    re_flux = mp.get_fluxes(refl)

    # for wavelength calculation
    fluxfreq = mp.get_flux_freqs(refl)

    wl =[]
    rs =[]
    ts =[]
    for i in range(nfreq):
        wl.append(1/fluxfreq[i])
        rs.append(re_flux[i]/tran_flux[i])
        ts.append(tran_flux[i])

    ax.plot(wl, rs, label='reflected', color='red')
    ax.plot(wl, ts, label='transmitted', color='blue')
    ax.legend()
    ax.set_xlabel('Wavelength (um)')
    ax.set_ylabel('Flux')
    ax.grid(True)
# defing the boundary to be used for the reflected flux
refl_fl = mp.FluxRegion(center=mp.Vector3(0.01*d, 0), size=mp.Vector3(0, 2*h)) 
# defing the boundary to be used for the transmitted flux
tran_fl = mp.FluxRegion(center=mp.Vector3(0.9*d, 0), size=mp.Vector3(0, 2*h))
pt = mp.Vector3(d,0) # defining the point where the flux has to be calculated

#to plot the reflected and transmitted plot
plt_rt(axs[0,1])

def plt_realtime_ez(ax):
    field_data = np.zeros((int(cell.x*resolution), int(cell.y*resolution)))
    image = ax.imshow(field_data, interpolation='spline36', cmap='RdBu', alpha=0.5)
    ax.set_title('real time Ez Fields')
    ax.set_xlabel(' um in x direction')
    ax.set_ylabel(' um in y direction')
    ax.set_aspect('equal', 'box')

    #update function for animation
    def update(frame):
        sim.run(until = frame)
        ez_data = sim.get_array(center=mp.Vector3(0,0), component=mp.Ez, size = cell,)
        image.set_data(ez_data.T)
        return image,
    ani = FuncAnimation(fig, update, frames=np.arange(0,10,1), interval=1, blit=True, repeat=False)
    plt.colorbar(image, ax =ax, label='Ez')

plt_realtime_ez(axs[1,0])

plt.tight_layout()
plt.show()

