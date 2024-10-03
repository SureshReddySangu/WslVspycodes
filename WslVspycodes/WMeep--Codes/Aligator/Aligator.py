import meep as mp
from meep import mpb
import matplotlib.pyplot as plt
import numpy as np
from geo_uti import simunits, drawblock, makemono
from plot_util import plot_geometry, plot_te_frequencies
from simu_uti import simu_te, simu_tm
from tofile import te_file, tm_file
#--------------- passign the constsnts-------#
num_bands = 4
resolution = 32
# n_si3n4 = 1.99
# #----------- physical paramters in um---------#
# a = 0.350 # pithc of the unit cell
# pie = np.pi 
# amp = 0.124 # the amplitude of the modulation
# wid = 0.199 # width of the nanobeam
# thk = 0.200 # thickness of the nanobeam
# dx = 1/resolution # the differential step size
# g = 0
#-----------k-points----------------#
k_points = [mp.Vector3(kx) for kx in np.linspace(0.45, 0.5, 5)]
default_materail = mp.Medium(index=1)
#----------------geoemtry list-------------------#
geo = makemono(0)

#---------------- simulation cell size ---------------------------#
cell_in_x = 1
ini, termi = 4,5
#-----------------------run simulation--------------------------------#
#--------------TE mode-------------------#
te_freqs, te_gaps, eps, cell_in_y, te_frqs_list, te_gaps_list = simu_te(geo, k_points, resolution, num_bands, default_materail, ini, termi)
#-------------TM mode ---------------#
tm_freqs, tm_gaps, eps, cell_in_y, tm_frqs_list, tm_gaps_list = simu_tm(geo,  k_points, resolution, num_bands, default_materail, ini, termi)
#--------------------------****************-----------------#

#------------------ plot the 2d view of the geometry epsilon ----------------#
# plot_geometry(eps, resolution, cell_in_y )
#------------------------ plot he eigenfrequcies and gaps---------#
# Plotting the Eigne frequnices and gaps in real units 
conv_te_freqs, conv_te_gaps= plot_te_frequencies(te_frqs_list, te_gaps_list, k_points, 0.370)

# write to the text file
te_file(te_frqs_list, te_gaps_list)
tm_file(tm_frqs_list, tm_gaps_list)