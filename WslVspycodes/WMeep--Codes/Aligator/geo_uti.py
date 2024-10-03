import numpy as np
import meep as mp
from meep import mpb

#----------- physical paramters in um---------#
resolution = 5
n_si3n4 = 1.9935
# a = 0.370 # pithc of the unit cell
pie = np.pi 
amp = 0.124 # the amplitude of the modulation
# wid = 0.280 # width of the nanobeam
thk = 0.200 # thickness of the nanobeam
dx = (1/resolution)/2 # the differential step size
g = 0
#-----------------------Conversion of physical units into the meep units---------------#
def simunits(x, a):
    return x/a

#-------------------Making the aligator geometry ------#
def drawblock(x, wid, a):
    
    dz = wid + amp * np.cos(2 * pie * x)
    ce = (dz + g) / 2
    return [
        mp.Block(
            center=mp.Vector3(x, 0, simunits(ce,a)),
            size=mp.Vector3(1 * dx, simunits(thk,a), simunits(dz,a)),
            material=mp.Medium(index=n_si3n4),
        ),
        mp.Block(
            center=mp.Vector3(x, 0, -simunits(ce,a)),
            size=mp.Vector3(1 * dx, simunits(thk,a), simunits(dz,a)),
            material=mp.Medium(index=n_si3n4),
        ),
    ]

# Recursively define the geometry list
geometry_list=[]
def makemono(x, wid, a):
    if x >= 1:
        return
    geometry_list.extend(drawblock(x, wid, a))
    # makegeo(x + dx, wid, amp, pie, g, thk, n_si3n4, dx, a, geometry_list)
    makemono(x + dx, wid, a)
    return geometry_list