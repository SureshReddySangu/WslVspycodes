import numpy as np
import meep as mp
from meep import mpb

#----------- physical paramters in um---------#
resolution = 32
n_si3n4 = 1.9935
a = 0.350 # pithc of the unit cell
pie = np.pi 
amp = 0.124 # the amplitude of the modulation
wid = 0.280 # width of the nanobeam
thk = 0.200 # thickness of the nanobeam
dx = 1/resolution # the differential step size
g = 0
#-----------------------Conversion of physical units into the meep units---------------#
def simunits(x):
    return x/a

#-------------------Making the aligator geometry ------#
def drawblock(x):
    dz = wid + amp * np.cos(2 * pie * x)
    ce = (dz + g) / 2
    return [
        mp.Block(
            center=mp.Vector3(x, 0, simunits(ce)),
            size=mp.Vector3(1 * dx, simunits(thk), simunits(dz)),
            material=mp.Medium(index=n_si3n4),
        ),
        mp.Block(
            center=mp.Vector3(x, 0, -simunits(ce)),
            size=mp.Vector3(1 * dx, simunits(thk), simunits(dz)),
            material=mp.Medium(index=n_si3n4),
        ),
    ]

# Recursively define the geometry list
geometry_list=[]
def makegeo(x ):
    if x >= 1:
        return
    geometry_list.extend(drawblock(x))
    # makegeo(x + dx, wid, amp, pie, g, thk, n_si3n4, dx, a, geometry_list)
    makegeo(x + dx)
    return geometry_list