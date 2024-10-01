import meep as mp
from meep import mpb 
import numpy as np
#---------------- simulation cell size ---------------------------#

cell_in_x = 1

def simu(geo,  k_points, resolution, num_bands, default_materail, ini, termi):
    te_frqs_list=[]
    te_gaps_list=[]
    for i in range(ini, termi+1):
        cell_in_y=i
        cell_in_z =i
    # Geometric latticd
        geometry_lattice = mp.Lattice(
            size = mp.Vector3(cell_in_x, cell_in_y, cell_in_z), # simualtion cell size
            basis1=mp.Vector3(1) # basis vesctor
            )
        #----------------mode solver-------------------#
        ms = mpb.ModeSolver(
            geometry=geo,
            geometry_lattice=geometry_lattice,
            k_points= k_points,
            resolution= resolution,
            num_bands=num_bands,
            default_material= default_materail
            )
        ms.run_te()
        t_freqs = ms.all_freqs
        gaps = ms.gap_list
        eps = ms.get_epsilon()
        te_frqs_list.append(t_freqs)
        te_gaps_list.append(gaps)
    return t_freqs, gaps, eps, cell_in_y, te_frqs_list, te_gaps_list
