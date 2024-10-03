import meep as mp
from meep import mpb 
import numpy as np
#---------------- simulation cell size ---------------------------#

cell_in_x = 1

def simu_te(geo,  k_points, resolution, num_bands, default_materail, cell_size):
    te_frqs_list=[]
    te_gaps_list=[]
    cell_in_y=cell_size
    cell_in_z = cell_size
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
    ms.run()
    te_freqs = ms.all_freqs
    te_gaps = ms.gap_list
    eps = ms.get_epsilon()
    te_frqs_list.append(te_freqs)
    te_gaps_list.append(te_gaps)
    return te_freqs, te_gaps, eps, cell_in_y, te_frqs_list, te_gaps_list


#---------------- simulation cell size ---------------------------#

cell_in_x = 1

def simu_tm(geo,  k_points, resolution, num_bands, default_materail, ini, termi):
    tm_frqs_list=[]
    tm_gaps_list=[]
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
        ms.run_tm()
        tm_freqs = ms.all_freqs
        tm_gaps = ms.gap_list
        eps = ms.get_epsilon()
        tm_frqs_list.append(tm_freqs)
        tm_gaps_list.append(tm_gaps)
    return tm_freqs, tm_gaps, eps, cell_in_y, tm_frqs_list, tm_gaps_list
