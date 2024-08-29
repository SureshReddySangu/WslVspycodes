import meep as mp
from meep import mpb
import matplotlib.pyplot as plt
import numpy as np

num_bands = 4
resolution = 32
n_lo =1.99

geometry_lattice = mp.Lattice(size=mp.Vector3(0,4,4),
                              basis1=mp.Vector3(1, 0))  # 1d cell

k_points = [mp.Vector3(kx) for kx in np.linspace(0, 0.5, 5)]

geometry = [mp.Block(
    size=mp.Vector3(mp.inf, 0.2,0.5),
    center=mp.Vector3(z=0.5),
    material=mp.Medium(index=n_lo)
),
    mp.Block(
    size=mp.Vector3(mp.inf, 0.2,0.5),
    center=mp.Vector3(z=-0.5),
    material=mp.Medium(index=n_lo)
)]
ms =mpb.ModeSolver(
    geometry=geometry,
    geometry_lattice=geometry_lattice,
    k_points=k_points,
    num_bands=num_bands,
    resolution=resolution,
)
ms.run()
t_frqs = ms.all_freqs
t_frqs = ms.all_freqs
gaps = ms.gap_list
eps11=ms.get_epsilon()
plt.imshow(eps11, cmap='RdBu', alpha=0.5)
plt.colorbar()
plt.show()

mpb.output_efield(ms,1)