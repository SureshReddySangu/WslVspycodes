import math
import meep as mp
from meep import mpb
from scipy.optimize import minimize_scalar
import numpy as np
import matplotlib.pyplot as plt

k_points = [mp.Vector3(),          # Gamma
            mp.Vector3(0.5),       # X
            mp.Vector3(0.5, 0.5),  # M
            mp.Vector3()]          # Gamma
num_bands = 8

# Initialize lists to store the results
radius_list = []
gap_size_list = []
ratio_list = []

def first_tm_gap(r):
    # Vary ax and ay from 0.1 to 1 in even steps
    for ax in np.arange(1, 2, 0.25):
        # ax and ay are always equal
        a = mp.Vector3(ax, ax)
        
        # Calculate the ratio of 'r' to 'a.x'
        ratio = r / ax
        
        # Update the geometry and geometry_lattice
        ms.geometry = [mp.Cylinder(r, material=mp.Medium(epsilon=12))]
        ms.geometry_lattice = mp.Lattice(size=a)
        
        ms.run_tm()
        gap = -1 * ms.retrieve_gap(1) # return the gap from TM band 1 to TM band 2
        radius_list.append(r)
        gap_size_list.append(gap)
        ratio_list.append(ratio)
    return gap

geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1))
resolution = 32
ms = mpb.ModeSolver(num_bands=num_bands,
                    k_points=k_points,
                    resolution=resolution)

#print_heading("Square lattice of rods: TE bands")
#ms.run_te()

# Run the optimization
result = minimize_scalar(first_tm_gap, method='bounded', bounds=[0.1, 0.5], options={'xatol': 0.1})

# Sort the radius_list and ratio_list in ascending order
radius_list, ratio_list, gap_size_list = zip(*sorted(zip(radius_list, ratio_list, gap_size_list)))

print("radius at maximum: {}".format(result.x))
print("gap size at maximum: {}".format(result.fun * -1))
print("All radius values: {}".format(radius_list))
print("All ratios: {}".format(ratio_list))
print("All gaps: {}".format(gap_size_list))

# Write the radius values, gap sizes, and ratios to a text file
#with open('output.txt', 'w') as f:
  #  f.write("Radius values: " + str(radius_list) + "\n")
  #  f.write("Gap sizes: " + str(gap_size_list) + "\n")
  #  f.write("Ratios: " + str(ratio_list) + "\n")

# Plot a graph of ratio vs gap size
#plt.figure(figsize=(10, 6))
#plt.plot(ratio_list, gap_size_list, marker='o')
#plt.xlabel('Ratio (r/a)')
#plt.ylabel('Gap Size')
#plt.title('Gap Size vs Ratio')
#plt.grid(True)
#plt.show()
