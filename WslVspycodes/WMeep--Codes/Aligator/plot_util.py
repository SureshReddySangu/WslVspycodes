import matplotlib.pyplot as plt
import numpy as np

# Function to plot the 2D view of the geometry
def plot_geometry(eps, resolution, cell_in_y):
    slic_in_y = int((resolution * cell_in_y) / 2)  # Calculate the slice position
    print(slic_in_y)
    print(eps.shape)
    # Plot the 2D slice of the geometry
    plt.imshow(eps[:, slic_in_y,:], cmap='hot')  # Now it works because eps_array is a NumPy array
    plt.colorbar()
    plt.title("2D View of the Geometry")
    plt.show()

# Function to plot the eigenfrequencies and gaps
def plot_frequencies(te_frqs_list, te_gaps_list, k_points, a):
    #Conversion of the egien frequnices isnto the real units
    c=3.0e8
    for t_freqs in te_frqs_list:
        t_freqs = np.array(t_freqs)
        conv_tfreqs = (t_freqs*c)/(a*10**(-6))
    #Conversion of the gaps into the real units
    conv_gaps =[]
    for gap in te_gaps_list:
        print(gap)
        g1 = (float(gap[1][1])*c)/(a*10**(-6))  
        g2 = (float(gap[2][2])*c)/(a*10**(-6))
        gapratio = 100 * (gap[2][2] - gap[1][1]) / ((gap[1][1] + gap[2][2]) / 2)
        conv_gaps.append((gapratio, g1, g2))

    # Plot eigenfrequencies
    fig, ax = plt.subplots()
    for i,tmz in zip(range(len(conv_tfreqs)), conv_tfreqs):
        ax.scatter([i]*len(tmz), tmz, color ='red', label ="TE")
    ax.plot(conv_tfreqs, color ='red')
    ax.set_xticks(range(len(k_points)))
    o_k= np.linspace(0.45, 0.5, 5)
    ax.set_xticklabels([f"{k:.2f}" for k in o_k])
    ax.set_xlabel("k-points")
    ax.set_ylabel("Frequancies in Thz")
    ax.grid(True)
    for gap in conv_gaps:
        if gap[0]>0:
            ax.fill_between(range(len(k_points)), gap[1], gap[2], color='blue', alpha =0.5)
    plt.title("Eigenfrequencies and Gaps")
    plt.show()

# def plot_range_freq
