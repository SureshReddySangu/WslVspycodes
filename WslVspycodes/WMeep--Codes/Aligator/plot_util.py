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

# Function to plot the TE eigenfrequencies and gaps
def plot_te_frequencies(te_frqs_list, te_gaps_list, k_points, a):
    #Conversion of the egien frequnices isnto the real units
    c=3.0e8
    conv_te_freqs =[]
    te_frqs_list = np.array(te_frqs_list)
    print(type(te_frqs_list))
    for te_freqs in te_frqs_list:
        conv_te_freqs.append((te_freqs * c)/(a * 1e-6))
    conv_te_freqs = np.array(conv_te_freqs)
    print(type(conv_te_freqs))
    print(conv_te_freqs.shape)
    #Conversion of the gaps into the real units
    conv_te_gaps =[]
    te_gaps_list = np.array(te_gaps_list)
    for gap, i in zip(te_gaps_list, range(len(te_gaps_list))):
        print(gap[0][1])
        g1 = ((gap[i][1])*c)/(a*1e-6)  
        g2 = (float(gap[i][2])*c)/(a*1e-6)
        gapratio = 100*(g2 - g1) / ((g2 + g1) / 2)
        conv_te_gaps.append((gapratio, g1, g2))
    
    # Plot eigenfrequencies
    fig, ax = plt.subplots()
    # Using a for loop to scatter and plot each converted frequency set
    print(conv_te_freqs.shape)
    
    for frq in conv_te_freqs:
        print(frq)
        for i,tmz in zip(range(len(frq)), frq):
            ax.scatter([i]*len(tmz), tmz, color ='red', label ="TE")
        # ax.plot(frq, color ='red')
    # for i, freqs in enumerate(conv_te_freqs):
    #     ax.scatter([i] * len(freqs), freqs, color='red', label="TE" if i == 0 else "")
    # ax.plot([i] * len(freqs), freqs, color='red')
    ax.set_xticks(range(len(k_points)))
    o_k = np.array(k_points)
    o_k =[i[0] for i in o_k]
    ax.set_xticklabels([f"{k:.2f}" for k in o_k])
    ax.set_xlabel("k-points")
    ax.set_ylabel("Frequancies in Thz")
    ax.grid(True)
    for gap in conv_te_gaps:
        if gap[0]>1:
            ax.fill_between(range(len(k_points)), gap[1], gap[2], color='blue', alpha =0.5)
    plt.title("Eigenfrequencies and Gaps")
    # plt.show()
    print(conv_te_freqs)
    print(conv_te_gaps)
    return conv_te_freqs, conv_te_gaps


# Function to plot the TM eigenfrequencies and gaps
def plot_tm_frequencies(tm_frqs_list, tm_gaps_list, k_points, a):
    #Conversion of the egien frequnices isnto the real units
    c=3.0e8
    t_freqs = np.array(t_freqs)
    for t_freqs in tm_frqs_list:
        conv_tfreqs = (t_freqs*c)/(a*10**(-6))
    #Conversion of the gaps into the real units
    conv_tm_gaps =[]
    print(tm_gaps_list)
    for gap in tm_gaps_list:
        g1 = (float(gap[1])*c)/(a*10**(-6))  
        g2 = (float(gap[2])*c)/(a*10**(-6))
        gapratio = 100*(g2 - g1) / ((g2 + g1) / 2)
        conv_tm_gaps.append((gapratio, g1, g2))
    print(conv_tfreqs)
    print(len(conv_tm_gaps))
    # Plot eigenfrequencies
    fig, ax = plt.subplots()
    for i,tmz in zip(range(len(conv_tfreqs)), conv_tfreqs):
        ax.scatter([i]*len(tmz), tmz, color ='red', label ="TE")
    ax.plot(conv_tfreqs, color ='red')
    ax.set_xticks(range(len(k_points)))
    o_k = np.array(k_points)
    o_k =[i[0] for i in o_k]
    ax.set_xticklabels([f"{k:.2f}" for k in o_k])
    ax.set_xlabel("k-points")
    ax.set_ylabel("Frequancies in Thz")
    ax.grid(True)
    for gap in conv_tm_gaps:
        if gap[0]>1:
            ax.fill_between(range(len(k_points)), gap[1], gap[2], color='blue', alpha =0.5)
    plt.title("Eigenfrequencies and Gaps")
    plt.show()
    return conv_tfreqs, conv_tm_gaps