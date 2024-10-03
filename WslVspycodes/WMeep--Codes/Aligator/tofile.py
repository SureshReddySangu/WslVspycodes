import numpy as np
def te_file(conv_te_freqs, conv_te_gaps, resolution, wid, a, cell_in_y):
    # Write the frequency list to the file
    c = 3e8
    # a =0.370
    with open("te_freqs.txt", "a") as f:
        f.write(f"resolution: {float(resolution)}, a: {float(a)}, wid: {float(wid)}, cellsize:{cell_in_y}\n")
        for item in conv_te_freqs:
            item = np.array(item)
            f.write(str(item) + "\n")  # Convert each item to string and write it on a new line
            

    # Write the gaps list to the file
    with open("te_gaps.txt", "a") as f1:
        f1.write(f"resolution: {float(resolution)}, a: {float(a)}, wid: {float(wid)}, cellsize:{cell_in_y}\n")
        for gap in conv_te_gaps:
            f1.write(str(gap) + "\n")  # Convert each gap to string and write it on a new line

def tm_file(tm_frqs_list, tm_gaps_list, resolution, wid, a, cell_in_y):
    # Write the frequency list to the file
    c = 3e8
    a =0.370
    with open("tm_freqs.txt", "a") as f:
        f.write(f"resolution: {float(resolution)}, a: {float(a)}, wid: {float(wid)}, cellsize:{cell_in_y}\n")
        for item in tm_frqs_list:
            item = np.array(item)
            conv_item = (item*c)/(a*10**(-6))
            f.write(str(conv_item) + "\n")  # Convert each item to string and write it on a new line
            

    # Write the gaps list to the file
    conv_gaps1 =[]
    with open("tm_gaps.txt", "a") as f1:
        for gap in tm_gaps_list:
            g1 = (float(gap[0][0])*c)/(a*10**(-6))  
            g2 = (float(gap[1][1])*c)/(a*10**(-6))
            gapratio = 100 * (gap[1][1] - gap[0][0]) / ((gap[0][0] + gap[1][1]) / 2)
            conv_gaps1.append((gapratio, g1, g2))
            f1.write(str(conv_gaps1) + "\n")  # Convert each gap to string and write it on a new line
