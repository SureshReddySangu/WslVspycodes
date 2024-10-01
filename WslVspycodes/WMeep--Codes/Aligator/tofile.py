import numpy as np
def toofile(te_frqs_list, te_gaps_list):
    # Write the frequency list to the file
    c = 3e8
    a =0.35
    with open("freqs.txt", "w") as f:
        for item in te_frqs_list:
            item = np.array(item)
            conv_item = (item*c)/(a*10**(-6))
            f.write(str(conv_item) + "\n")  # Convert each item to string and write it on a new line
            

    # Write the gaps list to the file
    conv_gaps1 =[]
    with open("gaps.txt", "w") as f1:
        for gap in te_gaps_list:
            g1 = (float(gap[1][1])*c)/(a*10**(-6))  
            g2 = (float(gap[2][2])*c)/(a*10**(-6))
            gapratio = 100 * (gap[2][2] - gap[1][1]) / ((gap[1][1] + gap[2][2]) / 2)
            conv_gaps1.append((gapratio, g1, g2))
            f1.write(str(conv_gaps1) + "\n")  # Convert each gap to string and write it on a new line
