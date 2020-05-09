import numpy as np
from matplotlib import pyplot as plt

def plot_and_or_xor(matrix, predict, weights=None, title="Prediction Matrix"):
    fig,ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    map_min=-1.5
    map_max=2.5
    res=0.5
    x = np.linspace(-1.5,2.5,8)
    plt.plot(x, -((weights[0]+weights[1]*x)/weights[2]), '-g', label='Decision')

    c1_data=[[],[]]
    c0_data=[[],[]]
    for i in range(len(matrix)):
        cur_x1 = matrix[i][1]
        cur_x2 = matrix[i][2]
        cur_y  = matrix[i][-1]
        if cur_y==1:
            c1_data[0].append(cur_x1)
            c1_data[1].append(cur_x2)
        else:
            c0_data[0].append(cur_x1)
            c0_data[1].append(cur_x2)

    plt.xticks(np.arange(map_min,map_max,res))
    plt.yticks(np.arange(map_min,map_max,res))
    plt.xlim(map_min,map_max-0.5)
    plt.ylim(map_min,map_max-0.5)

    c0s = plt.scatter(c0_data[0],c0_data[1],s=40.0,c='r',label='Class -1')
    c1s = plt.scatter(c1_data[0],c1_data[1],s=40.0,c='b',label='Class 1')

    plt.legend(fontsize=10,loc=1)
    plt.show()
    return