import sys, getopt
import numpy as np
import matplotlib.pyplot as plt


def read_from_file(f_name):
    with open(f_name) as f:
        str_list = f.readlines() 
        arr_episodes = []
        for element in [str_.strip() for str_ in str_list]:
            try:      
                arr_episodes.append(float(element))
            except:
                pass

        return arr_episodes


def plot_data(data):
    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)
    ax.plot(data, linestyle='-')
    plt.ylabel('Total Reward')
    plt.xlabel('Episode')
    plt.margins(0.1)

if __name__=="__main__":
    _, args = getopt.getopt(sys.argv[1:], "")
    try:
        result_file = args[0]
    except IndexError:
        print('plot.py <path_to_results_txt_file>')

    e = read_from_file(result_file)
    plot_data(e)
    plt.show()
