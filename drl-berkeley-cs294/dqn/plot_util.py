import os
import pickle
import numpy as np
import pandas as pd


log_dir = "/home/tasuku/workspace/rldm/drl-berkeley/homework/hw3/logs"

def load_data(file_name):
    """
    Returns data associated with file_name.
    :param - file_name: path to file name
    :returns: a binary file
    """
    with open(file_name, "rb") as f:
        return pickle.loads(f.read())


if __name__=="__main__":
    final = "stats-20170403-22-59-45-3500000.log"
    file_name = os.path.join(log_dir, final)
    ave_rewards = load_data(file_name)
    print ave_rewards
    t = np.arange(0,len(ave_rewards['ave_reward']))


    print t
    df = pd.DataFrame()
    df['100 step rolling'] = ave_rewards['ave_reward']
    print df
    plt = df.set_index(t).plot(title='100 step Average Return')
    plt.set_xlabel("Iteration (100k)")
    plt.set_ylabel("Average Rewards")
    fig = plt.get_figure()
    fig.savefig('DQN-ave-reward')
