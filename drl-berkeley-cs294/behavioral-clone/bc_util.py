"""
Utility functions used in behaveclone.py
# Date: 3/15/2017
# Author: Tasuku Miura
"""
import pickle
import numpy as np

def get_data_stats(data):
    """
    Gets mean and std used for data normalization.
    :param data: numpy.array of data.
    :return: tuple consisting of (mean,std) of data.
    """
    return data.mean(axis=0), data.std(axis=0)
    

def load_rollout(filename):
    """
    Extract data related to rollout generated by expert policy.
    :param filename - str, name of file.
    :return data - A tuple of lists of observations and actions.
    """
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())
    
    return data['observations'], data['actions']



def generate_batches(obs, actions, batch_size):
    """
    Generates random batches of the input data.
    :param obs: The observations.
    :param actions: Actions generated from policy observing obs.
    :param batch_size: The size of each minibatch.
    :yield: A tuple (observations, actions) of type numpy.darray
    """
    num = len(obs)

    while True:
        #generate a random batch of indices of size batch_size
        idxs = np.random.choice(num, batch_size)

        batch_obs, batch_actions = obs[idxs], actions[idxs].astype(float)
        #reduce dimensions so data "works" with model
        batch_actions = [act.flatten() for act in batch_actions]

        yield np.asarray(batch_obs), np.asarray(batch_actions)
    