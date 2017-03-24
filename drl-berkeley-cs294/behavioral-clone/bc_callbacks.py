#!/usr/bin/env python
import gym
import numpy as np
from keras.callbacks import Callback


class CloneStats(Callback):
    def __init__(self, envname, mean, std):
        self.env = gym.make(envname)
        self.mean = mean
        self.std = std
        self.max_steps = self.env.spec.timestep_limit
        self.ave_reward = []
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        rewards = []
        for i in range(10):
            done = False
            totalr = 0
            steps = 0
            obs = self.env.reset()
            
            while not done:
                obs = (obs - self.mean) / self.std
                action = self.model.predict(np.reshape(np.array(obs),
                                                       (1,len(obs))))

                
                obs, r, done, _ = self.env.step(action)
                totalr += r
                steps += 1

                if steps > self.max_steps:
                    break

            rewards.append(totalr)
        
        print ("\n{}".format(np.mean(rewards)))
        self.ave_reward.append(np.mean(rewards))
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
