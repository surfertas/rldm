#!/usr/bin/env python
import gym
import numpy as np
from collections import deque
from keras.callbacks import Callback
from bc_util import lstm_reshape

class CloneStats(Callback):
    def __init__(self, envname, mean, std, timesteps=0):
        self.env = gym.make(envname)
        self.mean = mean
        self.std = std
        self.timesteps = timesteps
        self.max_steps = self.env.spec.timestep_limit
        self.ave_reward = []
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        rewards = []
        for i in range(100):
            done = False
            totalr = 0
            steps = 0
            obs = self.env.reset()
            rollout = deque([(obs - self.mean)/self.std]*self.timesteps)
            assert(len(rollout) == self.timesteps)

            while not done:
                obs = (obs - self.mean) / self.std
                #if timesteps specified implies lstm model was used.
                if self.timesteps != 0:
                    rollout.popleft()
                    rollout.append(obs)
                    x = lstm_reshape(rollout, self.timesteps, len(obs))
                    action = self.model.predict(x)
                else:
                    action = self.model.predict(np.reshape(np.array(obs),
                                                          (1,len(obs))))

                
                obs, r, done, _ = self.env.step(action)
                totalr += r
                steps += 1

                if steps > self.max_steps:
                    break

            rewards.append(totalr)
       
        #need to reset state, as LSTM stateful=True
        if self.timesteps != 0:
            self.model.reset_states() 

        print ("\n{}".format(np.mean(rewards)))
        self.ave_reward.append(np.mean(rewards))
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
