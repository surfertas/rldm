# OpenGym LunarLander-v2
# DQN w/ Priority Replay
# Author: Tasuku Miura
# Date: 2016.12.22
#
# -----------------------------------
import numpy as np
import random
from collections import deque
from sklearn import neural_network
from sklearn.externals import joblib
from tqdm import tqdm
import gym

seed = 555
np.set_printoptions(precision=2)

FILE_NAME = 'results.txt' 
MODEL_NAME = 'model.sav'

class AGENT(object):
      

    def __init__(self, num_inputs, actions, discount_rate):
        self.model = neural_network.MLPRegressor(hidden_layer_sizes=(200,200),
                                                 warm_start=True)
        self._NINPUTS = num_inputs
        self._NACTIONS = len(actions)
        self._GAMMA = discount_rate
        self._EPSILON_DECAY = 0.99
        self._EPSILON_LOWERBOUND = 0.10
        self._MEMORY_CAPACITY = 100000
        self._BATCH_SIZE = 128
        self._REPLAY = 4
        self._memory = deque()
        self._first_experiences= deque()
        self._accum_rewards = 0.
        self.epsilon = 1.0
        self.returns_history = []
        self.explore = True
        self.experience_count = 0

        self._update(np.random.rand(1, self._NINPUTS),
                     np.random.rand(1, self._NACTIONS))


    def __call__(self, v):
        return self.model.predict([v])[0]

    def _select_action(self, Q):
        """Select action based on epsilon-greedy policy.

        Args:
            Q: Array of state-action values.

        Returns:
            a: Selected action.

        """
        if (np.random.random() < self.epsilon) or self.explore: 
            a = env.action_space.sample()
        else:
            a = np.argmax(np.array(Q))       

        return a
          
    def _update(self, X, Y):
        """Updates model with training data.

        Args: 
            X: Array of state_values.
            Y: Q values associated with respective state_values.

        """    
        self.model.partial_fit(X, Y)
    
    def _replay(self):
        """Updates model, by training on batches. Data selected
        at random uniformly.

        """
        if len(self._memory) < self._BATCH_SIZE:   
            batch_size = len(self._memory)
        else:
            batch_size = self._BATCH_SIZE

        self._first_experiences.extend(random.sample(self._memory, batch_size))

        x_train = []
        y_train = []
        for memory_tuple in self._first_experiences:
            experience = memory_tuple
            (sm0, a, rwd, sm1, _) = experience

            Q = agent(sm0)
            Q_next = agent(sm1)
            Q_max = Q_next[np.argmax(Q_next)]

            if abs(rwd) != 100.0:
                Q[a] = (rwd + self._GAMMA * (Q_max))
            else:
                Q[a] = rwd
       
            x_train.append(sm0)
            y_train.append(Q)

        self._update(np.array(x_train), np.array(y_train))
       
        #reset first experiences
        self._first_experiences.clear()

    def _process_rewards(self, reward, done):
        """Accumulates rewards for respective episode, appends
        to history if episode is done.

        Args:
            reward: Reward for one step.
            done: Indicator variable to signal if finished.

        """             
        self._accum_rewards += reward
        
        if done: 
            self.returns_history.append(self._accum_rewards)
            self._accum_rewards = 0.

    def act(self, Q):
        """Calls action function.
            
        Args: 
            Q: Array of state-action values.

        """
        return self._select_action(Q)

    def train(self, transition):
        """Process rewards, and prepares memory, then trains
        by calling replay method.

        Args:
            transition: a tuple of variables for one iteration.
        
        """
        (s0, a, r, s1, done) = transition
        self._process_rewards(r, done)

        #store in memory
        self._memory.append(transition)
        if len(self._memory) > self._MEMORY_CAPACITY:
            self._memory.popleft()
        
        if self.experience_count % self._REPLAY != 0:
            self._first_experiences.append(transition)
        else:
            self._replay()

        if done and not self.explore:
            self.epsilon = max(self._EPSILON_LOWERBOUND, 
                               self.epsilon * self._EPSILON_DECAY)

   
#Support methods for Qlearner
def write_to_file(f_name, string_):
    with open(f_name, 'a+') as f:
        f.write("{}\n".format(string_))

def process_stats(agent, file_name):
    ret_mean = np.mean(agent.returns_history[-100:])

    if ret_mean > 200.0:
        joblib.dump(agent.model, open(file_name, 'wb'))
        exit()

    print("Epoch: {} Reward: {} Ave: {} Eps: {}".format(
          epoch, agent.returns_history[-1], ret_mean, agent.epsilon)) 

    write_to_file(FILE_NAME, agent.returns_history[-1])


if __name__=="__main__":
    np.random.seed(seed)
    file_name = MODEL_NAME
    ACTIONS = ['NOOP',  'LEFT', 'MAIN', 'RIGHT']
    NUM_INPUTS = 8
    EPOCHS = 2000

    agent = AGENT(NUM_INPUTS, ACTIONS, 0.99)
    env = gym.make('LunarLander-v2')
    write_to_file(FILE_NAME,'NEW EXPERIMENT')

    for epoch in tqdm(range(EPOCHS)):
        #initialize state, sample from environment        
        s0 = env.reset()

        if epoch == 20:
            agent.explore = False

        iteration = 0
        done = False
        while not done:
            iteration += 1
            agent.experience_count += 1
            Q = agent(s0)

            a = agent.act(Q)
            s1, r, d, _ = env.step(a) 
            
            if d or iteration > 1000:
                done = True

            agent.train((s0, a, r, s1, done))
            s0 = np.copy(s1)

            if done:
                process_stats(agent, file_name) 
                break

             
    
