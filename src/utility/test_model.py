import sys, getopt
import numpy as np
from sklearn import neural_network
from sklearn.externals import joblib
import gym

np.set_printoptions(precision=2)

if __name__=="__main__":
    _, args = getopt.getopt(sys.argv[1:], "")
        
    try: 
        model_file = args[0]
    except IndexError:
        print('test_model.py <path_to_model>')
        sys.exit(2)
 
    model = joblib.load(model_file)
 
    print("loaded model...")

    env = gym.make('LunarLander-v2')
    for episode in xrange(20):
        s0 = env.reset()
        ret = 0
        done = False

        while not done:
            Q = model.predict([s0])[0]
            a = np.argmax(Q)
            s1, r, done, _ = env.step(a)
            ret += r
            env.render(mode='human')
            s0 = np.copy(s1)

        print("episode: {} R: {} Ave: {}".format(episode, r, ret))
    
