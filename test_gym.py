import gym
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')
import time 

# Number of steps you run the agent for 
num_steps = 1500

env = gym.make('MountainCar-v0')
obs = env.reset()

# Observation and action space 
obs_space = env.observation_space
action_space = env.action_space
print("The observation space: {}".format(obs_space))
print("The action space: {}".format(action_space))

for each_action in range(num_steps):

    action = env.action_space.sample() # make this more intelligent

    # apply the action
    obs, reward, done, trunct, info = env.step(action)
    print("The new observation is {}".format(obs)+" with reward being {}".format(reward) + "also whether its done {}".format(done))

    # Render the env
    env.render()

    # Wait a bit before the next frame unless you want to see a crazy fast video
    time.sleep(0.01)

    # If the epsiode is up, then start another one
    if done:
        env.reset()
