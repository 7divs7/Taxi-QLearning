import re
import numpy as np
import gym
import random

# import and initialize environment
env = gym.make('Taxi-v3')
#env.render()

# initialize state and action size
action_size = env.action_space.n
state_size = env.observation_space.n
#print(action_size, state_size)

# initialize q-table
q_table = np.zeros((state_size, action_size))
print(q_table)

# initialize hyperparameters
total_episodes = 50000
total_test_episodes = 100
max_steps = 99      # episode must end if steps > max_steps

learning_rate = 0.7
discount_factor = 0.618

# exploration parameters
epsilon = 1.0       # exploration rate
max_epsilon = 1.0   # exploration probability at start
min_epsilon = 0.01  # min exploration probability
decay_rate = 0.01   # exponential decay rate for exploration

# TRAIN THE MODEL
for episode in range(total_episodes):
    # Reset the env
    state = env.reset()
    step = 0
    done = False

    for step in range(max_steps):
        # choose an action a in current state s
        # eploration vs exploitation
        exp_exp_tradeoff = random.uniform(0, 1)   # randomly select a number

        # if num > epsilon --> exploit
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(q_table[state, :])
        
        # else explore
        else:
            action = env.action_space.sample()

        # take the action and observe the reward and next state
        new_state, reward, done, info = env.step(action)

        # update Q(s,a) as per Bellman eqn
        q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[new_state, :]) - q_table[state, action])
        
        state = new_state

        # if done --> finish episode
        if done == True:
            break

    episode += 1

    # reduce epsilon (more exploit and less explore)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode)


# TESTING
env.reset()
rewards = []

for episode in range(total_test_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    print("************************************************")
    print("EPISODE: ", episode)

    for step in range(max_steps):
        env.render()
        # take action that has max reward
        action = np.argmax(q_table[state, :])

        new_state, reward, done, info = env.step(action)

        total_rewards += reward

        if done:
            rewards.append(total_rewards)
            print("Score: ", total_rewards)
            break

        state = new_state

env.close()
print("Score over time: ", str(sum(rewards)/total_test_episodes))