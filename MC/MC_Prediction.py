#!/usr/bin/env python
# coding=utf-8
'''
Author:Yuying Chen
Date:2017年04月05日 星期三 15时57分04秒
Info:
'''
# %matplotlib inline
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')

import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict

if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()

def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    
    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        discount_factor: Lambda discount factor.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # The final value function
    V = defaultdict(float)
    
    # Implement this!
    for i_episode in range(1, num_episodes + 1):
        # print episode we're on
        if i_episode % 1000 == 0:     # calculate by 1000
            # print("\rEpisode {}/{}.",format(i_episode, num_episodes), end = "")
            # output is being buffered, unless flush sys.stdout after each print you won't see the output immediately
            sys.stdout.flush()
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        for t in range(100):  # 100 times sampling
            action = policy(state)  # get the action under the state
            next_state, reward, done, _ = env.step(action)  # get next state
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        # # First Visit  Monte-Carlo Policy evaluation
        # # Find all states the we've visited in this episode
        # # We convert each state to a tuple so that we can use it as a dict key
        # # set will clear the duplicated elements automatically
        # states_in_episode = set([tuple(x[0]) for x in episode])
        # for state in states_in_episode:
        #     # Find the first occurance of the state in the episode
        #     first_occurance_idx = next(i for i, x in enumerate(episode) if x[0] == state)
        #     # Sum up all rewards since the first occurance
        #     G = sum([x[2] * (discount_factor ** i) for i, x in enumerate(episode[first_occurance_idx:])])
        #     # Calculate average return for this state over all sampled episodes
        #     returns_sum[state] += G
        #     returns_count[state] += 1.0
        #     V[state] = returns_sum[state] / returns_count[state]

        # Every-Visit  Monte-Carlo Policy evaluation
        states_in_episode = [tuple(x[0]) for x in episode]
        state_max = len(states_in_episode)
        G = np.zeros(shape=state_max)
        G[state_max-1] = (discount_factor ** (state_max-1))*episode[state_max-1][2]
        if (state_max > 1):
            for i in range(1,state_max):
                G[state_max-i-1] = G[state_max-i]+(discount_factor ** (state_max-i-1))*episode[state_max-i-1][2]
        for i in range(state_max):
            state = states_in_episode[i]
            returns_sum[state] += G[i]
            returns_count[state] += 1
        for state in states_in_episode:
            V[state] = returns_sum[state] / returns_count[state]
    return V


def sample_policy(observation):
    """
    A policy that sticks if the player score is > 20 and hits otherwise.
    """
    score, dealer_score, usable_ace = observation
    return np.array([1.0, 0.0]) if score >= 20 else np.array([0.0, 1.0])


V_10k = mc_prediction(sample_policy, env, num_episodes=10000)
plotting.plot_value_function(V_10k, title="10,000 Steps")

V_500k = mc_prediction(sample_policy, env, num_episodes=500000)
plotting.plot_value_function(V_500k, title="500,000 Steps")
