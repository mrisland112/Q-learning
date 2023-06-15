import gym
import numpy as np
import pygame
from PIL import Image

#Get discrete state: return the corresponding index in q_table
def get_discrete_state(state, bins, s_dim):
  state_idx = [0] * s_dim

  for i in range(s_dim):
    state_idx[i] = np.digitize(state[i], bins[i]) - 1
  return tuple(state_idx)

#Main function
def main():
  #Parameters
  n_episode  = 4 #Number of episodes
  n_bins    = 16  #Number of bins

  #Create CartPole-v0 environment
  env = gym.make('CartPole-v0')
  s_dim = env.observation_space.shape[0] #continue
  a_dim = env.action_space.n       #discrete
  print(s_dim)
  print(a_dim)

  #Create 5D Q-table: (n_bins, n_bins, n_bins, n_bins, a_dim)
  bins = [
    np.linspace(-2.4, 2.4, n_bins),    #Cart position (discrete state 1)
    np.linspace(-4.0, 4.0, n_bins),    #Cart velocity (discrete state 2)
    np.linspace(-0.418, 0.418, n_bins),  #Pole angle (discrete state 3)
    np.linspace(-4.0, 4.0, n_bins),    #Pole velocity at tip (discrete state 4)
  ]
  q_table = np.random.uniform(low=-2, high=0, size=([n_bins]* s_dim + [a_dim]))
  print([n_bins]* s_dim + [a_dim])
  print(get_discrete_state([-2.2, 2.1, 0.3, 1.2], bins, s_dim))
  
  #Load Q-table
  with open("qtable.npy", "rb") as f:
    q_table = np.load(f)

  #Start testing
  #For each episode
  for i_episode in range(n_episode):
    state = env.reset()
    discrete_state = get_discrete_state(state, bins, s_dim)
    total_reward = 0
    episode_len = 0

    #For each time step
    while True:
      env.render()
      print("[{:2d}/{:2d}] length = {:d}, total_reward= {:.3f}".format(
          i_episode+1, n_episode, episode_len, total_reward
      ))
      #Choose action : a = argmax_a' Q(s, a')
      action = np.argmax(q_table[discrete_state])

      #Interact with the environment: put(s, a ,r ,s') to the replay buffer
      next_state, reward, done, _ = env.step(action)
      next_discrete_state = get_discrete_state(next_state, bins, s_dim)
      
      #Q-learning (Update)
      discrete_state = next_discrete_state
      total_reward += reward
      episode_len += 1

      if done:
        print("[{:2d}/{:2d}] length = {:d}, total_reward= {:.3f}".format(
            i_episode+1, n_episode, episode_len, total_reward
        ))
        break    
  env.close()

if __name__ == '__main__':
  main()