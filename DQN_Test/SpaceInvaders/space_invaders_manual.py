import gym
import cv2
import numpy as np
from PIL import Image
env = gym.make('SpaceInvaders-v0')
env = env.unwrapped
print(env.action_space.n)
print(env.get_action_meanings()) 

def play(episode=1000):
  for i in range(episode):
    observation = env.reset()
    while True:
      # env.render()
      img = Image.fromarray(observation, "RGB")
      img = img.resize((observation.shape[1]*2, observation.shape[0]*2), resample=Image.NEAREST)
      img = np.array(img)
      img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
      # observation = cv2.resize(observation, (observation.shape[1]*2, observation.shape[0]*2), cv2.INTER_NEAREST)
      cv2.imshow("Tetris Game", img)
      key = cv2.waitKey(20)
      action = 0 # NOOP
      if ord("k") == key:
        action = 1 # FIRE
      if ord("d") == key:
        action = 2 # RIGHT
      if ord("a") == key:
        action = 3 # LEFT
      if ord("e") == key:
        action = 4 # RIGHTFIRE
      if ord("q") == key: 
        action = 5 # LEFTFIRE  

      observation, reward, done, info = env.step(action)
      if done:
        print("Episode {} finished".format(i))
        break
  env.close()

if __name__ == "__main__":
  play()