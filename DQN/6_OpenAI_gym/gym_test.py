import gym

env = gym.make('CartPole-v0')
# env = gym.make('MountainCar-v0')
# env = gym.make('SpaceInvaders-v0')

env = env.unwrapped                     # 打开包装
print(env.action_space)                 # 动作空间
print(env.action_space.n)               # 输出动作个数[0, n-1]
print(env.env.get_action_meaning())     # 得到动作含义[0, n-1]
print(env.action_space.sample())        # 从动作空间中随机选取一个动作 (0或1)
print(env.observation_space)            # 查看状态空间
print(env.observation_space.shape[0])   # 输出列数，即输出状态个数
print(env.observation_space.high)       # 查看状态的最高值
print(env.observation_space.low)        # 查看状态的最低值


for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(observation)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
env.close()