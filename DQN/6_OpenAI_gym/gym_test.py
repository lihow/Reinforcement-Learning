import gym
from gym import envs

# ref https://www.cnblogs.com/ailitao/p/11047311.html
# env = gym.make('CartPole-v0')
# env = gym.make('MountainCar-v0')
# env = gym.make('SpaceInvaders-v0')
# env = gym.make('Assault-ramDeterministic-v0') # 小游戏2d飞行打怪
# env = gym.make('Assault-ramNoFrameskip-v4')
# env = gym.make('BankHeist-ramDeterministic-v0') # 2d迷宫
# env = gym.make('Bowling-ram-v4') # 小游戏弹球
# env = gym.make('Breakout-ram-v0') # 打砖块
# env = gym.make('Enduro-ramDeterministic-v0') # 摩托大战 赛车
# env = gym.make('Enduro-v4') # 摩托大战 赛车 **
# env = gym.make('FishingDerby-v0') # 钓鱼
# env = gym.make('Frostbite-ram-v0') # 跳水
# env = gym.make('Gopher-ram-v0') # 小游戏鱼儿
# env = gym.make('Gravitar-v4') # 2d射击类小游戏
# env = gym.make('Hero-ramNoFrameskip-v0') # 小游戏类似挖宝藏
# env = gym.make('IceHockey-ram-v0') # 打冰球
# env = gym.make('Krull-ramDeterministic-v0') #小游戏打怪
# env = gym.make('KungFuMaster-ram-v0') # 功夫游戏
# env = gym.make('MontezumaRevenge-ram-v0') # 蒙特祖玛的复仇 **
# env = gym.make('MontezumaRevenge-ramDeterministic-v0')
# env = gym.make('MsPacman-ram-v0')  # 吃豆人游戏
# env = gym.make('NameThisGame-ram-v0') # 2d打怪
# env = gym.make('Phoenix-ram-v0') # 2d射击类小游戏
# env = gym.make('Pitfall-ram-v0') # 玛雅人的冒险 *
# env = gym.make('Pong-ram-v0') # 乒乓球游戏 **
# env = gym.make('Pooyan-ram-v0')  # 射击类游戏
# env = gym.make('PrivateEye-ram-v0') # 小游戏
# env = gym.make('Qbert-ram-v0') # 波特Q精灵跳塔
# env = gym.make('Riverraid-ram-v0') # 河流突袭 单人射击
# env = gym.make('RoadRunner-ram-v0') # 通道奔跑者 奔跑
# env = gym.make('Robotank-ram-v0') # 小游戏打靶
# env = gym.make('Seaquest-ram-v0') # 深海游弋 射击类游戏
# env = gym.make('Skiing-ram-v0') # 滑雪游戏 **
# env = gym.make('Solaris-ram-v0') # 沙罗曼蛇 射击游戏 **
# env = gym.make('StarGunner-ram-v0') # 射击游戏
# env = gym.make('Tennis-ram-v0') # 网球游戏 **
# env = gym.make('TimePilot-ram-v0') # 时空战机
# env = gym.make('Tutankham-ram-v0') # 图坦卡门 迷宫射击
# env = gym.make('UpNDown-ram-v0') # 开车游戏 **
# env = gym.make('Venture-ram-v0') # (未知游戏)
# env = gym.make('VideoPinball-ram-v0') # 类似桌上弹球 **
# env = gym.make('WizardOfWor-ram-v0') # 探索射击 **
# env = gym.make('YarsRevenge-ram-v0') # 亚尔的复仇 飞行射击
env = gym.make('Zaxxon-ram-v0') # 飞行射击


 
env = env.unwrapped                     # 打开包装
print(env.action_space)                 # 动作空间
print(env.action_space.n)               # 输出动作个数[0, n-1]
print(env.get_action_meanings())     # 得到动作含义[0, n-1]
print(env.action_space.sample())        # 从动作空间中随机选取一个动作 (0或1)
print(env.observation_space)            # 查看状态空间
print(env.observation_space.shape[0])   # 输出列数，即输出状态个数
print(env.observation_space.high)       # 查看状态的最高值
print(env.observation_space.low)        # 查看状态的最低值
# print(envs.registry.all())              # 输出可以用的模拟环境

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