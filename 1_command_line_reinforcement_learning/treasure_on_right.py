import numpy as np
import pandas as pd
import time

# -o---T

N_STATES = 6 # 1维世界的宽度 
ACTIONS = ['left', 'right'] # 探索者的可用动作
EPSILON = 0.9 # 贪婪度 greedy
ALPHA = 0.1 # 学习率
GAMMA = 0.9 # 奖励递减值
MAX_EPISODES = 13 #最大回合数
FRASH_TIME = 0.3 # 移动间隔时间

def build_q_table(n_states, actions):
  table = pd.DataFrame(np.zeros((n_states, len(actions))), columns=actions,)
  return table

'''
print(build_q_table(N_STATES, ACTIONS))
   left  right
0   0.0    0.0
1   0.0    0.0
2   0.0    0.0
3   0.0    0.0
4   0.0    0.0
5   0.0    0.0
'''

def choose_action(state, q_table):
  state_actions = q_table.iloc[state, :] # 选出这个state的所有action值
  if (np.random.uniform() > EPSILON) or (state_actions.all() == 0): # 非贪婪 or 这个state还没有探索过
    action_name = np.random.choice(ACTIONS)
  else:
    action_name = state_actions.idxmax()
  return action_name

def get_env_feedback(S, A):
  # This is how agent will interact with environment
  if A == 'right': # move right
    if S == N_STATES - 2: # terminate
      S_ = 'terminal'
      R = 1
    else:
      S_ = S + 1
      R = 0
  else:
    R = 0
    if S == 0:
      S_ = S # reach the wall
    else:
      S_ = S - 1
  return S_, R

def update_env(S, episode, step_counter):
  # This is how enviroment be updated
  env_list = ['-']*(N_STATES-1)+['T'] # '-----T' our enviroment
  if S == 'terminal':
    interaction = 'Episode %s : total_steps = %s' % (episode+1, step_counter)
    print('\r{}'.format(interaction), end='')
    time.sleep(2)
    print('\r         ', end='')
  else:
    env_list[S] = 'o'
    interaction = ''.join(env_list)
    print('\r{}'.format(interaction), end='')
    time.sleep(FRASH_TIME)

def rl():
  q_table = build_q_table(N_STATES, ACTIONS)
  for episode in range(MAX_EPISODES):
    step_counter = 0
    S = 0 # 初始位置
    is_terminated = False
    update_env(S, episode, step_counter)
    while not is_terminated:
      A = choose_action(S, q_table)
      S_, R = get_env_feedback(S, A)
      q_predict = q_table.loc[S, A]
      if S_ != 'terminal':
        q_target = R + GAMMA * q_table.iloc[S_, :].max()
      else:
        q_target = R
        is_terminated = True

      q_table.loc[S, A] += ALPHA * (q_target - q_predict)
      S = S_

      update_env(S, episode, step_counter+1)
      step_counter += 1
  return q_table

if __name__ == "__main__":
  q_table = rl()
  print("\r\nQ-table:\n")
  print(q_table)