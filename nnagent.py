# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from util import nearestPoint
from game import Directions
import numpy as np
import game
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
from torch.autograd import Variable
import pickle

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # Undo volatility (which was used to prevent unnecessary gradients)
    expected_state_action_values = Variable(expected_state_action_values.data)

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        #todo: sort shapes
        self.fc1 = nn.Linear(1844, 50)
        self.fc2 = nn.Linear(50, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


use_cuda = False
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
policy_net = DQN()
target_net = DQN()
#policy_net=torch.load('policy_net')
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

if use_cuda:
    policy_net.cuda()
    target_net.cuda()

optimizer = optim.RMSprop(policy_net.parameters(),lr=0.001)
memory = ReplayMemory(5000)
with open("memo.file", "rb") as f:
    memory = pickle.load(f)


BATCH_SIZE = 128
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
GAMMA = 0.9

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'NNAgent', second = 'NNAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]


class NNAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """
  def update_reward(self,gameState):
      self.reward=0
      self.reward+=0.01*self.getScore(gameState)
      self.reward-=0.001*len(self.getFood(gameState).asList())
      self.reward+=0.001*len(self.getFoodYouAreDefending(gameState).asList())
      self.reward-=0.0001*self.time

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    self.old_state= None
    self.old_action=None
    self.name='Steven'
    self.gamma=0.95
    self.reward=None
    self.time=0
    self.alpha=0.00001
    self.old_q = None
    self.epsilon = 0.2


    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''

  def state_to_input(self,gameState):
    rows=len(gameState.data.layout.walls.data)
    cols=len(gameState.data.layout.walls.data[0])
    wall=np.zeros(rows*cols,dtype=int)
    count=0
    for w_r in gameState.data.layout.walls.data:
        for w_c in w_r:
            if w_c is True:
                wall[count]=1
            count+=1
    food=np.zeros((rows,cols),dtype=int)
    for capsule in gameState.data.capsules:
        food[capsule[0],capsule[1]]=-1
    food=food.reshape(-1,1)
    count=0
    for f_r in gameState.data.food:
        for f_c in f_r:
            if food[count] == 0:
                if f_c is True:
                    food[count]=1
            count+=1
    wall=np.append(wall,food)
    agents=np.zeros((rows,cols),dtype=int)
    for i in range(4):
        pos=gameState.getAgentPosition(i)
        if pos:
            if gameState.isOnRedTeam(i):
                if self.red:
                    agents[pos[0],pos[1]]=1
                else:
                    agents[pos[0], pos[1]] = -1
            else:
                if self.red:
                    agents[pos[0],pos[1]]=-1
                else:
                    agents[pos[0], pos[1]] = 1
    #todo:add probabilities
    agents=agents.reshape(-1,1)
    wall=np.append(wall,agents)
    agent_state=np.zeros(4,dtype=int)
    scared=np.zeros(4,dtype=int)
    for i in range(4):
        if gameState.getAgentState(i).isPacman:
            agent_state[i]=1
        else:
            agent_state[i] = -1
        if gameState.getAgentState(i).scaredTimer >0:
            scared[i]=1
    wall = np.append(wall, agent_state)
    wall = np.append(wall, scared)

    return wall

  def pick_best_allowed_action(self,Q_values,allowed_actions):
      actions=['North','South','East','West','Stop']
      value_list=list(zip(actions,Q_values))
      while True:
        action=max(value_list, key=lambda x: x[1])
        if action[0] in allowed_actions:
            return action[0]
        else:
            value_list.remove(action)

  def action_to_int(self,action):
      if action =='North':
          return 0
      elif action =='South':
          return 1
      elif action =='East':
          return 2
      elif action =='West':
          return 3
      elif action =='Stop':
          return 4




  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    self.time += 1
    state = self.state_to_input(gameState)
    if self.time > 1:
        self.update_reward(gameState)
        reward = Tensor([self.reward])
        memory.push(torch.from_numpy(self.old_state).unsqueeze(0).type(Tensor), LongTensor([[self.old_action]]), torch.from_numpy(state).unsqueeze(0).type(Tensor), reward)
        optimize_model()
        if self.time % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())


    actions = gameState.getLegalActions(self.index)
    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
    Q = policy_net(
        # Variable(self.state_to_input(gameState), volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
        Variable(torch.from_numpy(state), volatile=True).type(FloatTensor)).data
    Q = Q.numpy()
    if np.random.random() > self.epsilon:
        action=self.pick_best_allowed_action(Q,actions)
    else:
        action = random.choice(actions)

    self.old_state=state
    self.old_action=self.action_to_int(action)
    return action

  def finalUpdate(self,winner,gameState):
      state = self.state_to_input(gameState)
      self.update_reward(gameState)
      if winner=='Red':
        if self.red:
            self.reward+=1
        else:
            self.reward -= 1
      elif winner=='Blue':
          if self.red:
              self.reward -= 1
          else:
              self.reward += 1
      else:
          self.reward -= 0.1

      reward = Tensor([self.reward])
      memory.push(torch.from_numpy(self.old_state).unsqueeze(0).type(Tensor), LongTensor([[self.old_action]]),
                  torch.from_numpy(state).unsqueeze(0).type(Tensor), reward)
      optimize_model()
      torch.save(policy_net,'policy_net')
      print(policy_net.state_dict())
      with open("memo.file", "wb") as f:
          pickle.dump(memory, f, pickle.HIGHEST_PROTOCOL)
