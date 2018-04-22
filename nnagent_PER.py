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
    if MEMORY.tree.samples < BATCH_SIZE:
        return
    transitions = MEMORY.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    indices=[x[0] for x in transitions]
    priorities=[x[1] for x in transitions]
    tr=[x[2] for x in transitions]

    batch = Transition(*zip(*tr))
    P=priorities/MEMORY.tree.total()
    w=(MEMORY.tree.capacity*P)**(-MEMORY.b)
    max_w=np.max(w)
    w=w/max_w
    w_t = torch.from_numpy(w).type(FloatTensor).view(-1,1)
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
    state_action_values.data=state_action_values.data*w_t

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    index = policy_net(non_final_next_states).max(1)[1]
    next_state_values[non_final_mask] = target_net(non_final_next_states)[np.arange(index.size()[0]), index]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # Undo volatility (which was used to prevent unnecessary gradients)
    expected_state_action_values = Variable(expected_state_action_values.data.view(-1,1)*w_t)
    errors=np.abs((expected_state_action_values-state_action_values).data.numpy())

    for i in range(BATCH_SIZE):
        MEMORY.update(indices[i],errors[i])
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

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )

        self.samples=0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        if self.samples<self.capacity:
            self.samples+=1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

class ReplayMemory(object):
    e = 0.01
    a = 0.6
    b= 0.9

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.counter = 1

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append((idx,p, data))

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

    def count1(self):
        self.counter += 1

    def reset_counter(self):
        self.counter = 1


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        # todo: sort shapes
        self.conv1 = nn.Conv2d(6, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc3 = nn.Linear(30*14*32, 256)
        self.fc4 = nn.Linear(256, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x=x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

policy_net = DQN()
target_net = DQN()

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
BATCH_SIZE = 32
TARGET_UPDATE = 10
GAMMA = 0.99
MEMORY = ReplayMemory(10000)
REPLAY_PERIOD=4
load_memory=1
load_net=1

if load_memory == 1:
    try:
        with open("silver_memo_per.file", "rb") as f:
            MEMORY = pickle.load(f)
            MEMORY.reset_counter()
            print('MEMORY LOADED')
    except:
        print('COULDNT LOAD MEMORY')

if load_net == 1:
    try:
        policy_net = torch.load('silver_net_per')
        print('NET LOADED')
    except:
        print('COULDNT LOAD NET')
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

if use_cuda:
    policy_net.cuda()
    target_net.cuda()

optimizer = optim.RMSprop(policy_net.parameters(), lr=0.0002)


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='NNAgent', second='DefensiveReflexAgent'):
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

    def update_reward(self, gameState):

        reward= -0.1 #for time step
        if self.old_action==4:
            reward-=0.1

        food_in_belly = gameState.getAgentState(self.index).numCarrying
        food_returned = gameState.getAgentState(self.index).numReturned
        if food_in_belly==0:
            reward+=(self.last_distance-self.distance)

        if food_in_belly>self.last_food_in_belly:
            reward+=10*(food_in_belly-self.last_food_in_belly)
        reward+=10*(food_returned-self.last_food_returned)
        self.last_food_in_belly=food_in_belly
        self.last_food_returned=food_returned
        return reward

        '''
        self.reward+=0.01*self.getScore(gameState)
        self.reward-=0.001*len(self.getFood(gameState).asList())
        self.reward+=0.001*len(self.getFoodYouAreDefending(gameState).asList())
        self.reward-=0.00001*self.time'''
        #self.reward-=0.01*minDistance


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
        self.old_state = None
        self.old_action = None
        self.name = 'Steven'
        self.reward = 0
        self.time = 0
        self.update=0
        # self.alpha=0.00001
        self.old_q = None
        self.epsilon = 0.3

        ###Parameters for reward
        self.last_distance = 0
        self.last_food_in_belly=0
        self.last_food_returned=0

        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        '''

    def state_to_input(self, gameState):
        walls = np.array(gameState.data.layout.walls.data, dtype=int)
        capsules = np.zeros(walls.shape, dtype=int)
        c1=self.getCapsulesYouAreDefending(gameState)
        for c in c1:
            capsules[c]+=1
        c2=self.getCapsules(gameState)
        for c in c2:
            capsules[c] -= 1
        f1=np.array(self.getFoodYouAreDefending(gameState).data,dtype=int)
        f2=np.array(self.getFood(gameState).data,dtype=int)
        food = f1-f2
        my_agent=np.zeros(walls.shape, dtype=int)
        my_pos=gameState.getAgentPosition(self.index)
        if gameState.getAgentState(self.index).isPacman:
            my_agent[my_pos]=1
        else:
            my_agent[my_pos] = -1
        opponents = np.zeros(walls.shape, dtype=int)
        my_mate = np.zeros(walls.shape, dtype=int)

        for i in range(4):
            pos = gameState.getAgentPosition(i)
            if pos and i!=self.index:
                if gameState.isOnRedTeam(i):
                    if self.red:
                        if gameState.getAgentState(i).isPacman:
                            my_mate[pos] = 1
                        else:
                            my_mate[pos] = -1
                    else:
                        if gameState.getAgentState(i).isPacman:
                            opponents[pos] = 1
                        else:
                            opponents[pos] = -1
                else:
                    if self.red:
                        if gameState.getAgentState(i).isPacman:
                            opponents[pos] = 1
                        else:
                            opponents[pos] = -1
                    else:
                        if gameState.getAgentState(i).isPacman:
                            my_mate[pos] = 1
                        else:
                            my_mate[pos] = -1


        # todo:add probabilities,scared
        state_tensor=np.stack((walls,food,capsules,my_agent,my_mate,opponents))
        return state_tensor

    def pick_best_allowed_action(self, Q_values, allowed_actions):
        actions = ['North', 'South', 'East', 'West', 'Stop']
        value_list = list(zip(actions, Q_values.ravel()))
        while True:
            action = max(value_list, key=lambda x: x[1])
            if action[0] in allowed_actions:
                return action[0]
            else:
                value_list.remove(action)

    def action_to_int(self, action):
        if action == 'North':
            return 0
        elif action == 'South':
            return 1
        elif action == 'East':
            return 2
        elif action == 'West':
            return 3
        elif action == 'Stop':
            return 4

    def index_to_action(self, index):
        actions = ['North', 'South', 'East', 'West', 'Stop']
        return actions[index]

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        foodList = self.getFood(gameState).asList()
        myPos = gameState.getAgentState(self.index).getPosition()
        minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
        self.distance=minDistance
        self.time += 1
        state = self.state_to_input(gameState)
        if self.time > 1:
            reward=self.update_reward(gameState)
            reward_t = Tensor([reward])
            Q=policy_net(Variable(torch.from_numpy(state).unsqueeze(0).type(Tensor), volatile=True).type(FloatTensor))
            Q_index = Q.max(1)[1]
            Q_t=target_net(Variable(torch.from_numpy(state).unsqueeze(0).type(Tensor), volatile=True).type(FloatTensor))[np.arange(Q_index.size()[0]), Q_index].data.numpy()[0]
            error=np.abs(reward+GAMMA*Q_t-self.old_q)
            transition=Transition(torch.from_numpy(self.old_state).unsqueeze(0).type(Tensor), LongTensor([[self.old_action]]),
                        torch.from_numpy(state).unsqueeze(0).type(Tensor), reward_t)
            MEMORY.add(error,transition)
            if self.time % REPLAY_PERIOD ==0:
                optimize_model()
                self.update+=1
            if self.update % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        actions = gameState.getLegalActions(self.index)
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
        Q = policy_net(
            # Variable(self.state_to_input(gameState), volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
            Variable(torch.from_numpy(state).unsqueeze(0).type(Tensor), volatile=True).type(FloatTensor)).data
        Q = Q.numpy()[0]
        index = np.argmax(Q)
        action = self.index_to_action(index)

        if np.random.random() > self.epsilon:
            if action not in actions:
                self.old_action = self.action_to_int(action)
                action = random.choice(actions)
            else:
                self.old_action = self.action_to_int(action)

        else:
            action = random.choice(actions)
            self.old_action = self.action_to_int(action)

        self.old_q=Q[self.old_action]

        foodList = self.getFood(gameState).asList()
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            myPos = gameState.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            self.last_distance = minDistance

        self.old_state = state
        self.last_distance=self.distance
        return action

    def finalUpdate(self, winner, gameState):
        state = self.state_to_input(gameState)
        reward=self.update_reward(gameState)
        if winner=='Red':
          if self.red:
              reward+=1
          else:
              reward -= 1
        elif winner=='Blue':
            if self.red:
                reward -= 1
            else:
                reward += 1
        else:
            reward -= 0.1

        error=np.abs(reward-self.old_q)
        reward = Tensor([reward])
        transition = Transition(torch.from_numpy(self.old_state).unsqueeze(0).type(Tensor),
                                LongTensor([[self.old_action]]),
                                None, reward)
        MEMORY.add(error,transition)
        if MEMORY.counter>0 and MEMORY.counter%20==0:
            with open("silver_memo_per.file", "wb") as f:
                pickle.dump(MEMORY, f, pickle.HIGHEST_PROTOCOL)
                print('SAVING MEMORY')
            torch.save(policy_net, 'silver_net_per')
            print('SAVING NET')
            print('Iteration ',MEMORY.counter)
        MEMORY.count1()


class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def registerInitialState(self, gameState):
        self.name = 'paolo'
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        food_in_belly = gameState.getAgentState(self.index).numCarrying
        if food_in_belly>0:
            a=0
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        features['successorScore'] = -len(foodList)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance
        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 100, 'distanceToFood': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
