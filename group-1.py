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


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        # todo: sort shapes
        self.conv1 = nn.Conv2d(7, 16, 3)
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


use_cuda = False
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
REPLAY_PERIOD=4
load_memory=1
load_net=1

if load_net == 1:
    policy_net = torch.load('silver_net_per')


if use_cuda:
    policy_net.cuda()
    target_net.cuda()


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='NNAgent', second='NNAgent'):

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


class NNAgent(CaptureAgent):



    def registerInitialState(self, gameState):

        self.old_state = None
        self.old_action = None
        self.name = 'Petter'
        self.reward = 0
        self.time = 0
        self.update=0
        self.old_q = None
        self.epsilon = 0

        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

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
        opponents = np.zeros(walls.shape,dtype=int)
        opponents_prob= np.zeros(walls.shape)
        my_mate = np.zeros(walls.shape, dtype=int)
        is_scared=np.zeros(walls.shape)

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


            if gameState.getAgentState(i).scaredTimer > 0:
                if gameState.isOnRedTeam(i):
                    if self.red:
                        if i !=self.index:
                            is_scared-=my_mate
                        else:
                            is_scared-=my_agent
                    else:
                        if pos:
                            is_scared-=opponents
                        #else:
                            #is_scared+=opponents_prob

                else:
                    if self.red:
                        if pos:
                            is_scared-=opponents
                        #else:
                            #is_scared+=opponents_prob
                    else:
                        if i !=self.index:
                            is_scared-=my_mate
                        else:
                            is_scared-=my_agent

        #scared
        state_tensor=np.stack((walls,food,capsules,my_agent,my_mate,opponents,is_scared))
        return state_tensor

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
        state = self.state_to_input(gameState)

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
                action = random.choice(actions)

        else:
            action = random.choice(actions)

        return action


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
