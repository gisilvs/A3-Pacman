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
    self.name='Steven'
    self.gamma=0.95
    self.reward=None
    self.time=0
    self.alpha=0.00001
    self.old_q = None
    self.epsilon = 0.2
    self.weights = np.loadtxt('weights.txt')#np.random.normal(0,0.1,3)
    self.old_features = None

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




  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    self.weights = np.loadtxt('weights.txt')
    self.time += 1
    actions = gameState.getLegalActions(self.index)
    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
    if np.random.random() > self.epsilon:
        Q = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == Q]
    else:
        Q = np.random.choice(values)
        bestActions = [a for a, v in zip(actions, values) if v == Q]

    foodLeft = len(self.getFood(gameState).asList())

    action = random.choice(bestActions)

    if self.time > 1:
        self.update_reward(gameState)
        Q_plus = self.reward + self.gamma * Q
        self.update_weights(Q_plus)
    self.old_q = Q
    self.old_features = self.getFeatures(gameState, action)
    # self.old_features.normalize()
    self.old_features = np.array((list(self.old_features.values())))
    if gameState.isOver():
        a = 0
    if self.final(gameState):
        a = 0
    # self.old_features=np.array((list(self.getFeatures(gameState,action).values())))
    return action

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    successor=gameState.generateSuccessor(self.index, action)
    input=self.state_to_input(successor)
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights