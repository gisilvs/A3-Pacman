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
               first = 'DummyAgent', second = 'DummyAgent'):
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

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
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

    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.old_features = None
    '''
    Your initialization code goes here, if you need any.
    '''

  def update_reward(self,gameState):
      self.reward=0
      self.reward+=10*self.getScore(gameState)
      self.reward-=len(self.getFood(gameState).asList())
      self.reward +=len(self.getFoodYouAreDefending(gameState).asList())
      self.reward-=0.1*self.time

  def update_weights(self,Q_plus):
      self.weights=self.weights+self.alpha*(Q_plus-self.old_q)*self.old_features
      np.savetxt('weights.txt', self.weights)

  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    self.weights = np.loadtxt('weights.txt')
    self.time+=1
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

    action=random.choice(bestActions)

    if self.time>1:
        self.update_reward(gameState)
        Q_plus=self.reward+self.gamma*Q
        self.update_weights(Q_plus)
    self.old_q=Q
    self.old_features=self.getFeatures(gameState,action)
    #self.old_features.normalize()
    self.old_features=np.array((list(self.old_features.values())))
    if gameState.isOver():
        a=0
    if self.final(gameState):
        a=0
    #self.old_features=np.array((list(self.getFeatures(gameState,action).values())))
    return action

  def finalUpdate(self,winner):
      self.weights = np.loadtxt('weights.txt')
      if winner=='Red':
        if self.red:
            Q_plus=+100
        else:
            Q_plus= -100
        self.update_weights(Q_plus)
      elif winner=='Blue':
          if self.red:
              Q_plus = -100
          else:
              Q_plus = +100
          self.update_weights(Q_plus)
      else:
          self.update_weights(-10)

      np.savetxt('weights.txt',self.weights)
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
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    otherfood= self.getFoodYouAreDefending(successor).asList()
    features['successorScore'] = -len(foodList)+len(otherfood)  # self.getScore(successor)
    #features['scared']=successor.getAgentState(self.index).scaredTimer

    if successor.getAgentState(self.index).isPacman:
        features['ghost']=0
    else:
        features['ghost']=1
    team=self.getTeam(successor)
    if self.index != team[0]:
        mate_idx=team[0]
    else:
        mate_idx=team[1]

    if successor.getAgentState(mate_idx).isPacman:
        features['mate_ghost']=0
    else:
        features['mate_ghost']=1
    #features['scared_mate']=successor.getAgentState(mate_idx).scaredTimer
    dists=successor.getAgentDistances()
    opponents=self.getOpponents(successor)
    features['distance1']=dists[opponents[0]]
    if dists[opponents[0]] ==0 or dists[opponents[1]] == 0:
        a=0
    #features['scared_0']=successor.getAgentState(opponents[0]).scaredTimer
    #features['scared_1'] = successor.getAgentState(opponents[0]).scaredTimer
    if successor.getAgentState(opponents[0]).isPacman:
        features['pac1']=1
    else:
        features['pac1']=0
    if successor.getAgentState(opponents[1]).isPacman:
        features['pac2']=1
    else:
        features['pac2']=0
    features['distance2']=dists[opponents[1]]
    # Compute distance to the nearest food

    if len(foodList) > 0:  # This should always be True,  but better safe than sorry
        myPos = successor.getAgentState(self.index).getPosition()
        minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
        if not minDistance:
            features['distanceToFood'] = -1
        else:
            features['distanceToFood'] = minDistance

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman]
    #features['numInvaders'] = len(invaders)

    features['bias'] = 1
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': self.weights[0],'ghost':self.weights[1], 'mate_ghost':self.weights[2],
            'distance1': self.weights[3], 'distance2': self.weights[4],
            'pac1': self.weights[5], 'pac2': self.weights[6],
            'distanceToFood': self.weights[7],'bias': self.weights[-1]}

    '''
    You should change this in your own agent.
    '''


class ReflexCaptureAgent(CaptureAgent):

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
    self.name='Alfredo'
    self.start=gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)


  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
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
        dist = self.getMazeDistance(self.start,pos2)
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