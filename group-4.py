from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
         first = 'NotSoOffensiveAgent', second = 'DefensiveAgent'):
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

class SmartAgent(CaptureAgent):
  """
  A base class for search agents that chooses score-maximizing actions.
  """

  def registerInitialState(self, gameState):

    CaptureAgent.registerInitialState(self, gameState)
    self.boundary_top = True
    if gameState.getAgentState(self.index).getPosition()[0] == 1:
      self.isRed = True
    else:
      self.isRed = False

    self.boundaries = self.boundaryTravel(gameState)
    self.treeDepth = 3

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [action for action, value in zip(actions, values) if value == maxValue]

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

  def boundaryTravel(self, gameState):
    return (0, 0), (0, 0)


class NotSoOffensiveAgent(SmartAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

  def getAction(self, gameState):
    """
    Returns the expectimax action using self.depth and self.evaluationFunction
    All ghosts should be modeled as choosing uniformly at random from their
    legal moves.
    """
    opponents = {}
    for enemy in self.getOpponents(gameState):
      opponents[enemy] = gameState.getAgentState(enemy).getPosition()
    directions = {'north': (0, 1), 'south': (0, -1), 'east': (1, 0), 'west': (-1, 0)}
    ghost_weights = {'distance': 5, 'scared': 5}

    def getGhostActions(current_pos):
      walls = gameState.getWalls().asList()

      max_x = max([wall[0] for wall in walls])
      max_y = max([wall[1] for wall in walls])

      actions = []
      for direction in directions:
        action = directions[direction]
        new_pos = (int(current_pos[0] + action[0]), int(current_pos[1] + action[1]))
        if new_pos not in walls:
          if (1 <= new_pos[0] < max_x) and (1 <= new_pos[1] < max_y):
            actions.append(direction.title())

      return actions

    def getNewPosition(current_pos, action):
        act = directions[[direction for direction in directions if str(action).lower() == direction][0]]
        return (current_pos[0] + act[0], current_pos[1] + act[1])

    def expectation(gamestate, position, legalActions):
      ghost_dict = {}
      for action in legalActions:
        newPos = getNewPosition(position, action)
        ghost_dict[action] = self.getMazeDistance(position, newPos)*ghost_weights['distance']
      min_action = min(ghost_dict)
      for action in ghost_dict:
        if ghost_dict[action] == min_action:
          ghost_dict[action] = .8
        else:
          ghost_dict[action] = .2/len(legalActions)
      return ghost_dict

    def ghostEval(gamestate, opponents, opponent):
      newPos = opponents[opponent]
      enemy = gamestate.getAgentState(opponent)
      myPos = gamestate.getAgentState(self.index).getPosition()

      if enemy.scaredTimer != 0:
        distance = -self.getMazeDistance(myPos, newPos)*ghost_weights['distance']
      else:
        distance = self.getMazeDistance(myPos, newPos)*ghost_weights['distance']

      return distance

    def minimax(gamestate, depth, agent, opponents, alpha=-float('inf'), beta=float('inf')):
      """
      """
      # Get legal moves per agent
      legalActions = [action for action in gamestate.getLegalActions(self.index) if action != Directions.STOP]

      # Generate optimal action recursively
      actions = {}
      if agent == self.index:
        maxVal = -float('inf')
        for action in legalActions:
          eval = self.evaluate(gamestate, action)
          if depth == self.treeDepth:
            value = eval
          else:
            value = eval+minimax(self.getSuccessor(gamestate, action), depth, agent+1, opponents, alpha, beta)
          maxVal = max(maxVal, value)
          if beta < maxVal:
            return maxVal
          else:
            alpha = max(alpha, maxVal)
          if depth == 1:
            actions[value] = action
        if depth == 1:
          return actions[maxVal]
        return maxVal
      else:
        minVal = float('inf')
        for opponent in opponents:
          if gamestate.getAgentState(opponent).getPosition() is not None:
            legalActions = getGhostActions(opponents[opponent])
            expectations = expectation(gamestate, opponents[opponent], legalActions)
            for action in legalActions:
              new_opponents = opponents.copy()
              new_opponents[opponent] = getNewPosition(opponents[opponent], action)
              ghost_val = ghostEval(gamestate, new_opponents, opponent)*expectations[action]
              value = ghost_val + minimax(gamestate, depth+1, self.index, new_opponents, alpha, beta)
              minVal = min(minVal, value)
              if minVal < alpha:
                return minVal
              else:
                beta = min(beta, minVal)
        if minVal == float('inf'):
          return 0
        return minVal

    return minimax(gameState, 1, self.index, opponents)


  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()    

    # Computes distance to enemy ghosts
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

    features['invaderDistance'] = 0.0
    if len(invaders) > 0:
        features['invaderDistance'] = min([self.getMazeDistance(myPos, invader.getPosition()) for invader in invaders]) + 1

    if len(ghosts) > 0:
      ghostEval = 0.0
      scaredDistance = 0.0
      regGhosts = [ghost for ghost in ghosts if ghost.scaredTimer == 0]
      scaredGhosts = [ghost for ghost in ghosts if ghost.scaredTimer > 0]
      if len(regGhosts) > 0: 
        ghostEval = min([self.getMazeDistance(myPos, ghost.getPosition()) for ghost in regGhosts])
        if ghostEval <= 1:  ghostEval = -float('inf')
         
      if len(scaredGhosts) > 0: 
        scaredDistance = min([self.getMazeDistance(myPos, ghost.getPosition()) for ghost in scaredGhosts])
      if scaredDistance < ghostEval or ghostEval == 0:
        if scaredDistance == 0: features['ghostScared'] = -10
      features['distanceToGhost'] = ghostEval 

    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    if len(foodList) > 0:
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
      features['foodRemaining'] = len(foodList)

    # Compute distance to capsules
    capsules = self.getCapsules(gameState)
    if len(capsules) > 0:
      minDistance = min([ self.getMazeDistance(myPos, capsule) for capsule in capsules ])
      if minDistance == 0: minDistance = -100
      features['distanceToCapsules'] = minDistance

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'invaderDistance': -50, 'distanceToFood': -1, 'foodRemaining': -1, 'distanceToGhost': 2, 'ghostScared': -1, 'distanceToCapsules': -1, 'stop': -100, 'reverse': -20}



class DefensiveAgent(CaptureAgent):
  def __init__(self, index):
        self.index = index
        self.observationHistory = []

  def getSuccessor(self,gameState,action):
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != util.nearestPoint(pos):
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self,gameState,action):
    features = self.getFeatures(gameState,action)
    weights = self.getWeights(gameState,action)
    return features*weights

  def chooseAction(self,gameState):
    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState,a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions,values) if v == maxValue]
    return random.choice(bestActions)

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)
    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1
    teamNums = self.getTeam(gameState)
    features['Distancefromstart'] = self.getMazeDistance(gameState.getInitialAgentPosition(teamNums[0]), gameState.getInitialAgentPosition(teamNums[1]))
    features['stayApart'] = self.getMazeDistance(gameState.getAgentPosition(teamNums[0]), gameState.getAgentPosition(teamNums[1]))
    if(len(invaders) != 0):
      features['stayApart'] = 0
    return features

  def getWeights(self,gameState, action):
    return {'Distancefromstart': 5, 'numInvaders': -2000, 'onDefense': 400, 'stayApart': 4, 'invaderDistance':-800, 'stop':-10,'reverse':-2}