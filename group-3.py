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
import numpy as np
from game import Directions
import game


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed, first='OffenseAgent', second='DefenseAgent'):
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
trackers = [None, None]


class BaseAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    enemypos = [(-1,-1), (-1,-1)]

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
        global trackers
        CaptureAgent.registerInitialState(self, gameState)
        self.distancer.getMazeDistances()
        if self.red:
            trackers[0] = Tracking(gameState, gameState.blueTeam[0])
            trackers[1] = Tracking(gameState, gameState.blueTeam[1])
        else:
            trackers[0] = Tracking(gameState, gameState.redTeam[0])
            trackers[1] = Tracking(gameState, gameState.redTeam[1])

        self.enemies = self.getOpponents(gameState)
        '''
        Your initialization code goes here, if you need any.
        '''

    def chooseAction(self, gameState):
        global trackers
        t1 = time.time()
        self.enemypos[0] = trackers[0].trackPosition(self, gameState)
        self.enemypos[1] = trackers[1].trackPosition(self, gameState)
        simulationState = gameState.deepCopy()

        #print (self.enemypos)

        for i in range(len(self.enemypos)):
          conf = game.Configuration(self.enemypos[i], Directions.STOP)
          simulationState.data.agentStates[self.enemies[i]] = game.AgentState(conf, simulationState.isRed(self.enemypos[i]) != simulationState.isOnRedTeam(self.enemies[i]))


        move = self.expecti_maximize(simulationState, 3)[1]
        t2 = time.time()
        return move

    def expecti_maximize (self, gameState, depth):
        moves = gameState.getLegalActions(self.index)

        #print("Max depth:" + str(depth))
        if len(moves) > 1:
          moves.remove(Directions.STOP)

        if depth == 0 or gameState.isOver() or not moves:
          return self.evaluationFunction(gameState), None

        cost = []
        for move in moves:
          successor = gameState.generateSuccessor(self.index, move)
          cost.append((self.expecti_minimize(successor, self.enemies[0], depth-1)[0], move))
        return max(cost)

    def expecti_minimize(self, gameState, enemy, depth):
        #print("Min depth:" + str(depth))
        #print(enemy)
        moves = gameState.getLegalActions(enemy)

        if depth == 0 or gameState.isOver() or not moves:
          return self.evaluationFunction(gameState), None

        cost = []

        for move in moves:
          successor = gameState.generateSuccessor(enemy, move)
          if enemy == max(self.enemies):
            cost.append((self.expecti_maximize(successor, depth -1)[0], move))
          else: 
            cost.append((self.expecti_minimize(successor, enemy + 2, depth-1)[0], move))
        return sum(map(lambda x: float(x[0]) / len(cost), cost)), None

    def evaluationFunction(self, gameState):
        util.raiseNotDefined()


class OffenseAgent(BaseAgent):

  def registerInitialState(self, gameState):
    return BaseAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    return BaseAgent.chooseAction(self, gameState)

  def evaluationFunction(self, gameState):
    position = gameState.getAgentPosition(self.index)
    foods = self.getFood(gameState).asList()


    base_distance = abs(position[0] - gameState.getInitialAgentPosition(self.index)[0])
    enemy_distances = [self.distancer.getDistance(position, e) for e in self.enemypos]
    closest_enemy = min(enemy_distances)
    if (closest_enemy > 5):
      closest_enemy = 0

    if (gameState.getAgentState(self.index).numCarrying >= 5):
      return 1000*closest_enemy  - 2*base_distance
    else:
      food_distances = [self.distancer.getDistance(position, food) for food in foods]
      closest_food = min(food_distances) if len(food_distances) else 0
      return 2 * self.getScore(gameState) - 100 * len(foods) - 3 * closest_food + 2 * base_distance + 10 * closest_enemy
      


class DefenseAgent(BaseAgent):

  def registerInitialState(self, gameState):
    return BaseAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    return BaseAgent.chooseAction(self, gameState)

  def evaluationFunction(self, gameState):
    position = gameState.getAgentPosition(self.index)

    if gameState.getAgentState(self.index).isPacman:
        return -1000000

    enemy_pacmans = [gameState.getAgentState(enemy).isPacman for enemy in self.enemies]
    enemy_distances = [self.distancer.getDistance(position, e) for e in self.enemypos]

    return -100000 * len(enemy_pacmans) - 10* min(enemy_distances)


class Tracking:
    def __init__(self, gameState, index):
        self.index = index
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 0]
        self.beliefs = self.initializeBeliefs(gameState)
        self.transition = self.initializeTransitionMatrix()

    def initializeBeliefs(self, gameState):
        self.prevConfig = gameState.data.layout.agentPositions[self.index][1]
        return self.certainDistribution(gameState.data.layout.agentPositions[self.index][1])

    def initializeTransitionMatrix(self):
        transition = np.zeros((len(self.legalPositions), len(self.legalPositions)))
        for i, pos_i in enumerate(self.legalPositions, start=0):
            for j, pos_j in enumerate(self.legalPositions, start=0):
                if self.manhattanDistance(self.legalPositions[i], self.legalPositions[j]) <= 1:
                    transition[i][j] = 1.0

        return transition / transition.sum(axis=1)[:, None]

    def getObservationDistribution(self, p_pac, observation):
        distances = np.apply_along_axis(lambda x: self.manhattanDistance(x, p_pac), 1, np.asarray(self.legalPositions))
        min_obs = observation - 7
        max_obs = observation + 7
        dist = np.where(np.logical_and(np.greater(distances, min_obs),
                                       np.less(distances, max_obs),
                                       np.greater(distances, 5)), 1, 0)
        np_sum = np.sum(dist)
        return dist / np_sum

    def trackPosition(self, agent, gameState):
        obs = agent.getCurrentObservation()
        prev = agent.getPreviousObservation()
        my_pos = obs.data.agentStates[agent.index].configuration.pos
        configuration = obs.data.agentStates[self.index].configuration

        if my_pos == self.prevConfig:
            self.beliefs = self.initializeBeliefs(gameState)

        if configuration is not None:
            distribution = self.certainDistribution(configuration.pos)
            self.prevConfig = configuration.pos
            # if prev is not None and len(obs.data.capsules) != len(prev.data.capsules) \
            #        and obs.data.agentStates[self.index].isPacman:
            #    distribution = self.certainDistribution()
        #elif prev is not None and np.sum(obs.data.food.data) != np.sum(prev.data.food.data) \
         #       and obs.data.agentStates[self.index].numCarrying - prev.data.agentStates[self.index].numCarrying == 1:
         #   diff = np.array(prev.data.food.data) ^ np.array(obs.data.food.data)
         #   index = np.where(diff == 1)
         #   if len(index[0]) == 1:
         #       distribution = self.certainDistribution((index[0][0], index[1][0]))
          #  else:
          #      distribution = self.certainDistribution((index[0][0], index[1][0]))
        else:
            distribution = self.getObservationDistribution(my_pos, gameState.getAgentDistances()[self.index])

        return self.calculateForwardPass(distribution)


    def certainDistribution(self, targetPos):
        distribution = np.zeros(len(self.legalPositions))
        for num, pos in enumerate(self.legalPositions, start=0):
            if pos == targetPos:
                distribution[num] = 1.0
        return distribution


    def calculateForwardPass(self, obs):
        length = self.beliefs.size
        update = np.zeros(length)
        for i in np.arange(0, length):
            total = 0
            for j in np.arange(0, length):
                total += self.transition[j, i] * self.beliefs[j]
            update[i] = total * obs[i]
        update_sum = update.sum()
        if update_sum != 0:
            self.beliefs = update / update_sum
        return self.legalPositions[np.argmax(self.beliefs)]


    def manhattanDistance(self, xy1, xy2):
        return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
