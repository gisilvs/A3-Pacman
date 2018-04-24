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
from game import Directions
import game
from capture import SIGHT_RANGE


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed, first = 'DefensiveReflexAgent', second = 'OffensiveReflexAgent'):
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
class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """

        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        #start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        #print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != util.nearestPoint(pos):
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

    def isOnRed(self, gameState, xCoord):
        """ say whether the given x coordinate is on the red teams side"""
        halfway = int(gameState.data.layout.width / 2)
        return xCoord < halfway


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)

        self.analyzeMap(gameState)

        CaptureAgent.registerInitialState(self, gameState)


    def analyzeMap(self, gameState):
        isRed = gameState.isOnRedTeam(self.index) # is the agent red?
        halfway = int(gameState.data.layout.width / 2)
        walls = gameState.data.layout.walls

        deadEnds = {}  # Dictionary with coordinates as key and object as value
        self.carryLimit = len(self.getFood(gameState).asList())//4
        highWays = []
        highWayDeadEnd = {}

        # x boundaries for the opponents side
        if isRed:
            XhomePos = halfway-2
            x_start = halfway
            x_stop = gameState.data.layout.width -1
        else:
            XhomePos = halfway+1
            x_start = 0
            x_stop = halfway-1

        y_stop = gameState.data.layout.height-1

        for y in range(gameState.data.layout.height):
            if not gameState.hasWall(XhomePos, y):
                self.homePos = (XhomePos, y)
                break

        for x_coord in range(x_start, x_stop):
            for y_coord in range(y_stop):

                if walls[x_coord][y_coord]:
                    continue

                elif self.numSurroundingWalls(gameState, [x_coord, y_coord]) == 3:
                    key = (x_coord, y_coord)
                    deadEnd = [key]
                    deadEnds[key] = deadEnd

                    x = x_coord
                    y = y_coord

                    while True:
                        highWay = ()
                        if not walls[x+1][y] and (x+1, y) not in deadEnds:
                            if self.numSurroundingWalls(gameState, [x+1, y]) == 2:
                                x = x+1
                            else:
                                highWay = (x+1, y)

                        elif not walls[x-1][y] and (x-1, y) not in deadEnds:
                            if self.numSurroundingWalls(gameState, [x-1, y]) == 2:
                                x = x-1
                            else:
                                highWay = (x-1, y)

                        elif not walls[x][y+1] and (x, y+1) not in deadEnds:
                            if self.numSurroundingWalls(gameState, [x, y+1]) == 2:
                                y = y+1
                            else:
                                highWay = (x, y+1)

                        elif not walls[x][y-1] and (x, y-1) not in deadEnds:
                            if self.numSurroundingWalls(gameState, [x, y-1]) == 2:
                                y = y-1
                            else:
                                highWay = (x, y-1)


                        if highWay in highWays:
                            highWayDeadEnd[highWay].append((x, y))
                            break

                        elif len(highWay) > 0:
                            highWays.append(highWay)
                            highWayDeadEnd[highWay] = [(x, y)]
                            break

                        key = (x, y)
                        deadEnd.append(key)
                        deadEnds[key] = deadEnd

        # Goes through the highways and checks which are adjecent to several dead ends
        for highWay in highWays:
            if len(highWayDeadEnd[highWay]) > 1:
                if len(highWayDeadEnd[highWay]) == 2 and self.numSurroundingWalls(gameState, [highWay[0], highWay[1]]) == 0:
                    continue
                else:
                    firstDEList = deadEnds[highWayDeadEnd[highWay][0]]
                    for restOutlet in highWayDeadEnd[highWay][1:]:
                        DEList = deadEnds[restOutlet]
                        for posDE in DEList:
                            firstDEList.append(posDE)
                            deadEnds[posDE] = firstDEList
                    firstDEList.append(highWay)
                    deadEnds[highWay] = firstDEList

                    x = highWay[0]
                    y = highWay[1]
                    while True:
                        if not walls[x + 1][y] and (x + 1, y) not in deadEnds:
                            if self.numSurroundingWalls(gameState, [x + 1, y]) == 2:
                                x = x + 1
                            else:
                                break

                        elif not walls[x - 1][y] and (x - 1, y) not in deadEnds:
                            if self.numSurroundingWalls(gameState, [x - 1, y]) == 2:
                                x = x - 1
                            else:
                                break

                        elif not walls[x][y + 1] and (x, y + 1) not in deadEnds:
                            if self.numSurroundingWalls(gameState, [x, y + 1]) == 2:
                                y = y + 1
                            else:
                                break

                        elif not walls[x][y - 1] and (x, y - 1) not in deadEnds:
                            if self.numSurroundingWalls(gameState, [x, y - 1]) == 2:
                                y = y - 1
                            else:
                                break

                        key = (x, y)
                        firstDEList.append(key)
                        deadEnds[key] = firstDEList

        self.deadEnds = deadEnds


    def numSurroundingWalls(self, gameState, coords):
        numWalls = 0
        if gameState.hasWall(int(coords[0]+1), int(coords[1])):
            numWalls += 1
        if gameState.hasWall(int(coords[0]-1), int(coords[1])):
            numWalls += 1
        if gameState.hasWall(int(coords[0]), int(coords[1]+1)):
            numWalls += 1
        if gameState.hasWall(int(coords[0]), int(coords[1]-1)):
            numWalls += 1
        return numWalls

    def makeObservations(self, gameState, successor):

        """
        score [INT]: The current score
        isAtStart [BOOL]: Is the agent at its start position, it it probably dead
        minGhostDist [INT]: The distance to the ghost that is closest and within 5 steps (maze distance)
        fussyGhostDists [LIST]: Fussy distances to every opponent ghost
        minFoodDist [INT]: Distance to the closest food
        inDeadEnd [BOOL]: Is the agent in a dead end
        numFoodInDE [INT]: How many foods is in the dead end the agent is in (if in DE)
        possibleToEscape [BOOL]: If it is possible to escape a dead end
        safeToEnter [BOOL]: If it is safe to enter a dead end
        numCarriedFood [INT]: The amount of carried food by the agent
        activePowerpill [BOOL]: If any opponent ghost is scared
        minCapsulesDist [INT]: The min distance to a power pill
        """



        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        ######## SCORE ########
        score = self.getScore(successor)

        ######## IS AT START (DEAD) ########
        isAtStart = myPos == self.start

        ######## GHOSTDIST ########
        defenders = [a for a in enemies if not a.isPacman and a.scaredTimer < 5 and a.getPosition() != None]

        if myPos != self.start and len(defenders) > 0:
            minGhostDist = min([self.getMazeDistance(myPos, a.getPosition()) for a in defenders])
        else:
            minGhostDist = SIGHT_RANGE +1

        ######## FUSSYGHOSTDIST ########
        fussyGhostDists = [successor.getAgentDistances()[i]
                           for a, i in zip(enemies, self.getOpponents(successor)) if not a.isPacman]

        ######## CLOSESTS FOOD ########
        foodList = self.getFood(gameState).asList()

        if len(foodList) > 0:
            minFoodDist = min([self.getMazeDistance(myPos, food) for food in foodList])
        else:
            minFoodDist = 0

        ######## IS IN DEAD END ########
        inDeadEnd = myPos in self.deadEnds

        ######## IS IN DEAD END WITH FOOD SOMEWHERE ########
        numFoodInDE = 0
        if inDeadEnd:
            for position in self.deadEnds[myPos]:
                if gameState.hasFood(position[0], position[1]):
                    numFoodInDE += 1

        ######## POSSIBLE TO ESCAPE DEAD END ########
        possibleToEscape = True
        if inDeadEnd and len(defenders):
            distToOutlet = self.getMazeDistance(myPos, (self.deadEnds[myPos][-1][0], self.deadEnds[myPos][-1][1]))
            if minGhostDist - distToOutlet > 1:
                possibleToEscape = True
            else:
                possibleToEscape = False

        ######## SAFE TO ENTER DEAD END ########
        safeToEnter = True
        if inDeadEnd and len(defenders):
            safeToEnter = False


        ######## CARRIED FOOD ########
        numCarriedFood = gameState.data.agentStates[self.index].numCarrying

        ######## POWERPILL ACTIVE (MIN ONE GHOST SCARED) ########
        activePowerpill = False
        for enemy in enemies:
            if enemy.scaredTimer > 0:
                activePowerpill = True
                break

        ######## DISTANCE TO POWERPILL ########
        capsulesList = self.getCapsules(gameState)

        if len(capsulesList) > 0 and not activePowerpill:
            minCapsulesDist = min([self.getMazeDistance(myPos, capsule) for capsule in capsulesList])
        else:
            minCapsulesDist = 0

        return [score, isAtStart, minGhostDist, fussyGhostDists, minFoodDist, inDeadEnd,
                numFoodInDE, possibleToEscape, safeToEnter, numCarriedFood, activePowerpill, minCapsulesDist]



    def getFeatures(self, gameState, action):

        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        observations = self.makeObservations(gameState, successor)

        features["score"] = observations[0]
        features["isAtStart"] = int(observations[1])
        features["minGhostDist"] = observations[2]
        #features["fussyGhostDists"] = sum(observations[3])/len(observations[3])
        features["minFoodDist"] = observations[4]
        if observations[5]: # Is in dead end
            if observations[6] == 0:
                features["inDeadEnd"] = 1
        features["possibleToEscapeDE"] = int(observations[7])
        features["SafeToEnterDE"] = int(observations[8])

        if observations[9] >= self.carryLimit or len(self.getFood(gameState).asList()) <= 2: # Check the number of food carrying
            features["distHome"] = self.getMazeDistance(self.homePos, myPos)

        features["minDistPowerPill"] = observations[11]
        if action.lower() == "stop":
            features['stop'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'score': 100, 'minFoodDist': -1, 'distHome': -10, 'minGhostDist': 100, 'inDeadEnd': -2000,
                "possibleToEscapeDE": 100, "SafeToEnterDE": 100, "stop":-100, "isAtStart": -1000,
                'minDistPowerPill': -2}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    c ould be like.  It is not the best or only way to make
    such an agent.
    """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)

        if self.red:
            self.foodLocations = gameState.getRedFood().asList()
        else:
            self.foodLocations = gameState.getBlueFood().asList()


        self.investigationTarget = self.foodLocations[0]
        self.huntingTarget = True

        CaptureAgent.registerInitialState(self, gameState)

    def makeObservations(self, gameState, successor):
        """
        numInvaders [INT]: The number of invaders
        distToMissingFood [INT]: Distance to position where a food has disappeared most recently
        minIntruderDist [INT]: The shortest distance to the intruder, if in range
        isScared [BOOL]: Has the opponent taken a PowerPill
        isPacMan [BOOL]: Is the agent a Pacman?
        minDistPowerPill [INT]: The shortest distance to one of our PowerPills

        """
        currentState = gameState.getAgentState(self.index)
        successorState = successor.getAgentState(self.index)
        successorPos = successorState.getPosition()
        currentPos = currentState.getPosition()
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        intruders = [a for a in enemies if a.isPacman and a.getPosition() != None]


        ######## NUMBER OF INVADERS ########
        #numInvaders = len(intruders)

        ######## DISTANCE TO MOST REACENTLY EATEN FOOD ########
        if self.red:
            currFoodLoc = gameState.getRedFood().asList()
        else:
            currFoodLoc = gameState.getBlueFood().asList()

        # if any food has disappeared
        if len(currFoodLoc) < len(self.foodLocations):
            for food in self.foodLocations:
                if food not in currFoodLoc:
                    self.investigationTarget = food
                    self.huntingTarget = True
                    break

        distToMissingFood = self.getMazeDistance(successorPos, self.investigationTarget)
        self.foodLocations = currFoodLoc

        ######## REACHED LOST FOOD TARGET ########
        if currentPos == self.investigationTarget:
            self.huntingTarget = False

        ######## DISTANCE TO OPPONENT PACMAN ########

        if len(intruders) > 0:
            minIntruderDist = min([self.getMazeDistance(successorPos, a.getPosition()) for a in intruders])
        else:
            minIntruderDist = 0

        ######## IS SCARED? ########
        isScared = currentState.scaredTimer > 0

        ######## CHECKS IF IT IS PACMAN ########
        isPacMan = successorState.isPacman

        ######## MIN DISTANCE TO OWN POWER PILL ########
        if self.red:
            currPowerPillLoc = gameState.getRedCapsules()
        else:
            currPowerPillLoc = gameState.getBlueCapsules()
        if len(currPowerPillLoc) > 0:
            minDistPowerPill = min([self.getMazeDistance(successorPos, capsule) for capsule in currPowerPillLoc])
        else:
            minDistPowerPill = 10


        return [distToMissingFood, minIntruderDist, isScared, isPacMan, minDistPowerPill]

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        observations = self.makeObservations(gameState, successor)

        if self.huntingTarget:
            features["distToMissingFood"] = observations[0]
        else:
            features["distToMissingFood"] = observations[4]
        features["minIntruderDist"] = observations[1]
        if observations[2]:
            features["minIntruderDist"] = -observations[1]

        features["isPacman"] = int(observations[3])

        return features

    def getWeights(self, gameState, action):
        return {'distToMissingFood': -10, 'minIntruderDist': -20, "isPacman": -100}
