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
import distanceCalculator

from capture import SONAR_NOISE_RANGE

import sys

from enum import Enum, Flag

import b3

import copy

# Classes decorated @genetic
GA_Classes = []

def createTeam(firstIndex, secondIndex, isRed, tree = None):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.
    """

    team = Team([Agent(firstIndex), Agent(secondIndex)], tree)


    return team.agents

class Mode(Flag):
    Defense = 1
    Offense = 2


class PathFinder(object):
    """A useful 'distancer' class"""

    class PathData(object):

        def __init__(self, distance, action):
            self.distance = distance
            self.action = action

            self.weightedDistance = None
            self.weightedAction = None

            self.deadCutOff = None

    def __init__(self, layout):
        self.layout = layout

        self.data = {}

    def getDistance(self, start, goal):
        """Shortest path distance from start to goal"""

        return self.data[(start, goal)].distance

    def getWeightedDistance(self, start, goal):
        """Shortest path distance from start to goal, staying within own field"""

        return self.data[(start, goal)].weightedDistance

    def getNextAction(self, start, goal):
        """Get next action from start bringing us closer to goal"""

        return self.data[(start, goal)].action

    def getNextWeightedAction(self, start, goal):
        """Get next action from start bringing us closer to goal, staying within own field"""

        return self.data[(start, goal)].weightedAction

    def getDeadEnd(self, pos):
        """Get the entry point to the dead end, if pos is a dead end"""
        return self.data[(pos, pos)].deadCutOff

    def calculateDistances(self, isRed):
        nodes = self.layout.walls.asList(False)

        data = {}

        for source in nodes:
            dist = {}
            prev = {}
            closed = {}

            for node in nodes:
                dist[node] = float("inf")

            import util
            queue = util.PriorityQueue()
            queue.push(source, 0)
            dist[source] = 0

            while not queue.isEmpty():
                node = queue.pop()

                if node in closed:
                    continue

                closed[node] = True
                nodeDist = dist[node]
                adjacent = []
                x, y = node

                if not self.layout.isWall((x,y+1)):
                    adjacent.append((x,y+1))
                if not self.layout.isWall((x,y-1)):
                    adjacent.append((x,y-1) )
                if not self.layout.isWall((x+1,y)):
                    adjacent.append((x+1,y) )
                if not self.layout.isWall((x-1,y)):
                    adjacent.append((x-1,y))

                for other in adjacent:
                    if not other in dist:
                        continue

                    oldDist = dist[other]
                    newDist = nodeDist+1

                    if newDist < oldDist:
                        dist[other] = newDist
                        queue.push(other, newDist)
                        prev[other] = node

            for target in nodes:

                action = "Stop"

                if source != target:
                    first = target

                    while prev[first] != source:
                        first = prev[first]

                    diff = (first[0] - source[0], source[1] - first[1])

                    diff = (diff[0] > 0 and 1 or diff[0] < 0 and -1 or 0, diff[1] > 0 and 1 or diff[1] < 0 and -1 or 0)

                    if diff[0] == 1:
                        action = "East"
                    elif diff[1] == -1:
                        action = "North"
                    elif diff[0] == -1:
                        action = "West"
                    elif diff[1] == 1:
                        action = "South"

                data[(source, target)] = PathFinder.PathData(dist[target], action)

        minWeighted, maxWeighted = 0, int(self.layout.width/2)

        if isRed:
            minWeighted, maxWeighted = maxWeighted, self.layout.width

        for source in nodes:
            dist = {}
            prev = {}
            closed = {}

            for node in nodes:
                dist[node] = float("inf")

            import util
            queue = util.PriorityQueue()
            queue.push(source, 0)
            dist[source] = 0

            while not queue.isEmpty():
                node = queue.pop()

                if node in closed:
                    continue

                closed[node] = True
                nodeDist = dist[node]
                adjacent = []
                x, y = node

                if not self.layout.isWall((x,y+1)):
                    adjacent.append((x,y+1))
                if not self.layout.isWall((x,y-1)):
                    adjacent.append((x,y-1) )
                if not self.layout.isWall((x+1,y)):
                    adjacent.append((x+1,y) )
                if not self.layout.isWall((x-1,y)):
                    adjacent.append((x-1,y))

                for other in adjacent:
                    if not other in dist:
                        continue

                    oldDist = dist[other]
                    newDist = nodeDist+1

                    if minWeighted <= other[0] < maxWeighted:
                        newDist += 1

                    if newDist < oldDist:
                        dist[other] = newDist
                        queue.push(other, newDist)
                        prev[other] = node

            for target in nodes:

                action = "Stop"

                if source != target:
                    first = target

                    while prev[first] != source:
                        first = prev[first]

                    diff = (first[0] - source[0], source[1] - first[1])

                    diff = (diff[0] > 0 and 1 or diff[0] < 0 and -1 or 0, diff[1] > 0 and 1 or diff[1] < 0 and -1 or 0)

                    if diff[0] == 1:
                        action = "East"
                    elif diff[1] == -1:
                        action = "North"
                    elif diff[0] == -1:
                        action = "West"
                    elif diff[1] == 1:
                        action = "South"

                data[(source, target)].weightedDistance = dist[target]
                data[(source, target)].weightedAction = action

        self.data = data

    def __isWall(self, x, y):
        if x < 0 or y < 0 or x >= self.layout.width or y >= self.layout.height:
            return True

        return self.layout.walls[x][y]

    def calculateMinimumCut(self, fSink, isRed = True, closestCut = False):
        """Calculates the minimum cut with sources beginning in the opponent team, specify closestCut to make cuts closer to the sinks"""


        innerEdge = int(self.layout.width/2)-1
        intruderEdge = innerEdge+1

        if not isRed:
            innerEdge, intruderEdge = intruderEdge, innerEdge

        sourceOffset = 0
        
        # If there's food right on the border to the enemy, we have to pretend that the intruder edge is an inner one in order to place a barrier there
        for y in range(self.layout.height):
            if fSink(innerEdge, y):

                if isRed:
                    sourceOffset = 1
                else:
                    sourceOffset = -1

                break

        # The trick with closest cut is to swap sinks and sources, which means we must allow the intruder edge to be used
        if closestCut:
            if isRed:
                sourceOffset = 1
            else:
                sourceOffset = -1

        extendedInnerEdge = innerEdge+sourceOffset
        extendedIntruderEdge = intruderEdge+sourceOffset

        # Both foward and reverse flow are mapped here using a 4-index tuple
        # (technically flow may really be capacity in proper terms)
        flow = {}

        sink = (-1, -1)
        source = (-2, -2)

        def getFlow(key):
            if key in flow:
                return flow[key]

            # Residual flow for these are not equally directed
            if key[1] == source or key[0] == sink:
                return 0

            # Don't allow sources to saturate when sinks and sources are swapped
            if closestCut and key[0] == source:
                return 1000

            return 1

        while True:
            queue = []

            prev = {}

            # Closest cut; swap sinks and sources
            if closestCut:
                minX, maxX = 0, int(self.layout.width/2)+sourceOffset

                if not isRed:
                    minX, maxX = int(self.layout.width/2)+sourceOffset, self.layout.width

                for y in range(self.layout.height):
                    for x in range(minX, maxX):
                        if fSink(x, y) and getFlow((source, (x, y))) > 0:
                            prev[(x, y)] = source
                            queue.append((x, y))
            else:
                for y in range(self.layout.height):
                    if not self.__isWall(intruderEdge, y) and not self.__isWall(extendedInnerEdge, y) and getFlow((source, (extendedInnerEdge, y))) > 0:
                        prev[(extendedInnerEdge, y)] = source
                        queue.append((extendedInnerEdge, y))

            while queue:
                pos = queue.pop(0)

                x, y = pos
                
                # Closest cut; swap sinks and sources
                if closestCut:
                    if x == extendedInnerEdge and getFlow((pos, sink)) > 0:
                        prev[sink] = pos
                        queue.append(sink)
                else:
                    if fSink(x, y) and getFlow((pos, sink)) > 0:
                        prev[sink] = pos
                        queue.append(sink)

                edges = [(x+1, y), (x, y-1), (x-1, y), (x, y+1)]

                for vert in edges:
                    if vert not in prev and vert[0] != extendedIntruderEdge and not self.__isWall(vert[0], vert[1]) and getFlow((pos, vert)) > 0:
                        prev[vert] = pos
                        queue.append(vert)

                if sink in prev:
                    # Found an augmenting path

                    bottleneck = float("inf")

                    cur = sink

                    while cur in prev:
                        next = prev[cur]

                        bottleneck = min(bottleneck, getFlow((next, cur)))

                        cur = next

                    cur = sink

                    while cur in prev:
                        next = prev[cur]

                        flow[(next, cur)] = getFlow((next, cur)) - bottleneck
                        flow[(cur, next)] = getFlow((cur, next)) + bottleneck

                        cur = next

            if sink not in prev:
                break

        markedVertices = {source:True}

        queue = []
        
        # Closest cut; swap sinks and sources
        if closestCut:
            minX, maxX = 0, int(self.layout.width/2)+sourceOffset

            if not isRed:
                minX, maxX = int(self.layout.width/2)+sourceOffset, self.layout.width

            for y in range(self.layout.height):
                for x in range(minX, maxX):
                    if fSink(x, y):
                        queue.append((x, y))
        else:
            for y in range(self.layout.height):
                if not self.__isWall(intruderEdge, y) and not self.__isWall(extendedInnerEdge, y):
                    queue.append((extendedInnerEdge, y))

        while queue:
            pos = queue.pop(0)

            markedVertices[pos] = True

            x, y = pos
            
            edges = [(x+1, y), (x, y-1), (x-1, y), (x, y+1)]

            for vert in edges:
                if vert not in markedVertices and vert[0] != extendedIntruderEdge and not self.__isWall(vert[0], vert[1]) and getFlow((pos, vert)) > 0:
                    queue.append(vert)

        minimumCut = []

        for vertFrom in markedVertices:
            x, y = vertFrom
            
            edges = [(x+1, y), (x, y-1), (x-1, y), (x, y+1)]

            for vertTo in edges:
                if vertTo[0] != extendedIntruderEdge and not self.__isWall(vertTo[0], vertTo[1]) and vertTo not in markedVertices:
                    minimumCut.append((vertFrom, vertTo))

        return minimumCut, markedVertices

    def calculateBarriers(self, state, isRed):

        def isSink(x, y):
            return state.data.food[x][y] or (x, y) in state.data.capsules

        minimumCut, _ = self.calculateMinimumCut(isSink, isRed, closestCut=True)

        self.guardPositions = []

        for start, end in minimumCut:
            self.guardPositions.append(end)

        # No guard positions?!
        if len(self.guardPositions) == 0:
            print("Debug: No guard positions?!", file = sys.stderr)
            return

    def calculateDeadEnds(self):
        points = self.layout.walls.asList(False)

        intruderEdge = int(self.layout.width/2)

        bCalculated = {}

        # Gets the symmetric opponent point
        def getOpponentPoint(point):
            return (self.layout.width-point[0]-1, self.layout.height-point[1]-1)

        for point in points:

            if point not in bCalculated and point[0] < intruderEdge:
                
                def isSink(x, y):
                    return point[0] == x and point[1] == y

                minCut, sourceSet = self.calculateMinimumCut(isSink)

                cutOff = None

                if len(minCut) == 1:
                    cutOff = minCut[0][1]

                for target in points:
                    if target not in sourceSet and target[0] < intruderEdge:
                        bCalculated[target] = True

                        if cutOff:
                            self.data[(target, target)].deadCutOff = cutOff

                            opponent = getOpponentPoint(target)

                            self.data[(opponent, opponent)].deadCutOff = getOpponentPoint(cutOff)


class Condition(b3.Condition):

    def __init__(self, func):
        super().__init__()

        self.__func = func

    def tick(self, tick):
        return self.__func(tick)

class DefendAction(b3.Action):

    class GuardData(object):

        def __init__(self):
            self.index = []
            self.currentIndex = -1
            self.positions = []

        def ageIndicies(self, positions):

            i = 0

            # We keep an ordered list of guard positions here,
            # with newer positions last that get a slight priority
            while i < len(self.positions):

                if self.positions[i] not in positions:
                    self.positions.pop(i)
                    continue

                i += 1

            for pos in positions:
                if pos not in self.positions:
                    self.positions.append(pos)

            if len(self.index) < len(self.positions):
                for i in range(len(self.positions)-len(self.index)):
                    self.index.append(100)
            elif len(self.index) > len(self.positions):
                for i in range(len(self.index)-len(self.positions)):
                    self.index.pop()

            for i in range(len(self.index)):
                self.index[i] += 1

            if self.currentIndex >= len(self.index):
                self.currentIndex = -1

        def getIgnoredIndex(self):
            iMax = 0
            
            for i in range(len(self.index)):
                if self.index[i] >= self.index[iMax]:
                    iMax = i

            return iMax

        def visitIndex(self, index):
            self.index[index] = 0

    def __init__(self):
        super().__init__()

    def tick(self, tick):
        agent = tick.agent
        team = agent.team
        
        scaredTime = agent.scaredTimer

        closestEnemy = None
        enemyDist = float("inf")
                
        enemies = team.getEnemies()

        for enemy in enemies:
            if enemy.isPacman:
                dist = team.getWeightedDistance(enemy, agent, uncertainty=1)

                if dist < enemyDist:
                    enemyDist = dist
                    closestEnemy = enemy

        # If we're scared, try to play offensive if we're close to the opposing field or have enemies close
        if scaredTime > 0:
            distToPacman = float("inf")

            opponentEdge = int(tick.gameState.data.layout.width/2)

            if not agent.team.red:
                opponentEdge = opponentEdge-1

            for y in range(tick.gameState.data.layout.height):
                if not tick.gameState.data.layout.isWall((opponentEdge, y)):
                    dist = team.getDistance(agent.pos, (opponentEdge, y))

                    if dist < distToPacman:
                        dist = distToPacman

            if enemyDist < scaredTime*2 or distToPacman*2 < scaredTime:
                return b3.FAILURE

        guardData = tick.blackboard.get(agent.index, tick.tree, self)

        if guardData is None:
            guardData = DefendAction.GuardData()

        guardData.ageIndicies(agent.team.pather.guardPositions)

        guardPositions = guardData.positions

        # First time guarding or the guard position we were going for was removed
        if guardData.currentIndex == -1:
            guardData.currentIndex = guardData.getIgnoredIndex()

        distanceToGuardPos = team.getWeightedDistance(agent.pos, guardPositions[guardData.currentIndex])

        for i, guardPos in enumerate(guardPositions):
            if team.pather.getDistance(agent.pos, guardPos) <= 3:
                guardData.visitIndex(i)

        # Reached this position, move to the next one (if it's safe to leave it)
        if distanceToGuardPos == 0 and enemyDist > 3:
            guardData.currentIndex = guardData.getIgnoredIndex()

        # If we only have one position to guard, never leave it
        distGuard = 0

        if len(guardPositions) > 1:
            distGuard = team.getWeightedDistance(guardPositions[guardData.currentIndex-1], guardPositions[guardData.currentIndex])

        if closestEnemy:
            # Only chase enemy if we're close enough
            enemyDist = max(team.getWeightedDistance(closestEnemy, guardPositions[guardData.currentIndex]), team.getWeightedDistance(agent.pos, guardPositions[guardData.currentIndex])-2)

        if (enemyDist <= distGuard and distanceToGuardPos > 2) or closestEnemy and team.getDistance(closestEnemy, agent.pos) == 1:
            agent.setTargetPosition(team.getEnemyPosition(closestEnemy))
        else:
            agent.setTargetPosition(guardPositions[guardData.currentIndex])

        deadEnd = team.pather.getDeadEnd(team.getEnemyPosition(enemy))

        # If an enemy is trapped in a dead end, move in to block
        if closestEnemy and deadEnd and team.isPosInTeam(deadEnd):
            distMe = team.getDistance(agent.pos, deadEnd)
            distEnemy = team.getDistance(closestEnemy, deadEnd, uncertainty=1)

            distCapsules = float("inf")

            for capsule in tick.gameState.data.capsules:
                for enemy in enemies:
                    dist = team.getDistance(enemy, capsule, uncertainty=1)

                    if dist < distCapsules:
                        distCapsules = dist

            # If some enemy is close to a capsule, try to kill the trapped pacman before he gets it
            if distMe <= distEnemy and distCapsules > 3:
                agent.setTargetPosition(deadEnd)

        tick.blackboard.set(agent.index, guardData, tick.tree, self)

        return b3.SUCCESS

class Team(object):

    def __init__(self, agents, offensiveTree = None):
        self.agents = agents

        self.pather = None

        for i, agent in enumerate(agents):
            agent.team = self

            if i % 2 == 0:
                agent.mode = Mode.Offense
            else:
                agent.mode = Mode.Defense

            agent.tempGuard = 0

        self.numAgents = len(self.agents)

        self.bt = b3.BehaviorTree()
        self.btBoard = b3.Blackboard()

        self.bt.root = b3.Priority()

        self.bt.root.children.append(self.getDefenseRoot())

        if offensiveTree is None:
            offensiveTree = self.GA_Handcrafted()

        self.bt.root.children.append(offensiveTree)

        self.food = []
        self.capsules = []

        self.enemy = {}

        self.GA_Inherited = False
        self.debugPrint = ""

    def getDistance(self, *args, uncertainty = 0):
        processedArgs = []

        uncertainDistance = 0

        for i, arg in enumerate(args):

            iOther = len(args)-i-1

            if isinstance(arg, game.AgentState):
                # If the sonar is more reliable, return that distance instead
                if self.isUsingSonarTracking(arg) and isinstance(args[iOther], Agent):
                    return arg.sonarDistance[args[iOther].index] + uncertainty * self.getEnemyUncertainty(arg)

                pos = self.getEnemyPosition(arg)

                processedArgs.append(pos)

                uncertainDistance += uncertainty * self.getEnemyUncertainty(arg)
            elif isinstance(arg, Agent):
                processedArgs.append(arg.pos)
            else:
                processedArgs.append(arg)

        return self.pather.getDistance(*processedArgs)+uncertainDistance

    def getWeightedDistance(self, *args, uncertainty = 0):
        processedArgs = []

        uncertainDistance = 0

        for i, arg in enumerate(args):

            iOther = len(args)-i-1

            if isinstance(arg, game.AgentState):
                # If the sonar is more reliable, return that distance instead
                if self.isUsingSonarTracking(arg) and isinstance(args[iOther], Agent):
                    return arg.sonarDistance[args[iOther].index] + uncertainty * self.getEnemyUncertainty(arg)

                pos = self.getEnemyPosition(arg)

                processedArgs.append(pos)

                uncertainDistance += uncertainty * self.getEnemyUncertainty(arg)
            elif isinstance(arg, Agent):
                processedArgs.append(arg.pos)
            else:
                processedArgs.append(arg)

        return self.pather.getWeightedDistance(*processedArgs)+uncertainDistance

    def isPosInTeam(self, pos):
        """Returns whether or not pos is inside our team's playing field"""

        minX, maxX = 0, int(self.pather.layout.width/2)

        if not self.red:
            minX, maxX = maxX, self.pather.layout.width

        return minX <= pos[0] < maxX

    def getTeamFieldStart(self):
        """Returns the x edge where our team's field begins"""

        ourEdge = int(self.pather.layout.width/2)-1

        if not self.red:
            ourEdge = ourEdge+1

        return ourEdge

    def getEnemyFieldStart(self):
        """Returns the x edge where the enemy team's field begins"""

        enemyEdge = int(self.pather.layout.width/2)

        if not self.red:
            enemyEdge = enemyEdge-1

        return enemyEdge

    def tick(self, agent, gameState):
        """Own behavior for BehaviorTree.tick as we need more information passed to the tree"""


        tick = b3.Tick()
        tick.agent = agent
        tick.gameState = gameState
        tick.blackboard = self.btBoard
        tick.tree = self.bt
        tick.debug = self.bt.debug
        tick.lastNode = None

        # Tick node
        self.bt.root._execute(tick)

        return tick

    def getDefenseRoot(self):
        root = b3.Sequence()

        def defMode(tick):

            # Only defend if we have something to defend
            if (tick.agent.mode == Mode.Defense or tick.agent.tempGuard > 0) and tick.agent.team.pather.guardPositions:
                tick.agent.tempGuard -= 1
                return b3.SUCCESS

            return b3.FAILURE

        root.children.append(Condition(defMode))
        root.children.append(DefendAction())

        return root

    def getEnemies(self):
        return self.enemy.values()

    def getEnemyPosition(self, enemy):
        return enemy.estimatedPosition

    def isUsingSonarTracking(self, enemy):
        return enemy.estimatedUncertainty >= SONAR_NOISE_RANGE/2

    def getEnemyUncertainty(self, enemy):
        if self.isUsingSonarTracking(enemy):
            return SONAR_NOISE_RANGE/2
        else:
            return enemy.estimatedUncertainty

    def registerInitialState(self, agent, gameState, bRed):

        self.red = bRed

        if self.red:
            self.food = gameState.getRedFood().asList()
            self.capsules = gameState.getRedCapsules()
        else:
            self.food = gameState.getBlueFood().asList()
            self.capsules = gameState.getBlueCapsules()

        if not self.GA_Inherited:
            if self.pather is None:
                self.pather = PathFinder(gameState.data.layout)
                self.pather.calculateDistances(self.red)
                self.pather.calculateBarriers(gameState, self.red)
            else:
                self.pather.calculateDeadEnds()

        if not self.enemy:
            for opIndex in agent.getOpponents(gameState):

                enemy = gameState.getAgentState(opIndex)

                if not hasattr(enemy, "sonarDistance"):
                    enemy.sonarDistance = {}
                    
                enemy.sonarDistance[agent.index] = opIndex in gameState.agentDistances and gameState.agentDistances[opIndex] or 0
                enemy.estimatedUncertainty = 0
                enemy.estimatedPosition = enemy.start.getPosition()
                self.enemy[opIndex] = enemy

        else:
            for opIndex in agent.getOpponents(gameState):
                self.enemy[opIndex].sonarDistance[agent.index] = opIndex in gameState.agentDistances and gameState.agentDistances[opIndex] or 0

    def trackEnemies(self, agent, gameState):
        """Track where enemies are based on sonar, and if they've eaten food or capsules"""

        # The agent that moved before us
        opIndex = (agent.index + (self.numAgents*2)-1) % (self.numAgents*2)

        curFood = []
        curCapsules = []

        rebuildBarriers = False
        
        if self.red:
            curFood = gameState.getRedFood().asList()
            curCapsules = gameState.getRedCapsules()
        else:
            curFood = gameState.getBlueFood().asList()
            curCapsules = gameState.getBlueCapsules()

        enemy = gameState.getAgentState(opIndex)

        enemy.sonarDistance = self.enemy[opIndex].sonarDistance

        # Update with new observations
        enemy.sonarDistance[agent.index] = gameState.agentDistances[opIndex]

        self.enemy[opIndex].sonarDistance[agent.index] = enemy.sonarDistance[agent.index]

        # Update this so we don't accidentally forget that it's not valid anymore
        self.enemy[opIndex].configuration = enemy.configuration

        # We can see the enemy, simply set uncertainty to 0
        if enemy.getPosition():
            enemy.estimatedUncertainty = 0
            enemy.estimatedPosition = enemy.getPosition()
            self.enemy[opIndex] = enemy
        else:
            
            # If we were completely aware of the enemy by proximity, maybe he commited suicide
            if self.enemy[opIndex].estimatedUncertainty == 0:
                agentDist = float("inf")

                # Enemy was killed during the last turn, and thus might have exploded food
                if self.enemy[opIndex].estimatedPosition == self.enemy[opIndex].start.getPosition():
                    rebuildBarriers = True

                for a in self.agents:
                    dist = self.pather.getDistance(a.pos, self.enemy[opIndex].estimatedPosition)

                    if dist < agentDist:
                        agentDist = dist

                if agentDist <= 1:
                    enemy.estimatedUncertainty = 0
                    enemy.estimatedPosition = enemy.start.getPosition()

                    self.enemy[opIndex] = enemy

                    # Enemy might have eaten some food
                    rebuildBarriers = True


            # Add time since we last saw the enemy, indicating that it can be within this area
            self.enemy[opIndex].estimatedUncertainty += 1

        # If any food or capsules were taken, we know it was done during the opponent's turn, and we know which index it was

        # Check capsules first as they're much fewer
        for pos in self.capsules:
            if pos not in curCapsules:
                enemy.estimatedUncertainty = 0
                enemy.estimatedPosition = pos

                self.enemy[opIndex] = enemy

                rebuildBarriers = True
                break

        if not rebuildBarriers:
            for pos in self.food:
                if pos not in curFood:
                    enemy.estimatedUncertainty = 0
                    enemy.estimatedPosition = pos

                    self.enemy[opIndex] = enemy

                    rebuildBarriers = True
                    break

        if rebuildBarriers:
            self.pather.calculateBarriers(gameState, self.red)
                    
        self.food = curFood
        self.capsules = curCapsules

    def checkImminentDeath(self, agent, action):
        """Checks if this agent's action will lead to the imminent death of an enemy"""

        # No enemy will die by this move
        if action == "Stop":
            return

        dir = (1, 0)

        if action == "North":
            dir = (0, -1)
        elif action == "West":
            dir = (-1, 0)
        elif action == "South":
            dir = (0, 1)

        newPos = (agent.pos[0] + dir[0], agent.pos[1] + dir[1])

        for i in self.enemy:
            enemy = self.enemy[i]

            if enemy.getPosition() == newPos and self.isPosInTeam(newPos):
                enemy.estimatedUncertainty = 0
                enemy.estimatedPosition = enemy.start.getPosition()
                enemy.configuration = None

                self.enemy[i] = enemy


    def chooseAction(self, agent, gameState):
        # Must run first in this method! Ensures agent.pos is always where they are
        self.gameState = gameState

        self.trackEnemies(agent, gameState)

        agent.targetPosition = None
        
        agent.debugDraw(self.pather.guardPositions, (0, 0.5, 0), clear=True)
        
        # Draw dead ends
        #options = self.pather.layout.walls.asList(False)

        #dead = [point for point in options if self.pather.getDeadEnd(point)]
        #cut = [self.pather.getDeadEnd(point) for point in options if self.pather.getDeadEnd(point)]

        #agent.debugDraw(dead, (0, 0.5, 0), clear=True)
        #agent.debugDraw(cut, (0.5, 0.5, 0))
        # end dead end debugging

        tick = self.tick(agent, gameState)

        if tick.lastNode:
            self.debugPrint = tick.lastNode.__class__.__name__

        if agent.targetPosition is not None:
            action = None

            if agent.mode == Mode.Defense:
                action = self.pather.getNextWeightedAction(agent.pos, agent.targetPosition)
            else:
                action = self.pather.getNextAction(agent.pos, agent.targetPosition)

            self.checkImminentDeath(agent, action)

            return action

        return game.Directions.STOP

    def GA_Handcrafted(self):
        root = b3.Priority([b3.Sequence([InOwnTeam(), b3.Inverter(EnemiesFarAway(5)), BecomeDefender(10)])])

        root.children.append(b3.Sequence([EnemiesFarAway(3),
                                          b3.Sequence([
                                              b3.Priority([
                                                  b3.Sequence([FoodEaten(3), ReturnHome()]),
                                                  b3.Sequence([b3.Inverter(ScoreThreshold(1)), FoodEaten(1), ShouldContinueEating(False, 5, 2)]),
                                                  b3.Sequence([FoodEaten(1), ShouldContinueEating(False, 5, 5)]),
                                                  FoodEaten(0)]),
                                              b3.Priority([EatFood(True), EatFood(False)])])]))

        root.children.append(b3.Sequence([SafePowerUp(), AreEnemiesScared(5), EatFood(False), FoodEaten(3), ShouldContinueEating(False, 2, 5)]))
        root.children.append(b3.Sequence([FoodEaten(1), ReturnHome()]))
        root.children.append(b3.Sequence([AvoidEnemy(5)]))

        return root

    def GA_Generate(self, seed):
        random.seed(seed)

        composites = [b3.Priority, b3.Sequence]
        decorators = [b3.Limiter, b3.Repeater, b3.RepeatUntilFailure, b3.RepeatUntilSuccess]

        # For the b3 classes, we define the expected GenerateInstance function here
        for c in composites:
            c.GenerateInstance = lambda c=c: c()

        for d in decorators:
            d.GenerateInstance = lambda d=d: d(None, random.randint(1, 20)) # We don't want to use -1 for these as we might just get an infinite loop

        b3.Inverter.GenerateInstance = lambda: b3.Inverter()
        b3.MaxTime.GenerateInstance = lambda: b3.MaxTime()

        decorators += [b3.Inverter, b3.MaxTime]

        classes = composites + decorators + GA_Classes

        elementCount = random.randint(5, 100)

        # First element in elements will be the root, other elements will simply be a target for getting children
        elements = [random.choice(composites).GenerateInstance()]

        for _ in range(elementCount):
            parent = random.choice(elements)

            newObj = random.choice(classes).GenerateInstance()

            # This object can have children
            if isinstance(newObj, b3.Composite) or isinstance(newObj, b3.Decorator):
                elements.append(newObj)

            # Depending on the type, we can have as many children as we want, or only one
            if isinstance(parent, b3.Composite):
                parent.children.append(newObj)
            elif isinstance(parent, b3.Decorator):
                parent.child = newObj

                elements.remove(parent)

        return elements[0]

    def GA_Crossover(self, parent1, parent2, seed):
        random.seed(seed)

        result = copy.deepcopy(parent1)

        nodes1, nodes2 = [], []
        prev1 = {}

        def recurseThrough(nodes, prev, parent):

            if isinstance(parent, b3.Composite):
                for c in parent.children:
                    nodes.append(c)

                    if prev is not None:
                        prev[c] = parent

                    recurseThrough(nodes, prev, c)

            elif isinstance(parent, b3.Decorator):
                c = parent.child

                if c:
                    nodes.append(c)

                    if prev is not None:
                        prev[c] = parent

                    recurseThrough(nodes, prev, c)

        if not result.children:
            raise AttributeError("Parent 1 in crossover has no child nodes!")

        if not parent2.children:
            raise AttributeError("Parent 2 in crossover has no child nodes!")

        recurseThrough(nodes1, prev1, result)
        recurseThrough(nodes2, None, parent2)

        # Randomly select one node from each parent tree
        r1 = random.randrange(0, len(nodes1))
        r2 = random.randrange(0, len(nodes2))

        # As a decorator we only have one child
        if isinstance(prev1[nodes1[r1]], b3.Decorator):
            prev1[nodes1[r1]].child = copy.deepcopy(nodes2[r2])
        else:
            # Find which child we are in our parent
            r1c = prev1[nodes1[r1]].children.index(nodes1[r1])

            # Replace that child with the other child
            prev1[nodes1[r1]].children[r1c] = copy.deepcopy(nodes2[r2])

        return result

    def GA_Mutate(self, gene, seed):
        random.seed(seed)

        return self.GA_Crossover(gene, self.GA_Generate(random.randint(0, 10e20)), random.randint(0, 10e20))

    def GA_InheritCalculations(self, pather):
        self.GA_Inherited = True
        self.pather = pather

class Agent(CaptureAgent):
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

        self.red = gameState.isOnRedTeam(self.index)

        self.team.registerInitialState(self, gameState, self.red)

        self.distancer = self.team.pather

        import __main__

        if '_display' in dir(__main__):
            self.display = __main__._display

    @property
    def pos(self):
        return self.team.gameState.getAgentState(self.index).getPosition()

    @property
    def numCarrying(self):
        return self.team.gameState.getAgentState(self.index).numCarrying

    @property
    def scaredTimer(self):
        return self.team.gameState.getAgentState(self.index).scaredTimer

    def chooseAction(self, gameState):
        return self.team.chooseAction(self, gameState)

    def setTargetPosition(self, position):
        """Sets the target position of this agent, called by the behavior tree"""
        self.targetPosition = position

def genetic(cla):
    """Sets that this class can be used as an action or condition"""
    GA_Classes.append(cla)

    def debugEnter(self, tick):
        tick.lastNode = self

    cla._enter = debugEnter

    return cla

@genetic
class BecomeDefender(b3.Action):

    def __init__(self, time):
        super().__init__()

        self.time = time

    @staticmethod
    def GenerateInstance():
        return BecomeDefender(random.randint(10, 30))

    def tick(self, tick):
        agent = tick.agent
        team = agent.team

        layout = tick.gameState.data.layout

        homeEdge = team.getTeamFieldStart()

        bestDist = float("inf")

        if team.isPosInTeam(agent.pos):
            bestDist = 0
        else:
            for y in range(layout.height):
                curGoal = (homeEdge, y)

                if layout.isWall(curGoal):
                    continue

                curDist = team.pather.getDistance(agent.pos, curGoal)

                enemyDist = float("inf")

                for enemy in team.getEnemies():
                    dist = team.getDistance(enemy, curGoal, uncertainty=-1)

                    if dist < enemyDist:
                        dist = enemyDist

                if (curDist < enemyDist and curDist < bestDist):
                    bestDist = curDist

        if bestDist < self.time:
            agent.tempGuard = self.time
            return b3.SUCCESS

        return b3.FAILURE

@genetic
class ScoreThreshold(b3.Condition):
    """Have our team amassed at least X points?"""

    def __init__(self, threshold):
        super().__init__()

        self.threshold = threshold

    @staticmethod
    def GenerateInstance():
        return ScoreThreshold(random.randint(-10, 10))

    def tick(self, tick):
        agent = tick.agent
        team = agent.team

        score = tick.gameState.data.score

        if not team.red:
            score *= -1

        if score >= self.threshold:
            return b3.SUCCESS
        else:
            return b3.FAILURE

@genetic
class ShouldContinueEating(b3.Condition):
    """If either of the thresholds aren't satisfied, we return false"""

    def __init__(self, safe, foodDistance, homeDistance):
        super().__init__()

        self.safe = safe
        self.foodDistance = foodDistance
        self.homeDistance = homeDistance

    @staticmethod
    def GenerateInstance():
        return ShouldContinueEating(random.choice([True, False]), random.randint(0, 10), random.randint(0, 10))

    def tick(self, tick):
        agent = tick.agent
        team = agent.team

        layout = tick.gameState.data.layout

        homeDistance = float("inf")

        homeEdge = team.getTeamFieldStart()

        for y in range(layout.height):
            pos = (homeEdge, y)

            if layout.isWall(pos):
                continue

            dist = team.pather.getDistance(agent.pos, pos)

            if (dist < homeDistance):
                homeDistance = dist

        if self.homeDistance < homeDistance:
            return b3.FAILURE

        food = team.red and tick.gameState.getBlueFood().asList() or tick.gameState.getRedFood().asList()

        foodDistance = float("inf")
        foodHomeDistance = float("inf")

        for pos in food:
            if not self.safe or not team.pather.getDeadEnd(pos):
                dist = team.pather.getDistance(pos, agent.pos)

                homeDist = float("inf")

                for y in range(layout.height):
                    home = (homeEdge, y)

                    if layout.isWall(home):
                        continue

                    distHome = team.pather.getDistance(pos, home)

                    if (distHome < foodHomeDistance):
                        homeDist = dist

                if self.homeDistance < homeDist:
                    continue

                if dist < foodDistance and homeDist < foodHomeDistance:
                    foodDistance = dist
                    foodHomeDistance = homeDist

        if self.foodDistance < foodDistance:
            return b3.FAILURE

        return b3.SUCCESS

@genetic
class EatFood(b3.Action):

    def __init__(self, safe, random = False):
        super().__init__()

        self.safe = safe
        self.random = random

    @staticmethod
    def GenerateInstance():
        return EatFood(random.choice([True, False]), random.randint(0, 50))

    def open(self, tick):
        """Find closest / random food we want to go to"""


        agent = tick.agent
        team = agent.team

        layout = tick.gameState.data.layout

        food = team.red and tick.gameState.getBlueFood().asList() or tick.gameState.getRedFood().asList()

        candidates = [pos for pos in food if pos != agent.pos and (not self.safe or not team.pather.getDeadEnd(pos))]

        if candidates:
            if self.random > 0:
                tick.blackboard.set(agent.index, (random.choice(candidates), self.random), tick.tree, self)
            else:

                bestDist = float("inf")
                bestFood = None

                for pos in candidates:
                    dist = team.pather.getDistance(pos, agent.pos)

                    if dist < bestDist:
                        bestDist = dist
                        bestFood = pos

                tick.blackboard.set(agent.index, (bestFood, 5), tick.tree, self)

    def close(self, tick):
        if not self.random:
            return

        agent = tick.agent

        tick.blackboard.set(agent.index, None, tick.tree, self)

    def tick(self, tick):
        agent = tick.agent
        team = agent.team

        food = tick.gameState.data.food

        candidate = tick.blackboard.get(agent.index, tick.tree, self)

        x, y = int(agent.pos[0]), int(agent.pos[1])

        # If we stumbled across any food, return success to handle eating like the regular FindFood
        if food[x][y] and not team.isPosInTeam(agent.pos):
            # Might have to find a new food target
            self.open(tick)
            return b3.SUCCESS

        if candidate and candidate[1] > 0:
            x, y = candidate[0]

            tick.blackboard.set(agent.index, (candidate[0], candidate[1]-1), tick.tree, self)

            if food[x][y]:
                agent.setTargetPosition(candidate[0])

                return b3.RUNNING
            else:
                return b3.FAILURE

        return b3.FAILURE

@genetic
class FoodEaten(b3.Condition):
         
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    @staticmethod
    def GenerateInstance():
        return FoodEaten(random.randint(1, 16))

    def tick(self, tick):
        agent = tick.agent

        if (agent.numCarrying > self.threshold):
            return b3.SUCCESS
        else:
            return b3.FAILURE
    
@genetic
class EnemiesFarAway(b3.Condition):
    
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    @staticmethod
    def GenerateInstance():
        return EnemiesFarAway(random.randint(1, 10))
        
    def tick(self, tick):
        agent = tick.agent
        team = agent.team

        enemyDist = float("inf")

        for enemy in team.getEnemies():
            dist = team.getDistance(enemy, agent, uncertainty=-1)

            if dist < enemyDist:
                dist = enemyDist

        if (self.threshold < enemyDist):
            return b3.SUCCESS
            
        return b3.FAILURE
    
@genetic
class InOwnTeam(b3.Condition):
    
    def __init__(self):
        super().__init__()

    @staticmethod
    def GenerateInstance():
        return InOwnTeam()
        
    def tick(self, tick):
        agent = tick.agent
        team = agent.team

        if team.isPosInTeam(agent.pos):
            return b3.SUCCESS
        else:
            return b3.FAILURE
    
@genetic
class AreEnemiesScared(b3.Condition):
    
    def __init__(self, range):
        super().__init__()

        self.range = range

    @staticmethod
    def GenerateInstance():
        return AreEnemiesScared(random.randint(0, 10))
        
    def tick(self, tick):
        agent = tick.agent
        team = agent.team

        gameState = tick.gameState
        
        enemyDist = float("inf")
        closestScared = 0
                
        for i, enemy in team.enemy.items():
            dist = team.getDistance(enemy, agent, uncertainty=-1)

            if dist < enemyDist:
                dist = enemyDist
                closestScared = gameState.getAgentState(i).scaredTimer

        # Only go into a dead end if that's our only option...
        if enemyDist-closestScared > self.range:
            return b3.SUCCESS

        return b3.FAILURE
    
@genetic
class AvoidEnemy(b3.Action):
    
    def __init__(self, range):
        super().__init__()
        self.range = range

    @staticmethod
    def GenerateInstance():
        return AvoidEnemy(random.randint(2, 5))
        
    def tick(self, tick):
        agent = tick.agent
        team = agent.team

        layout = tick.gameState.data.layout

        minX = int(max(agent.pos[0]-self.range, 0))
        maxX = int(min(agent.pos[0]+self.range+1, layout.width))

        minY = int(max(agent.pos[1]-self.range, 0))
        maxY = int(min(agent.pos[1]+self.range+1, layout.height))

        closeEnemies = []
                
        for i, enemy in [(i, tick.gameState.getAgentState(i)) for i in agent.getOpponents(tick.gameState)]:
            if enemy.getPosition():
                dist = team.pather.getDistance(enemy.getPosition(), agent.pos)

                if dist < self.range:
                    closeEnemies.append(i)

        # No enemies to avoid!
        if not closeEnemies:
            return b3.SUCCESS

        if hasattr(tick, "calculatedSuccessorAvoidEnemy"):
            agent.setTargetPosition(tick.calculatedSuccessorAvoidEnemy)
        else:
            turns = [agent.index]

            turns.extend(closeEnemies)

            turns.sort()

            curTurn = turns.index(agent.index)

            def recurse(state, turnIndex, alpha, beta, level):
                score = 0

                agentState = state.getAgentState(agent.index)

                # We are officially untouchable! (unless we're scared)
                # But we avoid scoring high on this so that we leave other options open as well
                if team.isPosInTeam(agentState.getPosition()):
                    score = len(closeEnemies)*self.range/2
                # If we wound up in a dead end, we can't run away from the enemy!
                elif team.pather.getDeadEnd(agentState.getPosition()):
                    score -= 100

                enemyStates = [state.getAgentState(i) for i in turns if i != agent.index]

                for enemy in enemyStates:
                    score += team.pather.getDistance(enemy.getPosition(), agentState.getPosition())

                # Reached maximum depth, not going to search deeper
                if level == self.range or (turns[turnIndex] == agent.index and score < alpha) or (turns[turnIndex] != agent.index and score > beta):
                    return score

                actions = state.getLegalActions(turns[turnIndex])

                nextTurnIndex = turnIndex + 1

                if nextTurnIndex >= len(turns):
                    nextTurnIndex = 0

                bestAction = game.Directions.STOP
                bestScore = float("inf")

                if turnIndex == agent.index:
                    bestScore = -bestScore

                for curAction in actions:
                    curScore = recurse(state.generateSuccessor(turns[turnIndex], curAction), nextTurnIndex, alpha, beta, level+1)

                    if (turns[turnIndex] == agent.index and curScore > bestScore) or (turns[turnIndex] != agent.index and curScore < bestScore):
                        bestScore = curScore
                        bestAction = curAction

                    if turns[turnIndex] == agent.index:
                        alpha = max(alpha, bestScore)
                    else:
                        beta = min(beta, bestScore)

                return bestScore
            
            actions = tick.gameState.getLegalActions(agent.index)

            nextTurn = curTurn + 1

            if nextTurn >= len(turns):
                nextTurn = 0

            bestScore = -float("inf")
            bestAction = game.Directions.STOP

            for action in actions:
                score = recurse(tick.gameState.generateSuccessor(agent.index, action), nextTurn, alpha=bestScore, beta=float("inf"), level = 1)

                if score > bestScore:
                    bestScore = score
                    bestAction = action

            bestPos = game.Actions.directionToVector(bestAction)

            bestPos = (bestPos[0] + agent.pos[0], bestPos[1] + agent.pos[1])

            tick.calculatedSuccessorAvoidEnemy = bestPos

            agent.setTargetPosition(bestPos)

            return b3.RUNNING
            
        return b3.FAILURE
    
@genetic
class SafePowerUp(b3.Action):

    def __init__(self):
        super().__init__()

    @staticmethod
    def GenerateInstance():
        return SafePowerUp()
        
    def tick(self, tick):
        '''
        get power up positions enemy position and try check if you can get there safely
        '''
        agent = tick.agent
        team = agent.team
           
        
        if team.red:
            capsules = tick.gameState.getBlueCapsules()
        else:
            capsules = tick.gameState.getRedCapsules()
        
        for cap in capsules:
            capDist = team.pather.getDistance(agent.pos, cap)

            enemyDist = float("inf")

            for enemy in team.getEnemies():
                dist = team.getDistance(enemy, cap, uncertainty=-1)

                if dist < enemyDist:
                    dist = enemyDist

            if (capDist < enemyDist):
                agent.setTargetPosition(cap)
                return b3.SUCCESS
            
        return b3.FAILURE
 
@genetic
class ReturnHome(b3.Action):
    
    def __init__(self):
        super().__init__()

    @staticmethod
    def GenerateInstance():
        return ReturnHome()

    def tick(self, tick):
        agent = tick.agent
        team = agent.team
        gameState = tick.gameState
        layout = tick.gameState.data.layout

        if team.isPosInTeam(agent.pos):
            return b3.SUCCESS

        homeEdge = team.getTeamFieldStart()

        bestGoal = None
        bestDist = float("inf")
        

        for y in range(layout.height):
            curGoal = (homeEdge, y)

            if layout.isWall(curGoal):
                continue

            curDist = team.pather.getDistance(agent.pos, curGoal)

            enemyDist = float("inf")

            for enemy in team.getEnemies():
                if (enemy.scaredTimer > 1):
                    continue
                
                dist = team.getDistance(enemy, curGoal, uncertainty=-1)

                if dist < enemyDist:
                    dist = enemyDist

            if (curDist < enemyDist and curDist < bestDist):
                bestGoal = curGoal
                bestDist = curDist

        if bestGoal:
            agent.setTargetPosition(bestGoal)
            return b3.RUNNING
        
        return b3.FAILURE
    
class TimeToGoHome(b3.Condition):
    def __init__(self):
        super().__init__()

    @staticmethod
    def GenerateInstance():
        return ReturnHome()

    def tick(self, tick):
        agent = tick.agent
        team = agent.team
        layout = tick.gameState.data.layout        
        gameState = tick.gameState
        
        scared0 = 0
        scared1 = 0
        
        if (gameState.isOnRedTeam(agent.index)):
            enemies = gameState.getBlueTeamIndices()
        else:
            enemies = gameState.getRedTeamIndices()
            
        scared0 =  gameState.getAgentState(enemies[0]).scaredTimer
        scared1 =  gameState.getAgentState(enemies[1]).scaredTimer
        
        
        homeEdge = team.getTeamFieldStart()

        bestDist = float("inf")

        for y in range(layout.height):
            curGoal = (homeEdge, y)

            if layout.isWall(curGoal):
                continue

            curDist = team.pather.getDistance(agent.pos, curGoal)

            if (curDist < bestDist):
                bestDist = curDist

        if(bestDist >= min(scared0, scared1)):
            return b3.SUCCESS
        else:
            return b3.FAILURE
        