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


import random, time, io, json, copy

import b3
import numpy as np

from captureAgents import CaptureAgent
from game import Directions
from capture import GameState
import game
import util

from pdb import set_trace as bp

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='MyAgent', second='MyAgent'):
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

    names = {'Collect Food': CollectFood,
             'Return Food': ReturnFood,
             'Should Return Food': ShouldReturnFood,
             'Ghost Nearby': GhostNearby,
             'Capsule Close Enough': CapsuleCloseEnough,
             'Collect Capsule': CollectCapsule,
             'Be Offensive': BeOffensive,
             'Pacman Nearby': PacmanNearby,
             'Hunt Pacman': HuntPacman,
             'Patrol': Patrol,
             }

    data = {
  "id": "2a6f4e50-f239-431c-8805-5bff0db74283",
  "title": "A behavior tree",
  "description": "",
  "root": "1df39656-d347-4146-8057-b01cfdcb841b",
  "properties": {},
  "nodes": {
    "d0ff0db6-56d3-4845-8bcd-cecf0e17d845": {
      "id": "d0ff0db6-56d3-4845-8bcd-cecf0e17d845",
      "name": "Ghost Nearby",
      "title": "Ghost Nearby",
      "description": "",
      "properties": {},
      "display": {
        "x": -1188,
        "y": 216
      }
    },
    "5de96d10-0eb2-4c9a-b353-486dbb03a229": {
      "id": "5de96d10-0eb2-4c9a-b353-486dbb03a229",
      "name": "Return Food",
      "title": "Return Food",
      "description": "",
      "properties": {},
      "display": {
        "x": -564,
        "y": 348
      }
    },
    "ca27a2f4-0c5f-4009-ada4-e6bcc24da81e": {
      "id": "ca27a2f4-0c5f-4009-ada4-e6bcc24da81e",
      "name": "Collect Food",
      "title": "Collect Food",
      "description": "",
      "properties": {},
      "display": {
        "x": 60,
        "y": 216
      }
    },
    "83425c8e-c0cb-475e-91f1-d1346ddcb9b1": {
      "id": "83425c8e-c0cb-475e-91f1-d1346ddcb9b1",
      "name": "Should Return Food",
      "title": "Should Return Food",
      "description": "",
      "properties": {},
      "display": {
        "x": -360,
        "y": 348
      }
    },
    "381f7e98-091d-481d-9bbd-2a1809f8fe31": {
      "id": "381f7e98-091d-481d-9bbd-2a1809f8fe31",
      "name": "Priority",
      "title": "Priority",
      "description": "Offensive sub-tree",
      "properties": {},
      "display": {
        "x": -528,
        "y": -48
      },
      "children": [
        "232f8e3a-ff3c-46cf-9f2e-a7b83a655da8",
        "8ccd1db2-a6bf-4a89-851b-2715b025af33"
      ]
    },
    "232f8e3a-ff3c-46cf-9f2e-a7b83a655da8": {
      "id": "232f8e3a-ff3c-46cf-9f2e-a7b83a655da8",
      "name": "Sequence",
      "title": "Sequence",
      "description": "",
      "properties": {},
      "display": {
        "x": -960,
        "y": 84
      },
      "children": [
        "d0ff0db6-56d3-4845-8bcd-cecf0e17d845",
        "e4dfc945-a0e9-4263-816f-618f4df94cdb"
      ]
    },
    "8ccd1db2-a6bf-4a89-851b-2715b025af33": {
      "id": "8ccd1db2-a6bf-4a89-851b-2715b025af33",
      "name": "Priority",
      "title": "Priority",
      "description": "",
      "properties": {},
      "display": {
        "x": -96,
        "y": 84
      },
      "children": [
        "127e4bb2-e2ac-41be-8a67-d14ced0d7a8c",
        "ca27a2f4-0c5f-4009-ada4-e6bcc24da81e"
      ]
    },
    "127e4bb2-e2ac-41be-8a67-d14ced0d7a8c": {
      "id": "127e4bb2-e2ac-41be-8a67-d14ced0d7a8c",
      "name": "Sequence",
      "title": "Sequence",
      "description": "",
      "properties": {},
      "display": {
        "x": -252,
        "y": 216
      },
      "children": [
        "83425c8e-c0cb-475e-91f1-d1346ddcb9b1",
        "040dad1e-4468-4b6e-92a5-c3a6f620acfc"
      ]
    },
    "040dad1e-4468-4b6e-92a5-c3a6f620acfc": {
      "id": "040dad1e-4468-4b6e-92a5-c3a6f620acfc",
      "name": "Return Food",
      "title": "Return Food",
      "description": "",
      "properties": {},
      "display": {
        "x": -144,
        "y": 348
      }
    },
    "32057ac9-6304-41ec-94f7-9c7babfbdaea": {
      "id": "32057ac9-6304-41ec-94f7-9c7babfbdaea",
      "name": "Capsule Close Enough",
      "title": "Capsule Close Enough",
      "description": "",
      "properties": {},
      "display": {
        "x": -984,
        "y": 468
      }
    },
    "e4dfc945-a0e9-4263-816f-618f4df94cdb": {
      "id": "e4dfc945-a0e9-4263-816f-618f4df94cdb",
      "name": "Priority",
      "title": "Priority",
      "description": "",
      "properties": {},
      "display": {
        "x": -720,
        "y": 216
      },
      "children": [
        "bb4ed532-4ce7-4e58-8457-fb65c4c3ee22",
        "5de96d10-0eb2-4c9a-b353-486dbb03a229"
      ]
    },
    "bb4ed532-4ce7-4e58-8457-fb65c4c3ee22": {
      "id": "bb4ed532-4ce7-4e58-8457-fb65c4c3ee22",
      "name": "Sequence",
      "title": "Sequence",
      "description": "",
      "properties": {},
      "display": {
        "x": -876,
        "y": 348
      },
      "children": [
        "32057ac9-6304-41ec-94f7-9c7babfbdaea",
        "bd2fd080-37dd-4a61-b0bb-04ee622535b2"
      ]
    },
    "bd2fd080-37dd-4a61-b0bb-04ee622535b2": {
      "id": "bd2fd080-37dd-4a61-b0bb-04ee622535b2",
      "name": "Collect Capsule",
      "title": "Collect Capsule",
      "description": "",
      "properties": {},
      "display": {
        "x": -768,
        "y": 468
      }
    },
    "1df39656-d347-4146-8057-b01cfdcb841b": {
      "id": "1df39656-d347-4146-8057-b01cfdcb841b",
      "name": "Priority",
      "title": "Priority",
      "description": "",
      "properties": {},
      "display": {
        "x": -108,
        "y": -312
      },
      "children": [
        "44e4ca83-46ae-4597-8ee9-17f84fe3dc26",
        "c27acbb2-4902-4575-823b-01addeffa020"
      ]
    },
    "44e4ca83-46ae-4597-8ee9-17f84fe3dc26": {
      "id": "44e4ca83-46ae-4597-8ee9-17f84fe3dc26",
      "name": "Sequence",
      "title": "Sequence",
      "description": "",
      "properties": {},
      "display": {
        "x": -960,
        "y": -180
      },
      "children": [
        "9e4ba8ed-35fb-421c-a1d1-4341bde5f69b",
        "381f7e98-091d-481d-9bbd-2a1809f8fe31"
      ]
    },
    "9e4ba8ed-35fb-421c-a1d1-4341bde5f69b": {
      "id": "9e4ba8ed-35fb-421c-a1d1-4341bde5f69b",
      "name": "Be Offensive",
      "title": "Be Offensive",
      "description": "",
      "properties": {},
      "display": {
        "x": -1392,
        "y": -48
      }
    },
    "c27acbb2-4902-4575-823b-01addeffa020": {
      "id": "c27acbb2-4902-4575-823b-01addeffa020",
      "name": "Priority",
      "title": "Priority",
      "description": "Defensive sub-tree",
      "properties": {},
      "display": {
        "x": 732,
        "y": -180
      },
      "children": [
        "f761a81a-b5a0-432a-bde3-bc047237dda8",
        "1b0f4524-f329-4860-84a8-730e921f93b5",
        "aeedbe6f-4817-41a5-88df-178017f76171"
      ]
    },
    "f761a81a-b5a0-432a-bde3-bc047237dda8": {
      "id": "f761a81a-b5a0-432a-bde3-bc047237dda8",
      "name": "Sequence",
      "title": "Sequence",
      "description": "",
      "properties": {},
      "display": {
        "x": 372,
        "y": -48
      },
      "children": [
        "3c039fca-feb1-474d-8c0f-7bfbcc7ecf4b",
        "8cd2f0b6-e6ed-4155-900d-69963aedcd4d"
      ]
    },
    "3c039fca-feb1-474d-8c0f-7bfbcc7ecf4b": {
      "id": "3c039fca-feb1-474d-8c0f-7bfbcc7ecf4b",
      "name": "Pacman Nearby",
      "title": "Pacman Nearby",
      "description": "",
      "properties": {},
      "display": {
        "x": 264,
        "y": 84
      }
    },
    "8cd2f0b6-e6ed-4155-900d-69963aedcd4d": {
      "id": "8cd2f0b6-e6ed-4155-900d-69963aedcd4d",
      "name": "Hunt Pacman",
      "title": "Hunt Pacman",
      "description": "",
      "properties": {},
      "display": {
        "x": 480,
        "y": 84
      }
    },
    "1b0f4524-f329-4860-84a8-730e921f93b5": {
      "id": "1b0f4524-f329-4860-84a8-730e921f93b5",
      "name": "Patrol",
      "title": "Patrol",
      "description": "",
      "properties": {},
      "display": {
        "x": 684,
        "y": -48
      }
    },
    "3ab5d788-061d-4c1f-8670-dcbc2bb8e094": {
      "id": "3ab5d788-061d-4c1f-8670-dcbc2bb8e094",
      "name": "Collect Food",
      "title": "Collect Food",
      "description": "",
      "properties": {},
      "display": {
        "x": 1308,
        "y": 84
      }
    },
    "b4de14f8-7884-4b2d-9924-4e60b0a004ef": {
      "id": "b4de14f8-7884-4b2d-9924-4e60b0a004ef",
      "name": "Should Return Food",
      "title": "Should Return Food",
      "description": "",
      "properties": {},
      "display": {
        "x": 888,
        "y": 216
      }
    },
    "aeedbe6f-4817-41a5-88df-178017f76171": {
      "id": "aeedbe6f-4817-41a5-88df-178017f76171",
      "name": "Priority",
      "title": "Priority",
      "description": "",
      "properties": {},
      "display": {
        "x": 1152,
        "y": -48
      },
      "children": [
        "eb8d75b8-2093-4440-8a0f-182eaa2e1428",
        "3ab5d788-061d-4c1f-8670-dcbc2bb8e094"
      ]
    },
    "eb8d75b8-2093-4440-8a0f-182eaa2e1428": {
      "id": "eb8d75b8-2093-4440-8a0f-182eaa2e1428",
      "name": "Sequence",
      "title": "Sequence",
      "description": "",
      "properties": {},
      "display": {
        "x": 996,
        "y": 84
      },
      "children": [
        "b4de14f8-7884-4b2d-9924-4e60b0a004ef",
        "7afbb7fd-8af9-40c2-8e6f-f0003151de1c"
      ]
    },
    "7afbb7fd-8af9-40c2-8e6f-f0003151de1c": {
      "id": "7afbb7fd-8af9-40c2-8e6f-f0003151de1c",
      "name": "Return Food",
      "title": "Return Food",
      "description": "",
      "properties": {},
      "display": {
        "x": 1104,
        "y": 216
      }
    }
  },
  "display": {
    "camera_x": 194,
    "camera_y": 319,
    "camera_z": 0.75,
    "x": -108,
    "y": -432
  },
  "custom_nodes": [
    {
      "name": "Ghost Nearby",
      "category": "condition",
      "title": None,
      "description": None,
      "properties": {}
    },
    {
      "name": "Return Food",
      "category": "action",
      "title": None,
      "description": None,
      "properties": {}
    },
    {
      "name": "Collect Food",
      "category": "action",
      "title": None,
      "description": None,
      "properties": {}
    },
    {
      "name": "Should Return Food",
      "category": "condition",
      "title": None,
      "description": None,
      "properties": {}
    },
    {
      "name": "Capsule Close Enough",
      "category": "condition",
      "title": None,
      "description": None,
      "properties": {}
    },
    {
      "name": "Collect Capsule",
      "category": "action",
      "title": None,
      "description": None,
      "properties": {}
    },
    {
      "name": "Be Offensive",
      "category": "condition",
      "title": None,
      "description": None,
      "properties": {}
    },
    {
      "name": "Pacman Nearby",
      "category": "condition",
      "title": None,
      "description": None,
      "properties": {}
    },
    {
      "name": "Hunt Pacman",
      "category": "action",
      "title": None,
      "description": None,
      "properties": {}
    },
    {
      "name": "Patrol",
      "category": "action",
      "title": None,
      "description": None,
      "properties": {}
    }
  ]
}

    firstTree = b3.BehaviorTree()
    secondTree = b3.BehaviorTree()

    firstTree.load(data, names)
    secondTree.load(data, names)

    # Create agents
    firstAgent = eval(first)(firstIndex)
    secondAgent = eval(second)(secondIndex)

    firstAgent.otherAgent = secondAgent
    secondAgent.otherAgent = firstAgent

    firstAgent.isOffensive = False
    secondAgent.isOffensive = False

    firstAgent.tree = firstTree
    secondAgent.tree = secondTree

    # Blackboard
    blackboard = b3.Blackboard()
    firstAgent.blackboard = blackboard
    secondAgent.blackboard = blackboard

    # Debug
    debug = False
    firstTree.debug = debug
    secondTree.debug = debug

    # Debugger
    if debug:
        debugger = Debugger()
        firstAgent.debugger = debugger
        secondAgent.debugger = debugger

    return [firstAgent, secondAgent]


##########
# Agents #
##########

class MyAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.foodPath = []
        self.simAnn = SimulatedAnnealing()
        self.mapWidth = gameState.getWalls().width
        self.mapHeight = gameState.getWalls().height

        if self.tracker is None:
            self.blackboard.set('tracker', Tracker(self, gameState))

        if self.debug:
            # Setup will run twice, but oh well.
            self.debugger.setup(self.display, self.mapWidth, self.mapHeight)

    def chooseAction(self, gameState):

        #print('Moves left: {}'.format(gameState.data.timeleft))
        
        try:
            # Tracker stuff
            self.tracker.update(self, gameState)
            if self.debug and not self.red:
                #self.tracker.update(self, gameState)
                opponents = self.getOpponents(gameState)
            
                positions = self.tracker.getOpponentPos(opponents[1])
                self.debugger.draw(positions, [0,0,1], gameState, duration=2)
                positions = self.tracker.getOpponentPos(opponents[0])
                self.debugger.draw(positions, [0,1,0], gameState, duration=2)

            # Prepare next tick.
            self.nextMove = None
            self.blackboard.set('gameState', gameState)

            # Tick the behaviour tree.
            self.tree.tick(self, self.blackboard)

            # Display debugging
            if self.debug:
                self.debugger.updateDisplay(gameState)

            # nextMove should have been set by a behaviour during tree traversal.
            assert self.nextMove is not None
        except:
            return random.choice(gameState.getLegalActions(self.index))
        return self.nextMove

    @property
    def debug(self):
        return self.tree.debug

    @property
    def tracker(self):
        return self.blackboard.get('tracker')
    
    def getIsPacman(self, gameState):
        return gameState.getAgentState(self.index).isPacman

    def planPath(self, start, goal, gameState):
        path = []
        directions = [(1,0), (0,1), (-1,0), (0,-1)]
        currPos = start
        currDist = self.getMazeDistance(start, goal)
        while True:
            path.append(currPos)
            if currPos == goal:
                break
            for dir in directions:
                newPos = (currPos[0]+dir[0], currPos[1]+dir[1])
                if not gameState.hasWall(newPos):
                    newDist = self.getMazeDistance(newPos, goal)
                    if newDist < currDist:
                        currDist = newDist
                        currPos = newPos
                        break

        return path

    def distCapsule(self, gameState):
        minDist = np.inf
        minPos = None
        for capsulePos in gameState.getCapsules:
            capsuleDist = self.getMazeDistance(self.getPosition(gameState), capsulePos)
            if capsuleDist < minDist:
                minDist = capsuleDist
                minPos = capsulePos
        return minDist, minPos

    def distHome(self, gameState):
        
        walls = gameState.getWalls()
        borderIndex = int(self.mapWidth/2)
        if self.red:
            borderIndex -= 1
        currentPos = self.getPosition(gameState)
        minDist = np.inf
        minPos = None

        for height in range(walls.height):
            if walls[borderIndex][height]:
                continue
            dist = self.getMazeDistance(currentPos, (borderIndex, height))
            if dist < minDist:
                minDist = dist
                minPos = (borderIndex, height)

        return minDist, minPos

    def distOpponents(self, gameState):
        retList = []
        opponents = self.getOpponents(gameState)
        for opponent in opponents:
            oppPos = gameState.getAgentPosition(opponent)
            if oppPos is None:
                retList.append((np.inf, None))
            else:
                oppDist = self.getMazeDistance(self.getPosition(gameState), oppPos)
                retList.append((oppDist, oppPos))

        if retList[0][0] > retList[1][0]:
            retList.reverse()

        return retList

    def moveTowards(self, pos, gameState):
        """Helper"""
        actions = gameState.getLegalActions(self.index)
        currentPos = self.getPosition(gameState)
        dist = self.getMazeDistance(currentPos, pos)
        for action in actions:
            successor = gameState.generateSuccessor(self.index, action)
            newDist = self.getMazeDistance(self.getPosition(successor), pos)
            if newDist < dist:
                return action
        
        return 'Stop'

    def getPosition(self, gameState):
        return gameState.getAgentPosition(self.index)

    def getFoodCount(self, gameState):
        return gameState.getAgentState(self.index).numCarrying

    def getMapHeight(self, gameState):
        return gameState.getWalls().height

    def getMapWidth(self, gameState):
        return gameState.getWalls().width

    def getFoodRemaining(self, gameState):
        foodList = []
        food = self.getFood(gameState)
        for i in range(food.width):
            for j in range(food.height):
                if food[i][j]:
                    foodList.append((i, j))

        return foodList


##############
# Behaviours #
##############

class CollectFood(b3.Action):
    """A behaviour for collecting food."""

    def tick(self, tick):
        agent = tick.target
        gameState = tick.blackboard.get('gameState')

        # Set working parameters in SA for offensive/defensive.
        agent.simAnn.setParameters(agent.isOffensive)

        # Remove already eaten food from foodPath.
        i = 0
        while i < len(agent.foodPath):
            foodPos = agent.foodPath[i]
            if gameState.hasFood(foodPos[0], foodPos[1]):
                i += 1
            else:
                agent.foodPath.pop(i)

        # Plan foodPath.
        newFoodPath = agent.simAnn.planFoodPath(agent, gameState)
        newFoodPath = ['break'] + newFoodPath + ['break']
        newEnergy = agent.simAnn.energy(newFoodPath, agent, gameState)
        oldFoodPath = ['break'] + agent.foodPath + ['break']
        oldEnergy = agent.simAnn.energy(oldFoodPath, agent, gameState)
        if newEnergy < oldEnergy:
            agent.foodPath = newFoodPath[1:-1]

        # Set nextMove as moving towards start of foodPath.
        agent.nextMove = agent.moveTowards(agent.foodPath[0], gameState)

        if tick.debug:
            for i in range(len(agent.foodPath)):
                red = (1 - i/len(agent.foodPath))
                agent.debugger.draw(agent.foodPath[i], [red, 0, 0], gameState)

        return b3.RUNNING


class ReturnFood(b3.Action):
    """A behaviour for returning food."""

    def tick(self, tick):
        agent = tick.target
        gameState = tick.blackboard.get('gameState')

        # Find shortest way home.
        homeDist, homePos = agent.distHome(gameState)
        
        # Set nextMove as moving towards start of foodPath.
        if agent.getIsPacman(gameState):
            goalPos = homePos
        else: 
            goalPos = agent.foodPath[0]

        # Get one opponent position.
        opponents = agent.getOpponents(gameState)
        for opponent in opponents:
            oppPos = agent.tracker.getExactOppPos(opponent)
            if not oppPos is None:
                break
        
        currentPos = agent.getPosition(gameState)
        if oppPos is None:
            ghostSafePos = getPossiblePositions(currentPos, gameState.getWalls())
        else:
            ghostSafePos = []
            currentGhostDist = agent.getMazeDistance(currentPos, oppPos)
            for pos in getPossiblePositions(currentPos, gameState.getWalls()):
                if agent.getMazeDistance(pos, oppPos) > min(currentGhostDist, 5):
                    ghostSafePos.append(pos)
            if len(ghostSafePos) == 0: # no safe dir, move home
                agent.nextMove = agent.moveTowards(homePos, gameState)
                return b3.RUNNING

        currentGoalDist = agent.getMazeDistance(currentPos, goalPos)
        for safePos in ghostSafePos:
            if agent.getMazeDistance(safePos, goalPos) < currentGoalDist:
                agent.nextMove = agent.moveTowards(safePos, gameState)
                return b3.RUNNING

        agent.nextMove = agent.moveTowards(random.choice(ghostSafePos), gameState)

        return b3.RUNNING


class CollectCapsule(b3.Action):
    """A behaviour for collecting a capsule."""

    def tick(self, tick):
        agent = tick.target
        gameState = tick.blackboard.get('gameState')

        # TODO: Write behaviour logic here

        return b3.FAILURE


class HuntPacman(b3.Action):

    def tick(self, tick):
        agent = tick.target
        gameState = tick.blackboard.get('gameState')
        
        opponents = agent.getOpponents(gameState)
        for opponent in opponents:
            oppPos = gameState.getAgentPosition(opponent)
            if oppPos is None:
                continue
            elif gameState.getAgentState(opponent).isPacman:
                agent.nextMove = agent.moveTowards(oppPos, gameState)
                return b3.RUNNING

        # From now on we try to intercept at border.
        for opponent in opponents:
            oppPos = gameState.getAgentPosition(opponent)
            if not oppPos is None:
                break

        walls = gameState.getWalls()
        borderIndex = int(agent.mapWidth/2)
        if agent.red:
            borderIndex -= 1
        currentPos = oppPos
        minDist = np.inf
        minPos = None

        for height in range(walls.height):
            if walls[borderIndex][height]:
                continue
            dist = agent.getMazeDistance(currentPos, (borderIndex, height))
            if dist < minDist:
                minDist = dist
                minPos = (borderIndex, height)

        agent.nextMove = agent.moveTowards(minPos, gameState)

        return b3.RUNNING


class Patrol(b3.Action):

    def tick(self, tick):
        agent = tick.target
        gameState = tick.blackboard.get('gameState')
        
        if agent.red:
            borderIndex = agent.mapWidth/2 - 1
        else:
            borderIndex = agent.mapWidth/2
        couldBeOnOurSide = False
        patrolTarget = None
        targetDist = 2000
        agentPos = agent.getPosition(gameState)
        opponents = agent.getOpponents(gameState)
        for opponent in opponents:
            oppPositions = agent.tracker.getOpponentPos(opponent)
            for oppPos in oppPositions:
                # RED
                if agent.red: 
                    if oppPos[0] <= borderIndex:
                        couldBeOnOurSide = True
                        dist = agent.getMazeDistance(agentPos, oppPos)
                        if dist < targetDist:
                            patrolTarget = oppPos
                            targetDist = dist
                    elif not couldBeOnOurSide and borderIndex < oppPos[0] <= borderIndex + 5:
                        dist = oppPos[0] - borderIndex + 1000
                        if dist < targetDist:
                            patrolTarget = oppPos
                            targetDist = dist
                # BLUE
                else: 
                    if oppPos[0] >= borderIndex:
                        couldBeOnOurSide = True
                        dist = agent.getMazeDistance(agentPos, oppPos)
                        if dist < targetDist:
                            patrolTarget = oppPos
                            targetDist = dist
                    elif not couldBeOnOurSide and borderIndex > oppPos[0] >= borderIndex + 5:
                        dist = borderIndex - oppPos[0] + 1000
                        if dist < targetDist:
                            patrolTarget = oppPos
                            targetDist = dist

        if patrolTarget is None:
            return b3.FAILURE
        else:
            agent.nextMove = agent.moveTowards(patrolTarget, gameState)

        return b3.RUNNING


##############
# Conditions #
##############

class ShouldReturnFood(b3.Condition):
    """A condition for deciding to return food."""

    def tick(self, tick):
        agent = tick.target
        gameState = tick.blackboard.get('gameState')

        if agent.getFoodCount(gameState) == 0:
            return b3.FAILURE
        if len(agent.getFoodRemaining(gameState)) <= 2:
            return b3.SUCCESS

        cost = 0
        cost += agent.distHome(gameState)[0]
        cost -= agent.getFoodCount(gameState)
        if len(agent.foodPath) > 0:
            cost -= agent.getMazeDistance(agent.getPosition(gameState), agent.foodPath[0])

        if cost < 0:
            return b3.SUCCESS
        else:
            return b3.FAILURE


class GhostNearby(b3.Condition):
    """A condition for deciding to react to opponent ghost."""

    def tick(self, tick):
        agent = tick.target
        gameState = tick.blackboard.get('gameState')      
        oppTuples = agent.distOpponents(gameState)
        
        if oppTuples[0][0] < 5: 
            return b3.SUCCESS
        else: 
            return b3.FAILURE


class PacmanNearby(b3.Condition):
    """A condition for deciding to react to opponent ghost."""

    def tick(self, tick):
        agent = tick.target
        gameState = tick.blackboard.get('gameState')      
        oppTuples = agent.distOpponents(gameState)
        
        if oppTuples[0][0] < 5:
            return b3.SUCCESS
        else:
            return b3.FAILURE


class CapsuleCloseEnough(b3.Condition):
    """A condition for deciding to collect a capsule."""

    def tick(self, tick):
        agent = tick.target
        gameState = tick.blackboard.get('gameState')

        # TODO: Write condition logic here

        return b3.FAILURE


class BeOffensive(b3.Condition):
    """A condition for deciding to be offensive."""
    
    def tick(self, tick):
        agent = tick.target
        gameState = tick.blackboard.get('gameState')

        if not agent.getIsPacman(gameState):
            oppTuples = agent.distOpponents(gameState)
            if oppTuples[0][0] < 4:
                agent.isOffensive = False
                return b3.FAILURE

        if not agent.otherAgent.isOffensive:
            agent.isOffensive = True
            return b3.SUCCESS

        return b3.FAILURE


###########
# Helpers #
###########

class SimulatedAnnealing:

    def __init__(self):
        self.maxIter = 1000
        self.t0 = 50
        self.alpha = 0.99
        self.foodCoeff = -5
        self.distCoeff = 1
        self.discountCoeff = 0.98
        self.distIncreaseCoeff = 1.05

    def planFoodPath(self, agent, gameState, shouldPlot=False):
        state = []
        state.append('break')
        state.append('break')
        state += agent.getFoodRemaining(gameState)
        
        if shouldPlot:
            EValues = []
        currentState = self.getInitialState(state)
        currentE = self.energy(currentState, agent, gameState)
        for i in range(self.maxIter):
            temp = self.temperature(i)
            candidateState = self.neighbour(currentState)
            candidateE = self.energy(candidateState, agent, gameState)
            if self.prob(currentE, candidateE, temp) > random.uniform(0, 1):
                currentState = candidateState
                currentE = candidateE
            if shouldPlot:
                EValues.append(currentE)

        if shouldPlot:
            x = np.arange(self.maxIter)
            y = EValues
            plt.clf()
            plt.plot(x, y)
            plt.show(block=False)

        startAndGoalIndex = []
        for i, item in enumerate(currentState):
            if type(item) is str:
                startAndGoalIndex.append(i)

        returnState = currentState[startAndGoalIndex[0]+1:startAndGoalIndex[1]]
        #bp()
        return returnState

    def energy(self, state, agent, gameState):
        state = self.trimState(state, agent, gameState)
        if len(state) <= 2:
            return np.inf

        def getTotalDistance(state):
            dist = 0
            for i in range(1, len(state)):
                dist += agent.getMazeDistance(state[i-1], state[i])*self.distIncreaseCoeff
            return dist
        
        energy = 0
        energy += len(state)-1 * self.foodCoeff
        energy += getTotalDistance(state) * self.distCoeff
        return energy

    def prob(self, currentE, candidateE, temp):
        return np.exp((currentE - candidateE) / temp)

    def temperature(self, i):
        return self.t0 * self.alpha**i

    def neighbour(self, state):
        index = random.randint(0, len(state) - 1)
        item = state[index]
        neighbour = state[:index] + state[index+1:]
        insertIndex = random.randint(0, len(neighbour) - 1)
        neighbour.insert(insertIndex, item)
        return neighbour

    def getInitialState(self, state):
        random.shuffle(state)
        return state

    def trimState(self, state, agent, gameState):
        startAndGoalIndex = []
        for i, item in enumerate(state):
            if isinstance(item, str):
                startAndGoalIndex.append(i)
        energyState = [agent.getPosition(gameState)]
        energyState += state[startAndGoalIndex[0]+1:startAndGoalIndex[1]]

        return energyState

    def setParameters(self, isOffensive):
        if isOffensive:
            self.foodCoeff = -5
            self.distCoeff = 1
            self.discountCoeff = 0.98
            self.distIncreaseCoeff = 1.05
        else:
            self.foodCoeff = -2
            self.distCoeff = 1.5
            self.discountCoeff = 0.98
            self.distIncreaseCoeff = 1.05


class Tracker:

    def __init__(self, agent, gameState):
        self.lastUpdated = 1201
        self.walls = gameState.getWalls()
        self.width = agent.mapWidth
        self.height = agent.mapHeight
        self.gridDict = {}

        self.falseGrid = []
        for i in range(self.width):
            column = []
            for j in range(self.height):
                column.append(False)
            self.falseGrid.append(column)

        opponents = agent.getOpponents(gameState)
        for opponent in opponents:
            grid = copy.deepcopy(self.falseGrid)
            self.gridDict[opponent] = grid

        team = agent.getTeam(gameState)   
        for opponent in opponents:
            for ourAgent in team:
                ourPos = gameState.getAgentPosition(ourAgent)
                self.gridDict[opponent][self.width-ourPos[0]-1][self.height-ourPos[1]-1] = True

    def update(self, agent, gameState):
        dt = self.lastUpdated - gameState.data.timeleft
        opponents = agent.getOpponents(gameState)

        # Opponents are moving.
        if dt%4 == 2:
            opponent = (agent.index - 1)%4
            self.moveOneStep(opponent)
        for i in range(dt//4):
            for opponent in opponents:
                self.moveOneStep(opponent)

        # Observation
        for opponent in opponents:
            oppPos = gameState.getAgentPosition(opponent)
            if not oppPos is None:
                self.addPositiveRestriction(oppPos, opponent)
            else:
                noisyDist = gameState.getAgentDistances()[opponent]
                largeRadius = noisyDist + 6
                smallRadius = max([noisyDist - 7, 5])
                largeManSquare = getManhattanSquare(agent.getPosition(gameState), largeRadius, self.width, self.height)
                smallManSquare = getManhattanSquare(agent.getPosition(gameState), smallRadius, self.width, self.height)
                self.addPositiveRestriction(largeManSquare, opponent)
                self.addNegativeRestriction(smallManSquare, opponent)

        for opponent in opponents:
            positions = self.getOpponentPos(opponent)
            if len(positions) == 0:
                agentState = gameState.getAgentState(opponent)
                startPos = agentState.start.pos
                self.gridDict[opponent][startPos[0]][startPos[1]] = True

        self.lastUpdated = gameState.data.timeleft
                
    def moveOneStep(self, opponent):
        oppGrid = self.gridDict[opponent]
        posSet = set()
        for i in range(self.width):
            for j in range(self.height):
                if oppGrid[i][j] is True:
                    positions = getPossiblePositions((i, j), self.walls)
                    for pos in positions:
                        posSet.add(pos)

        for pos in posSet:
            oppGrid[pos[0]][pos[1]] = True

    def addPositiveRestriction(self, positions, opponent):
        if not type(positions) is list:
            positions = [positions]

        oldGrid = self.gridDict[opponent]
        self.gridDict[opponent] = copy.deepcopy(self.falseGrid)
        for pos in positions:
            if oldGrid[pos[0]][pos[1]]:
                self.gridDict[opponent][pos[0]][pos[1]] = True

    def addNegativeRestriction(self, positions, opponent):
        if not type(positions) is list:
            positions = [positions]

        for pos in positions:
            self.gridDict[opponent][pos[0]][pos[1]] = False

    def getOpponentPos(self, opponent):
        positions = []
        oppGrid = self.gridDict[opponent]
        for i in range(self.width):
            for j in range(self.height):
                if oppGrid[i][j] is True:
                    positions.append((i,j))
        return positions

    def getOpponentGrid(self, opponent):
        return self.gridDict[opponent]

    def getExactOppPos(self, opponent):
        oppPositions = self.getOpponentPos(opponent)
        if len(oppPositions) == 1:
            return oppPositions[0]
        else:
            return None


class Debugger:

    def __init__(self):
        self.blank = [0, 0, 0]

    def setup(self, display, mapWidth, mapHeight):
        self.display = display
        self.width = mapWidth
        self.height = mapHeight

        self.grid = []
        for i in range(self.width):
            column = []
            for j in range(self.height):
                column.append([])
            self.grid.append(column)

        self.isModified = []
        for i in range(self.width):
            column = []
            for j in range(self.height):
                column.append(False)
            self.isModified.append(column)


    def draw(self, cells, color, gameState, duration=4, prio=0):
        """Queue debug cells."""
        if not type(cells) is list:
            cells = [cells]

        lastUntil = gameState.data.timeleft - duration
        
        for cell in cells:
            drawOrder = (color, lastUntil, prio)
            self.grid[cell[0]][cell[1]].append(drawOrder)
            self.isModified[cell[0]][cell[1]] = True

    def updateDisplay(self, gameState):
        """Updates displayed debug cells."""
        timeleft = gameState.data.timeleft
        for i in range(self.width):
            for j in range(self.height):
                drawOrders = self.grid[i][j]
                
                # Remove expired draw orders.
                k = 0
                while k < len(drawOrders):
                    drawOrder = drawOrders[k]
                    if drawOrder[1] < timeleft:
                        k += 1
                    else:
                        drawOrders.pop(k)
                        self.isModified[i][j] = True

                # Redraw according to most prioritised color.
                if self.isModified[i][j]:
                    #topColor = self.blank
                    #topPrio = -1
                    #for drawOrder in drawOrders:
                    #    if drawOrder[2] > topPrio:
                    #        topPrio = drawOrder[2]
                    #        topColor = drawOrder[0]
                            
                    topColors = [self.blank]
                    topPrio = -1
                    for drawOrder in drawOrders:
                        if drawOrder[2] > topPrio:
                            topPrio = drawOrder[2]
                            topColors = [drawOrder[0]]
                        elif drawOrder[2] == topPrio:
                            topColors.append(drawOrder[0])
                    
                    topColor = self.blank[:]
                    for color in topColors:
                        topColor[0] += color[0]
                        topColor[1] += color[1]
                        topColor[2] += color[2]

                    topColor[0] /= len(topColors)
                    topColor[1] /= len(topColors)
                    topColor[2] /= len(topColors)

                    self.display.debugDraw([(i,j)], topColor)
                    self.isModified[i][j] = False


def getPossiblePositions(pos, walls):
    directions = [(1,0), (0,1), (-1,0), (0,-1)]
    possiblePos = []
    for dir in directions:
        newPos = (pos[0]+dir[0], pos[1]+dir[1])
        if not walls[newPos[0]][newPos[1]]:
            possiblePos.append(newPos)

    return possiblePos

def getManhattanSquare(pos, dist, width, height):
    positions = []
    for i in range(-dist, dist+1):
        x = pos[0]+i
        if x >= width or x < 0:
            continue
        for j in range(-dist, dist+1):
            y = pos[1]+j
            if y >= height or y < 0:
                continue
            if abs(i)+abs(j) <= dist:
                positions.append((x, y))
    
    return positions
