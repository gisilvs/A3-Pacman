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

class DecisionTree :
  levels = 3

  def __init__(self,game_state,root_agent_object,offensive):
      agent_that_moved_here = (root_agent_object.index + 3) % 4

      self.ai_agent = root_agent_object
      self.offensive = offensive

      self.root_node = DecisionTreeNode(game_state,agent_that_moved_here,root_agent_object,self.offensive)

      self.add_children(self.root_node, 1)

  def add_children(self,tree_node,level):
    if level > self.levels:
      return

    future_states = []
    agent_to_move = (tree_node.agent_that_moved_here + 1) % 4

    if (tree_node.node_state.data.agentStates[agent_to_move].configuration == None):
      future_states.append(tree_node.node_state) # I.E current state
    else:
      actions = tree_node.node_state.getLegalActions(agent_to_move)
      for i in actions:
        future_states.append(tree_node.node_state.generateSuccessor(agent_to_move, i))

    for i in range(len(future_states)):
      next_node = tree_node.add_child(future_states[i], agent_to_move)

      self.add_children(next_node,level + 1)


  def get_action(self):
    self.propagate_tree(self.root_node,0)

    child_scores = []
    for i in self.root_node.children:
      child_scores.append(i.propagated_score)

    max_index = child_scores.index(max(child_scores))

    actions = self.root_node.node_state.getLegalActions((self.root_node.agent_that_moved_here + 1) % 4)

    return actions[max_index]


  def propagate_tree(self,current_node,level):
    if level == self.levels:
      current_node.propagated_score = current_node.node_score
      return current_node.propagated_score

    if (current_node.agent_that_moved_here == self.root_node.agent_that_moved_here or (current_node.agent_that_moved_here + 2) % 4 == self.root_node.agent_that_moved_here):
      enemy_team = False
    else:
      enemy_team = True

    child_scores = []

    for i in current_node.children:
      child_score = self.propagate_tree(i,level+1)
      child_scores.append(child_score)

    if enemy_team:
      current_node.propagated_score = min(child_scores)
    else:
      current_node.propagated_score = max(child_scores)

    return current_node.propagated_score




class DecisionTreeNode :
  def __init__(self,game_state,agent_that_moved_here,root_agent_object,offensive):
    self.node_state = game_state
    self.agent_that_moved_here = agent_that_moved_here #The index of the agent that moved to this state
    self.root_agent_object = root_agent_object
    self.offensive = offensive
    self.node_score = self.evaluate_state()
    self.propagated_score = 0
    self.children = []


  def add_child(self,game_state,index):
    child_node = DecisionTreeNode(game_state,index,self.root_agent_object,self.offensive)
    self.children.append(child_node)

    return child_node

  def evaluate_state(self):
    if (self.agent_that_moved_here == self.root_agent_object.index or (self.agent_that_moved_here + 2) % 4 == self.root_agent_object.index):
      enemy_team = False
    else:
      enemy_team = True

    agent_positions = []

    for i in range(0,4):
      if (self.node_state.data.agentStates[i].configuration != None):
        agent_positions.append(self.node_state.data.agentStates[i].configuration.pos)
      else:
        agent_positions.append(self.root_agent_object.get_most_likely_distance_from_noisy_reading(i))

    my_index = self.agent_that_moved_here
    team_mates_index = (self.agent_that_moved_here + 2) % 4

    if self.offensive:
      state_value = self.evaluate_state_one_agent(agent_positions, my_index,enemy_team) + self.evaluate_state_one_agent(agent_positions,team_mates_index,enemy_team)
      maintain_distance_factor = self.get_maintain_distance_factor(agent_positions[my_index],agent_positions[team_mates_index])
      enemies_are_scared_factor = self.get_enemies_are_scared_factor(my_index)
      score_factor = self.get_score_factor()
      state_value += (score_factor + enemies_are_scared_factor)# + maintain_distance_factor)
    else:
      state_value = self.evalute_state_one_agent_defensive(agent_positions[my_index], my_index, enemy_team, agent_positions[team_mates_index]) + self.evalute_state_one_agent_defensive(agent_positions[team_mates_index],team_mates_index,enemy_team,agent_positions[team_mates_index])


    if (enemy_team): #If not enemy team is moving, then child nodes should have positive state values
      state_value = - state_value

    return state_value

  def evalute_state_one_agent_defensive(self, agent_position,agent_index,agent_is_on_enemy_team,team_mate_position):
    if self.check_if_valid_coordinates(self.root_agent_object.middle):
      distance_to_middle_factor = CaptureAgent.getMazeDistance(self.root_agent_object, agent_position,self.root_agent_object.middle)
    else:
      distance_to_middle_factor = 0
    if self.check_if_valid_coordinates(self.root_agent_object.upperHalf):
      distance_to_upper_half = CaptureAgent.getMazeDistance(self.root_agent_object, agent_position,self.root_agent_object.upperHalf)
    else:
      distance_to_upper_half = 0
    if self.check_if_valid_coordinates(self.root_agent_object.lowerHalf):
      distance_to_lower_half = CaptureAgent.getMazeDistance(self.root_agent_object, agent_position, self.root_agent_object.lowerHalf)
    else:
      distance_to_lower_half = 0
    enemies = [self.node_state.getAgentState(i) for i in CaptureAgent.getOpponents(self.root_agent_object, self.node_state)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    number_of_invaders = len(invaders)
    invader_distance = 0
    if len(invaders) > 0:
      dists = [CaptureAgent.getMazeDistance(self.root_agent_object, agent_position, a.getPosition()) for a in invaders]
      invader_distance = min(dists)

    distance_from_each_other = CaptureAgent.getMazeDistance(self.root_agent_object, agent_position, team_mate_position)

    state_value = number_of_invaders * (-9999) + invader_distance * (-1000) + distance_to_middle_factor * (-200) + \
                  distance_to_lower_half*(-50) + distance_to_upper_half*(-50) + distance_from_each_other * (200)
    return state_value

  def check_if_valid_coordinates(self, coordinates):
    valid = False
    invalid_coordinates = tuple((0, 0))
    if coordinates != invalid_coordinates:
      valid = True
    return valid

  def evaluate_state_one_agent(self,agent_positions,agent_index,agent_is_on_enemy_team):
    distance_to_food_factor = self.get_distances_to_food_factor(agent_positions[agent_index], agent_index,agent_is_on_enemy_team)

    return_with_food_factor = self.get_return_with_food_factor(agent_positions[agent_index], agent_index)

    distance_to_enemy_ghosts_factor = self.get_distance_to_enemy_ghosts_factor(agent_positions, agent_index)

    state_value = distance_to_food_factor + return_with_food_factor + distance_to_enemy_ghosts_factor#+ food_carried_factor

    return state_value

  def get_distance_to_enemy_ghosts_factor(self,agent_positions,agent_index):
    #return 0
    enemy_agent_index_1 = (agent_index + 1) % 4
    enemy_agent_index_2 = (agent_index + 3) % 4

    i_am_ghost = self.is_ghost(agent_index)
    enemy_1_is_ghost = self.is_ghost(enemy_agent_index_1)
    enemy_2_is_ghost = self.is_ghost(enemy_agent_index_2)

    if i_am_ghost or (not enemy_1_is_ghost and not enemy_2_is_ghost):
      return 0

    distance_enemy_1 = CaptureAgent.getMazeDistance(self.root_agent_object, agent_positions[agent_index], agent_positions[enemy_agent_index_1])
    distance_enemy_2 = CaptureAgent.getMazeDistance(self.root_agent_object, agent_positions[agent_index], agent_positions[enemy_agent_index_2])

    if enemy_1_is_ghost and enemy_2_is_ghost:
      distance = (distance_enemy_1 + distance_enemy_2) / 2
    elif enemy_1_is_ghost:
      distance = distance_enemy_1
    else:
      distance = distance_enemy_2

    if distance > 3:
      return 0

    if distance == 0:
      distance = 1
    distane_to_enemy_ghosts_factor = (-1 / distance) * 5

    return distane_to_enemy_ghosts_factor

  def is_ghost(self,agent_index):
    is_ghost = True

    if self.node_state.data.agentStates[agent_index].isPacman:
      is_ghost = False

    if self.node_state.data.agentStates[agent_index].scaredTimer > 0:
      is_ghost = False

    return is_ghost


  def get_distances_to_food_factor(self,agent_position,agent_index,enemy_team):
    if (enemy_team):
      food = CaptureAgent.getFoodYouAreDefending(self.root_agent_object,self.node_state)
    else:
      food = CaptureAgent.getFood(self.root_agent_object, self.node_state)

    distances = []

    blue_capsule = (-1,-1)
    red_capsule = (-1,-1)

    for c in self.node_state.data.capsules:
      if c[0] > self.node_state.data.layout.width / 2:
        blue_capsule = c
      else:
        red_capsule = c


    blue_side = agent_index % 2
    if blue_side:
     our_capsule = red_capsule
    else:
      our_capsule = blue_capsule

    for i in range(0,food.width):
      for j in range(0,food.height):
        if food.data[i][j] or our_capsule == (i,j):
          distance = CaptureAgent.getMazeDistance(self.root_agent_object, agent_position, (i,j))
          distances.append(distance)

    if (len(distances) == 0): return 1

    #distance_to_food_factor = sum(distances) / float(len(distances))
    distance_to_food_factor = min(distances)
    if (distance_to_food_factor == 0):
      distance_to_food_factor = 1

    distance_to_food_factor = 1 / distance_to_food_factor
    distance_to_food_factor = distance_to_food_factor * 1


    return distance_to_food_factor

  def get_score_factor(self):
    score_diff = CaptureAgent.getScore(self.root_agent_object,self.node_state)

    return 10000 * score_diff

  def get_food_carried_factor(self,agent_index):
    food_carrying = self.node_state.data.agentStates[agent_index].numCarrying

    return 1000 * food_carrying


  def get_maintain_distance_factor(self,agent_position_1,agent_position_2):
    distance_between = CaptureAgent.getMazeDistance(self.root_agent_object,agent_position_1,agent_position_2)

    return distance_between/300

  def get_enemies_are_scared_factor(self,agent_index):
    enemy_agent_index_1 = (agent_index + 1) % 4
    enemy_agent_index_2 = (agent_index + 3) % 4

    ret_val = 0

    if self.node_state.data.agentStates[enemy_agent_index_1].scaredTimer > 0:
      ret_val += 10
    if self.node_state.data.agentStates[enemy_agent_index_2].scaredTimer > 0:
      ret_val += 10

    return ret_val


  def get_return_with_food_factor(self,agent_position,agent_index):
    food_carrying = self.node_state.data.agentStates[agent_index].numCarrying

    if food_carrying == 0:
      return 0

    distance_home = self.get_closest_distance_to_home(agent_position,agent_index)

    if distance_home == 0:
      distance_home = 1

    return_with_food_factor = (1 / distance_home) * food_carrying * 100

    return return_with_food_factor


  def get_closest_distance_to_home(self,agent_position,agent_index):
    blue_side = agent_index % 2

    width = self.node_state.data.layout.width
    height = self.node_state.data.layout.height

    if blue_side:
      col = int(width/2)
    else:
      col = int(width/2 - 1)

    distances = []

    for i in range(height):
      if not self.node_state.data.layout.isWall((col,i)):
        dist = CaptureAgent.getMazeDistance(self.root_agent_object,agent_position,(col,i))
        distances.append(dist)

    return min(distances)

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
    CaptureAgent.registerInitialState(self, gameState)
    #Map without walls
    self.grid_to_work_with = []
    for x in range(gameState.data.layout.width):
      for y in range(gameState.data.layout.height):
        if not gameState.hasWall(x, y):
          self.grid_to_work_with.append((x, y))

    self.enemy_indexes = self.getOpponents(gameState)
    self.team_indexes = self.getTeam(gameState)
    self.offensive = False

    #Map choke points
    self.LayoutHalfWayPoint_Width = gameState.data.layout.width/2
    self.LayoutHalfWayPoint_Height = gameState.data.layout.height/2
    self.middle = (self.LayoutHalfWayPoint_Width, self.LayoutHalfWayPoint_Height)
    self.middle = self.find_place_in_grid(gameState, self.middle)
    # try features for the upper and lower half so the agent not in the middle could settle on either one of those points
    self.lowerHalf = (self.LayoutHalfWayPoint_Width, gameState.data.layout.height/2-0.25*self.LayoutHalfWayPoint_Height)
    self.lowerHalf = self.find_place_in_grid(gameState, self.lowerHalf)
    self.upperHalf = (self.LayoutHalfWayPoint_Width, gameState.data.layout.height/2+0.25*self.LayoutHalfWayPoint_Height)
    self.upperHalf = self.find_place_in_grid(gameState, self.upperHalf)

    min_value = min(self.enemy_indexes)
    if min == 0:
      self.enemy_to_update_possible_locations = self.index - 1
    else:
      self.enemy_to_update_possible_locations = self.index + 1

    self.grid_to_work_with = self.convert_tuples_to_list(self.grid_to_work_with)
    self.list_of_invaders = []
    self.emission_probabilties_for_each_location_for_each_agent = self.initialize_probabilty_list(gameState)

  def chooseAction(self, gameState):
    #First update enemy position
    offensive = self.offensive
    self.update_enemy_possible_locations_depending_on_round(self.enemy_to_update_possible_locations, gameState)
    if self.check_if_we_killed_an_enemy(gameState):
      self.offensive = True
    list_of_enemies_in_range = [i for i in self.enemy_indexes if gameState.getAgentState(i).getPosition() != None]
    for i in list_of_enemies_in_range:
      self.reset_agent_probabilties_when_we_know_the_true_position(i, list(gameState.getAgentState(i).getPosition()))
    #Get the position of the other agent at this gameState
    #we could swiss to attack if we kill some1 and then return to defensive stance when we have returned some pellets
    myState = gameState.getAgentState(self.index)
    myPos = myState.getPosition()
    otherDudePosition = self.get_other_agent_positon(myState, gameState)
    #We could do this for all actions in the tree which would maybe give better results
    for enemy in self.enemy_indexes:
      #Take both of our agents into account
      self.updateNoisyDistanceProbabilities(myPos, enemy, gameState)
      if otherDudePosition != 0:
        self.updateNoisyDistanceProbabilities(otherDudePosition, enemy, gameState)

    actions = gameState.getLegalActions(self.index)

    score = self.getScore(gameState)
    #print(score)
    if score > 0:
      self.offensive = False
    if myState.isPacman:
      self.offensive = True
    # we toggle defensive vs offensive if we killed someone
    # or if we are winning
    # i.e. if we kill some1 we go offensive
    # then if we are winning on the scoreboard we go defensive
    decision_tree = DecisionTree(gameState, self, self.offensive)
    return decision_tree.get_action()
    #updateNoisyDistanceProbabilities
  ###Utility functions###

  def find_place_in_grid(self, gameState, location):
    #we only have half of the board to work with so
    w = gameState.data.layout.width/2
    #change the tuple to a list
    location_to_work_with = list(location)
    if location not in self.grid_to_work_with:
      location_copy = location_to_work_with
      for i in range(1, 8):
        if self.red:
          location_copy[0] = location_copy[0] - i
        else:
          location_copy[0] = location_copy[0] + i
        if tuple(location_copy) in self.grid_to_work_with:
          return tuple(location_copy)
      for i in range(1, 8):
        location_copy[1] = location_copy[1] - i
        if tuple(location_copy) in self.grid_to_work_with:
          return tuple(location_copy)
        location_copy[1] = location_copy[1] + i
        if tuple(location_copy) in self.grid_to_work_with:
          return tuple(location_copy)
      #Next one above
      #Third one below
      #Then two to left etc etc
      return (0, 0) #got to figure this edge case out
    return location

  def convert_tuples_to_list(self, tuple):
    l = []
    for i in tuple:
      element = list(i)
      l.append(element)
    return l

  def getEnemyListIndex(self, enemy_we_are_checking):
    max_element = max(self.enemy_indexes)
    if enemy_we_are_checking == max_element:
      index = 1
    else:
      index = 0
    return index

  def get_other_agent_positon(self, myState, gameState):
    otherDudePosition = 0
    team = [gameState.getAgentState(i) for i in self.getTeam(gameState)]
    otherDude = [i for i in team if i != myState]
    # I still don't know wwhy I need this check but we need it
    if len(otherDude) > 0:
      otherDude = otherDude[0]
      otherDudePosition = otherDude.getPosition()
    return otherDudePosition

  def get_legal_actions_from_location(self, location):
    location = list(location)
    list_of_possible_coordinates = []
    #always possible to stop in place
    list_of_possible_coordinates.append(location)
    #now check up, down, left, right,
    up = [location[0], location[1]+1]
    down = [location[0], location[1]-1]
    left = [location[0]-1, location[1]]
    right = [location[0]+1, location[1]]
    candidates = [up, down, left, right]
    for candidate in candidates:
      if candidate in self.grid_to_work_with:
        list_of_possible_coordinates.append(candidate)
    return list_of_possible_coordinates

  def update_enemy_possible_locations_depending_on_round(self, enemy_to_update, gameState):
    # all legal actions for the enemy depending on possible locations
    list_index = self.getEnemyListIndex(enemy_to_update)
    all_locations_not_with_zero_probabilty = [i for i in self.emission_probabilties_for_each_location_for_each_agent[list_index] if i[1] != 0]
    all_possible_moves = []
    for i in all_locations_not_with_zero_probabilty:
      posible_moves_from_i = self.get_legal_actions_from_location(i[0])
      all_possible_moves.extend(posible_moves_from_i)
    number_of_possible_locations_to_move_to = len(all_possible_moves)
    for move in all_possible_moves:
      for i in self.emission_probabilties_for_each_location_for_each_agent[list_index]:
        if i[0] == move:
          i[1] = 1/number_of_possible_locations_to_move_to

  def initialize_probabilty_list(self, gameState):
    list_of_probabilities = []
    for enemy in self.enemy_indexes:
      all_locations = self.grid_to_work_with
      all_locations_plus_odds = []
      for location in all_locations:
        element = [location, 0]
        all_locations_plus_odds.append(element)
      list_of_probabilities.append(all_locations_plus_odds)
      # Because we know the inital position we can put the probabilty of the enemy being in that position
      initPos = list(gameState.getInitialAgentPosition(enemy))
      initPos = [initPos, 0]
      agent_list_index = self.getEnemyListIndex(enemy)
      starting_location_index = list_of_probabilities[agent_list_index].index(initPos)
      list_of_probabilities[agent_list_index][starting_location_index][1] = 1.0
    return list_of_probabilities

  def reset_agent_probabilties_when_we_know_the_true_position(self, enemy, true_position):
      index = self.getEnemyListIndex(enemy)
      listCopy = [i[0] for i in self.emission_probabilties_for_each_location_for_each_agent[index]]
      location_index = listCopy.index(true_position)
      self.emission_probabilties_for_each_location_for_each_agent[index][location_index][1] = 1.0
      #todo need to set all other odds to zero
      for i in range(len(self.emission_probabilties_for_each_location_for_each_agent[index])):
        if i != location_index:
          self.emission_probabilties_for_each_location_for_each_agent[index][i][1] = 0.0


  def check_if_we_killed_an_enemy(self, gameState):
    kill = False
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    old_invaderList = self.list_of_invaders
    new_invaderList = [e for e in enemies if e.isPacman]
    self.list_of_invaders = new_invaderList
    # we killed some one
    if len(new_invaderList) < len(old_invaderList):
      kill = True
      # get that agent and reset his probabilty pos to the inital position
      enemies_to_reset_locations_for = list(set(old_invaderList) - set(new_invaderList))
      enemy_indexes = self.enemy_indexes
      enemy_indexes_to_reset = []
      if len(enemies_to_reset_locations_for) > 0:
        for index in enemy_indexes:
          index_to_store = index
          state = gameState.getAgentState(index)
          for lad in enemies_to_reset_locations_for:
            if state == lad:
              enemy_indexes_to_reset.append(index_to_store)
      for enemy in enemy_indexes_to_reset:
        initPos = gameState.getInitialAgentPosition(enemy)
        initPos = [initPos[0], initPos[1]]
        self.reset_agent_probabilties_when_we_know_the_true_position(enemy, initPos)
    return kill

  def get_most_likely_distance_from_noisy_reading(self, enemy):
    index = self.getEnemyListIndex(enemy)
    listCopy = [i[1] for i in self.emission_probabilties_for_each_location_for_each_agent[index]]
    max_index = listCopy.index(max(listCopy))
    return tuple(self.emission_probabilties_for_each_location_for_each_agent[index][max_index][0])

  def updateNoisyDistanceProbabilities(self, mypos, enemy_we_are_checking, gameState):
    index = self.getEnemyListIndex(enemy_we_are_checking)
    distance_to_agents = gameState.getAgentDistances()
    distance_to_enemy = distance_to_agents[enemy_we_are_checking]
    counter = 0
    #Only check possible locations of the enemy in question
    for i in self.emission_probabilties_for_each_location_for_each_agent[index]:
      the_coordinates = tuple(i[0])
      distance = util.manhattanDistance(mypos, the_coordinates)
      location_probabilty = gameState.getDistanceProb(distance, distance_to_enemy)
      updated_probabilties_for_each_location = i[1] * location_probabilty
      self.emission_probabilties_for_each_location_for_each_agent[index][counter][1] = updated_probabilties_for_each_location
      counter += 1
