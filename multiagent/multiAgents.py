# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
        food_distances = [manhattanDistance(food, newPos) for food in newFood.asList()] # successor food distances 
        ghost_distances = [manhattanDistance(ghost.getPosition(), newPos) for ghost in newGhostStates] # successor ghost positions 
        current_state = currentGameState.getPacmanPosition() # current pacman state 

        if len(food_distances) == 0: # all food pellets have been collected                                                 
            return 100  

        if current_state == newPos: # avoid staying still 
            return -100 

        for ghost_distance in ghost_distances: # avoid getting closer to ghosts                                                
            if ghost_distance <= 1 and min(newScaredTimes) == 0: # run away if ghosts aren't scared 
                return -100
                                               
        remaining_foods = len(food_distances) # number of remaining food pellets.
        total_distance_to_foods = sum(food_distances) # total distance to all remaining food pellets. I added this one because remaining_foods was not enough itself and pacman got stuck.            

        #return 1 / remaining_foods                                                               
        return 1 / remaining_foods + 1 / total_distance_to_foods # we add there reciprocals to reward states that the two values are smaller. 



def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def is_terminal_state(gameState, agentIndex, depth): # terminal state checker 
            
            return len(gameState.getLegalActions(agentIndex)) == 0 or gameState.isWin() or gameState.isLose() or depth == self.depth
                
        
        def max_value(gameState, depth):
    
            if is_terminal_state(gameState, 0, depth):
                return self.evaluationFunction(gameState), None
            
            v = float("-inf") # initialize -inf for max 
            next_action = None

            for action in gameState.getLegalActions(0): # iterate all the actions 
                successor_state = gameState.generateSuccessor(0, action)
                successor_value = min_value(successor_state, 1, depth)[0] # call min for specific branch 
                if successor_value > v:
                    v = successor_value # find the maximum of all branches 
                    next_action = action # the required action to go there 

            return v, next_action
        
        def min_value(gameState, agentIndex, depth):

            if is_terminal_state(gameState, agentIndex, depth):
                return self.evaluationFunction(gameState), None 

            v = float("inf") # initialize +inf for min 
            next_action = None

            for action in gameState.getLegalActions(agentIndex): # iterate all the actions
                successor_state = gameState.generateSuccessor(agentIndex, action)

                if agentIndex == gameState.getNumAgents() - 1: # this is pacman! so we max 
                    successor_value = max_value(successor_state, depth + 1)[0]
                else: # rest are ghosts, we min 
                    successor_value = min_value(successor_state, agentIndex + 1, depth)[0]

                if successor_value < v:
                    v = successor_value # find minimum of all branches 
                    next_action = action # the required action to go there

            return v, next_action
        
        next_action = max_value(gameState, 0)[1]
        return next_action
                    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacman_position = currentGameState.getPacmanPosition()
    ghost_positions = currentGameState.getGhostPositions()
    ghost_states = currentGameState.getGhostStates()
    food_list = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()

    current_score = scoreEvaluationFunction(currentGameState)

    if currentGameState.isWin():
        return 1000
    elif currentGameState.isLose():
        return -1000
    
    closest_food = min([manhattanDistance(food, pacman_position) for food in food_list])
    closest_scared_ghost = min([manhattanDistance(ghost, pacman_position) for ghost in ghost_positions if ghost.scaredTimer == 0])
    closest_normal_ghost = min([manhattanDistance(ghost, pacman_position) for ghost in ghost_positions if ghost.scaredTimer > 0])

    current_score -= 1.5 * closest_food + 2 * (1.0 / closest_normal_ghost) + 2 * closest_scared_ghost + 20 * len(capsules) + 4 * len(food_list)
    return current_score



# Abbreviation
better = betterEvaluationFunction
