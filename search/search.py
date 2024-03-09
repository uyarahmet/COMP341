# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState()) 
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    
    from util import Stack
  
    frontier = Stack() # frontier will store states and their corresponding paths from start. 
    visited = [] # we will mark visited states to avoid infinite cycles.
    path = [] # we will keep track of the current path to determine each states path

    frontier.push([problem.getStartState(), []]) # [] denotes initial path, kept both values in a list. 

    while not frontier.isEmpty(): # this signals the termination condition, if the frontier is empty it means there is no solution

        state, path = frontier.pop() # the stack stores state and path values 

        if state not in visited:
            visited.append(state)
        else:
            continue # continue the loop if it's already visited! 

        if problem.isGoalState(state): # we check if the suffices the goal state 
            return path
        
        for neighbor in problem.getSuccessors(state): # if not, we continue with it's neighbors 
            if neighbor[0] not in visited: # check if visited 
                frontier.push([neighbor[0], path + [neighbor[1]]]) # add state and its new path to it
            
    
    

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""

    from util import Queue # We only change the data structure from stack to queue to change to BFS. 
  
    frontier = Queue() # frontier will store states and their corresponding paths from start. 
    visited = [] # we will mark visited states to avoid infinite cycles.
    path = [] # we will keep track of the current path to determine each states path

    frontier.push([problem.getStartState(), []]) # [] denotes initial path, kept both values in a list. 

    while not frontier.isEmpty(): # this signals the termination condition, if the frontier is empty it means there is no solution

        state, path = frontier.pop() # the queue stores state and path values 

        if state not in visited:
            visited.append(state)
        else:
            continue # continue the loop if it's already visited! 

        if problem.isGoalState(state): # we check if the suffices the goal state 
            return path
        
        for neighbor in problem.getSuccessors(state): # if not, we continue with it's neighbors 
            if neighbor[0] not in visited: # check if visited 
                frontier.push([neighbor[0], path + [neighbor[1]]]) # add state and its new path to it
        

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    
    from util import PriorityQueue # Now it is priority queue instead of normal queue.  
  
    frontier = PriorityQueue() # frontier will store states and their corresponding paths from start. 
    visited = [] # we will mark visited states to avoid infinite cycles.
    path = [] # we will keep track of the current path to determine each states path

    frontier.push([problem.getStartState(), [], 0], 0) # [] denotes initial path, 0 for starting value.  

    while not frontier.isEmpty(): # this signals the termination condition, if the frontier is empty it means there is no solution

        state, path, cost = frontier.pop() # the queue stores state and path values 

        if state not in visited:
            visited.append(state)
        else:
            continue # continue the loop if it's already visited! 

        if problem.isGoalState(state): # we check if the suffices the goal state 
            return path
        
        for neighbor in problem.getSuccessors(state): # if not, we continue with it's neighbors 
            if neighbor[0] not in visited: # check if visited 
                frontier.push([neighbor[0], path + [neighbor[1]], cost + neighbor[2]], cost + neighbor[2]) # add state, new path, and priority value.
                # not that we also store the priority value in our frontier so we can use it easily for its neighbor

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    #print(problem.getStartState()) (35, 1)
    #print(problem.getSuccessors(problem.getStartState())) [((35, 2), 'North', 1), ((34, 1), 'West', 1)]
   
    from util import PriorityQueue
  
    frontier = PriorityQueue() # frontier will store states and their corresponding paths from start. 
    visited = [] # we will mark visited states to avoid infinite cycles.
    path = [] # we will keep track of the current path to determine each states path

    frontier.push([problem.getStartState(), [], 0], 0 + heuristic(problem.getStartState(), problem)) # [] denotes initial path, 0 for starting value.  

    while not frontier.isEmpty(): # this signals the termination condition, if the frontier is empty it means there is no solution

        state, path, cost = frontier.pop() # the queue stores state and path values 

        if state not in visited:
            visited.append(state)
        else:
            continue # continue the loop if it's already visited! 

        if problem.isGoalState(state): # we check if the suffices the goal state 
            return path
        
        for neighbor in problem.getSuccessors(state): # if not, we continue with it's neighbors 
            if neighbor[0] not in visited: # check if visited 
                frontier.push([neighbor[0], path + [neighbor[1]], cost + neighbor[2]], cost + neighbor[2] + heuristic(neighbor[0], problem)) # add state, new path, and priority value.
                # not that the new priority value is f = g + h where g is the cumulative cost and h is the heuristic



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
