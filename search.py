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
from util import *

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

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    "*** YOUR CODE HERE ***"

    statesVisited = []
    fringe = Stack()
    
    if problem.isGoalState(problem.getStartState()):
        return []

    fringe.push((problem.getStartState(), []))
    
    while not fringe.isEmpty():
        currState, currPath = fringe.pop()

        if problem.isGoalState(currState):
            return currPath

        if (currState not in statesVisited):
            statesVisited.append(currState)
            successors = problem.getSuccessors(currState)
            if successors:
                for succ in successors:        
                    newPath = currPath + [succ[1]]
                    fringe.push((succ[0], newPath))
    
    return []


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    # queue structure: a list of tuples
    # tuple structure: (state, list representing path to state)
    queue = Queue()

    statesVisited = [] # a list of states visited
    resultPath = []    # path to state

    # Base Case
    if problem.isGoalState(problem.getStartState()):
        return []

    # queue will contain currentNode and currentPath to that node
    queue.push((problem.getStartState(), resultPath))

    # continue itterating till queue of nodes is empty or goal is reached
    while not queue.isEmpty():

        # grab the next set of node data
        currNode, currPath = queue.pop()

        # list of nodes visited
        statesVisited.append(currNode)

        # if we have found a solution then return
        if problem.isGoalState(currNode):
            return currPath

        # now get a list of successors
        successors = problem.getSuccessors(currNode)

        # if there are successors
        if successors:
            # for every successor found
            for succ in successors:
                # use successor coords to check against list of visited coords
                if succ[0] not in statesVisited:
                    statesVisited.append(succ[0])
                    newResult = currPath + [succ[1]]
                    queue.push((succ[0], newResult))
    
    # return empty list if no new path is found && queue isEmpty()
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    # queue holds a tuple containing the node, the path taken to the node, and the cost to get to the node on that path.
    # it also holds a priority (annoyingly, the same as the third element of the tuple). Min-heap, not max.
    # queue = PriorityQueue()
    # queue.push((problem.getStartState(), [], 0), 0)
    # # We don't need to bother looking at nodes we've already pulled out of the queue, we are guaranteed to have
    # # taken the best path to it already.
    # visited = [problem.getStartState()]

    # while not queue.isEmpty():
    #     current, path, cost = queue.pop()
    #     if problem.isGoalState(current):
    #         return path

    #     for neighbor in problem.getSuccessors(current):
    #         neighborState = neighbor[0]
    #         neighborPath = path + [neighbor[1]]
    #         neighborCost = cost + neighbor[2]
    #         if neighborState not in visited:
    #             visited.append(neighborState)
    #             queue.update((neighborState, neighborPath, neighborCost), neighborCost)

    fringe = PriorityQueue()
    statesVisited = []

    fringe.push((problem.getStartState(), [], 0), 0)

    while not fringe.isEmpty():
        currState, currPath, cost = fringe.pop()
        
        if problem.isGoalState(currState):
            return currPath
        
        if currState not in statesVisited:
            statesVisited.append(currState)
            successors = problem.getSuccessors(currState)
            if successors:
                for succ in successors:
                    newPath = currPath + [succ[1]]
                    fringe.update((succ[0], newPath, cost + succ[2]), cost + succ[2])

    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # queue holds a tuple containing the node, the path taken to the node, and the cost to get to the node on that path.
    # it also holds a priority, which is path cost plus the heuristic. Min-heap, not max.
    # queue = PriorityQueue()
    # current = problem.getStartState()
    # queue.push((current, [], 0), heuristic(current, problem))
    # # We don't need to bother looking at nodes we've already pushed into the queue, we are guaranteed to have
    # # taken the best path to it already.
    # visited = [problem.getStartState()]

    # while not queue.isEmpty():
    #     current, path, cost = queue.pop()
    #     if problem.isGoalState(current):
    #         return path

    #     for neighbor in problem.getSuccessors(current):
    #         neighborState = neighbor[0]
    #         neighborPath = path + [neighbor[1]]
    #         neighborCost = cost + neighbor[2]
    #         if neighborState not in visited:
    #             visited.append(neighborState)
    #             queue.update((neighborState, neighborPath, neighborCost),
    #                          neighborCost + heuristic(neighborState, problem))

    fringe = PriorityQueue()
    statesVisited = []

    fringe.push((problem.getStartState(), [], 0), 0)

    while not fringe.isEmpty():
        currState, currPath, cost = fringe.pop()
        
        if problem.isGoalState(currState):
            return currPath
        
        if currState not in statesVisited:
            statesVisited.append(currState)
            successors = problem.getSuccessors(currState)
            if successors:
                for succ in successors:
                    newPath = currPath + [succ[1]]
                    fringe.update((succ[0], newPath, cost + succ[2]), cost + succ[2] + heuristic(succ[0], problem))

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
