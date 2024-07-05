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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Base score from the current game state
        currentScore = successorGameState.getScore()

        if action == Directions.STOP:
            currentScore -= 100

        ghostPenalty = 0
        for ghostState in newGhostStates:
            distance = manhattanDistance(newPos, ghostState.configuration.getPosition())
            if ghostState.scaredTimer > 0:
                if distance > 0:
                    ghostPenalty += 200 / distance
            else:
                if distance < 2:
                    ghostPenalty += -1000

        foodReward = 0
        foodList = newFood.asList()

        if(foodList):
            minFoodDistance = min(manhattanDistance(newPos, food) for food in foodList)
            foodReward += 10 / minFoodDistance

        foodCountPenalty = len(foodList) * 10

        currentScore += ghostPenalty + foodReward - foodCountPenalty

        return currentScore

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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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

        _, move = self._getMaxAgentAction(0, self.depth, gameState)
        print(move)
        return move

    def _getMaxAgentAction(self, agentIndex: int, depth: int, gameState: GameState):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), 'NoMove'

        moves = gameState.getLegalActions(agentIndex)
        if Directions.STOP in moves:
            moves.remove(Directions.STOP)

        scoresWithMoves = []
        for move in moves:
            scoresWithMoves.append((self._getMinAgentAction(1, depth, gameState.generateSuccessor(agentIndex, move)), move))

        maxMove = max(scoresWithMoves)
        return maxMove[0][0], maxMove[1]

    def _getMinAgentAction(self, agentIndex: int, depth: int, gameState: GameState):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), 'NoMove'

        numAgents = gameState.getNumAgents()
        moves = gameState.getLegalActions(agentIndex)
        scoresWithMoves = []

        if agentIndex == numAgents - 1:
            for move in moves:
                scoresWithMoves.append(
                    (self._getMaxAgentAction(0, depth - 1, gameState.generateSuccessor(agentIndex, move)), move))
        else:
            for move in moves:
                scoresWithMoves.append(
                    (self._getMinAgentAction(agentIndex + 1, depth, gameState.generateSuccessor(agentIndex, move)), move))

        minMove = min(scoresWithMoves)
        return minMove[0][0], minMove[1]


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
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
