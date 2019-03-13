#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import numpy as np

class SARSAAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(SARSAAgent, self).__init__()
		self.learningRate = learningRate
		self.discountFactor = discountFactor
		self.epsilon = epsilon
		self.transition = {}
		self.previous_transition={}
		self.observation = None
		self.action=None
		self.table = {}
		self.reset()
	def learn(self):
		[previous_state,previous_action,previous_reward, pre_nextstate]=self.previous_transition
		[state, action, reward, nextState] = self.transition
		value = self.table[previous_state][previous_action]
		self.observation=nextState
		next_action=self.act()
		self.obervation = nextState
		self.table[previous_state][previous_action] += self.learningRate * (previous_reward
														  + self.discountFactor * self.table[state][action]
														  - self.table[previous_state][previous_action])
		return self.table[state][action] - value

	def act(self):
		if np.random.uniform() < self.epsilon:
			action = self.possibleActions[np.argmax(self.table[self.observation])]
		else:
			action = self.possibleActions[np.random.randint(5)]
		self.action=action
		return action

	def setState(self, state):
		self.observation=state
	def setExperience(self, state, action, reward, status, nextState):
		self.previous_transition=self.transition
		self.transition=[state,action,reward,nextState]

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		return self.learningRate,self.epsilon

	def toStateRepresentation(self, state):
		raise NotImplementedError

	def reset(self):
		for x in range(5):
			for y in range(6):
				self.table[(x, y)] = {}
				for action in self.possibleActions:
					self.table[(x, y)][action] = 0
		self.table['GOAL'] = {}
		self.table['OUT_OF_BOUNDS'] = {}
		for action in self.possibleActions:
			self.table['GOAL'][action] = 0
			self.table['OUT_OF_BOUNDS'][action] = 0


	def setLearningRate(self, learningRate):
		return self.learningRate

	def setEpsilon(self, epsilon):
		return self.epsilon
if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=500)

	args=parser.parse_args()
	
	numEpisodes = args.numEpisodes
	# Initialize connection to the HFO environment using HFOAttackingPlayer
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a SARSA Agent
	agent = SARSAAgent(0.1, 0.99,1.0)

	# Run training using SARSA
	numTakenActions = 0 
	for episode in range(numEpisodes):	
		agent.reset()
		status = 0

		observation = hfoEnv.reset()
		nextObservation = None
		epsStart = True

		while status==0:
			learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			agent.setLearningRate(learningRate)
			
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1

			nextObservation, reward, done, status = hfoEnv.step(action)
			print(obsCopy, action, reward, nextObservation)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			
			if not epsStart :
				agent.learn()
			else:
				epsStart = False
			
			observation = nextObservation

		agent.setExperience(agent.toStateRepresentation(nextObservation), None, None, None, None)
		agent.learn()

	
