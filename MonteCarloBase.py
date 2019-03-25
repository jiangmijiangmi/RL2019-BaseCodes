#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse

class MonteCarloAgent(Agent):
	def __init__(self, discountFactor, epsilon, initVals=0.0):
		super(MonteCarloAgent, self).__init__()
		self.discountFactor=discountFactor
		self.epsilon=epsilon
		self.transition={}
		self.observationn={}
		self.qtable={}
		self.returns={}
		self.pi={}
		for x in range(5):
			for y in range(6):
				self.qtable[(x, y)] = {}
				for action in self.possibleActions:
    				self.pi[(x, y)][action]=1/length(self.possibleActions)
					self.qtable[(x, y)][action] = 0
					self.returns[(x,y)][action]=0
		self.qtable['GOAL'] = {}
		
		for action in self.possibleActions:
			self.qtable['GOAL'][action] = 0
			self.qtable['OUT_OF_BOUNDS'][action] = 0
		



	def learn(self):
		T=self.transition.shape(0)
		[S,A,R]=self.transition
		pair=[S,A]
		t=T-1
		while t >=0:
    		G=self.discountFactor*G+R[t+1]
			if [S[t],A[t]] not in pair[0:t-1]:
				self.returns[S[t]][A[t]].append(G)
				self.qtable[S[t]][A[t]]=mean(self.returns[S[t]][A[t]])
				value=-10000
				for act in self.possibleAction:
    				if value<Q(S[t],a):
    					value=Q(S[t],a)
						A_star=a
    			for act in self.possibleActions:
    					if act==A_star:
    						self.pi[S[t]][act]=1-self.epsilon+self.epsilon/length(self.possibleActions)
						else:
    						self.pi[S[t]][act]=self.epsilon/length(self.possibleActions)


	def toStateRepresentation(self, state):
		if state in ['GOAL', 'OUT_OF_BOUNDS']:
			return state
		else:
			return state[0]

	def setExperience(self, state, action, reward, status, nextState):
		self.transition.append([state,action,reward])

	def setState(self, state):
		self.observation=state

	def reset(self):
		raise NotImplementedError

	def act(self):
		if np.random.uniform()<self.epsilon:
    		action=np.argmax(self.table[self.observation])
		else:
			action=self.action[np.random.randint(5)]
		return action


	def setEpsilon(self, epsilon):
		return epsilon

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		return self.epsilon


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=500)

	args=parser.parse_args()

	#Init Connections to HFO Server
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a Monte-Carlo Agent
	agent = MonteCarloAgent(discountFactor = 0.99, epsilon = 1.0)
	numEpisodes = args.numEpisodes
	numTakenActions = 0
	# Run training Monte Carlo Method
	for episode in range(numEpisodes):	
		agent.reset()
		observation = hfoEnv.reset()
		status = 0

		while status==0:
			epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1
			nextObservation, reward, done, status = hfoEnv.step(action)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			observation = nextObservation

		agent.learn()
