from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse

class QLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(QLearningAgent, self).__init__()
		self.learningRate=learningRate
		self.discountFactor=discountFactor
		self.epsilon=epsilon
		self.agent=Agent()
		self.action=Agent().possibleActions
		self.transition={}
		self.states={}
		self.observation={}
		self.table={}
		for x in range (6):
			for y in range(5):
				self.table[(x,y)]={}
				for action in self.action:
					self.table[(x,y)][action]=0
        


	def learn(self):
        
		[state, action,reward,nextState]=self.transition
		self.obervation=nextState
		next_action=self.act()
		self.table[state][action] += self.learningRate * (reward
                                                            + self.discountFactior * self.table[nextState][next_action] 
                                                            - self.table[state][action])

	def act(self):
                                                     
		if np.random.uniform()<self.epsilon:
			action=np.argmax(self.table[self.observation])
		else:
			action=self.action[np.random.randint(5)]
		return action

                                                     
                                                     
	def toStateRepresentation(self, state):
		return state[0]

	def setState(self, state):
         
		self.observation=state

	def setExperience(self, state, action, reward, status, nextState):
		self.transition=[state, action, reward, nextState]
                                                     
	def setLearningRate(self, learningRate):
		return learningRate

	def setEpsilon(self, epsilon):
		return epsilon

	def reset(self):
		raise NotImplementedError
		
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		return self.learningRate,self.epsilon

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=500)

	args=parser.parse_args()

	# Initialize connection with the HFO server
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a Q-Learning Agent
	agent = QLearningAgent(learningRate = 0.1, discountFactor = 0.99, epsilon = 1.0)
	numEpisodes = args.numEpisodes

	# Run training using Q-Learning
	numTakenActions = 0 
	for episode in range(numEpisodes):
		status = 0
		observation = hfoEnv.reset()
		
		while status==0:
			learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			agent.setLearningRate(learningRate)
			
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1
			
			nextObservation, reward, done, status = hfoEnv.step(action)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			update = agent.learn()
			
			observation = nextObservation
	

