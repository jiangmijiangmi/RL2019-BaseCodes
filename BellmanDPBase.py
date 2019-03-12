from MDP import MDP
import numpy as np
class BellmanDPSolver(object):
	def __init__(self, discountRate):
		self.MDP = MDP()
		self.states=MDP().S
		self.action=MDP().A
		self.discountRate=discountRate
		self.Values={}
		self.Values['GOAL']=0
		self.Values['OUT']=0
		self.Policy={}
		self.initVs()

	def initVs(self):
		for x in range(5):
			for y in range(5):
				self.Values[(x,y)]=0
				self.Policy[(x,y)]=["DRIBBLE_UP","DRIBBLE_DOWN","DRIBBLE_LEFT","DRIBBLE_RIGHT","SHOOT"]

	def BellmanUpdate(self):
		values={}
		for i in range(5):
			for j in range(5):
 				values[(i,j)]=-np.inf   
				for action in self.action:
					tmp_values=0.0
					nextstateprob=self.MDP.probNextStates((i,j), action)
					for nextstate in nextstateprob.keys():
						reward = self.MDP.getRewards((i,j),action,nextstate)
						prob=nextstateprob[nextstate]
						tmp_values=tmp_values+(prob*(reward + self.discountRate * self.Values[nextstate]))
					if values[(i,j)]<tmp_values:
						values[(i,j)]=tmp_values
 						self.Policy[(i,j)]=[action]
					elif values[(i,j)]==tmp_values:
						self.Policy[(i,j)].append(action)
				self.Values[(i,j)]=values[(i,j)]                          
		return self.Values,self.Policy
		
        
if __name__ == '__main__':
	solution = BellmanDPSolver(0.9)
	for i in range(200):
		values, policy = solution.BellmanUpdate()
	print("Values : ", values)
	print("Policy : ", policy)

