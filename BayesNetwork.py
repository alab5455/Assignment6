#Name: Ali Abu AlSaud
#Date: 10/12/2015
#Assignemnt 6: Bayes Network for Disease Predictor

import getopt
import sys
import array

#variables to determine the direction of reasoning
NoEvidence = 0
predictiveReasoning = 1
diagnosticReasoning = 2
Neighbors = 3
#variables to determine the direction of more complex reasoning
interCausal = 4
combine = 5
neither = 6
natural = 7


class Node(object):
	
	def __init__(node, name, MPN):
		node.name = name
		node.probs = {}
		node.parents = {}
		node.children = {}
		node.marginalProbDone = False
		node.marginalProbName = MPN
		node.marginalProbability = None

	def addProbability(node, key, value):
		node.probs[key] = value

	def addParent(node, Node):
		node.parents[Node.name] = Node

	def addChild(node, Node):
		node.children[Node.name] = Node

	def changeProbability(node, key, value):
		node.probs[key] = value


class BayesNetwork(object):

	def __init__(node):
		node.nodes = {}

	def addNode(node, Node):
		node.nodes[Node.name] = Node

	def solveMarginalProbability(node, N):
		if (len(N.parents) == 0):
			result = N.probs.values()
			if (len(result) != 1):
				print ("There is an Error")
			else:
				N.marginalProbability = result[0]
				N.marginalProbDone = True
		else:
			RV_marg_prob = 0.0
			rvp_marg_probs = {}
			for rvp in N.parents.values():
				if (rvp.marginalProbDone == False):
					node.solveMarginalProbability(rvp)
				rvp_marg_probs[rvp.marginalProbName] = rvp.marginalProbability

			for (key, val) in N.probs.items():
				cur = 1.0
				negate = False
				for char in key:
					if (char == '~'):
						negate = True
					else:
						if (negate == False):
							cur *= rvp_marg_probs[char]
						else:
							negate = False
							cur *= (1 - rvp_marg_probs[char])
				RV_marg_prob += val * cur

			N.marginalProbability = RV_marg_prob
			N.marginalProbDone = True

	def calculateMarginalProbability(node):
		for i in node.nodes.values():
			if i.marginalProbDone == False:
				node.solveMarginalProbability(i)
				
	def update_probability(node, arg, value):
		Node = None

		if (arg == "P"):
			Node = node.nodes["Pollution"]

		elif (arg == "S"):
			Node = node.nodes["Smoker"]
		
		Node.changeProbability(arg, newValue)

		for i in node.nodes.values():
			i.marginalProbDone = False
		
		node.calculateMarginalProbability()
		
	def decisionOfDirection(node, N1, N2):
		if (N1.name == N2.name):
			return noEvidence
			
		queue = []
		#if N2 is above N1, this is predictive reasoning
		queue.append(N1)
		while(len(queue) > 0):
			for i in queue:
				if (i.name == N2.name):
					return predictiveReasoning
				queue.remove(i)
				for item in i.parents.values():
					queue.append(item)
					
		Queue = []
		queue.append(N2)
		#If N1 is above N2, this is Diagnostic Reasoning
		while (len(queue) > 0):
			for i in queue:
				if (i.name == N1.name):
					return diagnosticReasoning
				queue.remove(i)
				for item in i.parents.values():
					queue.append(item)
					
		return Neighbors

	def calculateConditionalProbability(node, N1, N2, status1, status2):
		#solve P(N1 | N2)
		evidence = node.decisionOfDirection(N1, N2)
		
		if (evidence == NoEvidence):
			return 1
			
		elif (evidence == predictiveReasoning):
			if(N1.probs.has_key(N2.marginalProbName)):
				if status1 == '~':
					return 1 - N1.probs[status2 + N2.marginalProbName]
				else:
					return N1.probs[status2 + N2.marginalProbName]
			else:
				if not (N1.parents.has_key(N2.names)):
					N = None
					for i in N1.parents.values():
						N = i
					N10 = node.calculateConditionalProbability(N1, N, status1, '')
					N02 = node.calculateConditionalProbability(N, N2, '', status2)
					negateN10 = node.calculateConditionalProbability(N1, N, status1, '~')
					negateN02 = 1 - negateN10
					return ((N10 * N02) + (negateN10 + negateN02))
					
				else:
					NProbToSum = None
					NMarginaleConditioning = N2.name
					for i in N1.parents.values():
						if (i.name != N2.name):
							NProbToSum = (i.marginalProbName, i.marginalProbability)
							
					x = ((Status2 + N2.marginalProbName + NProbtoSum[0], NProbToSum[0] + status2 + N2.marginalProbName), N2.marginalProbability * NProbToSum[1])
					y = ((Status2 + N2.marginalProbName + '~' + NProbtoSum[0], '~' + NProbToSum[0] + status2 + N2.marginalProbName), N2.marginalProbability * (1 - NProbToSum[1]))
					
					R1 = x[1] * N1.probs.get(x[0][0], N1.probs.get(x[0][1], False))
					R2 = y[1] * N1.probs.get(y[0][0], N1.probs.get(y[0][1], False))
					
					if status1 == '':
						return ((R1 + R2) / N2.marginalProbability)
					else:
						return 1 - ((R1 + R2) / N2.marginalProbability) 
					
				
				
		elif (evidence == diagnosticReasoning):
			#P(A|B) = P(B|A) * P(A) / P(B)
			x = node.calculateConditionalProbability(N2, N1, status2, '')
			x = x * N1.marginalProbability
			y = N2.marginalProbability
			if(status2 == '~'):
				y = 1 - y
			x = x / y
			if(status1 == '~'):
				return (1-x)
			else:
				return x
			
		elif (evidence == Neighbors):
			commonParent = False
			N3 = None
			for Np1 in N1.parents.values():
				for Np2 in N2.parents.values():
					if Np1 == Np2:
						N3 = Np1
						commonParent = True
						break
			if(commonParent):
				N13 = node.calculateConditionalProbability(N1, N3, status1, '')
				N32 = node.calculateConditionalProbability(N3, N2, '', status2)
				negativeN13 = node.calculateConditionalProbability(N1, N3, status1, '~')
				negativeN32 = node.calculateConditionalProbability(N3, N2, '~', status2)
				
				return ((N13 * N32) + (negativeN13 * negativeN32))
				
			else:
				if (status1 == '~'):
					return 1 - N1.marginalProbability
				return N1.marginalProbability 
	
	def calculateJointProbability(node, N1, N2, status1, status2):
		if status2 == '':
			marginalProb = N2.marginalProbability
		else:
			marginalProb = 1 - N2.marginalProbability
		return marginalProb * node.calculateConditionalProbability(N1, N2, status1, status2)
	
	def evidenceWithMoreEvidence(node, N1, N2, N3):
		if (N1 == N2):
			return (natural, 2)
		elif (N1 == N3):
			return (natural, 3)
			
		#Intercausal rel
		if (N2.parents.has_key(N3.name)):
			return (interCausal, N2)
		elif (N3.parents.has_key(N2.name)):
			return (interCausal, N3)
			
		#combined rel
		arrow = node.decisionOfDirection(N2, N3)
		if (arrow != neighbors):
			if (len(N1.children.items()) == 0 and len(N3.children.items()) == 0):
				return (combine, (1, 3))
			elif (len(N1.children.items()) == 0 and len(N2.children.items()) == 0):
				return (combine, (1, 2))
			elif (len(N2.children.items()) == 0):
				return (combine, 2)
			else:
				return (combine, 3)
		
		#neither
		return (neither, None)
						
	def calculateConditionalonJoint(node, N1, N2, N3, status1, status2, status3):
		Narray = [N1, N2, N3]
		Nstatus = [status1, status2, status3]
		evidence = node.evidenceWithMoreEvidence(N1, N2, N3)
		#If P(c)|s,p)
		if (N1.probs.has_key(status2 + N2.marginalProbName + status3 + N3.marginalProbName) or N1.probs.has_key(status3 + N3.marginalProbName + status2 + N2.marginalProbName)):
			if (status1 == '~'):
				return 1 - N1.probs.get(status2 + N2.marginalProbName + status3 + N3.marginalProbName, N1.probs.get(status3 + N3.marginalProbName + status2 + N2.marginalProbName, False))
			else:
				return N1.probs.get(status2 + N2.marginalProbName + status3 + N3.marginalProbName, N1.probs.get(status3 + N3.marginalProbName + status2 + N2.marginalProbName, False))
		
		elif(evidence[0] == natural):
			if (status1 == Nstatus[evidence[1] - 1]):
				return 1.0
	
			return 0.0
				
		elif (evidence[0] == interCausal):
			Nroot = evidence[1]
			#P(X | c,s ) = P (x|c)
			if (Nroot == N2):
				if (N2.parents.has_key(N3.name) and N2.childern.has_key(N1.name)):
					return node.calculateConditionalProbability(N1, N2, status1, status2)
				elif (N2.parents.has_key(N3.name) and N3.childern.has_key(N1.name)):
					return node.calculateConditionalProbability(N1, N3, status1, status3)
			elif (Nroot == N3):
				if (N3.parents.has_key(N2.name) and N3.children.has_key(N1.name)):
					return node.calculateConditionalProbability(N1, N3, status1, status3)
				elif (N3.parents.has_key(N2.name) and N2.children.has_key(N1.name)):
					return node.calculateConditionalProbability(N1, N2, status1, status2)
					
			# P(P | c,s)
			notNroot = []
			NrootIdentity = None
			
			for i in range(len(Narray)):
				if Narray[i] == Nroot:
					NrootIdentitty = i
				else:
					notNroot.append(i)
			
			NrootParentEvidence = None
			if (NrootIdentity == 2):
				NrootParentEvidence = 1
			else:
				NrootParentEvidence = 2
				
			if (status1 == ''):
				prob = N1.marginalProbability
				
			else:
				prob = 1 - N1.marginalProbability
			
			mult = Nroot.probs.get(Nstatus[notNroot[0]] + Narray[notNroot[0]].marginalProbName + Nstatus[notNroot[1]] + Narray[notNroot[1]].marginalProbName, Nroot.probs.get(Nstatus[notNroot[1]] + Narray[notNroot[1]].marginalProbName + Nstatus[notNroot[0]] + Narray[notNroot[0]].marginalProbName, False))
			prob = prob * mult
			prob = prob / (node.calculateConditionalProbability(Nroot, Narray[NrootParentEvidence], Nstatus[NrootIdentity], Nstatus[NrootParentEvidence]))
			
			return prob
			
		elif (evidence[0] == combine):
			# P(D}X,P)
			if isinstance(evidence[1], tuple):
				NrootIdentity1 = evidence[1][0] - 1
				Nparents = N1.parents.values()
				NP = Nparents[0]
				NrootIdentity2 = evidence[1][1] - 1
				N2root = Narrau[NrootIdentity2]
				otherEvidence = None
				
				if (NrootIdentity2 == 2):
					otherEvidence = 1
				elif (NrootIdentity2 == 1):
					otherEvidence = 2
					
				N1GivenNP = node.calculateConditionalProbability(N1, NP, status1, '')
				NrootGivenNP = node.calculateConditionalProbability(N2root, NP, Nstatus[NrootIdentity2], '')
				N1GivenNotNP = node.calculateConditionalProbability(N1, NP, status1, '~')
				NrootGivenNotNP = node.calculateConditionalProbability(N2root, NP, Nstatus[NrootIdentity2], '~')
				NOther = node.calculateConditionalProbability(NP, Narray[otherEvidence], '', Nstatus[otherEvidence])
				notNOther = 1 - NOther
				NrootOther = node.calculateConditionalProbability(N2root, Narray[otherEvidence], Nstatus[NrootIdentity2], Nstatus[otherEvidence])
				
				return ((N1GivenNP * NrootGivenNP * NOther) + (N1GivenNotNP * NrootGivenNotNP * notNOther)) / NrootOther
			
			#default Case
			NrootIdentity = evidence[1] - 1
			otherEvidence = None
			if (NrootIdentity == 1):
				otherEvidence = 2
			elif (NrootIdentity == 2):
				otherEvidence = 1
			
			#Using Bayes Theorem:
			rootGivenS = node.calculateConditionalonJoint(Narray[NrootIdentity], Narray[(NrootIdentity + 1) % 3], Narray[(NrootIdentity + 2) % 3], Nstatus[NrootIdentity], Nstatus[(NrootIdentity + 1) % 3], Nstatus[(NrootIdentity + 2) % 3])
			rootGivenOther = node.calculateConditionalProbability(Narray[NrootIdentity], Narray[otherEvidence], Nstatus[NrootIdentity], Nstatus[OtherEvidence])
			
			prob = node.calculateConditionalProbability(N1, Narray[otherEvidence], status1, Nstatus[OtherEvidence])
			prob = (prob * rootGivenS) / rootGivenOther
			return prob
			
		elif (evidence[0] == neither):
			Nparents = N1.parents.values()
			NP = Nparents[0]
			
			N1GivenNP = node.calculateConditionalProbability(N1, NP, status1, '')
			N1GivenNotNP = node.calculateConditionalProbability(N1, NP, status1, '~')
			NPGivenN2N3 = node.calculateConditionalonJoint(NP, N2, N3, '', status2, status3)
			notNPGivenN2N3 = 1 - NPGivenN2N3
			
			return ((N1GivenNP * NPGivenN2N3) + (N1GivenNotNP * notNPGivenN2N3)) / (NPGivenN2N3 + notNPGivenN2N3)
			
	def calculateJointProbabilityWithThreeVariables(node, N1, N2, N3, status1, status2, status3):
		N1GivenN2N3 = node.calculateConditionalonJoint(N1, N2, N3, status1, status2, status3)
		N2GivenN3 = node.calculateConditionalProbability(N2, N3, status2, status2)
		jointProbability = N3.marginalProbability * N1GivenN2N3 * N2GivenN3
		if (status3 == ''):
			return jointProbability
		else:
			return 1 - jointProbability
			
	def nodeLooking(node, Node):
		for i in node.nodes.values():
			if (i.marginalProbName == Node.upper()):
				return i
		print("Invalid Node")
		sys.exit(2)
		
	def recursiveCombinations(node, Node, table):
		base = True
		for i in Node:
			if i.isupper():
				key1 = Node.replace(i, i.lower())
				i = Node.replace(i, '~' + i.lower())
				if not table.has_key(key1):
					node.recursiveCombinations(key1, table)
				if not table.has_key(i):
					node.recursiveCombinations(i, table)
				base = False
		if (base):
			table[Node] = False
			
	
		
	def bayesNetworkQuery(node, flag, Node):
		Node = Node.replace('/', '|')
		if (flag == '-m'):
			p = Node.find('~')
			if (len(Node) > 1 and p == -1):
				print("-m doesn't work with two variables")
				system.exit(2)
				
			printing = None
			NodeUpper = Node.isupper()
			if(NodeUpper):
				a = node.nodeLooking(Node)
				printing = a.marginalProbability
				print("Marginal Probability of"),Node
				print (Node.lower() + ":", printing, '~' + Node.lower() + ':', 1 - printing)
			else:
				if '~' in Node:
					a = node.nodeLooking(Node[1])
					printing = 1 - a.marginalProbability
				else:
					a = node.nodeLooking(Node)
					printing = a.marginalProbability
				print ("Marginal Probability of ", a + ":", printing)
		
		elif (flag == '-p'):
			node.update_probability(Node[0], float(Node[1:]))
			print ("New Probability of ", a[0].lower(), " is ", float(Node[1:]))
			
		elif (flag == '-g'):
			variation = {}
			node.recursiveCombinations(Node, variation)
			for i in variation.keys():
				node.conditionalHelper(flag, i)
				
		elif (flag == '-j'):
			variation = {}
			node.recursiveCombinations(Node, variation)
			for i in variation.keys():
				node.jointHelper(flag, i)
				
		else:
			assert False

	def conditionalHelper(node, flag, Node):
		p = Node.find("|")
		if (p == -1):
			print ("This is not a Conditional Probability")
			sys.exit(2)
			
		N1 = Node[:p]
		N1status = ''
		if ('~' in N1):
			N1 = N1[1]
			N1status = '~'
			
		conditionalNarray = []
		nconditionalNstatus = []
		
		positive = True
		for i in Node[p + 1: ]:
			if i == '~':
				positive = False
			else:
				conditionalNarray.append(i)
				if positive == False:
					positive = True
					conditionalNstatus.append('~')
				else:
					conditionalNstatus.append('')
					
		if (len(conditionalNarray) == 1):
			N1 = node.nodeLooking(N1)
			N2 = node.nodeLooking(conditionalNarray[0])
			print ("Probability of ", Node, " is ", round(node.calculateConditionalProbability(N1, N2, status1, conditionalNstatus[0]), 3))
		elif (len(conditionalNarray) == 2):
			N1 = node.nodeLooking(N1)
			N2 = node.nodeLooking(conditionalNarray[0])
			N3 = node.nodeLooking(conditionalNarray[1])
			print ("Probability of ", Node, " is ", round(node.calculateConditionalonJoint(N1, N2, N3, status1, conditionalNstatus[0], conditionalNstatus[1]), 3))
			
	def jointHelper(node, flag, Node):
		p = Node.find('|')
		if p > - 1:
			print("This is not a Joint Probability")
			sys.exit(2)
		
		Narray = []
		Nstatus = []
		positive = True
		for i in Node:
			if i == '~':
				positive = False
			else:
				Narray.append(i)
				if positive == False:
					positive = True
					Nstatus.append('~')
				else:
					Nstatus.append('')
					
		if (len(Narray) < 2 or len(Narray) > 3):
			print("Not valid command")
			sys.exit(2)
		elif (len(Narray) == 2):
			N1 = node.nodeLooking(Narray[0])
			N2 = node.nodeLooking(Narray[1])
			print ("Probability of ", Node, " is ", round(node.calculateJointProbability(N1, N2, Nstatus[0], Nstatus[1]), 3))
		elif (len(Narray) == 3):
			N1 = node.nodeLooking(Narray[0])
			N2 = node.nodeLooking(Narray[1])
			N3 = node.nodeLooking(Narray[2])
			print ("Probability of ", Node, " is ", round(node.calculateJointProbabilityWithThreeVariables(N1, N2, N3, Nstatus[0], Nstatus[1], Nstatus[2]), 3))
		
def buildBayesNetwork():
	P = Node('Pollution', 'P')
	P.addProbability('P', 0.9)
	S = Node('Smoker', 'S')
	S.addProbability('S', 0.3)
	C = Node('Cancer', 'C')
	C.addProbability('~PS', 0.05)
	C.addProbability('~P~S', 0.02)
	C.addProbability('PS', 0.03)
	C.addProbability('P~S', 0.001)
	X = Node('XRay', 'X')
	X.addProbability('C', 0.9)
	X.addProbability('~C', 0.2)
	D = Node('Dyspnoea', 'D')
	D.addProbability('C', 0.65)
	D.addProbability('~C', 0.3)

	C.addParent(P)
	P.addChild(C)
	C.addParent(S)
	S.addChild(C)
	X.addParent(C)
	C.addChild(X)
	D.addParent(C)
	C.addChild(D)

	BayesNet = BayesNetwork()
	BayesNet.addNode(P)
	BayesNet.addNode(S)
	BayesNet.addNode(C)
	BayesNet.addNode(X)
	BayesNet.addNode(D)
	BayesNet.calculateMarginalProbability()

	return BayesNet


flags = ':g:j:m:p:'

if __name__ == "__main__":
	try:
		opts, args = getopt.getopt(sys.argv[1:], flags)
	except getopt.GetoptError as err:
		print str(err)
		sys.exit(2)

	bayes_net = buildBayesNetwork()
	for o, a in opts:
		bayes_net.bayesNetworkQuery(o, a)
